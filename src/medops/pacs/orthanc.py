"""Orthanc REST client + the "1 Study = 1 Series" guard.

Mapping: case_id == DICOM PatientID (the anonymized ID stored in Orthanc). Nothing but
identifiers and series-level technical tags ever leaves this module; PatientName etc. are
never read into return values.
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import requests

from medops.gcp_auth import IdTokenAuth
from medops.ids import validate_case_id
from medops.settings import Settings

EXCLUDED_IMAGE_TYPES = {"LOCALIZER", "SECONDARY", "DERIVED"}


class PacsError(RuntimeError):
    pass


class CaseNotFoundError(PacsError):
    pass


class SeriesSelectionError(PacsError):
    """Zero or several candidate series: we never guess which one to analyze."""

    def __init__(self, message: str, series: list["SeriesInfo"]):
        self.series = series
        listing = "\n".join(
            f"  - {s.series_instance_uid}: {s.modality}, {s.num_instances}枚, "
            f"SliceThickness={s.slice_thickness}, ImageType={s.image_type}"
            f"{'' if s.is_candidate else '（候補外）'}"
            for s in series
        ) or "  （シリーズなし）"
        super().__init__(f"{message}\n見つかったシリーズ:\n{listing}")


@dataclass
class SeriesInfo:
    orthanc_id: str
    series_instance_uid: str
    modality: str
    num_instances: int
    slice_thickness: Optional[str]
    image_type: str
    series_description: Optional[str] = None

    @property
    def is_candidate(self) -> bool:
        types = {t.strip().upper() for t in self.image_type.split("\\") if t.strip()}
        return self.modality == "CT" and not (types & EXCLUDED_IMAGE_TYPES)

    def to_public(self) -> dict:
        d = asdict(self)
        d.pop("orthanc_id")
        d["is_candidate"] = self.is_candidate
        return d


@dataclass
class StudyInfo:
    orthanc_id: str
    study_instance_uid: str
    study_date: Optional[str]
    series: list[SeriesInfo] = field(default_factory=list)

    def to_public(self) -> dict:
        return {
            "study_instance_uid": self.study_instance_uid,
            "study_date": self.study_date,
            "series": [s.to_public() for s in self.series],
        }


def iso_date(dicom_date: Optional[str]) -> Optional[str]:
    """DICOM DA 'YYYYMMDD' -> 'YYYY-MM-DD' (None if absent or malformed)."""
    d = (dicom_date or "").strip()
    if len(d) != 8 or not d.isdigit():
        return None
    try:
        return date(int(d[:4]), int(d[4:6]), int(d[6:])).isoformat()
    except ValueError:
        return None


def study_entry(case_id: str, study_uid: str, study_date: Optional[str],
                modalities: list[str], instance_counts: list[int]) -> dict:
    return {"case_id": case_id, "study_instance_uid": study_uid,
            "study_date": iso_date(study_date),
            "modalities": sorted({m for m in modalities if m}),
            "series_count": len(instance_counts), "image_count": sum(instance_counts)}


def select_series(study: StudyInfo) -> SeriesInfo:
    """Exactly one candidate (CT, not LOCALIZER/SECONDARY/DERIVED) or an error. Never guess."""
    candidates = [s for s in study.series if s.is_candidate]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SeriesSelectionError(
            f"Study {study.study_instance_uid} に解析対象となる CT シリーズがありません。", study.series)
    raise SeriesSelectionError(
        f"Study {study.study_instance_uid} に候補シリーズが {len(candidates)} 件あります。"
        "現在は 1 Study = 1 Series のデータのみ対応しており、推測での選択はしません。",
        study.series)


class OrthancClient:
    def __init__(self, settings: Settings, timeout: float = 30.0, auth=None):
        self.base = settings.orthanc_url
        self.timeout = timeout
        self.session = requests.Session()
        if auth is not None:            # e.g. deploy tooling uploading to the cloud PACS
            self.session.auth = auth
        elif settings.orthanc_user:     # local Orthanc (docker compose): basic auth
            self.session.auth = (settings.orthanc_user, settings.orthanc_password or "")
        else:                           # Cloud Run PACS: IAM-protected, ID token
            self.session.auth = IdTokenAuth()

    # -- low level ------------------------------------------------------------------

    def _get(self, path: str, **kw):
        try:
            r = self.session.get(f"{self.base}{path}", timeout=self.timeout, **kw)
        except requests.RequestException as exc:
            raise PacsError(f"Orthanc ({self.base}) に接続できません: {exc}") from exc
        if r.status_code >= 400:
            raise PacsError(f"Orthanc GET {path} -> HTTP {r.status_code}")
        return r

    def _post(self, path: str, payload) -> requests.Response:
        try:
            r = self.session.post(f"{self.base}{path}", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise PacsError(f"Orthanc ({self.base}) に接続できません: {exc}") from exc
        if r.status_code >= 400:
            raise PacsError(f"Orthanc POST {path} -> HTTP {r.status_code}")
        return r

    def ping(self) -> dict:
        sysinfo = self._get("/system").json()
        return {"name": sysinfo.get("Name"), "version": sysinfo.get("Version")}

    # -- queries --------------------------------------------------------------------

    def list_case_ids(self) -> list[str]:
        ids = []
        for pid in self._get("/patients").json():
            tags = self._get(f"/patients/{pid}").json().get("MainDicomTags", {})
            if tags.get("PatientID"):
                ids.append(tags["PatientID"])
        return sorted(ids)

    def list_studies(self) -> list[dict]:
        """Catalog of every study: case_id, study date, modalities and image counts.

        Two queries in total (studies + series), so it stays cheap as the PACS grows.
        Only identifiers, dates, modalities and counts are returned (no names, no free text).
        """
        studies = self._post("/tools/find", {"Level": "Study", "Expand": True, "Query": {}}).json()
        series = self._post("/tools/find", {"Level": "Series", "Expand": True, "Query": {}}).json()
        by_study: dict[str, list[dict]] = {}
        for se in series:
            by_study.setdefault(se.get("ParentStudy"), []).append(se)
        out = []
        for st in studies:
            case_id = st.get("PatientMainDicomTags", {}).get("PatientID", "")
            try:
                validate_case_id(case_id)
            except ValueError:
                continue  # not addressable by the tools anyway
            ses = by_study.get(st["ID"], [])
            main = st.get("MainDicomTags", {})
            out.append(study_entry(
                case_id, main.get("StudyInstanceUID", ""), main.get("StudyDate"),
                [se.get("MainDicomTags", {}).get("Modality", "") for se in ses],
                [len(se.get("Instances", [])) for se in ses]))
        return sorted(out, key=lambda e: (e["case_id"], e["study_date"] or ""))

    def _series_info(self, orthanc_series_id: str) -> SeriesInfo:
        s = self._get(f"/series/{orthanc_series_id}").json()
        main = s.get("MainDicomTags", {})
        first = s["Instances"][0]
        tags = self._get(f"/instances/{first}/simplified-tags").json()
        return SeriesInfo(
            orthanc_id=orthanc_series_id,
            series_instance_uid=main.get("SeriesInstanceUID", ""),
            modality=main.get("Modality", tags.get("Modality", "")),
            num_instances=len(s["Instances"]),
            slice_thickness=tags.get("SliceThickness"),
            image_type=tags.get("ImageType", ""),
            series_description=main.get("SeriesDescription"),
        )

    def find_studies(self, case_id: str) -> list[StudyInfo]:
        validate_case_id(case_id)  # also guarantees no DICOM wildcard characters (* ?)
        hits = self._post("/tools/find", {
            "Level": "Study", "Expand": True, "CaseSensitive": True,
            "Query": {"PatientID": case_id},
        }).json()
        studies = []
        for st in hits:
            main = st.get("MainDicomTags", {})
            studies.append(StudyInfo(
                orthanc_id=st["ID"],
                study_instance_uid=main.get("StudyInstanceUID", ""),
                study_date=main.get("StudyDate"),
                series=[self._series_info(se) for se in st.get("Series", [])],
            ))
        return studies

    def resolve_series(self, case_id: str,
                       study_instance_uid: Optional[str] = None) -> tuple[StudyInfo, SeriesInfo]:
        studies = self.find_studies(case_id)
        if not studies:
            raise CaseNotFoundError(f"case_id={case_id} の Study が PACS にありません。")
        if study_instance_uid:
            studies = [s for s in studies if s.study_instance_uid == study_instance_uid]
            if not studies:
                raise CaseNotFoundError(
                    f"case_id={case_id} に StudyInstanceUID={study_instance_uid} がありません。")
        if len(studies) > 1:
            listing = "\n".join(f"  - {s.study_instance_uid} (StudyDate={s.study_date}, "
                                f"{len(s.series)} series)" for s in studies)
            raise PacsError(f"case_id={case_id} に Study が {len(studies)} 件あります。"
                            f"study_instance_uid を指定してください:\n{listing}")
        study = studies[0]
        return study, select_series(study)

    # -- retrieval ------------------------------------------------------------------

    def download_series(self, series: SeriesInfo, dest_dir: Path) -> Path:
        """Download every instance of the series into dest_dir (flat). Returns dest_dir."""
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "series.zip"
            r = self._get(f"/series/{series.orthanc_id}/archive", stream=True)
            with zip_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path) as zf:
                for i, info in enumerate(m for m in zf.infolist() if not m.is_dir()):
                    # flatten: never trust archive member paths
                    with zf.open(info) as src, (dest_dir / f"{i:05d}.dcm").open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        n = sum(1 for _ in dest_dir.iterdir())
        if n != series.num_instances:
            raise PacsError(f"取得枚数が一致しません: {n} / {series.num_instances}")
        return dest_dir

    def upload_dicom(self, path: Path) -> dict:
        try:
            r = self.session.post(f"{self.base}/instances", data=path.read_bytes(),
                                  headers={"Content-Type": "application/dicom"},
                                  timeout=self.timeout)
        except requests.RequestException as exc:
            raise PacsError(f"Orthanc ({self.base}) に接続できません: {exc}") from exc
        if r.status_code >= 400:
            raise PacsError(f"upload {path.name} -> HTTP {r.status_code}")
        return r.json()
