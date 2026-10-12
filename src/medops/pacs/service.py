"""PACS facade used by the job server and the agent tools (MOCK or Orthanc LIVE).

    ingest(case_id) = resolve the single candidate series -> download -> convert ->
                      data/nifti/{case_id}_0000.nii.gz (+ sidecar .json with the series UID)
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from medops.analysis.engine import nifti_path_for
from medops.ids import validate_case_id
from medops.pacs import convert
from medops.pacs.orthanc import (
    CaseNotFoundError, OrthancClient, PacsError, SeriesInfo, StudyInfo, iso_date, select_series,
    study_entry,
)
from medops.settings import Settings, on_cloud_run

Progress = Callable[[str], None]


@dataclass
class IngestResult:
    case_id: str
    study_instance_uid: str
    series_instance_uid: str
    nifti_path: Path
    cached: bool
    report: dict = field(default_factory=dict)
    study_date: Optional[str] = None      # ISO 'YYYY-MM-DD'
    modality: Optional[str] = None


def _noop(_: str) -> None:
    pass


# ----------------------------------------------------------------------------------------
# MOCK: a small built-in catalog so every upper layer works without Orthanc.
# ----------------------------------------------------------------------------------------

def _uid(*parts: str) -> str:
    h = int(hashlib.sha256("/".join(parts).encode()).hexdigest()[:24], 16)
    return f"2.25.{h}"


def _mock_study(case_id: str, series_specs: list[tuple[str, str, int, str]],
                study_date: str = "20260101") -> StudyInfo:
    return StudyInfo(
        orthanc_id=f"mock-{case_id}",
        study_instance_uid=_uid(case_id, "study"),
        study_date=study_date,
        series=[SeriesInfo(orthanc_id=f"mock-{case_id}-{i}",
                           series_instance_uid=_uid(case_id, "series", str(i)),
                           modality=mod, num_instances=n, slice_thickness=thick,
                           image_type=itype)
                for i, (mod, itype, n, thick) in enumerate(series_specs)],
    )


_AXIAL = ("CT", "ORIGINAL\\PRIMARY\\AXIAL", 600, "1")
_SCOUT = ("CT", "ORIGINAL\\PRIMARY\\LOCALIZER", 2, "1")
MOCK_CATALOG: dict[str, list[StudyInfo]] = {
    **{f"demo_{i:03d}": [_mock_study(f"demo_{i:03d}", [_SCOUT, _AXIAL],
                                     f"2026{9 + i // 5:02d}{i * 3:02d}")] for i in range(1, 9)},
    # Two axial reconstructions in one study -> must be refused, never guessed.
    "demo_multi": [_mock_study("demo_multi", [_AXIAL, ("CT", "ORIGINAL\\PRIMARY\\AXIAL", 300, "2")],
                               "20261001")],
}


class MockPacs:
    mode = "mock"

    def __init__(self, settings: Settings):
        self.settings = settings

    def ping(self) -> dict:
        return {"name": "MockPACS", "version": "mock"}

    def list_case_ids(self) -> list[str]:
        return sorted(MOCK_CATALOG)

    def list_studies(self) -> list[dict]:
        return [study_entry(cid, st.study_instance_uid, st.study_date,
                            [s.modality for s in st.series], [s.num_instances for s in st.series])
                for cid, studies in sorted(MOCK_CATALOG.items()) for st in studies]

    def find_studies(self, case_id: str) -> list[StudyInfo]:
        validate_case_id(case_id)
        return MOCK_CATALOG.get(case_id, [])

    def resolve_series(self, case_id: str, study_instance_uid: Optional[str] = None):
        studies = self.find_studies(case_id)
        if not studies:
            raise CaseNotFoundError(f"case_id={case_id} の Study が PACS にありません。")
        return studies[0], select_series(studies[0])

    def ingest(self, case_id: str, study_instance_uid: Optional[str] = None,
               progress: Progress = _noop) -> IngestResult:
        study, series = self.resolve_series(case_id, study_instance_uid)
        progress(f"[MOCK PACS] シリーズ {series.num_instances}枚 を取得（実ファイルなし）")
        return IngestResult(case_id, study.study_instance_uid, series.series_instance_uid,
                            nifti_path_for(self.settings, case_id), cached=False,
                            report={"mock": True}, study_date=iso_date(study.study_date),
                            modality=series.modality)


# ----------------------------------------------------------------------------------------
# LIVE: Orthanc
# ----------------------------------------------------------------------------------------

class OrthancPacs:
    mode = "live"

    def __init__(self, settings: Settings):
        settings.validate_live_pacs()
        self.settings = settings
        self.client = OrthancClient(settings)

    def ping(self) -> dict:
        return self.client.ping()

    def list_case_ids(self) -> list[str]:
        return self.client.list_case_ids()

    def list_studies(self) -> list[dict]:
        return self.client.list_studies()

    def find_studies(self, case_id: str) -> list[StudyInfo]:
        return self.client.find_studies(case_id)

    def resolve_series(self, case_id: str, study_instance_uid: Optional[str] = None):
        return self.client.resolve_series(case_id, study_instance_uid)

    def ingest(self, case_id: str, study_instance_uid: Optional[str] = None,
               progress: Progress = _noop) -> IngestResult:
        validate_case_id(case_id)
        progress("PACS で Study / シリーズを検索中")
        study, series = self.client.resolve_series(case_id, study_instance_uid)
        nifti = nifti_path_for(self.settings, case_id)

        cached = convert.read_sidecar(nifti)
        if cached and cached.get("series_instance_uid") == series.series_instance_uid:
            progress("変換済み NIfTI を再利用（同一シリーズ）")
            return IngestResult(case_id, study.study_instance_uid, series.series_instance_uid,
                                nifti, cached=True, report=cached,
                                study_date=iso_date(study.study_date), modality=series.modality)

        if on_cloud_run():
            # data/ is a GCS FUSE mount there: reading hundreds of small DICOMs back through it
            # stalls the conversion. Fetch and convert on the instance, publish only the NIfTI.
            with tempfile.TemporaryDirectory(prefix="medops-ingest-") as tmp:
                staged = Path(tmp) / "nifti" / nifti.name
                report = self._fetch_and_convert(series, Path(tmp) / "dicom", staged, progress)
                part = nifti.with_name(nifti.name + ".part")
                nifti.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged, part)
                part.replace(nifti)
        else:
            dicom_dir = self.settings.dicom_dir / case_id / series.series_instance_uid
            report = self._fetch_and_convert(series, dicom_dir, nifti, progress)
        if report.series_instance_uid != series.series_instance_uid:
            raise PacsError("変換したシリーズの UID が PACS の選択結果と一致しません")
        convert.write_sidecar(nifti, case_id, study.study_instance_uid, report)
        return IngestResult(case_id, study.study_instance_uid, series.series_instance_uid,
                            nifti, cached=False, report=report.to_dict(),
                            study_date=iso_date(study.study_date), modality=series.modality)

    def _fetch_and_convert(self, series: SeriesInfo, dicom_dir: Path, nifti: Path,
                           progress: Progress) -> convert.ConversionReport:
        progress(f"DICOM 取得中（{series.num_instances}枚）")
        self.client.download_series(series, dicom_dir)
        progress("DICOM → NIfTI 変換・幾何チェック中")
        return convert.dicom_to_nifti(dicom_dir, nifti)


def get_pacs(settings: Settings):
    return OrthancPacs(settings) if settings.pacs_mode == "live" else MockPacs(settings)
