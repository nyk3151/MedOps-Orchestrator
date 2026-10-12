"""DICOM series -> NIfTI ({case_id}_0000.nii.gz), the biggest new risk in this project.

Wrong orientation (LPS/RAS) or spacing silently corrupts measurements, so the geometry is
checked independently with pydicom BEFORE handing the files to ITK, and conversion refuses
anything it cannot represent faithfully as a regular 3D grid:

- exactly one SeriesInstanceUID, one ImageOrientationPatient, one Rows/Columns/PixelSpacing
- slice normal (from IOP) parallel to the slice stacking direction (no gantry tilt)
- uniform slice spacing, no duplicates / gaps (missing slices)

The volume is read with SimpleITK (GDCM) — the same ITK stack 3D Slicer uses — which keeps
DICOM's LPS physical space and writes the NIfTI qform/sform as RAS. The resulting spacing is
taken from the IPP positions, not SliceThickness (they differ for overlapping recons).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

SPACING_RTOL = 1e-3   # relative tolerance for uniform slice spacing
ORIENT_ATOL = 1e-4    # direction cosine tolerance


class ConversionError(RuntimeError):
    pass


@dataclass
class ConversionReport:
    series_instance_uid: str
    num_slices: int
    size: list[int]
    spacing_mm: list[float]
    origin_lps_mm: list[float]
    direction_lps: list[float]
    orientation_code: str          # ITK/DICOM LPS convention, e.g. "LPS" for a standard axial
    slice_thickness_mm: float | None
    hu_min: float
    hu_max: float
    pixel_type: str

    def to_dict(self) -> dict:
        return asdict(self)


def _orientation_code(direction: np.ndarray) -> str:
    """Letters the +i,+j,+k axes point TO, in LPS patient space (DICOM/ITK convention)."""
    letters = (("L", "R"), ("P", "A"), ("S", "I"))
    code = ""
    for col in range(3):
        v = direction[:, col]
        ax = int(np.argmax(np.abs(v)))
        code += letters[ax][0] if v[ax] > 0 else letters[ax][1]
    return code


def check_geometry(dicom_dir: Path) -> dict:
    """Independent geometry validation with pydicom (headers only)."""
    files = sorted(p for p in dicom_dir.iterdir() if p.is_file())
    if not files:
        raise ConversionError(f"DICOM ファイルがありません: {dicom_dir.name}")

    uids, iops, shapes, spacings, positions, thicknesses = set(), [], set(), set(), [], set()
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        uids.add(str(ds.SeriesInstanceUID))
        iops.append(np.array([float(x) for x in ds.ImageOrientationPatient]))
        shapes.add((int(ds.Rows), int(ds.Columns)))
        spacings.add(tuple(round(float(x), 6) for x in ds.PixelSpacing))
        positions.append(np.array([float(x) for x in ds.ImagePositionPatient]))
        if "SliceThickness" in ds and ds.SliceThickness not in (None, ""):
            thicknesses.add(round(float(ds.SliceThickness), 4))

    problems = []
    if len(uids) != 1:
        problems.append(f"SeriesInstanceUID が複数あります ({len(uids)})")
    iop0 = iops[0]
    if any(not np.allclose(i, iop0, atol=ORIENT_ATOL) for i in iops):
        problems.append("ImageOrientationPatient がスライス間で一致しません")
    if len(shapes) != 1:
        problems.append(f"Rows/Columns が一致しません: {sorted(shapes)}")
    if len(spacings) != 1:
        problems.append(f"PixelSpacing が一致しません: {sorted(spacings)}")
    if problems:
        raise ConversionError("; ".join(problems))

    row, col = iop0[:3], iop0[3:]
    normal = np.cross(row, col)
    proj = np.array([float(np.dot(p, normal)) for p in positions])
    order = np.argsort(proj)
    proj_sorted, pos_sorted = proj[order], np.array(positions)[order]
    if len(files) > 1:
        diffs = np.diff(proj_sorted)
        if np.any(diffs < 1e-6):
            raise ConversionError("同一位置のスライスが重複しています")
        step = float(np.median(diffs))
        if not np.allclose(diffs, step, rtol=SPACING_RTOL, atol=1e-4):
            raise ConversionError(
                f"スライス間隔が不均一です（欠損スライスの可能性）: min={diffs.min():.4f} "
                f"max={diffs.max():.4f} mm")
        stack = pos_sorted[-1] - pos_sorted[0]
        stack /= np.linalg.norm(stack)
        if abs(abs(float(np.dot(stack, normal))) - 1.0) > 1e-3:
            raise ConversionError("スライス積層方向が画像法線と平行ではありません（ガントリ傾斜）")
    else:
        step = float(next(iter(thicknesses), 1.0))

    return {
        "series_instance_uid": next(iter(uids)),
        "num_slices": len(files),
        "slice_spacing_mm": step,
        "pixel_spacing_mm": list(next(iter(spacings))),
        "first_position_lps": pos_sorted[0].tolist(),
        "normal_lps": normal.tolist(),
        "slice_thickness_mm": next(iter(thicknesses)) if len(thicknesses) == 1 else None,
    }


def dicom_to_nifti(dicom_dir: Path, out_path: Path) -> ConversionReport:
    geom = check_geometry(dicom_dir)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if len(series_ids) != 1:
        raise ConversionError(f"GDCM が {len(series_ids)} 個のシリーズを検出しました")
    filenames = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    if len(filenames) != geom["num_slices"]:
        raise ConversionError(
            f"GDCM の読み込み枚数 {len(filenames)} がファイル数 {geom['num_slices']} と一致しません")
    reader.SetFileNames(filenames)
    image = reader.Execute()  # applies RescaleSlope/Intercept -> HU

    # Cross-check ITK geometry against the independent pydicom computation.
    sp = image.GetSpacing()
    if not np.isclose(sp[2], geom["slice_spacing_mm"], rtol=SPACING_RTOL):
        raise ConversionError(f"z spacing 不一致: ITK={sp[2]} / IPP={geom['slice_spacing_mm']}")
    # DICOM PixelSpacing is (row spacing = y, column spacing = x); ITK spacing is (x, y).
    if not np.allclose(sp[:2], geom["pixel_spacing_mm"][::-1], rtol=1e-6):
        raise ConversionError(f"in-plane spacing 不一致: ITK={sp[:2]} / DICOM={geom['pixel_spacing_mm']}")
    direction = np.array(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction[:, 2], geom["normal_lps"], atol=1e-3):
        raise ConversionError("ITK のスライス方向が IOP 法線と一致しません")
    if not np.allclose(image.GetOrigin(), geom["first_position_lps"], atol=1e-3):
        raise ConversionError("ITK の原点が最初のスライスの ImagePositionPatient と一致しません")

    # Keep integer HU as int16 (what CT NIfTIs normally hold) when it is lossless.
    arr = sitk.GetArrayViewFromImage(image)
    hu_min, hu_max = float(arr.min()), float(arr.max())
    if image.GetPixelID() != sitk.sitkInt16 and np.all(np.mod(arr, 1) == 0) \
            and hu_min >= -32768 and hu_max <= 32767:
        image = sitk.Cast(image, sitk.sitkInt16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp.nii.gz")
    sitk.WriteImage(image, str(tmp), useCompression=True)
    tmp.replace(out_path)  # atomic: never leave a half-written input for the analysis

    return ConversionReport(
        series_instance_uid=geom["series_instance_uid"],
        num_slices=geom["num_slices"],
        size=list(image.GetSize()),
        spacing_mm=[round(x, 6) for x in image.GetSpacing()],
        origin_lps_mm=[round(x, 4) for x in image.GetOrigin()],
        direction_lps=[round(x, 6) for x in image.GetDirection()],
        orientation_code=_orientation_code(direction),
        slice_thickness_mm=geom["slice_thickness_mm"],
        hu_min=hu_min,
        hu_max=hu_max,
        pixel_type=image.GetPixelIDTypeAsString(),
    )


def write_sidecar(nifti_path: Path, case_id: str, study_uid: str, report: ConversionReport) -> Path:
    """{case_id}_0000.json next to the NIfTI: which series it came from (cache key + audit)."""
    sidecar = nifti_path.with_name(nifti_path.name.replace(".nii.gz", ".json"))
    sidecar.write_text(json.dumps({"case_id": case_id, "study_instance_uid": study_uid,
                                   **report.to_dict()}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return sidecar


def read_sidecar(nifti_path: Path) -> dict | None:
    sidecar = nifti_path.with_name(nifti_path.name.replace(".nii.gz", ".json"))
    if not sidecar.is_file() or not nifti_path.is_file():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))
