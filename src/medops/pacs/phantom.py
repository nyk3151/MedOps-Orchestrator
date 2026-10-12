"""Synthetic CT phantom as DICOM — redistributable demo data and conversion test fixture.

The phantom has an asymmetric body (patient-left marker, cranial marker) so orientation
errors are detectable, plus a contrast-filled "aorta" cylinder and a "spine". It is NOT
meant to be analyzable by the real nnU-Net models — it exercises PACS / conversion / MOCK
analysis only. All tags are synthetic; no real patient data is involved.

    python -m medops.pacs.phantom --upload            # load the demo cases into Orthanc
"""
from __future__ import annotations

import argparse
import datetime
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

LEFT_MARKER_HU = 1500    # placed at patient LEFT (+x in LPS)
CRANIAL_MARKER_HU = 2000  # placed at the most SUPERIOR slice region (+z in LPS)


@dataclass
class PhantomSpec:
    rows: int = 96
    cols: int = 96
    slices: int = 40
    pixel_spacing: float = 3.0     # mm (row spacing == column spacing)
    slice_spacing: float = 4.0     # mm
    flip_rows: bool = False        # IOP row direction = -x  (image stored right-to-left)
    flip_cols: bool = False        # IOP column direction = -y
    descending_z: bool = False     # instance 1 is the most superior slice
    image_type: str = "ORIGINAL\\PRIMARY\\AXIAL"
    gantry_tilt_mm_per_slice: float = 0.0   # shifts IPP along y to simulate tilt
    drop_slice: Optional[int] = None        # simulate a missing slice


def world_volume(spec: PhantomSpec) -> tuple[np.ndarray, np.ndarray]:
    """HU volume indexed [z, y, x] in a canonical LPS grid (+x=L, +y=P, +z=S), and origin."""
    z, y, x = np.mgrid[0:spec.slices, 0:spec.rows, 0:spec.cols]
    cx, cy = (spec.cols - 1) / 2, (spec.rows - 1) / 2
    xs, ys = (x - cx) * spec.pixel_spacing, (y - cy) * spec.pixel_spacing
    vol = np.full(x.shape, -1000, dtype=np.int16)
    body = (xs / (0.42 * spec.cols * spec.pixel_spacing)) ** 2 + \
           (ys / (0.32 * spec.rows * spec.pixel_spacing)) ** 2 <= 1
    vol[body] = 40
    spine = (xs ** 2 + (ys - 0.18 * spec.rows * spec.pixel_spacing) ** 2) <= (0.07 * spec.cols * spec.pixel_spacing) ** 2
    vol[spine] = 700
    radius = 12.0 + 3.0 * np.sin(z / max(spec.slices - 1, 1) * np.pi)  # aorta 24-30 mm
    aorta = ((xs - 0.06 * spec.cols * spec.pixel_spacing) ** 2 + ys ** 2) <= radius ** 2
    vol[aorta] = 300
    left = (xs > 0.30 * spec.cols * spec.pixel_spacing) & (np.abs(ys) < 6) & body
    vol[left] = LEFT_MARKER_HU
    cranial = (z >= spec.slices - 3) & (np.abs(xs) < 6) & (ys < -0.2 * spec.rows * spec.pixel_spacing) & body
    vol[cranial] = CRANIAL_MARKER_HU
    origin = np.array([-cx * spec.pixel_spacing, -cy * spec.pixel_spacing, -600.0])
    return vol, origin


def write_series(spec: PhantomSpec, out_dir: Path, *, patient_id: str, study_uid: str,
                 series_number: int = 1, shuffle_files: bool = False) -> str:
    """Write one series; returns its SeriesInstanceUID."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vol, origin = world_volume(spec)
    series_uid = generate_uid()
    frame_uid = generate_uid()
    row_dir = np.array([-1.0 if spec.flip_rows else 1.0, 0, 0])
    col_dir = np.array([0, -1.0 if spec.flip_cols else 1.0, 0])
    now = datetime.datetime(2026, 1, 1, 9, 0, 0)

    z_order = list(range(spec.slices))
    if spec.descending_z:
        z_order.reverse()
    if spec.drop_slice is not None:
        z_order.remove(spec.drop_slice)
    rng = np.random.default_rng(0)
    file_order = rng.permutation(len(z_order)) if shuffle_files else np.arange(len(z_order))

    for inst_no, k in enumerate(z_order, start=1):
        img = vol[k]
        if spec.flip_rows:
            img = img[:, ::-1]
        if spec.flip_cols:
            img = img[::-1, :]
        # first stored pixel (row 0, col 0) position in LPS
        x0 = origin[0] + (spec.cols - 1) * spec.pixel_spacing if spec.flip_rows else origin[0]
        y0 = origin[1] + (spec.rows - 1) * spec.pixel_spacing if spec.flip_cols else origin[1]
        ipp = [x0, y0 + k * spec.gantry_tilt_mm_per_slice, origin[2] + k * spec.slice_spacing]

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientID = patient_id
        ds.PatientName = "PHANTOM^SYNTHETIC"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.StudyDate = ds.SeriesDate = now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")
        ds.Modality = "CT"
        ds.ImageType = spec.image_type.split("\\")
        ds.SeriesDescription = "Synthetic phantom"
        ds.SeriesNumber = series_number
        ds.InstanceNumber = inst_no
        ds.ImageOrientationPatient = [*map(float, row_dir), *map(float, col_dir)]
        ds.ImagePositionPatient = [float(v) for v in ipp]
        ds.PixelSpacing = [spec.pixel_spacing, spec.pixel_spacing]
        ds.SliceThickness = spec.slice_spacing
        ds.Rows, ds.Columns = spec.rows, spec.cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 0
        ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
        ds.PixelData = (img.astype(np.int32) + 1024).astype(np.uint16).tobytes()
        ds.save_as(out_dir / f"f{int(file_order[inst_no - 1]):04d}.dcm", enforce_file_format=True)
    return series_uid


# Demo cases for Orthanc (judges' quick start). One multi-series study shows the guard.
DEMO_CASES = {
    "phantom_001": [PhantomSpec()],
    "phantom_002": [PhantomSpec(descending_z=True), PhantomSpec(image_type="ORIGINAL\\PRIMARY\\LOCALIZER", slices=1)],
    "phantom_003": [PhantomSpec(flip_rows=True)],
    "phantom_multi": [PhantomSpec(), PhantomSpec(slice_spacing=2.0, slices=60)],
}


def upload_demo_cases() -> None:
    from medops.pacs.orthanc import OrthancClient
    from medops.settings import load_settings

    settings = load_settings()
    client = OrthancClient(settings)
    print(f"Orthanc: {client.ping()} at {settings.orthanc_url}")
    existing = set(client.list_case_ids())
    for case_id, specs in DEMO_CASES.items():
        if case_id in existing:
            print(f"  {case_id}: already in Orthanc, skipped")
            continue
        study_uid = generate_uid()
        with tempfile.TemporaryDirectory() as tmp:
            n = 0
            for i, spec in enumerate(specs, start=1):
                d = Path(tmp) / str(i)
                write_series(spec, d, patient_id=case_id, study_uid=study_uid, series_number=i)
                for f in sorted(d.iterdir()):
                    client.upload_dicom(f)
                    n += 1
        print(f"  {case_id}: uploaded {len(specs)} series / {n} instances")


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic CT phantom (DICOM)")
    ap.add_argument("--upload", action="store_true", help="upload the demo cases to Orthanc")
    ap.add_argument("--out", type=Path, help="write one phantom series to this directory")
    args = ap.parse_args()
    if args.upload:
        upload_demo_cases()
    elif args.out:
        uid = write_series(PhantomSpec(), args.out, patient_id="phantom_local", study_uid=generate_uid())
        print(f"wrote series {uid} to {args.out}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
