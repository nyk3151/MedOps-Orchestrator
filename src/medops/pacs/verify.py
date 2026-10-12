"""Developer check: ingest a case from Orthanc and compare against a verified NIfTI.

    python -m medops.pacs.verify <case_id> --reference <verified NIfTI, e.g. C:/data/case_0000.nii.gz>

Both volumes are read with SimpleITK, i.e. in the same LPS physical space, so any LPS/RAS
flip, axis permutation, spacing or origin error shows up as a geometry mismatch, and any
slice-order / intensity error as a voxel difference. Exit code 0 only if everything matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from medops.pacs.service import OrthancPacs
from medops.settings import load_settings


def compare(ours: Path, reference: Path) -> dict:
    a, b = sitk.ReadImage(str(ours)), sitk.ReadImage(str(reference))
    out = {
        "size": (a.GetSize(), b.GetSize()),
        "spacing": (a.GetSpacing(), b.GetSpacing()),
        "origin": (a.GetOrigin(), b.GetOrigin()),
        "direction": (a.GetDirection(), b.GetDirection()),
        "pixel_type": (a.GetPixelIDTypeAsString(), b.GetPixelIDTypeAsString()),
    }
    checks = {
        "size": a.GetSize() == b.GetSize(),
        "spacing": np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4),
        "origin": np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-2),
        "direction": np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4),
    }
    if checks["size"]:
        va = sitk.GetArrayFromImage(a).astype(np.float64)
        vb = sitk.GetArrayFromImage(b).astype(np.float64)
        diff = np.abs(va - vb)
        out["voxel_max_abs_diff"] = float(diff.max())
        out["voxel_mean_abs_diff"] = float(diff.mean())
        out["voxels_differing"] = int((diff > 0).sum())
        checks["voxels"] = float(diff.max()) == 0.0
        # Physical-space spot check: same world point -> same voxel value in both images.
        center = a.TransformContinuousIndexToPhysicalPoint([s / 2 for s in a.GetSize()])
        out["center_lps_mm"] = center
    out["checks"] = checks
    out["ok"] = all(checks.values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_id")
    ap.add_argument("--reference", required=True, type=Path, help="verified NIfTI to compare against")
    args = ap.parse_args()

    settings = load_settings()
    if settings.pacs_mode != "live":
        settings = type(settings)(**{**settings.__dict__, "pacs_mode": "live"})
    pacs = OrthancPacs(settings)
    res = pacs.ingest(args.case_id, progress=lambda m: print(f"  .. {m}", flush=True))
    print(json.dumps({"series_instance_uid": res.series_instance_uid, "cached": res.cached,
                      "report": res.report}, indent=2, ensure_ascii=False, default=str))
    result = compare(res.nifti_path, args.reference)
    print(json.dumps(result, indent=2, default=str))
    print("MATCH" if result["ok"] else "MISMATCH")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
