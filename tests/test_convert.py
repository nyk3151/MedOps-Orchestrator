"""DICOM -> NIfTI conversion: orientation / spacing must survive every storage layout."""
import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.uid import generate_uid

from medops.pacs.convert import ConversionError, dicom_to_nifti
from medops.pacs.phantom import CRANIAL_MARKER_HU, LEFT_MARKER_HU, PhantomSpec, world_volume, write_series


def _convert(tmp_path, spec, **kw):
    d = tmp_path / "dcm"
    write_series(spec, d, patient_id="t", study_uid=generate_uid(), **kw)
    out = tmp_path / "t_0000.nii.gz"
    report = dicom_to_nifti(d, out)
    return sitk.ReadImage(str(out)), report


@pytest.mark.parametrize("spec,shuffle", [
    (PhantomSpec(), False),
    (PhantomSpec(descending_z=True), False),
    (PhantomSpec(flip_rows=True), False),
    (PhantomSpec(flip_cols=True, descending_z=True), True),
    (PhantomSpec(flip_rows=True, flip_cols=True), True),
])
def test_layouts_reconstruct_the_same_world(tmp_path, spec, shuffle):
    img, report = _convert(tmp_path, spec, shuffle_files=shuffle)
    expected, origin = world_volume(spec)

    # Resample to canonical LPS axes; must equal the world volume voxel-for-voxel.
    canon = sitk.DICOMOrient(img, "LPS")
    assert np.allclose(canon.GetDirection(), np.eye(3).ravel())
    assert np.allclose(canon.GetOrigin(), origin, atol=1e-3)
    assert np.allclose(canon.GetSpacing(), [spec.pixel_spacing, spec.pixel_spacing, spec.slice_spacing])
    assert np.array_equal(sitk.GetArrayFromImage(canon), expected)

    # Physical spot checks: patient-left and cranial markers are where they should be.
    def hu_at(pt):
        return img[img.TransformPhysicalPointToIndex(pt)]
    x_left = 0.36 * spec.cols * spec.pixel_spacing
    z_mid = origin[2] + spec.slices / 2 * spec.slice_spacing
    assert hu_at((x_left, 0.0, z_mid)) == LEFT_MARKER_HU
    assert hu_at((-x_left, 0.0, z_mid)) != LEFT_MARKER_HU
    z_top = origin[2] + (spec.slices - 1) * spec.slice_spacing
    y_ant = -0.25 * spec.rows * spec.pixel_spacing
    assert hu_at((0.0, y_ant, z_top)) == CRANIAL_MARKER_HU
    assert hu_at((0.0, y_ant, origin[2])) != CRANIAL_MARKER_HU

    assert report.pixel_type == "16-bit signed integer"
    assert report.hu_min == -1000


def test_missing_slice_is_refused(tmp_path):
    with pytest.raises(ConversionError, match="不均一"):
        _convert(tmp_path, PhantomSpec(drop_slice=10))


def test_gantry_tilt_is_refused(tmp_path):
    with pytest.raises(ConversionError, match="ガントリ傾斜"):
        _convert(tmp_path, PhantomSpec(gantry_tilt_mm_per_slice=0.5))


def test_mixed_series_in_one_folder_is_refused(tmp_path):
    d = tmp_path / "dcm"
    uid = generate_uid()
    write_series(PhantomSpec(), d / "a", patient_id="t", study_uid=uid)
    write_series(PhantomSpec(), d / "b", patient_id="t", study_uid=uid)
    flat = tmp_path / "flat"
    flat.mkdir()
    for i, f in enumerate(sorted(d.rglob("*.dcm"))):
        f.rename(flat / f"{i:04d}.dcm")
    with pytest.raises(ConversionError, match="SeriesInstanceUID"):
        dicom_to_nifti(flat, tmp_path / "x.nii.gz")
