import pytest

from medops.pacs.orthanc import SeriesInfo, SeriesSelectionError, StudyInfo, select_series
from medops.pacs.service import MockPacs


def _s(i, modality="CT", itype="ORIGINAL\\PRIMARY\\AXIAL", n=300):
    return SeriesInfo(f"o{i}", f"1.2.{i}", modality, n, "1", itype)


def test_single_candidate_ignoring_scout_and_dose_report():
    study = StudyInfo("o", "1.2", None, [
        _s(1, itype="ORIGINAL\\PRIMARY\\LOCALIZER", n=2),
        _s(2),
        _s(3, itype="DERIVED\\SECONDARY\\DOSE_INFO", n=1),
        _s(4, modality="SR", itype="", n=1),
    ])
    assert select_series(study).series_instance_uid == "1.2.2"


def test_two_candidates_is_an_error_listing_all_series():
    study = StudyInfo("o", "1.2", None, [_s(1), _s(2, n=150)])
    with pytest.raises(SeriesSelectionError) as ei:
        select_series(study)
    msg = str(ei.value)
    assert "1.2.1" in msg and "1.2.2" in msg and "150枚" in msg and "ImageType" in msg


def test_no_candidate_is_an_error():
    with pytest.raises(SeriesSelectionError):
        select_series(StudyInfo("o", "1.2", None, [_s(1, itype="DERIVED\\SECONDARY")]))


def test_mock_pacs(settings):
    pacs = MockPacs(settings)
    assert "demo_001" in pacs.list_case_ids()
    res = pacs.ingest("demo_001")
    assert res.series_instance_uid.startswith("2.25.")
    with pytest.raises(SeriesSelectionError):
        pacs.ingest("demo_multi")
