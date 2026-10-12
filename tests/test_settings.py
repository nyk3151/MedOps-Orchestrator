import pytest

from medops import settings as settings_mod
from medops.ids import normalize_levels, validate_case_id
from medops.settings import REPO_ROOT, SettingsError, load_settings


def test_zero_config_is_mock_and_repo_relative(monkeypatch):
    monkeypatch.delenv("MEDOPS_DATA_DIR")
    monkeypatch.delenv("MEDOPS_MOCK_DELAY_S")
    s = load_settings()
    assert not s.analysis_live
    assert s.pacs_mode == "mock" and s.llm_mode == "mock"
    assert s.data_dir == REPO_ROOT / "data"
    assert s.nifti_dir == REPO_ROOT / "data" / "nifti"
    assert s.database_url.startswith("sqlite:///")
    assert s.pipeline_script == REPO_ROOT / "scripts" / "run_pipeline.py"


def test_priority_env_over_yaml_over_default(monkeypatch, tmp_path):
    y = tmp_path / "settings.yaml"
    y.write_text("orthanc_url: http://yaml:8042\npacs_mode: live\nnifti_dir: rel/nifti\n",
                 encoding="utf-8")
    monkeypatch.setenv("MEDOPS_SETTINGS", str(y))
    s = load_settings()
    assert s.orthanc_url == "http://yaml:8042" and s.sources["orthanc_url"] == "yaml"
    assert s.nifti_dir == (REPO_ROOT / "rel" / "nifti").resolve()  # relative to repo, not cwd
    monkeypatch.setenv("MEDOPS_ORTHANC_URL", "http://env:8042")
    s = load_settings()
    assert s.orthanc_url == "http://env:8042" and s.sources["orthanc_url"] == "env"


def test_live_cannot_be_enabled_from_yaml(monkeypatch, tmp_path):
    y = tmp_path / "settings.yaml"
    y.write_text("analysis_live: true\n", encoding="utf-8")
    monkeypatch.setenv("MEDOPS_SETTINGS", str(y))
    with pytest.raises(SettingsError, match="未知のキー"):
        load_settings()


def test_live_validation_lists_every_missing_item(monkeypatch):
    s = load_settings(live=True)
    with pytest.raises(SettingsError) as ei:
        s.validate_live_analysis()
    msg = str(ei.value)
    for key in ("slicer_exe", "module_dir", "aorta_model", "tl_model"):
        assert key in msg


def test_live_validation_passes_with_complete_config(monkeypatch, tmp_path):
    exe = tmp_path / "Slicer.exe"; exe.write_text("")
    mod = tmp_path / "mod"; mod.mkdir(); (mod / "AorticPipeline.py").write_text("")
    a = tmp_path / "a"; a.mkdir()
    t = tmp_path / "t"; t.mkdir()
    monkeypatch.setenv("MEDOPS_SLICER_EXE", str(exe))
    monkeypatch.setenv("MEDOPS_MODULE_DIR", str(mod))
    monkeypatch.setenv("MEDOPS_AORTA_MODEL", str(a))
    monkeypatch.setenv("MEDOPS_TL_MODEL", str(t))
    monkeypatch.setenv("MEDOPS_ANALYSIS_LIVE", "1")
    s = load_settings()
    assert s.analysis_live
    s.validate_live_analysis()


def test_gemini_validation(monkeypatch):
    with pytest.raises(SettingsError, match="GOOGLE_CLOUD_PROJECT"):
        load_settings().validate_gemini()


@pytest.mark.parametrize("bad", ["../x", "a/b", "a b", "", "case;rm"])
def test_case_id_rejects_traversal(bad):
    with pytest.raises(ValueError):
        validate_case_id(bad)


def test_levels():
    assert normalize_levels(None) == ["T8", "T12"]
    assert normalize_levels(["t8", "T8", "L1"]) == ["T8", "L1"]
    with pytest.raises(ValueError):
        normalize_levels(["T13"])
