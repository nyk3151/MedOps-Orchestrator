from medops.analysis.engine import canonical_level, normalize_rows


def test_aorticpipeline_display_labels_are_canonicalized():
    # AorticPipeline's FinalResults uses 'Th8'/'Th12' (verified on real CT)
    assert canonical_level("Th8") == "T8" and canonical_level("Th12") == "T12"
    assert canonical_level("T8") == "T8" and canonical_level("L4") == "L4"
    row = normalize_rows([{"level": "Th8", "aorta_area_mm2": 616.0, "aorta_dia_mm": 28.01,
                           "tl_area_mm2": 156.25, "fl_area_mm2": 459.75, "tl_ratio_%": 25.4}])[0]
    assert row["level"] == "T8" and row["tl_ratio_pct"] == 25.4


def test_measurement_window_fits_the_cpr_section():
    """A ROI larger than the straightened section makes AorticPipeline start the window at a
    negative index and measure only a corner (2026-10-13). Keep ROI <= CPR slice size."""
    from medops.analysis import engine
    assert engine.DEFAULT_ROI_SIZE <= engine.DEFAULT_CPR_SLICE_SIZE
