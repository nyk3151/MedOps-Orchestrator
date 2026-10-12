# Runs INSIDE 3D Slicer's bundled Python (not the medops venv). Do not import medops here.
#
#   Slicer.exe --no-splash --no-main-window --python-script run_pipeline.py <abs path to config.json>
#
# Verified constraints (see CLAUDE.md):
#   - config is passed as ONE positional argument; "--foo" args are eaten by Slicer's parser
#   - stdout never reaches the caller -> everything is logged to a file (run.log next to config.json)
#   - cwd is the Slicer install dir -> all paths in config.json are absolute
#   - results are extracted right after runPipeline(), BEFORE slicer.util.exit()
#
# Ported from the previous project's run_pipeline.py (verified on real CT). Changes:
#   - log file lives in the job directory instead of a hard-coded absolute path
#   - slicer_version is added to results.json (traceability)
#   - after results.json: views.npz (measured cross-sections + CPR strip) and the Slicer
#     scene (scene.mrb) for later review; failures there never change the result
import sys, json, traceback, datetime, math
from pathlib import Path

LOG = Path(sys.argv[1]).resolve().parent / "run.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} {msg}\n")

def extract_results():
    import slicer
    finalTable = slicer.mrmlScene.GetFirstNodeByName("FinalResults")
    if finalTable is None:
        return None
    t = finalTable.GetTable()
    hasTl = t.GetNumberOfColumns() >= 5
    rows = []
    for i in range(t.GetNumberOfRows()):
        aorta = float(t.GetValue(i, 1).ToFloat())
        rec = {
            "level": t.GetValue(i, 0).ToString(),
            "aorta_area_mm2": round(aorta, 2),
            "aorta_dia_mm": round(2 * math.sqrt(aorta / math.pi), 2) if aorta > 0 else 0.0,
        }
        if hasTl:
            rec["tl_area_mm2"] = round(float(t.GetValue(i, 2).ToFloat()), 2)
            rec["fl_area_mm2"] = round(float(t.GetValue(i, 3).ToFloat()), 2)
            rec["tl_ratio_%"]  = round(float(t.GetValue(i, 4).ToFloat()), 1)
        rows.append(rec)
    return rows

def export_views(out_dir, results):
    """Small 2-D arrays for the dashboard's analysis view -> views.npz (< 1 MB).

    Uses the nodes AorticPipeline leaves in the scene: the measured cross-section at each
    level (Slice_/Mask_ nodes, straightened space) and the straightened volume + labelmaps
    for a long-axis CPR strip. Read-only: nothing in the scene is changed.
    """
    import numpy as np
    import slicer
    from slicer.util import arrayFromVolume

    def node(name):
        return slicer.mrmlScene.GetFirstNodeByName(name)

    arrays, levels = {}, []
    masks = [n.GetName() for n in slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")]
    for r in results or []:
        lv = r["level"]
        sl, am = node(f"Slice_{lv}"), node(f"Mask_whole_aorta_{lv}")
        if sl is None or am is None:
            continue
        levels.append(lv)
        arrays[f"slice_{lv}"] = arrayFromVolume(sl)[0].astype("int16")
        arrays[f"aorta_{lv}"] = (arrayFromVolume(am)[0] > 0).astype("uint8")
        tl = [m for m in masks if m.startswith("Mask_") and m.endswith(f"_{lv}")
              and m != f"Mask_whole_aorta_{lv}"]
        if tl:
            arrays[f"tl_{lv}"] = (arrayFromVolume(node(tl[0]))[0] > 0).astype("uint8")

    # z index of each level along the straightened volume (Z_index column of the area tables)
    z_index = {}
    for t in slicer.util.getNodesByClass("vtkMRMLTableNode"):
        tab = t.GetTable()
        cols = [tab.GetColumnName(c) for c in range(tab.GetNumberOfColumns())]
        if cols[:2] == ["Level", "Z_index"]:
            for i in range(tab.GetNumberOfRows()):
                z_index[tab.GetValue(i, 0).ToString()] = int(tab.GetValue(i, 1).ToInt())

    straight = node("Straightened_Volume")
    if straight is not None:
        vol = arrayFromVolume(straight)                      # (z, y, x)
        mid = vol.shape[2] // 2
        arrays["cpr"] = vol[:, :, mid].astype("int16")
        sa, stl = node("StraightLabelmap_whole_aorta__"), None
        for m in masks:
            if m.startswith("StraightLabelmap_") and m != "StraightLabelmap_whole_aorta__":
                stl = node(m)
        if sa is not None and arrayFromVolume(sa).shape == vol.shape:
            arrays["cpr_aorta"] = (arrayFromVolume(sa)[:, :, mid] > 0).astype("uint8")
        if stl is not None and arrayFromVolume(stl).shape == vol.shape:
            arrays["cpr_tl"] = (arrayFromVolume(stl)[:, :, mid] > 0).astype("uint8")
        arrays["spacing"] = np.array(straight.GetSpacing(), dtype="float32")

    arrays["levels"] = np.array(levels)
    arrays["z_index"] = np.array([z_index.get(lv, -1) for lv in levels], dtype="int32")
    np.savez_compressed(out_dir / "views.npz", **arrays)
    return levels


code = 1
try:
    log(f"START argv={sys.argv}")
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    log(f"config loaded: {cfg}")

    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.json"

    import slicer
    sys.path.insert(0, cfg["module_dir"])
    from AorticPipeline import AorticPipelineLogic
    log("module imported")

    ctNode = slicer.util.loadVolume(cfg["input"])
    log(f"volume loaded: {ctNode.GetName()}")

    logic = AorticPipelineLogic()
    log("runPipeline start")
    logic.runPipeline(
        ctNode,
        aortaModelDir=cfg["aorta_model"], aortaFolds=cfg.get("aorta_folds", "0,1,2,3,4"),
        tlModelDir=cfg["tl_model"],       tlFolds=cfg.get("tl_folds", "0,1,2,3,4"),
        roiSize=cfg.get("roi_size", 100.0),
        cprSpacing=cfg.get("cpr_spacing", 0.5),
        cprSliceSize=cfg.get("cpr_slice_size", 100.0),  # = GUI default; must cover roi_size
        runTl=cfg.get("run_tl", True),
        vertebraCodes=cfg.get("vertebra_codes"),
    )
    log("runPipeline done")

    # ← ここで抽出
    results = extract_results()
    log(f"extracted {len(results) if results else 0} rows")

    if results is None:
        names = [n.GetName() for n in slicer.util.getNodesByClass("vtkMRMLTableNode")]
        log(f"FinalResults not found. tables in scene: {names}")

    result_path.write_text(
        json.dumps({"status": "success", "case_id": cfg.get("case_id"), "results": results,
                    "slicer_version": slicer.app.applicationVersion},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    code = 0

    # Extras for later review. results.json is already written: a failure here is logged
    # and never changes the outcome of the analysis.
    try:
        log(f"views exported: {export_views(out_dir, results)}")
    except Exception:
        log("views export failed\n" + traceback.format_exc())
    if cfg.get("save_scene", True):
        try:
            scene = out_dir / "scene.mrb"
            ok = slicer.util.saveScene(str(scene))
            log(f"scene saved: {scene.name} ok={ok} "
                f"size={scene.stat().st_size if scene.exists() else 0}")
        except Exception:
            log("scene save failed\n" + traceback.format_exc())
except Exception:
    tb = traceback.format_exc()
    log("ERROR\n" + tb)
    try:
        result_path.write_text(json.dumps({"status": "error", "trace": tb},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

log(f"EXIT {code}")
import slicer
slicer.util.exit(code)
