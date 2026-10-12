"""Analysis view: render what the pipeline measured, for review after the fact.

run_pipeline.py exports views.npz next to results.json (job directory):
  slice_<L> / aorta_<L> / tl_<L>   the measured cross-section at level <L> (straightened
                                   space, the exact plane the area was computed on)
  cpr / cpr_aorta / cpr_tl         long-axis strip through the straightened aorta
  levels, z_index, spacing         where each level sits along the strip

This module turns those arrays into PNGs (numpy + Pillow only). Images go to the UI only,
never to the LLM. In MOCK mode a synthetic views.npz is generated from the dummy areas so
the zero-config demo shows the same screen.
"""
from __future__ import annotations

import io
import math
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from medops.analysis.engine import canonical_level

WINDOW = (-150.0, 450.0)          # CT angiography-ish window (HU)
TL_RGB = (230, 57, 70)            # true lumen
FL_RGB = (42, 120, 214)           # false lumen (aorta minus true lumen)
EDGE_RGB = (255, 210, 63)         # aortic wall outline / level markers
ALPHA = 0.38
SECTION_PX = 360                  # displayed size of a cross-section
VIEWS_FILE = "views.npz"


def _gray(ct: np.ndarray) -> np.ndarray:
    lo, hi = WINDOW
    g = np.clip((ct.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255
    return np.repeat(g[..., None], 3, axis=2)


def _edge(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner


def _overlay(ct: np.ndarray, aorta: Optional[np.ndarray], tl: Optional[np.ndarray],
             scale: int) -> Image.Image:
    rgb = _gray(np.kron(ct, np.ones((scale, scale))) if scale > 1 else ct)
    up = (lambda m: np.kron(m, np.ones((scale, scale), dtype=np.uint8)).astype(bool)) \
        if scale > 1 else (lambda m: m.astype(bool))
    if aorta is not None:
        a = up(aorta)
        t = up(tl) & a if tl is not None else np.zeros_like(a)
        for mask, color in ((t, TL_RGB), (a & ~t, FL_RGB if tl is not None else None)):
            if color is not None and mask.any():
                rgb[mask] = (1 - ALPHA) * rgb[mask] + ALPHA * np.array(color)
        rgb[_edge(a)] = EDGE_RGB
    return Image.fromarray(rgb.astype(np.uint8))


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def has_views(job_dir: Path) -> bool:
    return (job_dir / VIEWS_FILE).is_file()


def render(job_dir: Path) -> dict:
    """{"sections": [{"level", "png"}], "cpr": png | None, "cpr_levels": [...], "fov_mm": float}"""
    with np.load(job_dir / VIEWS_FILE, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    levels = [str(x) for x in d.get("levels", [])]
    spacing = d.get("spacing", np.array([0.5, 0.5, 0.5]))
    sections = []
    for lv in levels:
        ct = d[f"slice_{lv}"]
        scale = max(1, SECTION_PX // max(ct.shape))
        img = _overlay(ct, d.get(f"aorta_{lv}"), d.get(f"tl_{lv}"), scale)
        sections.append({"level": canonical_level(lv), "png": _png(img),
                         "fov_mm": round(float(ct.shape[1] * spacing[0]), 1)})

    cpr_png, cpr_levels = None, []
    if "cpr" in d:
        cpr, ca, ct_ = d["cpr"], d.get("cpr_aorta"), d.get("cpr_tl")
        z_idx = [int(x) for x in d.get("z_index", [])]
        # keep the span that contains the aorta (plus a margin) so the strip stays readable
        rows = np.where(ca.any(axis=1))[0] if ca is not None and ca.any() else np.arange(cpr.shape[0])
        lo, hi = max(int(rows.min()) - 20, 0), min(int(rows.max()) + 20, cpr.shape[0] - 1)
        sl = slice(lo, hi + 1)
        # (z, y) -> horizontal strip: proximal (aortic root side) on the left
        img_arr = _overlay(cpr[sl].T, None if ca is None else ca[sl].T,
                           None if ct_ is None else ct_[sl].T, 2)
        arr = np.array(img_arr)
        for lv, zi in zip(levels, z_idx):
            if lo <= zi <= hi:
                x = (zi - lo) * 2
                arr[:, max(x - 1, 0):x + 2] = EDGE_RGB
                cpr_levels.append({"level": canonical_level(lv),
                                   "position": round((zi - lo) / max(hi - lo, 1), 3)})
        cpr_png = _png(Image.fromarray(arr))
    return {"sections": sections, "cpr": cpr_png, "cpr_levels": cpr_levels}


def write_synthetic(job_dir: Path, rows: list[dict], seed: str) -> None:
    """MOCK: a phantom with the dummy areas, so the demo has an analysis view too."""
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    n, px = 80, 0.5
    yy, xx = np.mgrid[0:n, 0:n]
    arrays, levels, z_index = {}, [], []
    length = 600
    cpr = rng.normal(40, 12, (length, n)).astype(np.float32)
    cpr_a = np.zeros((length, n), np.uint8)
    cpr_t = np.zeros((length, n), np.uint8)
    for i, r in enumerate(rows):
        lv = r["level"]
        radius = math.sqrt(max(r.get("aorta_area_mm2") or 400.0, 1.0) / math.pi) / px
        aorta = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2 <= radius ** 2).astype(np.uint8)
        ct = rng.normal(40, 12, (n, n)).astype(np.float32)
        ct[aorta.astype(bool)] = 300
        arrays[f"slice_{lv}"] = ct.astype(np.int16)
        arrays[f"aorta_{lv}"] = aorta
        if r.get("tl_area_mm2"):
            tl_r = math.sqrt(r["tl_area_mm2"] / math.pi) / px
            tl = ((yy - n / 2) ** 2 + (xx - n / 2 - (radius - tl_r)) ** 2 <= tl_r ** 2) & aorta.astype(bool)
            arrays[f"tl_{lv}"] = tl.astype(np.uint8)
        levels.append(lv)
        z = 150 + i * 250
        z_index.append(z)
    rad = np.interp(np.arange(length), [z for z in z_index] or [0],
                    [math.sqrt((r.get("aorta_area_mm2") or 400) / math.pi) / px for r in rows] or [20])
    for zi in range(length):
        band = np.abs(np.arange(n) - n / 2) <= rad[zi]
        cpr_a[zi, band] = 1
        cpr_t[zi, band & (np.arange(n) > n / 2 - rad[zi] * 0.2)] = 1 if rows and rows[0].get("tl_area_mm2") else 0
        cpr[zi, band] = 300
    arrays.update(cpr=cpr.astype(np.int16), cpr_aorta=cpr_a, cpr_tl=cpr_t,
                  spacing=np.array([px, px, px], np.float32),
                  levels=np.array(levels), z_index=np.array(z_index, np.int32))
    np.savez_compressed(job_dir / VIEWS_FILE, **arrays)
