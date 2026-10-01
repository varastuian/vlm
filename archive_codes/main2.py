#!/usr/bin/env python3
"""
Aradan change-detection pipeline — no VLM.

  1. Load Sentinel-2 before/after pairs from stac_cache/raw.
  2. Detect changed areas with DINOv2 + SAM.
  3. Export regions.geojson and a Leaflet viewer with the polygons drawn
     on a Google satellite basemap. Each popup shows the raw numeric
     evidence (score, DINO, dNDVI, dNDBI) so you can eyeball whether the
     region is real.

Requirements:
    pip install torch torchvision --break-system-packages
    pip install git+https://github.com/facebookresearch/segment-anything.git --break-system-packages
    # + numpy, opencv-python, rasterio, pillow
    # SAM checkpoint: sam_vit_b_01ec64.pth in working dir
"""

import glob
import itertools
import json
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np


# =====================================================================
# Inlined dino_sam_cd
# =====================================================================
_DINO_CACHE = {}
_SAM_CACHE = {}


def load_dino(model_name="dinov2_vits14", device=None):
    key = (model_name, device)
    if key in _DINO_CACHE:
        return _DINO_CACHE[key]
    try:
        import torch
    except ImportError:
        print("Warning: torch not available", file=sys.stderr)
        return None, None, None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not load DINOv2 ({e})", file=sys.stderr)
        return None, None, None
    model.eval().to(device)
    result = (model, device, 14)
    _DINO_CACHE[key] = result
    return result


def _preprocess_for_dino(img_bgr, patch_size):
    import torch
    H, W = img_bgr.shape[:2]
    new_h = max(patch_size, (H // patch_size) * patch_size)
    new_w = max(patch_size, (W // patch_size) * patch_size)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, (new_h, new_w)


def dino_patch_features(img_bgr, dino_model, device, patch_size=14):
    import torch
    tensor, (new_h, new_w) = _preprocess_for_dino(img_bgr, patch_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        out = dino_model.forward_features(tensor)
        tokens = out["x_norm_patchtokens"]
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
    h_p, w_p = new_h // patch_size, new_w // patch_size
    return tokens[0].reshape(h_p, w_p, -1).cpu().numpy()


def dino_change_map(before, after, dino_model, device, patch_size=14):
    f1 = dino_patch_features(before, dino_model, device, patch_size)
    f2 = dino_patch_features(after, dino_model, device, patch_size)
    if f1.shape[:2] != f2.shape[:2]:
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]), interpolation=cv2.INTER_LINEAR)
    cos_sim = np.sum(f1 * f2, axis=-1)
    cos_dist = 1.0 - cos_sim
    H, W = before.shape[:2]
    full = cv2.resize(cos_dist.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(full, 0, 2)


def load_sam(checkpoint_path, model_type="vit_b", device=None,
             points_per_side=64, pred_iou_thresh=0.75, min_mask_region_area=60):
    key = (checkpoint_path, model_type, device)
    if key in _SAM_CACHE:
        return _SAM_CACHE[key]
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        print("Warning: segment-anything / torch not available", file=sys.stderr)
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not load SAM checkpoint ({e})", file=sys.stderr)
        return None
    sam.to(device)
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    _SAM_CACHE[key] = gen
    return gen


def sam_change_regions(after, change_map, mask_generator, min_change_score=0.15,
                       min_area=40, max_regions=15, candidate_mask=None,
                       min_change_overlap=0.05):
    if candidate_mask is not None and candidate_mask.shape != change_map.shape:
        raise ValueError("candidate_mask and change_map must have the same shape")
    candidate = candidate_mask.astype(bool) if candidate_mask is not None else None

    masks = mask_generator.generate(after)
    scored = []
    for m in masks:
        seg = m["segmentation"]
        if seg.sum() < min_area:
            continue
        if candidate is not None and float(candidate[seg].mean()) < min_change_overlap:
            continue
        score = float(change_map[seg].mean())
        if score < min_change_score:
            continue
        x, y, w, h = m["bbox"]
        scored.append(((int(x), int(y), int(w), int(h)), score, seg))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:max_regions]


# =====================================================================
# Config — tuned for Aradan, Iran
# =====================================================================
CENTER = (35.2494, 52.4930)
SIZE_PX = 3072

PATCH_PX = 512
OVERLAP_PX = 128
DINO_THRESH = 0.15
INDEX_THRESH = 0.20          # directional dNDBI threshold
NDVI_THRESH = 0.15           # symmetric |dNDVI| threshold
MIN_SCORE = 0.15
MIN_AREA = 40
MAX_REGIONS = 15
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
DEBUG = True
OUT_DIR = "categorized_output"

NEEDED = ("red", "nir", "swir16", "visual")
SCL_INVALID = (3, 9)         # shadow + high-prob cloud only
RAW_RE = re.compile(r"^(S2[ABC]_\w{5}_(\d{8})_\d+)_L2A_(\w+)\.tif$")
BOA_OFFSET_DATE = "20220125"
DINO_MODEL = "dinov2_vits14"
SAM_TYPE, SAM_POINTS = "vit_b", 64
DINO_FULL = 0.6
INDEX_FULL = 0.4
W_DINO = 0.3                 # lean on the index signal
MIN_OVERLAP = 0.05
MAX_SEG_FRAC = 0.85          # keep large urban masks
TILE = 200
PANELS = ("BEFORE", "AFTER", "DINO", "INDEX")

VIEWER_DIR = "stac_cache/previews"
PREVIEW_MAX_SIZE = 1600

CATEGORY_COLORS = {
    # Only used for styling. Without the VLM, everything gets "_default".
    "building/construction":  "#ff3b30",
    "vegetation/agriculture": "#34c759",
    "road/infrastructure":    "#ff9500",
    "water/flooding":         "#007aff",
    "bare soil/land-use":     "#a2845e",
    "vehicle/object":         "#af52de",
    "damage/disaster":        "#ff2d55",
    "no significant change":  "#8e8e93",
    "other":                  "#ffcc00",
    "_default":               "#ff3b30",
}


# =====================================================================
# Raster helpers
# =====================================================================
def find_pair(path):
    folder, name = os.path.split(path)
    m = RAW_RE.match(name)
    if not m:
        sys.exit(f"Unrecognised file name: {name}")
    prefix, date, band = m.groups()
    tile = prefix.split("_")[1]

    others = {}
    for p in glob.glob(os.path.join(folder, f"S2?_{tile}_*_L2A_{band}.tif")):
        pm = RAW_RE.match(os.path.basename(p))
        if pm and pm.group(1) != prefix:
            others[pm.group(2)] = pm.group(1)
    if not others:
        sys.exit(f"No other date found for tile {tile} in {folder}")

    def days_apart(d):
        return abs((datetime.strptime(d, "%Y%m%d") - datetime.strptime(date, "%Y%m%d")).days)

    other_date = max(others, key=days_apart)
    before, after = sorted([(date, prefix), (other_date, others[other_date])])

    scenes = []
    for d, pre in (before, after):
        files = {b: os.path.join(folder, f"{pre}_L2A_{b}.tif") for b in NEEDED}
        scl = os.path.join(folder, f"{pre}_L2A_scl.tif")
        if os.path.exists(scl):
            files["scl"] = scl
        missing = [os.path.basename(f) for f in files.values() if not os.path.exists(f)]
        if missing:
            sys.exit(f"Missing files: {missing}")
        scenes.append({"id": pre, "date": d, **files})
    print(f"before={scenes[0]['id']}  after={scenes[1]['id']}")
    return scenes


def radiometry(path, date):
    import rasterio
    with rasterio.open(path) as src:
        scale, offset = src.scales[0], src.offsets[0]
    if scale == 1.0 and offset == 0.0:
        return 1e-4, (-0.1 if date >= BOA_OFFSET_DATE else 0.0)
    return scale, offset


INPUT_FILE_REF = [None]


def pick_input_file():
    candidates = sorted(glob.glob("stac_cache/raw/S2?_*_L2A_red.tif"))
    if not candidates:
        sys.exit("No S2 tiles in stac_cache/raw/. Download some first.")
    if CENTER is None:
        return candidates[0]

    import rasterio
    from rasterio.warp import transform as warp_transform

    lat, lon = CENTER
    for path in candidates:
        with rasterio.open(path) as src:
            x, y = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            xs, ys = x[0], y[0]
            left, bottom, right, top = src.bounds
            if left <= xs <= right and bottom <= ys <= top:
                print(f"Using tile containing Aradan: {os.path.basename(path)}")
                return path
    print(f"No tile contains {CENTER}, falling back to {candidates[0]}")
    return candidates[0]


def load_scene():
    import rasterio
    import rasterio.windows as rw
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform

    before, after = find_pair(INPUT_FILE_REF[0])

    with rasterio.open(before["red"]) as src:
        ref_transform, ref_crs = src.transform, src.crs
        if CENTER is None:
            col, row = src.width / 2, src.height / 2
        else:
            x, y = warp_transform("EPSG:4326", src.crs, [CENTER[1]], [CENTER[0]])
            col, row = ~src.transform * (x[0], y[0])
        half = SIZE_PX // 2
        win = rw.Window(int(col) - half, int(row) - half, SIZE_PX, SIZE_PX)
        win = win.intersection(rw.Window(0, 0, src.width, src.height))
    win = rw.Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
    H, W = int(win.height), int(win.width)
    print(f"Window: {W}x{H}px around {CENTER}")

    def read(path, resampling):
        with rasterio.open(path) as src:
            if src.crs != ref_crs:
                sys.exit(f"{os.path.basename(path)}: different CRS")
            if src.transform == ref_transform:
                w = win
            elif src.res[0] > 10:
                w = rw.from_bounds(*rw.bounds(win, ref_transform), transform=src.transform)
            else:
                sys.exit(f"{os.path.basename(path)}: 10 m grid differs from before scene")
            return src.read(window=w, out_shape=(src.count, H, W), resampling=resampling)

    arrays = {}
    meta = {"crs": ref_crs.to_string(),
            "transform": tuple(rw.transform(win, ref_transform))[:6]}
    for when, sc in (("before", before), ("after", after)):
        for band in ("red", "nir", "swir16"):
            arrays[(when, band)] = read(sc[band], Resampling.bilinear)[0]
        arrays[(when, "visual")] = np.ascontiguousarray(
            read(sc["visual"], Resampling.bilinear).transpose(1, 2, 0)[..., ::-1])
        if "scl" in sc:
            arrays[(when, "scl")] = read(sc["scl"], Resampling.nearest)[0]
        scale, offset = radiometry(sc["red"], sc["date"])
        meta[when] = {"id": sc["id"], "date": sc["date"],
                      "scale": scale, "offset": offset}
    return Scene(arrays, meta)


class Scene:
    def __init__(self, arrays, meta):
        self.raw, self.meta = arrays, meta
        self.h, self.w = arrays[("before", "red")].shape
        self.has_scl = ("before", "scl") in arrays and ("after", "scl") in arrays
        self.shift = self._global_shift()
        for when in ("before", "after"):
            m = meta[when]
            red = self.refl(when, "red", (slice(None, None, 8), slice(None, None, 8)))
            print(f"{when:6s} scale={m['scale']} offset={m['offset']} | "
                  f"red p1={np.percentile(red, 1):.3f} median={np.median(red):.3f}")
        print(f"cloud mask: {'SCL' if self.has_scl else 'OFF'} | "
              f"median shift removed: dNDVI={self.shift[0]:+.3f} dNDBI={self.shift[1]:+.3f}")

    def refl(self, when, band, sl):
        m = self.meta[when]
        return np.clip(self.raw[(when, band)][sl].astype(np.float32)
                       * m["scale"] + m["offset"], 0, 1)

    def rgb(self, when, sl):
        return self.raw[(when, "visual")][sl].copy()

    def indices(self, when, sl):
        red, nir, swir = (self.refl(when, n, sl) for n in ("red", "nir", "swir16"))
        return ((nir - red) / np.maximum(nir + red, 0.01),
                (swir - nir) / np.maximum(swir + nir, 0.01))

    def valid(self, sl):
        bad = np.zeros(self.raw[("before", "red")][sl].shape, bool)
        for when in ("before", "after"):
            bad |= self.raw[(when, "red")][sl] == 0
            if self.has_scl:
                bad |= np.isin(self.raw[(when, "scl")][sl], SCL_INVALID)
        return cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0

    def _global_shift(self):
        s = (slice(None, None, 8), slice(None, None, 8))
        nb, bb = self.indices("before", s)
        na, ba = self.indices("after", s)
        ok = self.valid(s)
        return float(np.median((na - nb)[ok])), float(np.median((ba - bb)[ok]))

    def latlon(self, col, row):
        try:
            from rasterio.transform import Affine
            from rasterio.warp import transform
        except ImportError:
            return None
        x, y = Affine(*self.meta["transform"]) * (col, row)
        lon, lat = transform(self.meta["crs"], "EPSG:4326", [x], [y])
        return lat[0], lon[0]


# =====================================================================
# Patch analysis
# =====================================================================
def patch_starts(n, size, step):
    if n <= size:
        return [0]
    s = list(range(0, n - size + 1, step))
    if s[-1] + size < n:
        s.append(n - size)
    return s


def index_composite(dndvi, dndbi):
    r = np.clip(dndbi / 0.4, 0, 1)
    g = np.clip(dndvi / 0.4, 0, 1)
    b = np.clip(-dndvi / 0.4, 0, 1)
    return (np.dstack([r, g, b]) * 255).astype(np.uint8)


def evidence_row(before, after, heat, idx_rgb, box, seg, pad=0.3):
    x, y, w, h = box
    H, W = before.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x1 = max(0, x - px); y1 = max(0, y - py)
    x2 = min(W, x + w + px); y2 = min(H, y + h + py)
    seg_t = cv2.resize(seg[y1:y2, x1:x2].astype(np.uint8), (TILE, TILE),
                       interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(seg_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx = int((x + w / 2 - x1) / max(x2 - x1, 1) * TILE)
    cy = int((y + h / 2 - y1) / max(y2 - y1, 1) * TILE)
    panels = []
    for img, outline in ((before, True), (after, True), (heat, False), (idx_rgb, False)):
        p = cv2.resize(img[y1:y2, x1:x2], (TILE, TILE), interpolation=cv2.INTER_LINEAR)
        if outline:
            cv2.drawContours(p, cnts, -1, (0, 255, 255), 1)
        panels.append(p)
    cv2.drawMarker(panels[1], (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    return np.hstack(panels)


def save_debug(path, before, after, valid, heat, idx_rgb, cand, found_masks):
    def gray(m):
        return cv2.cvtColor((m > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    after_o = after.copy()
    for seg in found_masks:
        cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(after_o, cnts, -1, (0, 255, 255), 1)
    tiles = [(before, "before"), (after_o, "after + SAM segments"),
             (gray(valid), "valid (white=clear)"), (heat, "DINO"),
             (idx_rgb, "INDEX (r=dNDBI g=dNDVI b=-dNDVI)"),
             (gray(cand), "candidates")]
    for img, label in tiles:
        cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(path, np.vstack([np.hstack([t[0] for t in tiles[:3]]),
                                 np.hstack([t[0] for t in tiles[3:]])]))


def detect_patch(scene, y0, x0, n, models):
    sl = (slice(y0, min(y0 + PATCH_PX, scene.h)),
          slice(x0, min(x0 + PATCH_PX, scene.w)))

    before, after = scene.rgb("before", sl), scene.rgb("after", sl)
    valid = scene.valid(sl)
    nb, bb = scene.indices("before", sl)
    na, ba = scene.indices("after", sl)
    dndvi = cv2.GaussianBlur(na - nb - scene.shift[0], (5, 5), 0) * valid
    dndbi = cv2.GaussianBlur(ba - bb - scene.shift[1], (5, 5), 0) * valid
    idx_mag = np.maximum(np.abs(dndvi), np.abs(dndbi))

    dino_map = dino_change_map(before, after, *models["dino"])
    if dino_map.shape != valid.shape:
        dino_map = cv2.resize(dino_map, (valid.shape[1], valid.shape[0]))
    dino_map = dino_map * valid

    # Directional candidate: built-up/bare GAIN, or any strong NDVI change.
    built_gain = (dndbi > INDEX_THRESH).astype(np.uint8) * 255
    veg_change = (np.abs(dndvi) > NDVI_THRESH).astype(np.uint8) * 255
    dino_hit = (dino_map > DINO_THRESH).astype(np.uint8) * 255
    cand = cv2.bitwise_or(cv2.bitwise_or(built_gain, veg_change), dino_hit)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    found = []
    if np.count_nonzero(cand) >= MIN_AREA:
        score_map = (W_DINO * np.clip(dino_map / DINO_FULL, 0, 1)
                     + (1 - W_DINO) * np.clip(idx_mag / INDEX_FULL, 0, 1)).astype(np.float32)
        found = sam_change_regions(
            after, score_map, models["sam"], min_change_score=MIN_SCORE,
            min_area=MIN_AREA, max_regions=MAX_REGIONS, candidate_mask=cand,
            min_change_overlap=MIN_OVERLAP)
    found = [(b, s, np.asarray(m).astype(bool)) for b, s, m in found]
    too_big = [f for f in found if f[2].mean() > MAX_SEG_FRAC]
    found = [f for f in found if f[2].any() and f[2].mean() <= MAX_SEG_FRAC]

    print(f"  valid={valid.mean():.0%}  DINO p99={np.percentile(dino_map, 99):.2f}  "
          f"|dNDVI| p99={np.percentile(np.abs(dndvi), 99):.2f}  "
          f"dNDBI p99={np.percentile(dndbi, 99):.2f}  "
          f"cand={np.count_nonzero(cand)}px  SAM kept={len(found)} "
          f"dropped_too_big={len(too_big)}")

    heat = cv2.applyColorMap((np.clip(dino_map / 0.8, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    idx_rgb = index_composite(dndvi, dndbi)
    if DEBUG:
        save_debug(os.path.join(OUT_DIR, "debug", f"patch_{n:02d}.png"),
                   before, after, valid, heat, idx_rgb, cand, [f[2] for f in found])

    regions = []
    for box, score, seg in found:
        cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
        regions.append({
            "box": (box[0] + x0, box[1] + y0, box[2], box[3]),
            "score": float(score),
            "dino": float(dino_map[seg].mean()),
            "dndvi": float(dndvi[seg].mean()),
            "dndbi": float(dndbi[seg].mean()),
            "row": evidence_row(before, after, heat, idx_rgb, box, seg),
            "contours": list(cnts),
        })
    return regions


def merge_regions(regions, overlap=0.3):
    def overlap_of_smaller(a, b):
        iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        return 0.0 if iw <= 0 or ih <= 0 else iw * ih / min(a[2] * a[3], b[2] * b[3])

    groups = []
    for r in sorted(regions, key=lambda r: r["box"][2] * r["box"][3], reverse=True):
        for g in groups:
            if overlap_of_smaller(r["box"], g["box"]) >= overlap:
                g["score"] = max(g["score"], r["score"])
                g["contours"] = g["contours"] + r["contours"]
                break
        else:
            groups.append(dict(r))
    return sorted(groups, key=lambda g: g["score"], reverse=True)[:MAX_REGIONS]


# =====================================================================
# GeoJSON export
# =====================================================================
def contours_to_geojson(scene, regions):
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp_transform

    transform = Affine(*scene.meta["transform"])
    crs = scene.meta["crs"]

    def ring_to_lonlat(ring_px):
        gx, gy = [], []
        for x, y in ring_px:
            tx, ty = transform * (float(x), float(y))
            gx.append(tx); gy.append(ty)
        lons, lats = warp_transform(crs, "EPSG:4326", gx, gy)
        return [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]

    features = []
    tile = scene.meta["before"]["id"].split("_")[1]

    for i, r in enumerate(regions, start=1):
        for c in r["contours"]:
            ring_px = c.reshape(-1, 2).tolist()
            if len(ring_px) < 3:
                continue
            ring = ring_to_lonlat(ring_px)
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            x, y, w, h = r["box"]
            ll = scene.latlon(x + w / 2, y + h / 2)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "tile": tile, "index": i,
                    "score": round(r["score"], 3),
                    "dino": round(r["dino"], 3),
                    "dndvi": round(r["dndvi"], 3),
                    "dndbi": round(r["dndbi"], 3),
                    "lat": round(ll[0], 6) if ll else None,
                    "lon": round(ll[1], 6) if ll else None,
                },
            })
    return {"type": "FeatureCollection", "features": features}


# =====================================================================
# Preview JPGs (auto-detect input range)
# =====================================================================
def make_preview(tif_path, jpg_path):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from PIL import Image

    with rasterio.open(tif_path) as src:
        scale = min(1.0, PREVIEW_MAX_SIZE / max(src.width, src.height))
        width = max(1, int(src.width * scale))
        height = max(1, int(src.height * scale))

        if src.count >= 3:
            data = src.read([1, 2, 3],
                            out_shape=(3, height, width),
                            resampling=Resampling.bilinear)
            data = np.moveaxis(data, 0, -1)
        else:
            data = src.read(1, out_shape=(height, width),
                            resampling=Resampling.bilinear)
            data = np.stack([data, data, data], axis=-1)

        data = data.astype(np.float32)
        finite = data[np.isfinite(data)]
        p_hi = float(np.percentile(finite, 99)) if finite.size else 1.0

        if p_hi <= 1.5:
            # Reflectance. Shared stretch preserves color.
            lo, hi = 0.0, 0.30
            data = np.clip((data - lo) / (hi - lo), 0, 1)
        else:
            # 8-bit (0-255) or 16-bit. Shared percentile stretch.
            lo = float(np.percentile(finite, 2))
            hi = float(np.percentile(finite, 98))
            if hi > lo:
                data = np.clip((data - lo) / (hi - lo), 0, 1)
            else:
                data = data / max(p_hi, 1.0)

        data = np.nan_to_num(data)
        data = (data * 255).astype(np.uint8)

        Image.fromarray(data, "RGB").save(jpg_path, "JPEG", quality=92)

        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        west, south, east, north = bounds
        return {
            "crs": str(src.crs),
            "width": src.width, "height": src.height,
            "west": west, "south": south, "east": east, "north": north,
            "lat": (south + north) / 2, "lon": (west + east) / 2,
        }


# =====================================================================
# Leaflet viewer (no VLM)
# =====================================================================
def write_viewer(folder, scenes, change_geojson):
    from pathlib import Path

    html_file = Path(folder) / "index.html"

    js_scenes = []
    for idx, sc in enumerate(scenes):
        info = sc["info"]
        js_scenes.append(
            "{"
            f"id:{idx},"
            f'tile:"{sc["tile"]}",'
            f'before:"{sc["before_jpg"]}",'
            f'after:"{sc["after_jpg"]}",'
            f'beforeDate:"{sc["before_date"]}",'
            f'afterDate:"{sc["after_date"]}",'
            f'west:{info["west"]}, south:{info["south"]},'
            f'east:{info["east"]}, north:{info["north"]},'
            f'center:[{info["lat"]},{info["lon"]}]'
            "}")
    js_scenes_str = "[\n" + ",\n".join(js_scenes) + "\n]"
    js_change_str = json.dumps(change_geojson, ensure_ascii=False)

    html_doc = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Aradan — change regions</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html, body { margin:0; padding:0; height:100%; font-family: Arial, sans-serif; background:#f4f4f4; }
#map { position:absolute; top:0; left:0; right:0; bottom:0; z-index:1; }
#sidebar {
    position:absolute; top:10px; right:10px; width:400px;
    max-height: calc(100% - 20px); overflow-y:auto;
    background: rgba(255,255,255,0.96);
    border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,0.3);
    padding:15px; z-index:1000; font-size:13px;
}
#sidebar h1 { font-size:16px; margin:0 0 12px 0; padding-bottom:10px; border-bottom:1px solid #ddd; }
#sidebar h2 { font-size:14px; margin:16px 0 8px 0; color:#222; }
.card { background:white; padding:10px; margin-bottom:12px; border-radius:8px;
        box-shadow:0 1px 4px #ccc; cursor:pointer; transition: box-shadow 0.2s; }
.card:hover { box-shadow:0 2px 8px rgba(0,0,0,0.25); }
.card.active { outline:3px solid #4285f4; }
.card h2 { margin:0 0 8px 0; font-size:14px; color:#333; }
.images { display:flex; gap:8px; }
.image-box { flex:1; text-align:center; }
.image-box h3 { margin:0 0 4px 0; font-size:11px; color:#666; }
.image-box img { width:100%; display:block; border-radius:4px; border:1px solid #ccc; }
.info { margin-top:8px; font-family: monospace; font-size:11px; color:#555; line-height:1.5; }
.region-row { background:white; padding:8px; margin-bottom:6px; border-radius:6px;
              box-shadow:0 1px 3px #ddd; cursor:pointer; border-left:4px solid #ff3b30;
              font-size:12px; }
.region-row:hover { box-shadow:0 1px 6px rgba(0,0,0,0.25); }
.region-row .stats { font-family:monospace; font-size:11px; color:#555; }
@media (max-width: 900px) { #sidebar { width: calc(100% - 20px); max-height:45%;
                                         top:auto; bottom:10px; } }
</style>
</head>
<body>
<div id="map"></div>
<div id="sidebar">
<h1>Aradan — change regions (no VLM)</h1>
<div class="legend" style="background:#fafafa;border:1px solid #ddd;border-radius:6px;padding:8px;margin-bottom:12px;font-size:11px;font-family:monospace;">
<div><b>Evidence to verify each region:</b></div>
<div>score  = combined confidence (DINO + index)</div>
<div>DINO   = semantic change, 0 = same, &gt;0.4 = strong</div>
<div>dNDVI  = + veg gain, - veg loss</div>
<div>dNDBI  = + more built-up/bare, - less built-up</div>
</div>
<div id="scenes"></div>
<h2 id="regions-title" style="display:none;">Detected change regions</h2>
<div id="regions"></div>
</div>
<script>
const scenes = __SCENES__;
const changeGeoJSON = __CHANGE__;

const map = L.map('map', { zoomControl:true, worldCopyJump:true, preferCanvas:true })
    .setView([35.2494, 52.4930], 12);

L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    { maxZoom: 21, attribution: '&copy; Google' }).addTo(map);

const rectangles = {};
const allBounds = [];
const scenesEl = document.getElementById('scenes');

function popupHtml(scene) {
    return `
        <div style="font-weight:bold;text-align:center;">${scene.tile}</div>
        <img style="width:260px;display:block;border-radius:4px;margin:4px 0;" src="${scene.before}">
        <div style="font-weight:bold;font-size:12px;text-align:center;">BEFORE — ${scene.beforeDate}</div>
        <img style="width:260px;display:block;border-radius:4px;margin:4px 0;" src="${scene.after}">
        <div style="font-weight:bold;font-size:12px;text-align:center;">AFTER — ${scene.afterDate}</div>
    `;
}

scenes.forEach(scene => {
    const bounds = [[scene.south, scene.west], [scene.north, scene.east]];
    allBounds.push(bounds);
    const rect = L.rectangle(bounds, {
        color: '#4285f4', weight: 2, fillOpacity: 0.04, dashArray: '6 6'
    }).addTo(map);
    rect.bindPopup(popupHtml(scene), { maxWidth: 300, minWidth: 280 });
    rect.on('click', () => setActiveCard(scene.id));
    rectangles[scene.id] = rect;

    scenesEl.innerHTML += `
        <div class="card" id="card-${scene.id}" onclick="focusScene(${scene.id})">
            <h2>${scene.tile}</h2>
            <div class="images">
                <div class="image-box"><h3>BEFORE — ${scene.beforeDate}</h3><img src="${scene.before}"></div>
                <div class="image-box"><h3>AFTER — ${scene.afterDate}</h3><img src="${scene.after}"></div>
            </div>
            <div class="info">Center: ${scene.center[0].toFixed(5)}, ${scene.center[1].toFixed(5)}</div>
        </div>`;
});

if (allBounds.length > 0) {
    map.fitBounds(allBounds, { padding: [40, 40] });
}

const regionLayer = L.geoJSON(changeGeoJSON, {
    style: function() {
        return { color: '#ff3b30', weight: 2, fillColor: '#ff3b30',
                 fillOpacity: 0.30, dashArray: '3 3' };
    },
    onEachFeature: function(feature, layer) {
        const p = feature.properties || {};
        const title = `#${p.index || '?'}`;
        const stats = [
            p.score !== undefined ? `score=${(+p.score).toFixed(2)}` : null,
            p.dino !== undefined ? `DINO=${(+p.dino).toFixed(2)}` : null,
            p.dndvi !== undefined ? `dNDVI=${(+p.dndvi).toFixed(2)}` : null,
            p.dndbi !== undefined ? `dNDBI=${(+p.dndbi).toFixed(2)}` : null,
        ].filter(Boolean).join('  ');
        layer.bindPopup(
            `<div style="font-weight:bold;text-align:center;">${title}</div>` +
            `<div style="font-family:monospace;font-size:12px;">${stats}</div>`);
        layer.on('mouseover', () => layer.setStyle({ fillOpacity: 0.55, weight: 3 }));
        layer.on('mouseout',  () => layer.setStyle({ fillOpacity: 0.30, weight: 2 }));
    }
});

const regionsEl = document.getElementById('regions');
const regionsTitle = document.getElementById('regions-title');
let regionIndex = 0;

if (changeGeoJSON.features && changeGeoJSON.features.length > 0) {
    regionLayer.addTo(map);
    regionsTitle.style.display = 'block';

    changeGeoJSON.features.forEach(f => {
        const p = f.properties || {};
        const rowId = `region-${regionIndex++}`;
        const stats = [
            p.score !== undefined ? `score=${(+p.score).toFixed(2)}` : null,
            p.dino !== undefined ? `DINO=${(+p.dino).toFixed(2)}` : null,
            p.dndvi !== undefined ? `dNDVI=${(+p.dndvi).toFixed(2)}` : null,
            p.dndbi !== undefined ? `dNDBI=${(+p.dndbi).toFixed(2)}` : null,
        ].filter(Boolean).join('  ');

        regionsEl.innerHTML += `
            <div class="region-row" id="${rowId}">
                <div><b>#${p.index}</b></div>
                <div class="stats">${stats}</div>
            </div>`;

        document.getElementById(rowId).addEventListener('click', () => {
            try {
                map.fitBounds(L.geoJSON(f).getBounds(),
                    { padding: [60, 60], maxZoom: 17 });
            } catch(e) {}
            regionLayer.eachLayer(l => {
                if (l.feature === f) {
                    l.openPopup();
                    l.setStyle({ fillOpacity: 0.55, weight: 3 });
                    setTimeout(() => l.setStyle({ fillOpacity: 0.30, weight: 2 }), 1500);
                }
            });
        });
    });

    try {
        map.fitBounds(regionLayer.getBounds(), { padding: [40, 40], maxZoom: 15 });
    } catch(e) {}
}

L.control.layers(null, { 'Change regions': regionLayer },
    { collapsed:false, position:'topleft' }).addTo(map);

function setActiveCard(id) {
    document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
    const el = document.getElementById('card-' + id);
    if (el) { el.classList.add('active'); el.scrollIntoView({ behavior:'smooth', block:'nearest' }); }
}
function focusScene(id) {
    const scene = scenes.find(s => s.id === id);
    if (!scene) return;
    map.fitBounds([[scene.south, scene.west], [scene.north, scene.east]],
                  { padding:[40,40] });
    rectangles[id].openPopup();
    setActiveCard(id);
}
</script>
</body>
</html>
"""

    html_doc = (html_doc
                .replace("__SCENES__", js_scenes_str)
                .replace("__CHANGE__", js_change_str))

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return html_file


# =====================================================================
# Sanity check
# =====================================================================
def sanity_check(scene):
    corners = {"top-left": (0, 0), "top-right": (scene.w, 0),
               "bottom-left": (0, scene.h), "bottom-right": (scene.w, scene.h)}
    print(f"\nScene corners (should sit around Aradan {CENTER}):")
    for name, (c, r) in corners.items():
        lat, lon = scene.latlon(c, r)
        print(f"  {name:12s}: {lat:.5f}, {lon:.5f}  "
              f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}")
    lat, lon = scene.latlon(scene.w / 2, scene.h / 2)
    d = ((lat - CENTER[0]) * 111_320) ** 2 + \
        ((lon - CENTER[1]) * 111_320 * np.cos(np.radians(lat))) ** 2
    print(f"CENTER round-trip: requested {CENTER}, window center is "
          f"{lat:.5f}, {lon:.5f} ({d ** 0.5:.0f} m away, should be near 0)")


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs(os.path.join(OUT_DIR, "debug"), exist_ok=True)
    os.makedirs(VIEWER_DIR, exist_ok=True)

    INPUT_FILE_REF[0] = pick_input_file()
    print(f"Input: {INPUT_FILE_REF[0]}")

    scene = load_scene()
    sanity_check(scene)

    print("\nLoading DINOv2 and SAM...")
    dino = load_dino(DINO_MODEL)
    if dino[0] is None:
        sys.exit("DINOv2 failed to load.")
    sam = load_sam(SAM_CHECKPOINT, model_type=SAM_TYPE,
                   points_per_side=SAM_POINTS, min_mask_region_area=MIN_AREA)
    if sam is None:
        sys.exit(f"SAM failed to load (checkpoint: {SAM_CHECKPOINT}).")
    models = {"dino": dino, "sam": sam}

    step = max(PATCH_PX - OVERLAP_PX, 1)
    grid = list(enumerate(
        itertools.product(patch_starts(scene.h, PATCH_PX, step),
                          patch_starts(scene.w, PATCH_PX, step)), start=1))

    regions = []
    for n, (y0, x0) in grid:
        print(f"[patch {n}/{len(grid)}] x={x0} y={y0}")
        regions += detect_patch(scene, y0, x0, n, models)

    regions = merge_regions(regions)
    print(f"\n{len(regions)} region(s) after merging patches.")

    overlay = scene.rgb("after", (slice(None), slice(None)))
    for i, r in enumerate(regions, start=1):
        cv2.drawContours(overlay, r["contours"], -1, (0, 0, 255), 2)
        cv2.putText(overlay, f"#{i}", (r["box"][0], max(14, r["box"][1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "overlay.png"), overlay)

    if not regions:
        print("No regions. Check debug sheets or lower DINO_THRESH / MIN_SCORE.")
        return

    # Text summary
    lines = []
    for i, r in enumerate(regions, start=1):
        x, y, w, h = r["box"]
        ll = scene.latlon(x + w / 2, y + h / 2)
        where = (f"({ll[0]:.5f}, {ll[1]:.5f}) "
                 f"[https://www.google.com/maps?q={ll[0]:.6f},{ll[1]:.6f}]"
                 if ll else f"pixel bbox x={x} y={y} w={w} h={h}")
        lines.append(
            f"#{i} @ {where} score={r['score']:.2f} "
            f"DINO={r['dino']:.2f} dNDVI={r['dndvi']:+.2f} dNDBI={r['dndbi']:+.2f}")
    print("\n--- Regions ---\n" + "\n".join(lines))
    with open(os.path.join(OUT_DIR, "regions.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # GeoJSON
    change_geojson = contours_to_geojson(scene, regions)
    geojson_path = os.path.join(OUT_DIR, "regions.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(change_geojson, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(change_geojson['features'])} polygon(s) to {geojson_path}")

    # Previews
    before_prefix = scene.meta["before"]["id"]
    after_prefix = scene.meta["after"]["id"]
    raw_dir = os.path.dirname(INPUT_FILE_REF[0])
    tile = before_prefix.split("_")[1]

    previews = []
    for prefix, date, tag in (
        (before_prefix, scene.meta["before"]["date"], "BEFORE"),
        (after_prefix,  scene.meta["after"]["date"],  "AFTER"),
    ):
        visual = os.path.join(raw_dir, f"{prefix}_L2A_visual.tif")
        jpg = os.path.join(VIEWER_DIR, f"{tile}_{date}_{tag}.jpg")
        if os.path.exists(visual):
            info = make_preview(visual, jpg)
            previews.append({"jpg": os.path.basename(jpg), "info": info, "date": date})
        else:
            print(f"  no visual for {prefix}, skipping preview")

    scenes_for_viewer = []
    if len(previews) == 2:
        scenes_for_viewer.append({
            "tile": tile,
            "before_date": previews[0]["date"],
            "after_date":  previews[1]["date"],
            "before_jpg": previews[0]["jpg"],
            "after_jpg":  previews[1]["jpg"],
            "info": previews[0]["info"],
        })

    html_file = write_viewer(VIEWER_DIR, scenes_for_viewer, change_geojson)
    print(f"\nHTML viewer: {html_file}")
    print(f"xdg-open {html_file}")


if __name__ == "__main__":
    main()