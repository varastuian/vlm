#!/usr/bin/env python3
"""
STAC change detection: DINOv2 + SAM, with NDVI/NDBI as supporting evidence,
processed in small patches so it fits on a laptop.

Flow
  1. fetch_pair()    download ONLY the area window from Sentinel-2 COGs, once,
                     and cache it as .npy files (re-runs never touch the network)
  2. Scene           memory-maps the cache; patches are sliced lazily from disk
  3. detect_patch()  per patch: DINO change map + index deltas -> fused score
                     -> SAM proposes object-shaped regions
  4. merge_regions() de-duplicate regions found in overlapping patches
  5. classify()      montage (before | after | DINO | index) -> Ollama VLM

Only dino_sam_cd.py is still needed from your old code base.
"""
import argparse
import base64
import gc
import hashlib
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import requests

import dino_sam_cd as dsc  # load_dino, dino_change_map, load_sam, sam_change_regions

BANDS = ("red", "green", "blue", "nir", "swir16", "scl")
SCL_INVALID = (0, 1, 3, 8, 9, 10)  # no-data, saturated, cloud shadow, cloud (med/high), cirrus
MAX_SEG_FRAC = 0.35                # drop SAM segments covering >35% of a patch (terrain/fields)
TILE = 200                         # px per panel in the VLM montage
PANELS = ("BEFORE", "AFTER", "DINO", "INDEX")


# --------------------------------------------------------------------------- #
# 1. STAC download + cache
# --------------------------------------------------------------------------- #
def fetch_pair(args):
    """Download the before/after window once and cache it. Returns the cache folder."""
    key = hashlib.md5(json.dumps([
        round(args.lat, 5), round(args.lon, 5), args.size_px, args.before, args.after,
        args.max_cloud, args.collection, args.stac_url]).encode()).hexdigest()[:12]
    folder = os.path.join(args.cache_dir, key)
    if os.path.exists(os.path.join(folder, "meta.json")):  # meta.json is written last
        print(f"Using cached scenes: {folder}")
        return folder
    os.makedirs(folder, exist_ok=True)

    import rasterio
    import rasterio.windows as rw
    from pystac_client import Client
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform

    client = Client.open(args.stac_url)
    point = {"type": "Point", "coordinates": [args.lon, args.lat]}

    def pick(dates, grid=None):
        items = list(client.search(
            collections=[args.collection], intersects=point, datetime=dates,
            query={"eo:cloud_cover": {"lt": args.max_cloud}}, max_items=50).items())
        if grid:  # same MGRS tile => identical pixel grid, so before/after align exactly
            items = [i for i in items if i.properties.get("grid:code") == grid]
        if not items:
            sys.exit(f"No scene for {dates}" + (f" on tile {grid}" if grid else "")
                     + f" with cloud cover < {args.max_cloud}%.")
        return min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))

    items = {"before": pick(args.before)}
    items["after"] = pick(args.after, items["before"].properties.get("grid:code"))
    for when, it in items.items():
        print(f"{when:6s}: {it.id}  {it.properties['datetime'][:10]}  "
              f"cloud={it.properties.get('eo:cloud_cover', '?')}%")

    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                       GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES")
    with env:
        # Reference 10 m grid + window centred on the point (clipped to the tile).
        with rasterio.open(items["before"].assets["red"].href) as src:
            ref_transform, ref_crs = src.transform, src.crs
            x, y = warp_transform("EPSG:4326", src.crs, [args.lon], [args.lat])
            col, row = ~src.transform * (x[0], y[0])
            half = args.size_px // 2
            win = rw.Window(int(col) - half, int(row) - half, args.size_px, args.size_px)
            win = win.intersection(rw.Window(0, 0, src.width, src.height))
        win = rw.Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
        H, W = int(win.height), int(win.width)
        if (H, W) != (args.size_px, args.size_px):
            print(f"Note: point is near the tile edge, window clipped to {W}x{H}px.")

        def read(item, name):
            resampling = Resampling.nearest if name == "scl" else Resampling.bilinear
            with rasterio.open(item.assets[name].href) as src:
                if src.crs != ref_crs:
                    sys.exit("Before/after scenes use different CRS; pick another date range.")
                if src.res[0] == 10 and src.transform != ref_transform:
                    sys.exit("Before/after 10 m grids differ; pick another date range.")
                if src.transform == ref_transform:
                    w = win
                else:  # 20 m bands (swir16, scl): same ground window, upsampled to 10 m
                    w = rw.from_bounds(*rw.bounds(win, ref_transform), transform=src.transform)
                return src.read(1, window=w, out_shape=(H, W), resampling=resampling)

        for when, it in items.items():
            for name in BANDS:
                print(f"  downloading {when}/{name}")
                np.save(os.path.join(folder, f"{when}_{name}.npy"), read(it, name))

    def radiometry(item):
        rb = (item.assets["red"].extra_fields.get("raster:bands") or [{}])[0]
        try:
            baseline = float(item.properties.get("s2:processing_baseline") or 0)
        except ValueError:
            baseline = 0.0
        return {"scale": rb.get("scale", 1e-4),
                "offset": rb.get("offset", -0.1 if baseline >= 4.0 else 0.0)}

    meta = {
        "size": [H, W],
        "crs": ref_crs.to_string(),
        "transform": list(tuple(rw.transform(win, ref_transform))[:6]),
        "before": {"id": items["before"].id, **radiometry(items["before"])},
        "after": {"id": items["after"].id, **radiometry(items["after"])},
    }
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return folder


# --------------------------------------------------------------------------- #
# 2. Scene: lazy, memory-mapped access to the cache
# --------------------------------------------------------------------------- #
class Scene:
    def __init__(self, folder):
        with open(os.path.join(folder, "meta.json")) as f:
            self.meta = json.load(f)
        self.h, self.w = self.meta["size"]
        self.raw = {(when, b): np.load(os.path.join(folder, f"{when}_{b}.npy"), mmap_mode="r")
                    for when in ("before", "after") for b in BANDS}

    def _refl(self, when, band, sl):
        m = self.meta[when]
        arr = self.raw[(when, band)][sl].astype(np.float32)
        return np.clip(arr * m["scale"] + m["offset"], 0.0, 1.0)

    def rgb(self, when, sl):
        """BGR uint8 with a FIXED stretch, so before/after (and all patches) are comparable."""
        b, g, r = (self._refl(when, n, sl) for n in ("blue", "green", "red"))
        img = np.clip(np.dstack([b, g, r]) / 0.3, 0, 1) ** 0.7
        return (img * 255).astype(np.uint8)

    def indices(self, when, sl):
        red, nir, swir = (self._refl(when, n, sl) for n in ("red", "nir", "swir16"))
        ndvi = (nir - red) / np.maximum(nir + red, 0.01)
        ndbi = (swir - nir) / np.maximum(swir + nir, 0.01)
        return ndvi, ndbi

    def valid(self, sl):
        """False where either date has cloud / shadow / no-data (slightly dilated)."""
        bad = (np.isin(self.raw[("before", "scl")][sl], SCL_INVALID)
               | np.isin(self.raw[("after", "scl")][sl], SCL_INVALID)).astype(np.uint8)
        bad = cv2.dilate(bad, np.ones((5, 5), np.uint8))
        return bad == 0

    def latlon(self, col, row):
        from rasterio.transform import Affine
        from rasterio.warp import transform
        x, y = Affine(*self.meta["transform"]) * (col, row)
        lon, lat = transform(self.meta["crs"], "EPSG:4326", [x], [y])
        return lat[0], lon[0]


def _starts(n, size, step):
    if n <= size:
        return [0]
    s = list(range(0, n - size + 1, step))
    if s[-1] + size < n:
        s.append(n - size)
    return s


# --------------------------------------------------------------------------- #
# 3. Per-patch detection: DINO + indices -> fused score -> SAM regions
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    box: tuple      # (x, y, w, h) in full-scene pixels
    score: float
    stats: dict     # mean DINO / dNDVI / dNDBI inside the SAM segment
    row: np.ndarray # small before|after|dino|index strip for the VLM montage
    contours: list # segment outline in full-scene pixels (for the overlay)


def index_composite(dndvi, dndbi):
    """BGR: red = vegetation loss, green = vegetation gain, blue = more built-up/bare."""
    b = np.clip(dndbi / 0.4, 0, 1)
    g = np.clip(dndvi / 0.4, 0, 1)
    r = np.clip(-dndvi / 0.4, 0, 1)
    return (np.dstack([b, g, r]) * 255).astype(np.uint8)


def evidence_row(before, after, heat, idx_rgb, box, seg, pad=0.3):
    x, y, w, h = box
    H, W = before.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x1, y1, x2, y2 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    seg_t = cv2.resize(seg[y1:y2, x1:x2].astype(np.uint8), (TILE, TILE),
                       interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(seg_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for img, outline in ((before, True), (after, True), (heat, False), (idx_rgb, False)):
        p = cv2.resize(img[y1:y2, x1:x2], (TILE, TILE), interpolation=cv2.INTER_LINEAR)
        if outline:
            cv2.drawContours(p, cnts, -1, (0, 255, 255), 1)
        panels.append(p)
    return np.hstack(panels)


def detect_patch(scene, y0, x0, models, args):
    sl = (slice(y0, min(y0 + args.patch_px, scene.h)), slice(x0, min(x0 + args.patch_px, scene.w)))
    before, after = scene.rgb("before", sl), scene.rgb("after", sl)
    valid = scene.valid(sl)
    nb, bb = scene.indices("before", sl)
    na, ba = scene.indices("after", sl)
    dndvi = cv2.GaussianBlur(na - nb, (5, 5), 0) * valid
    dndbi = cv2.GaussianBlur(ba - bb, (5, 5), 0) * valid
    idx_mag = np.maximum(np.abs(dndvi), np.abs(dndbi))

    dino_map = dsc.dino_change_map(before, after, *models["dino"])
    if dino_map.shape != valid.shape:
        dino_map = cv2.resize(dino_map, (valid.shape[1], valid.shape[0]))
    dino_map = dino_map * valid

    # A pixel is a candidate if EITHER the semantic or the spectral signal fires.
    cand = ((dino_map > args.dino_thresh) | (idx_mag > args.index_thresh)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if np.count_nonzero(cand) < args.min_area:
        return []

    # Fused score used by SAM to rank segments: DINO leads, indices back it up.
    score_map = (0.7 * np.clip(dino_map / 0.6, 0, 1)
                 + 0.3 * np.clip(idx_mag / 0.4, 0, 1)).astype(np.float32)
    found = dsc.sam_change_regions(
        after, score_map, models["sam"], min_change_score=args.min_score,
        min_area=args.min_area, max_regions=args.per_patch, candidate_mask=cand,
        min_change_overlap=args.min_overlap)
    if not found:
        return []

    heat = cv2.applyColorMap((np.clip(dino_map / 0.8, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    idx_rgb = index_composite(dndvi, dndbi)

    regions = []
    for box, score, seg in found:
        seg = np.asarray(seg).astype(bool)
        if not seg.any() or seg.mean() > MAX_SEG_FRAC:
            continue
        cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
        regions.append(Region(
            box=(box[0] + x0, box[1] + y0, box[2], box[3]),
            score=float(score),
            stats={"dino": float(dino_map[seg].mean()),
                   "dndvi": float(dndvi[seg].mean()),
                   "dndbi": float(dndbi[seg].mean())},
            row=evidence_row(before, after, heat, idx_rgb, box, seg),
            contours=list(cnts)))
    return regions


def _overlap_of_smaller(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / min(aw * ah, bw * bh)


def merge_regions(regions, max_regions, overlap=0.5):
    """Patches overlap, so the same object can be found twice: keep the best-scoring copy."""
    kept = []
    for r in sorted(regions, key=lambda r: r.score, reverse=True):
        if all(_overlap_of_smaller(r.box, k.box) < overlap for k in kept):
            kept.append(r)
        if len(kept) == max_regions:
            break
    return kept


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# 4. VLM classification (batched so the 4B model sees only a few regions at once)
# --------------------------------------------------------------------------- #
def build_montage(regions, first):
    rows = []
    for i, r in enumerate(regions, start=first):
        bar = np.full((22, TILE * len(PANELS), 3), 255, np.uint8)
        for c, name in enumerate(PANELS):
            cv2.putText(bar, f"#{i} {name}", (c * TILE + 5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows += [bar, r.row]
    return np.vstack(rows)


def make_prompt(regions, first):
    stats = "\n".join(
        f"  #{i}: DINO={r.stats['dino']:.2f}  dNDVI={r.stats['dndvi']:+.2f}  "
        f"dNDBI={r.stats['dndbi']:+.2f}" for i, r in enumerate(regions, start=first))
    return (
        f"Sentinel-2 imagery (10 m/pixel). The image has {len(regions)} rows, regions "
        f"#{first}-#{first + len(regions) - 1}. Each row has 4 panels: BEFORE, AFTER, DINO, INDEX. "
        "A yellow outline on BEFORE/AFTER marks the candidate segment.\n"
        "DINO = semantic change heatmap (red/yellow = content differs, blue = same).\n"
        "INDEX = spectral change: red = vegetation loss, green = vegetation gain, "
        "blue = more built-up/bare surface, magenta = vegetation replaced by built-up/bare "
        "(typical construction or clearing), black = no index change.\n"
        "Mean values inside each segment (DINO > 0.4 is strong; negative dNDVI = vegetation "
        "loss; positive dNDBI = more built-up/bare):\n" + stats + "\n"
        "For EACH region give one line:\n"
        "  #<number>: <category> — <one-sentence description>\n"
        "Category must be one of: building/construction, vegetation/agriculture, "
        "road/infrastructure, water/flooding, bare soil/land-use, vehicle/object, "
        "damage/disaster, no significant change, other.\n"
        "Judge from BEFORE vs AFTER first; DINO and INDEX are supporting evidence only. "
        "If before and after look the same, answer 'no significant change'.")


def ask_ollama(prompt, image_bgr, args):
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode montage")
    r = requests.post(f"{args.ollama_url}/api/generate", timeout=args.ollama_timeout, json={
        "model": args.model, "prompt": prompt, "stream": False, "think": False,
        "images": [base64.b64encode(buf.tobytes()).decode()],
        "keep_alive": args.ollama_keep_alive,
        "options": {"num_predict": args.ollama_max_tokens},
    })
    r.raise_for_status()
    body = r.json()
    text = (body.get("response") or "").strip()
    if not text:
        hint = (" The model only produced 'thinking' -- use the non-thinking "
                "`qwen3-vl:4b-instruct` tag or raise --ollama-max-tokens."
                if body.get("thinking") else "")
        raise RuntimeError(f"Ollama returned no text (done_reason={body.get('done_reason')}).{hint}")
    return text


def classify(regions, args):
    """Save one montage per batch; ask the VLM unless --no-describe. Returns {number: text}."""
    answers = {}
    for start in range(0, len(regions), args.vlm_batch):
        chunk, first = regions[start:start + args.vlm_batch], start + 1
        montage = build_montage(chunk, first)
        cv2.imwrite(os.path.join(args.out, f"montage_{first:02d}.png"), montage)
        if args.no_describe:
            continue
        print(f"Asking {args.model} about regions #{first}-#{first + len(chunk) - 1}...")
        try:
            text = ask_ollama(make_prompt(chunk, first), montage, args)
        except (requests.exceptions.RequestException, RuntimeError) as e:
            print(f"VLM step failed ({e}). Montages and locations are still saved; "
                  f"re-run to retry (scenes are cached).", file=sys.stderr)
            break
        for line in text.splitlines():
            m = re.match(r"^\W*#?(\d+)\s*[:.)]\s*(.*)$", line.strip())
            if m:
                answers[int(m.group(1))] = m.group(2)
    return answers


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="STAC change detection with DINOv2 + SAM (+ NDVI/NDBI)")
    g = ap.add_argument_group("scene (STAC)")
    g.add_argument("--lat", type=float, default=35.2472)
    g.add_argument("--lon", type=float, default=52.4921)
    g.add_argument("--before", default="2020-01-01/2020-03-01", help="STAC datetime range")
    g.add_argument("--after", default="2026-01-01/2026-03-01", help="STAC datetime range")
    g.add_argument("--size-px", type=int, default=2048,
                   help="Area size in 10 m pixels (2048 ~ 20 km). Cached on disk, so this can "
                        "be big; memory use is set by --patch-px, not this.")
    g.add_argument("--max-cloud", type=float, default=30, help="Max scene cloud cover %%")
    g.add_argument("--cache-dir", default="stac_cache")
    g.add_argument("--collection", default="sentinel-2-l2a")
    g.add_argument("--stac-url", default="https://earth-search.aws.element84.com/v1")

    g = ap.add_argument_group("patching (lower --patch-px if you run out of memory)")
    g.add_argument("--patch-px", type=int, default=512)
    g.add_argument("--overlap-px", type=int, default=64)

    g = ap.add_argument_group("detection")
    g.add_argument("--dino-model", default="dinov2_vits14")
    g.add_argument("--dino-thresh", type=float, default=0.3,
                   help="DINO cosine-distance threshold for candidate pixels")
    g.add_argument("--index-thresh", type=float, default=0.2,
                   help="|dNDVI| or |dNDBI| threshold for candidate pixels")
    g.add_argument("--sam-checkpoint", default="sam_vit_b_01ec64.pth")
    g.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    g.add_argument("--sam-points", type=int, default=24, help="SAM grid density (lower = faster)")
    g.add_argument("--min-score", type=float, default=0.25,
                   help="Min mean fused score (0-1) for a SAM segment")
    g.add_argument("--min-overlap", type=float, default=0.10,
                   help="Min fraction of a SAM segment that must lie inside the candidate mask")
    g.add_argument("--min-area", type=int, default=100, help="Min segment area in pixels")
    g.add_argument("--per-patch", type=int, default=6, help="Max regions kept per patch")
    g.add_argument("--max-regions", type=int, default=12, help="Max regions in the final report")

    g = ap.add_argument_group("VLM (Ollama)")
    g.add_argument("--model", default="qwen3-vl:4b-instruct")
    g.add_argument("--ollama-url", default="http://localhost:11434")
    g.add_argument("--ollama-timeout", type=int, default=900)
    g.add_argument("--ollama-keep-alive", default="10m")
    g.add_argument("--ollama-max-tokens", type=int, default=1024)
    g.add_argument("--vlm-batch", type=int, default=4, help="Regions per VLM call")
    g.add_argument("--no-describe", action="store_true")

    ap.add_argument("--out", default="categorized_output")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene = Scene(fetch_pair(args))
    print(f"Scene {scene.w}x{scene.h}px | before={scene.meta['before']['id']} "
          f"after={scene.meta['after']['id']}")

    print("Loading DINOv2 and SAM (once)...")
    dino = dsc.load_dino(args.dino_model)  # (model, device, patch_size)
    if dino[0] is None:
        sys.exit("DINOv2 failed to load.")
    sam = dsc.load_sam(args.sam_checkpoint, model_type=args.sam_model_type,
                       points_per_side=args.sam_points, min_mask_region_area=args.min_area)
    if sam is None:
        sys.exit(f"SAM failed to load (checkpoint: {args.sam_checkpoint}).")
    models = {"dino": dino, "sam": sam}

    step = max(args.patch_px - args.overlap_px, 1)
    grid = list(itertools.product(_starts(scene.h, args.patch_px, step),
                                  _starts(scene.w, args.patch_px, step)))
    regions = []
    for n, (y0, x0) in enumerate(grid, start=1):
        found = detect_patch(scene, y0, x0, models, args)
        print(f"[{n}/{len(grid)}] patch x={x0} y={y0}: {len(found)} region(s)")
        regions += found
        free_memory()

    regions = merge_regions(regions, args.max_regions)
    print(f"{len(regions)} region(s) after merging patches.")

    # Overlay of the whole scene (uint8, fine even for several thousand px per side).
    overlay = scene.rgb("after", (slice(None), slice(None)))
    for i, r in enumerate(regions, start=1):
        cv2.drawContours(overlay, r.contours, -1, (0, 0, 255), 2)
        cv2.putText(overlay, f"#{i}", (r.box[0], max(14, r.box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(args.out, "overlay.png"), overlay)
    print(f"Saved overlay -> {os.path.join(args.out, 'overlay.png')}")

    if not regions:
        print("No candidate regions found. Try lowering --dino-thresh / --min-score.")
        return

    answers = classify(regions, args)

    lines = []
    for i, r in enumerate(regions, start=1):
        x, y, w, h = r.box
        lat, lon = scene.latlon(x + w / 2, y + h / 2)
        line = (f"#{i} @ ({lat:.5f}, {lon:.5f}) [https://www.google.com/maps?q={lat:.6f},{lon:.6f}] "
                f"score={r.score:.2f} DINO={r.stats['dino']:.2f} "
                f"dNDVI={r.stats['dndvi']:+.2f} dNDBI={r.stats['dndbi']:+.2f}")
        if i in answers:
            line += f" | {answers[i]}"
        lines.append(line)
    result = "\n".join(lines)
    print("\n--- Regions ---")
    print(result)
    with open(os.path.join(args.out, "regions.txt"), "w") as f:
        f.write(result + "\n")


if __name__ == "__main__":
    main()