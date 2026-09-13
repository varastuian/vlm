#!/usr/bin/env python3
"""
categorized_change.py — detect ANY kind of change (buildings, vegetation /
deforestation, roads, vehicles, other objects...) between a before/after
image pair, and have Qwen classify + describe each one. No training needed.

Why: bit_cd_infer.py's BIT_CD model is a *building-only* specialist — it was
trained exclusively on LEVIR-CD and structurally cannot recognize other
categories of change, no matter how you tune it at inference time. To get
general-purpose coverage without training a new model, this script combines
THREE independent detectors, OR'd together so any one of them can propose a
region:

  1. Generic pixel/structure diff (SSIM + Lab color distance) — doesn't
     care WHAT changed, only THAT something did. Catches anything visible
     in a plain photo or true-color satellite composite.
  2. Spectral index change (NDVI/NDBI, STAC mode only) — catches changes
     invisible in true-color RGB but obvious in near-infrared/SWIR
     reflectance, e.g. vegetation stress/loss that hasn't visibly browned yet.
  3. BIT_CD (--use-bit-cd) — a real building-change specialist. Its mask is
     OR'd into region proposal too, not just used as a label, so a building
     change the other two detectors miss still gets surfaced.

Each candidate region is then cropped (before/after/change-graph) and sent
to Qwen, which classifies it (building / vegetation-deforestation /
road-infrastructure / vehicle-object / water / other) and describes it —
zero-shot, using Qwen's general world knowledge rather than training data.

Usage:
  python categorized_change.py --before before.png --after after.png --out results
  python categorized_change.py --before before.png --after after.png --use-bit-cd
  python categorized_change.py --lat 40.6 --lon 15.05 \
      --before-start 2019-01-01 --before-end 2019-03-01 \
      --after-start 2022-01-01 --after-end 2022-03-01 --use-bit-cd
"""

import argparse
import base64
import json
import os
import sys

# BIT_CD was trained on 256x256 tiles. A STAC crop smaller than that gets
# padded up by border-replication in bit_cd_infer.predict_change_mask, which
# means the model mostly sees stretched padding rather than real scene
# content and finds nothing. At Sentinel-2's 10m/pixel resolution, a crop
# needs at least this many km across to reach 256px.
BIT_CD_MIN_TILE_PX = 256
SENTINEL2_RES_M = 10
BIT_CD_MIN_BUFFER_KM = round(BIT_CD_MIN_TILE_PX * SENTINEL2_RES_M / 1000, 2)  # 2.56

import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity as ssim

# Reuse the alignment logic already validated in main2.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main2 import load_and_align, align_images  # noqa: E402


def compute_generic_change_mask(before, after, blur_ksize=5, ssim_thresh=0.88,
                                 color_thresh=18, min_area=100):
    """
    Category-agnostic change mask combining two independent signals so
    neither one's blind spot silently drops real changes:
      - grayscale SSIM: catches structural/shape/texture changes
        (construction, demolition, new objects, roads)
      - Lab color distance: catches same-luminance HUE changes that SSIM
        misses entirely (e.g. vegetation turning brown/bare — a classic
        deforestation signature can have nearly identical grayscale
        structure but a big color shift)
    A pixel is flagged as changed if EITHER signal fires.
    """
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    g1b = cv2.GaussianBlur(g1, (blur_ksize, blur_ksize), 0)
    g2b = cv2.GaussianBlur(g2, (blur_ksize, blur_ksize), 0)

    _, diff_map = ssim(g1b, g2b, full=True)
    dissimilarity = ((1.0 - diff_map) * 127.5).astype(np.uint8)
    thresh_val = int((1.0 - ssim_thresh) * 127.5)
    _, struct_mask = cv2.threshold(dissimilarity, thresh_val, 255, cv2.THRESH_BINARY)

    lab1 = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.GaussianBlur(lab1, (blur_ksize, blur_ksize), 0)
    lab2 = cv2.GaussianBlur(lab2, (blur_ksize, blur_ksize), 0)
    color_dist = np.linalg.norm(lab1 - lab2, axis=2)
    _, color_mask = cv2.threshold(color_dist, color_thresh, 255, cv2.THRESH_BINARY)
    color_mask = color_mask.astype(np.uint8)

    mask = cv2.bitwise_or(struct_mask, color_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    changed_pct = 100.0 * np.count_nonzero(mask) / mask.size
    return mask, boxes, changed_pct


def compute_index_change_map(before_idx, after_idx, ndvi_thresh=0.15, ndbi_thresh=0.15):
    """
    Given before/after NDVI/NDBI/NDWI dicts (from stac_fetch.fetch_scene_with_indices),
    return the raw deltas plus a binary mask of pixels where NDVI or NDBI
    moved past threshold — fed into region proposal alongside the generic
    SSIM/color mask so spectral-only changes (invisible in the true-color
    composite) still get picked up as candidate regions.
    """
    dndvi = after_idx["ndvi"] - before_idx["ndvi"]
    dndbi = after_idx["ndbi"] - before_idx["ndbi"]
    index_mask = (((np.abs(dndvi) > ndvi_thresh) | (np.abs(dndbi) > ndbi_thresh))
                  .astype(np.uint8) * 255)
    return dndvi, dndbi, index_mask


def _colorize_diverging_full(delta, pos_bgr, neg_bgr):
    """delta: full-scene float array. Positive values tint pos_bgr, negative
    values tint neg_bgr, magnitude controls intensity. Zero delta = black."""
    pos = np.clip(delta, 0, 1)
    neg = np.clip(-delta, 0, 1)
    img = np.zeros((*delta.shape, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = pos * (pos_bgr[c] / 255.0) + neg * (neg_bgr[c] / 255.0)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _colorize_single_sided_full(delta, bgr):
    """delta: full-scene float array. Only positive values shown, tinted bgr."""
    pos = np.clip(delta, 0, 1)
    img = np.zeros((*delta.shape, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = pos * (bgr[c] / 255.0)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def compute_generic_heatmap_full(before, after, blur_ksize=5):
    """
    Full-scene generic change-intensity heatmap (grayscale SSIM dissimilarity
    + Lab color distance, JET-colorized) — the same two signals used for
    region proposal, precomputed once here so every region's panel crops
    from a consistent full-scene map instead of recomputing SSIM on tiny
    per-region crops (which is noisier on small windows).
    """
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    g1b = cv2.GaussianBlur(g1, (blur_ksize, blur_ksize), 0)
    g2b = cv2.GaussianBlur(g2, (blur_ksize, blur_ksize), 0)
    _, diff_map = ssim(g1b, g2b, full=True)
    struct_dissim = 1.0 - diff_map

    lab1 = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_dist = np.linalg.norm(lab1 - lab2, axis=2)
    color_norm = color_dist / (color_dist.max() + 1e-6)

    combined = np.clip(0.5 * struct_dissim + 0.5 * color_norm, 0, 1)
    heat = (combined * 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)


def merge_overlapping_boxes(boxes, pad=10, iou_merge_thresh=0.05):
    """Merge boxes that overlap or sit close together (after padding), so one
    real changed object doesn't get split into several fragments."""
    if not boxes:
        return []
    padded = [(x - pad, y - pad, x + w + pad, y + h + pad) for (x, y, w, h) in boxes]
    merged = True
    while merged:
        merged = False
        out = []
        used = [False] * len(padded)
        for i in range(len(padded)):
            if used[i]:
                continue
            x1, y1, x2, y2 = padded[i]
            for j in range(i + 1, len(padded)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = padded[j]
                # overlap test
                if x1 < bx2 and bx1 < x2 and y1 < by2 and by1 < y2:
                    x1, y1, x2, y2 = min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2)
                    used[j] = True
                    merged = True
            out.append((x1, y1, x2, y2))
            used[i] = True
        padded = out
    return [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in padded]


def select_top_regions(boxes, max_regions=10):
    """Keep the largest N regions so the VLM prompt/montage stays manageable."""
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    return boxes_sorted[:max_regions]


def crop_with_context(img, box, pad_ratio=0.3, tile_size=200):
    x, y, w, h = box
    H, W = img.shape[:2]
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)


def crop_region_bounds(img_shape, box, pad_ratio=0.3):
    """Same padded-region math as crop_with_context, but returns the pixel
    bounds instead of the crop — used to slice the SAME region out of both
    images and the diff mask consistently."""
    x, y, w, h = box
    H, W = img_shape[:2]
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    return x1, y1, x2, y2


def change_graph_tile(before, after, box, layers, pad_ratio=0.3, tile_size=200):
    """
    Build ONE composite "change graph" for a region: a fixed 2x2 grid so
    Qwen sees every available signal at once, at consistent positions:
        top-left      = BIT_CD building-change mask (white = flagged)
        top-right     = NDVI change (green = vegetation gain, red = loss)
        bottom-left   = NDBI change (blue = new built-up/bare surface)
        bottom-right  = generic structural/color change heatmap (always present)
    Any signal not computed for this run (e.g. --use-bit-cd not passed, or
    not in STAC/index mode) shows as solid gray in that quadrant — gray
    means "not computed", NOT "no change" (which shows as black/dark).
    """
    x1, y1, x2, y2 = crop_region_bounds(before.shape, box, pad_ratio)
    half = tile_size // 2
    gray = np.full((half, half, 3), 128, dtype=np.uint8)

    def crop_or_gray(full_img):
        if full_img is None:
            return gray
        crop = full_img[y1:y2, x1:x2]
        return cv2.resize(crop, (half, half), interpolation=cv2.INTER_NEAREST)

    top_left = crop_or_gray(layers.get("bit_cd"))
    top_right = crop_or_gray(layers.get("ndvi"))
    bottom_left = crop_or_gray(layers.get("ndbi"))
    bottom_right = crop_or_gray(layers.get("generic"))  # always present upstream

    top = np.hstack([top_left, top_right])
    bottom = np.hstack([bottom_left, bottom_right])
    return np.vstack([top, bottom])


def build_montage(before, after, boxes, tile_size=200, label_h=28, layers=None):
    """
    N x 3 grid: row i = [before_crop_i | after_crop_i | change_graph_i], each
    row numbered. The third column is a 2x2 composite of every available
    change signal (BIT_CD / NDVI / NDBI / generic — see change_graph_tile)
    so Qwen can weigh them together instead of picking one signal to show.
    """
    layers = layers or {}
    n = len(boxes)
    cell_h = tile_size + label_h
    montage = np.full((cell_h * n, tile_size * 3, 3), 255, dtype=np.uint8)

    for i, box in enumerate(boxes):
        b_crop = crop_with_context(before, box, tile_size=tile_size)
        a_crop = crop_with_context(after, box, tile_size=tile_size)
        graph = change_graph_tile(before, after, box, layers, tile_size=tile_size)
        y0 = i * cell_h
        cv2.putText(montage, f"#{i+1} BEFORE", (5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(montage, f"#{i+1} AFTER", (tile_size + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(montage, f"#{i+1} CHANGE GRAPH", (tile_size * 2 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        montage[y0 + label_h:y0 + cell_h, 0:tile_size] = b_crop
        montage[y0 + label_h:y0 + cell_h, tile_size:tile_size * 2] = a_crop
        montage[y0 + label_h:y0 + cell_h, tile_size * 2:tile_size * 3] = graph

    return montage


def draw_overlay(after, boxes, source_labels=None):
    overlay = after.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        color = (0, 0, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"#{i+1}"
        if source_labels and source_labels[i]:
            label += f" [{'/'.join(source_labels[i])}]"
        cv2.putText(overlay, label, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return overlay


def encode_image_b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def classify_and_describe(montage, n_regions, source_labels=None, model="qwen3-vl",
                           ollama_url="http://localhost:11434", available_signals=None):
    hint_note = ""
    if source_labels and any(source_labels):
        parts = [f"#{i+1} ({'/'.join(tags)})" for i, tags in enumerate(source_labels) if tags]
        if parts:
            hint_note = ("\nNote: region(s) " + ", ".join(parts) + " were independently "
                         "flagged by a specialist detector (building = BIT_CD building-change "
                         "model, index = spectral NDVI/NDBI change) — treat as a hint, not a "
                         "certainty; verify visually.")

    available_signals = available_signals or set()
    computed = ", ".join(sorted(available_signals)) if available_signals else "generic only"
    not_computed = {"bit_cd", "ndvi", "ndbi"} - available_signals
    gray_note = (f" Signals NOT computed this run: {', '.join(sorted(not_computed))} — their "
                 f"quadrant will show as solid gray, meaning 'not available', NOT 'no change'."
                 if not_computed else "")

    map_explainer = (
        "a CHANGE GRAPH: a fixed 2x2 grid combining every available change signal so you can "
        "weigh them together, always in this layout:\n"
        "  top-left     = BIT_CD building-change mask (white = flagged as a building change)\n"
        "  top-right    = NDVI change (green = vegetation gain, red = vegetation loss/deforestation)\n"
        "  bottom-left  = NDBI change (blue = new built-up/bare surface, i.e. construction)\n"
        "  bottom-right = generic structural/color change heatmap (red/yellow = strong change, "
        "always present)\n"
        f"Computed this run: {computed}.{gray_note}\n"
        "Weigh whichever quadrants are populated together — e.g. strong top-right RED plus little "
        "else usually means vegetation loss/deforestation; strong bottom-left BLUE plus top-left "
        "white usually means new construction."
    )

    prompt = (
        f"This image is a grid of {n_regions} numbered rows, each with 3 crops of "
        f"the same candidate changed region: BEFORE, AFTER, and {map_explainer}\n"
        "For EACH numbered region, on its own line, give:\n"
        "  #<number>: <category> — <one-sentence description>\n"
        "Category must be one of: building (construction/demolition), "
        "vegetation (growth/loss/deforestation), road/infrastructure, "
        "vehicle/object, water, no significant change, other.\n"
        "If the change graph shows only weak/scattered/gray signal with no clear "
        "before/after difference, say 'no significant change' rather than "
        "inventing a category."
        + hint_note
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image_b64(montage)],
        "stream": False,
    }
    resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=240)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def get_bit_cd_mask(before, after, bit_cd_repo=None):
    """
    Run BIT_CD tiled inference and return its full building-change mask
    (same H x W as the input images, 0/255). This now feeds directly into
    region proposal (OR'd with the generic SSIM/color mask and, in STAC
    mode, the index mask) instead of only labeling regions found by other
    detectors — so a building change BIT_CD catches but the generic/index
    signals miss still shows up as a candidate region.
    """
    try:
        import bit_cd_infer as bit
        import torch
    except ImportError:
        print("Warning: bit_cd_infer.py / torch not available, skipping --use-bit-cd",
              file=sys.stderr)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = bit_cd_repo or bit.DEFAULT_VENDOR_PATH
    checkpoint = os.path.join(repo, "checkpoints", "BIT_LEVIR", "best_ckpt.pt")
    model = bit.load_bit_cd_model(repo, checkpoint, "base_transformer_pos_s4_dd8_dedim8", device)
    mask, _, _ = bit.predict_change_mask(model, before, after, device)
    return mask


def label_box_sources(boxes, bit_cd_mask=None, index_mask=None, overlap_thresh=0.15):
    """For each final region, note which detector(s) actually fired there —
    used both for the overlay label and as a hint in the VLM prompt."""
    labels = []
    for (x, y, w, h) in boxes:
        tags = []
        if bit_cd_mask is not None:
            region = bit_cd_mask[y:y + h, x:x + w]
            if region.size and np.count_nonzero(region) / region.size > overlap_thresh:
                tags.append("building")
        if index_mask is not None:
            region = index_mask[y:y + h, x:x + w]
            if region.size and np.count_nonzero(region) / region.size > overlap_thresh:
                tags.append("index")
        labels.append(tags)
    return labels


def main():
    ap = argparse.ArgumentParser(description="Category-agnostic change detection + VLM classification")
    ap.add_argument("--before", help="Path to the 'before' image (omit if using --lat/--lon)")
    ap.add_argument("--after", help="Path to the 'after' image (omit if using --lat/--lon)")

    stac_group = ap.add_argument_group("STAC input (alternative to --before/--after)")
    stac_group.add_argument("--lat", type=float, default=35.2472)
    stac_group.add_argument("--lon", type=float, default=52.4921)
    stac_group.add_argument("--before-start", default="2020-01-01")
    stac_group.add_argument("--before-end", default="2020-03-01")
    stac_group.add_argument("--after-start", default="2026-01-01")
    stac_group.add_argument("--after-end", default="2026-03-01")
    stac_group.add_argument("--buffer-km", type=float, default=1.0,
                             help="Crop size around the point, in km")
    stac_group.add_argument("--cache-dir", default="stac_cache",
                             help="Where fetched scenes are cached — repeat queries for the "
                                  "same lat/lon/dates/area skip the network entirely")
    stac_group.add_argument("--max-cloud", type=float, default=30,
                             help="Max acceptable cloud cover %% when picking a scene")
    stac_group.add_argument("--stac-collection", default="sentinel-2-l2a")
    stac_group.add_argument("--stac-url", default="https://earth-search.aws.element84.com/v1")

    ap.add_argument("--out", default="categorized_output")
    ap.add_argument("--ssim-thresh", type=float, default=0.88,
                     help="Lower = more sensitive to structural changes")
    ap.add_argument("--color-thresh", type=float, default=18.0,
                     help="Lab color distance threshold, lower = more sensitive to hue/color "
                          "changes (e.g. vegetation turning brown)")
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--max-regions", type=int, default=10,
                     help="Cap on regions sent to the VLM per call")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--use-bit-cd", action="store_true",
                     help="Also run BIT_CD and OR its building-change mask into region "
                          "proposal (not just a label) — catches building changes the "
                          "generic/index detectors miss, alongside everything else they find")
    ap.add_argument("--bit-cd-repo", default=None)
    ap.add_argument("--model", default="qwen3-vl:4b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--no-describe", action="store_true")
    args = ap.parse_args()

    if args.use_bit_cd and (args.lat is not None or args.lon is not None):
        if args.buffer_km < BIT_CD_MIN_BUFFER_KM:
            print(f"Note: --use-bit-cd needs at least a {BIT_CD_MIN_TILE_PX}x{BIT_CD_MIN_TILE_PX}px "
                  f"crop to work properly (BIT_CD's training resolution) — at Sentinel-2's "
                  f"~{SENTINEL2_RES_M}m/pixel, that means >= {BIT_CD_MIN_BUFFER_KM}km. Your "
                  f"--buffer-km {args.buffer_km} would produce a "
                  f"~{int(args.buffer_km*1000/SENTINEL2_RES_M)}x{int(args.buffer_km*1000/SENTINEL2_RES_M)}px "
                  f"crop, which gets stretched with border-replicated padding and won't give BIT_CD "
                  f"anything real to look at. Bumping --buffer-km to {BIT_CD_MIN_BUFFER_KM} for this run.")
            args.buffer_km = BIT_CD_MIN_BUFFER_KM

    os.makedirs(args.out, exist_ok=True)
    ndvi_layer_full = None
    ndbi_layer_full = None
    index_mask = None

    if args.lat is not None or args.lon is not None:
        missing = [n for n, v in [
            ("--lat", args.lat), ("--lon", args.lon),
            ("--before-start", args.before_start), ("--before-end", args.before_end),
            ("--after-start", args.after_start), ("--after-end", args.after_end),
        ] if v is None]
        if missing:
            ap.error(f"--lat/--lon mode also needs: {', '.join(missing)}")

        from stac_fetch import fetch_before_after_with_indices
        before, after, before_idx, after_idx, before_meta, after_meta = fetch_before_after_with_indices(
            args.lat, args.lon, args.before_start, args.before_end,
            args.after_start, args.after_end, buffer_km=args.buffer_km,
            cache_dir=args.cache_dir, collection=args.stac_collection,
            stac_url=args.stac_url, max_cloud=args.max_cloud,
        )
        print(f"Before scene: {before_meta['item_id']} ({before_meta['datetime']})")
        print(f"After scene:  {after_meta['item_id']} ({after_meta['datetime']})")
        cv2.imwrite(os.path.join(args.out, "before_source.png"), before)
        cv2.imwrite(os.path.join(args.out, "after_source.png"), after)

        dndvi, dndbi, index_mask = compute_index_change_map(before_idx, after_idx)
        ndvi_layer_full = _colorize_diverging_full(dndvi, pos_bgr=(0, 255, 0), neg_bgr=(0, 0, 255))
        ndbi_layer_full = _colorize_single_sided_full(dndbi, bgr=(255, 0, 0))
        cv2.imwrite(os.path.join(args.out, "ndvi_change.png"), ndvi_layer_full)
        cv2.imwrite(os.path.join(args.out, "ndbi_change.png"), ndbi_layer_full)

        # STAC crops are already geo-aligned to the same lat/lon window/CRS —
        # ECC alignment is unnecessary here and can misfire on seasonal
        # vegetation texture changes, so it's skipped in this mode.
        args.no_align = True
    elif args.before and args.after:
        before, after = load_and_align(args.before, args.after)
        if args.use_bit_cd and min(before.shape[:2]) < BIT_CD_MIN_TILE_PX:
            print(f"Warning: your image is {before.shape[1]}x{before.shape[0]}px — smaller than "
                  f"the {BIT_CD_MIN_TILE_PX}x{BIT_CD_MIN_TILE_PX}px BIT_CD was trained on. It will "
                  f"be padded with replicated edge pixels rather than resized, so BIT_CD is mostly "
                  f"looking at stretched padding and likely won't find real changes here. Use a "
                  f"larger source image, or crop/tile a bigger region, for --use-bit-cd to work well.")
    else:
        ap.error("Provide either --before/--after, or --lat/--lon + date ranges")

    if not args.no_align:
        aligned, ok = align_images(before, after)
        if ok:
            after = aligned
            print("Alignment: converged.")
        else:
            print("Alignment: did not converge, continuing unaligned.")

    mask, raw_boxes, changed_pct = compute_generic_change_mask(
        before, after, ssim_thresh=args.ssim_thresh, color_thresh=args.color_thresh,
        min_area=args.min_area
    )

    bit_cd_mask = None
    if args.use_bit_cd:
        print("Running BIT_CD (building-change detector) as an additional detection source...")
        bit_cd_mask = get_bit_cd_mask(before, after, args.bit_cd_repo)

    # OR every available detector's mask together for region proposal: generic
    # SSIM/color (any visual change), spectral index (vegetation/built-up
    # change invisible in RGB), and BIT_CD (buildings specifically, including
    # ones the other two miss). A region only needs ONE detector to fire.
    combined_mask = mask
    if index_mask is not None:
        combined_mask = cv2.bitwise_or(combined_mask, index_mask)
    if bit_cd_mask is not None:
        combined_mask = cv2.bitwise_or(combined_mask, bit_cd_mask)

    if index_mask is not None or bit_cd_mask is not None:
        mask = combined_mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= args.min_area]
        changed_pct = 100.0 * np.count_nonzero(mask) / mask.size

    boxes = merge_overlapping_boxes(raw_boxes)
    boxes = select_top_regions(boxes, args.max_regions)

    print(f"Changed area: {changed_pct:.2f}% | {len(raw_boxes)} raw regions -> "
          f"{len(boxes)} after merge/cap")

    source_labels = label_box_sources(boxes, bit_cd_mask=bit_cd_mask, index_mask=index_mask)

    overlay = draw_overlay(after, boxes, source_labels)
    cv2.imwrite(os.path.join(args.out, "overlay.png"), overlay)
    cv2.imwrite(os.path.join(args.out, "mask.png"), mask)
    print(f"Saved overlay -> {os.path.join(args.out, 'overlay.png')}")

    if not boxes:
        print("No candidate regions found.")
        return

    generic_layer_full = compute_generic_heatmap_full(before, after)
    bit_cd_layer_full = cv2.cvtColor(bit_cd_mask, cv2.COLOR_GRAY2BGR) if bit_cd_mask is not None else None
    layers = {
        "bit_cd": bit_cd_layer_full,
        "ndvi": ndvi_layer_full,
        "ndbi": ndbi_layer_full,
        "generic": generic_layer_full,
    }
    available_signals = {name for name, layer in layers.items() if layer is not None}

    montage = build_montage(before, after, boxes, layers=layers)
    cv2.imwrite(os.path.join(args.out, "montage.png"), montage)

    if args.no_describe:
        return

    print(f"Asking {args.model} to classify + describe {len(boxes)} region(s)...")
    try:
        result = classify_and_describe(
            montage, len(boxes), source_labels=source_labels, model=args.model,
            ollama_url=args.ollama_url, available_signals=available_signals,
        )
        print("\n--- Per-region classification ---")
        print(result)
        with open(os.path.join(args.out, "regions.txt"), "w") as f:
            f.write(result)
    except requests.exceptions.RequestException as e:
        print(f"\nCould not reach Ollama at {args.ollama_url}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()