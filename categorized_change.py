import argparse
import base64
import json
import math
import os
import re
import sys

BIT_CD_MIN_TILE_PX = 256

import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity as ssim

# Reuse the alignment logic already validated in main2.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main2 import load_and_align, align_images  # noqa: E402
import dino_sam_cd as dsc  # noqa: E402


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


def compute_generic_change_score_full(before, after, blur_ksize=5):
    """
    Full-scene generic change-intensity score in [0, 1] (grayscale SSIM
    dissimilarity + Lab color distance, averaged). Raw float, no
    colorization — this is what compute_generic_heatmap_full() colorizes,
    and it also doubles as the fallback score map for SAM region scoring
    when --use-dino isn't enabled.
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

    return np.clip(0.5 * struct_dissim + 0.5 * color_norm, 0, 1)


def compute_generic_heatmap_full(before, after, blur_ksize=5):
    """JET-colorized version of compute_generic_change_score_full(), for the montage."""
    combined = compute_generic_change_score_full(before, after, blur_ksize)
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
    images, the diff mask, and any per-segment SAM mask consistently."""
    x, y, w, h = box
    H, W = img_shape[:2]
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    return x1, y1, x2, y2


def pixel_box_to_geo(box, image_shape, center_lat, center_lon, buffer_km):
    """
    Approximate lat/lon for a pixel box, given the STAC crop's known center
    point and total width in km (this pipeline's --buffer-km IS the full
    crop width, not a radius -- see the --target-px comment in main()).
    Uses a flat-earth/equirectangular approximation local to center_lat,
    which is accurate to a few meters over a crop this size (kilometers,
    not hundreds of km) -- fine for "which region is this" purposes, not
    for survey-grade coordinates.

    Caveat: this assumes the fetched crop is square and centered exactly
    on (center_lat, center_lon) with no extra internal padding/cropping.
    If stac_fetch.py's actual window differs (e.g. non-square, or offset
    to align to pixel/tile boundaries), these will be off by however much
    that window differs from the naive assumption. Sanity-check by
    confirming the center of the FULL image (not a region) maps back to
    very close to (center_lat, center_lon) -- it does by construction here,
    so the real thing to check is whether before_source.png's real-world
    footprint matches --buffer-km.
    """
    x, y, w, h = box
    H, W = image_shape[:2]
    meters_per_px = (buffer_km * 1000.0) / W
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * math.cos(math.radians(center_lat)))

    def px_to_ll(px, py):
        east_m = (px - W / 2.0) * meters_per_px
        north_m = (H / 2.0 - py) * meters_per_px  # image rows increase southward
        return (center_lat + north_m * deg_per_m_lat,
                center_lon + east_m * deg_per_m_lon)

    lat_c, lon_c = px_to_ll(x + w / 2.0, y + h / 2.0)
    lat_n, lon_w = px_to_ll(x, y)
    lat_s, lon_e = px_to_ll(x + w, y + h)
    return {
        "center_lat": lat_c, "center_lon": lon_c,
        "lat_north": lat_n, "lat_south": lat_s,
        "lon_west": lon_w, "lon_east": lon_e,
    }


def merge_location_into_result(result_text, boxes, region_geo):
    """
    Post-process the VLM's '#<n>: <category> — <description>' lines to
    prepend real-world location: lat/lon + a Google Maps link in STAC
    mode, or the pixel bbox as a fallback when there's no geo reference
    (plain --before/--after file mode).
    """
    pattern = re.compile(r"^#(\d+):\s*(.*)$")
    out_lines = []
    for line in result_text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            out_lines.append(line)
            continue
        idx = int(m.group(1)) - 1
        rest = m.group(2)
        if region_geo is not None and 0 <= idx < len(region_geo):
            g = region_geo[idx]
            maps_url = f"https://www.google.com/maps?q={g['center_lat']:.6f},{g['center_lon']:.6f}"
            out_lines.append(
                f"#{idx+1} @ ({g['center_lat']:.5f}, {g['center_lon']:.5f}) [{maps_url}]: {rest}"
            )
        elif 0 <= idx < len(boxes):
            x, y, w, h = boxes[idx]
            out_lines.append(f"#{idx+1} @ pixel bbox (x={x}, y={y}, w={w}, h={h}): {rest}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


# Fixed 2-row x 3-column layout for the change-graph quadrants. Order here
# controls draw order in change_graph_tile()/classify_and_describe(); keep
# the two lists in sync if you reorder.
CHANGE_GRAPH_LAYERS = ["bit_cd", "ndvi", "dino"]
CHANGE_GRAPH_LAYERS_ROW2 = ["ndbi", "generic", "sam_overlay"]


def change_graph_tile(before, after, box, layers, seg_mask=None, pad_ratio=0.3, tile_size=200):
    """
    Build ONE composite "change graph" for a region: a fixed 2x3 grid so
    Qwen sees every available signal at once, at consistent positions:
        top:    BIT_CD building-change mask | NDVI change | DINO semantic change
        bottom: NDBI change | generic structural/color heatmap | SAM segment overlay
    Any signal not computed for this run (BIT_CD/NDVI/NDBI need the right
    flags/mode; DINO needs --use-dino; SAM overlay needs --use-sam AND this
    particular region needs to have come from a SAM segment) shows as solid
    gray in that quadrant — gray means "not computed", NOT "no change"
    (which shows as black/dark). `generic` is always present.
    """
    x1, y1, x2, y2 = crop_region_bounds(before.shape, box, pad_ratio)
    half = tile_size // 2
    gray = np.full((half, half, 3), 128, dtype=np.uint8)

    def crop_or_gray(full_img):
        if full_img is None:
            return gray
        crop = full_img[y1:y2, x1:x2]
        return cv2.resize(crop, (half, half), interpolation=cv2.INTER_NEAREST)

    def sam_overlay_or_gray():
        if layers.get("sam_overlay_enabled") and seg_mask is not None:
            after_crop = after[y1:y2, x1:x2]
            seg_crop = seg_mask[y1:y2, x1:x2]
            blended = dsc.sam_mask_overlay(after_crop, seg_crop)
            return cv2.resize(blended, (half, half), interpolation=cv2.INTER_NEAREST)
        return gray

    row1 = np.hstack([
        crop_or_gray(layers.get("bit_cd")),
        crop_or_gray(layers.get("ndvi")),
        crop_or_gray(layers.get("dino")),
    ])
    row2 = np.hstack([
        crop_or_gray(layers.get("ndbi")),
        crop_or_gray(layers.get("generic")),  # always present upstream
        sam_overlay_or_gray(),
    ])
    return np.vstack([row1, row2])


def build_montage(before, after, boxes, tile_size=200, label_h=28, layers=None, seg_masks=None):
    """
    N x 3 grid: row i = [before_crop_i | after_crop_i | change_graph_i], each
    row numbered. The third column is a 2x3 composite of every available
    change signal (BIT_CD / NDVI / DINO / NDBI / generic / SAM overlay — see
    change_graph_tile) so Qwen can weigh them together instead of picking
    one signal to show.
    """
    layers = layers or {}
    seg_masks = seg_masks or [None] * len(boxes)
    n = len(boxes)
    cell_h = tile_size + label_h
    half = tile_size // 2
    graph_w = half * 3
    montage = np.full((cell_h * n, tile_size * 2 + graph_w, 3), 255, dtype=np.uint8)

    for i, box in enumerate(boxes):
        b_crop = crop_with_context(before, box, tile_size=tile_size)
        a_crop = crop_with_context(after, box, tile_size=tile_size)
        graph = change_graph_tile(before, after, box, layers, seg_mask=seg_masks[i],
                                   tile_size=tile_size)
        y0 = i * cell_h
        cv2.putText(montage, f"#{i+1} BEFORE", (5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(montage, f"#{i+1} AFTER", (tile_size + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(montage, f"#{i+1} CHANGE GRAPH", (tile_size * 2 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        montage[y0 + label_h:y0 + cell_h, 0:tile_size] = b_crop
        montage[y0 + label_h:y0 + cell_h, tile_size:tile_size * 2] = a_crop
        montage[y0 + label_h:y0 + cell_h, tile_size * 2:tile_size * 2 + graph_w] = graph

    return montage


def draw_overlay(after, boxes, source_labels=None, seg_masks=None):
    overlay = after.copy()
    seg_masks = seg_masks or [None] * len(boxes)
    for i, (x, y, w, h) in enumerate(boxes):
        color = (0, 0, 255)
        if seg_masks[i] is not None:
            contours, _ = cv2.findContours(seg_masks[i].astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
        else:
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
                           ollama_url="http://localhost:11434", available_signals=None,
                           timeout=240, keep_alive="10m"):
    hint_note = ""
    if source_labels and any(source_labels):
        parts = [f"#{i+1} ({'/'.join(tags)})" for i, tags in enumerate(source_labels) if tags]
        if parts:
            hint_note = ("\nNote: region(s) " + ", ".join(parts) + " were independently "
                         "flagged by a specialist detector (building = BIT_CD building-change "
                         "model, index = spectral NDVI/NDBI change, semantic = DINO embedding "
                         "change, sam = SAM segment selected by change score) — treat as a "
                         "hint, not a certainty; verify visually.")

    available_signals = available_signals or set()
    computed = ", ".join(sorted(available_signals)) if available_signals else "generic only"
    all_signals = {"bit_cd", "ndvi", "ndbi", "dino", "sam_overlay"}
    not_computed = all_signals - available_signals
    gray_note = (f" Signals NOT computed this run: {', '.join(sorted(not_computed))} — their "
                 f"quadrant will show as solid gray, meaning 'not available', NOT 'no change'."
                 if not_computed else "")

    map_explainer = (
        "a CHANGE GRAPH: a fixed 2x3 grid combining every available change signal so you can "
        "weigh them together, always in this layout:\n"
        "  top-left     = BIT_CD building-change mask (white = flagged as a building change)\n"
        "  top-middle   = NDVI change (green = vegetation gain, red = vegetation loss/deforestation)\n"
        "  top-right    = DINO semantic-change heatmap (a frozen vision model's embedding "
        "distance between before/after; red/yellow = the content at that spot is semantically "
        "different, not just visually noisier — more reliable than raw pixel difference for "
        "distinguishing real change from lighting/sensor/seasonal shifts)\n"
        "  bottom-left  = NDBI change (blue = new built-up/bare surface, i.e. construction)\n"
        "  bottom-middle= generic structural/color change heatmap (red/yellow = strong change, "
        "always present)\n"
        "  bottom-right = SAM segment overlay (yellow outline/fill = the exact object boundary "
        "this candidate region came from, when region proposal used Segment Anything)\n"
        f"Computed this run: {computed}.{gray_note}\n"
        "Weigh whichever quadrants are populated together — e.g. strong top-right (DINO) plus "
        "strong top-middle (NDVI) usually means vegetation loss/deforestation; strong bottom-left "
        "(NDBI) plus top-left (BIT_CD) usually means new construction. A region with a strong "
        "DINO signal but weak/gray everything else is still meaningful — it means the content "
        "changed semantically even if pixel-level structure/color didn't move much."
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
        "keep_alive": keep_alive,
    }
    resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def get_bit_cd_mask(before, after, bit_cd_repo=None):
    """
    Run BIT_CD tiled inference and return its full building-change mask
    (same H x W as the input images, 0/255). Feeds directly into region
    proposal (OR'd with the other masks) instead of only labeling regions
    found by other detectors.
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


def label_box_sources(boxes, bit_cd_mask=None, index_mask=None, dino_mask=None,
                       sam_scores=None, overlap_thresh=0.15):
    """For each final region, note which detector(s) actually fired there —
    used both for the overlay label and as a hint in the VLM prompt."""
    labels = []
    for i, (x, y, w, h) in enumerate(boxes):
        tags = []
        if bit_cd_mask is not None:
            region = bit_cd_mask[y:y + h, x:x + w]
            if region.size and np.count_nonzero(region) / region.size > overlap_thresh:
                tags.append("building")
        if index_mask is not None:
            region = index_mask[y:y + h, x:x + w]
            if region.size and np.count_nonzero(region) / region.size > overlap_thresh:
                tags.append("index")
        if dino_mask is not None:
            region = dino_mask[y:y + h, x:x + w]
            if region.size and np.count_nonzero(region) / region.size > overlap_thresh:
                tags.append("semantic")
        if sam_scores is not None and sam_scores[i] is not None:
            tags.append(f"sam:{sam_scores[i]:.2f}")
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
    stac_group.add_argument("--buffer-km", type=float, default=None,
                             help="Crop size around the point, in km. Overrides --target-px "
                                  "if both are given.")
    stac_group.add_argument("--target-px", type=int, default=1024,
                             help="Crop size in pixels (converted to km using Sentinel-2's "
                                  "~10m/pixel resolution). Defaults to 1024 — the same size "
                                  "as LEVIR-CD's raw scenes (and a clean 4x4 grid of BIT_CD's "
                                  "256px tiles). Ignored if --buffer-km is given.")
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

    dino_group = ap.add_argument_group("DINOv2 semantic change (zero-shot, no training needed)")
    dino_group.add_argument("--use-dino", action="store_true",
                             help="OR a DINOv2 patch-embedding change map into region proposal. "
                                  "Unlike SSIM/color, this compares learned semantic content, so "
                                  "it's less fooled by sensor calibration drift / seasonal color "
                                  "shift between the before/after scene, and works at whatever "
                                  "resolution Sentinel-2 gives you (no LEVIR-CD-style resolution "
                                  "requirement like BIT_CD has).")
    dino_group.add_argument("--dino-model", default="dinov2_vits14",
                             help="torch.hub DINOv2 variant (vits14/vitb14/vitl14/vitg14 — "
                                  "bigger = slower + slightly better features)")
    dino_group.add_argument("--dino-thresh", type=float, default=0.3,
                             help="Cosine-distance threshold (0-2 scale) for the DINO change "
                                  "mask; lower = more sensitive")

    sam_group = ap.add_argument_group("SAM region proposals (replaces contour/bbox proposal)")
    sam_group.add_argument("--use-sam", default=True, action="store_true",
                            help="Use Segment Anything to propose regions instead of contour "
                                 "boxes from the OR'd masks — gives object-shaped regions scored "
                                 "by mean change intensity (DINO if --use-dino is also set, else "
                                 "the generic heatmap) rather than rectangles.")
    sam_group.add_argument("--sam-checkpoint", default="sam_vit_b_01ec64.pth",
                            help="Path to a SAM checkpoint (.pth) — required if --use-sam is set")
    sam_group.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    sam_group.add_argument("--sam-min-change-score", type=float, default=0.2,
                            help="Minimum mean change score (DINO 0-2 scale if --use-dino, else "
                                 "generic heatmap 0-1 scale) for a SAM segment to be kept")
    sam_group.add_argument("--sam-points-per-side", type=int, default=24,
                            help="SAM automatic-mask-generator grid density; lower = faster, "
                                 "coarser segmentation")

    ap.add_argument("--model", default="qwen3-vl:4b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-timeout", type=int, default=900,
                     help="Seconds to wait for the Ollama response before giving up. Larger "
                          "montages (more regions, --use-sam often proposes more than the "
                          "contour fallback) take longer -- bump this if you see read timeouts, "
                          "e.g. --ollama-timeout 600.")
    ap.add_argument("--ollama-keep-alive", default="10m",
                     help="How long Ollama keeps the model loaded after this request, so back-"
                          "to-back runs don't pay model-load time again (Ollama's default is 5m).")
    ap.add_argument("--no-describe", action="store_true")
    args = ap.parse_args()

    if args.use_sam and not args.sam_checkpoint:
        ap.error("--use-sam requires --sam-checkpoint")

    from bit_cd_infer import min_buffer_km_for_tiles, SENTINEL2_RES_M
    stac_mode = args.lat is not None or args.lon is not None

    if stac_mode and args.buffer_km is None:
        args.buffer_km = round(args.target_px * SENTINEL2_RES_M / 1000, 2)
        print(f"Using --target-px {args.target_px} -> --buffer-km {args.buffer_km} "
              f"(Sentinel-2 ~{SENTINEL2_RES_M}m/pixel).")

    if args.use_bit_cd and stac_mode:
        min_km = min_buffer_km_for_tiles(BIT_CD_MIN_TILE_PX, n_tiles_per_side=1)
        if args.buffer_km < min_km:
            print(f"Note: --use-bit-cd needs at least a {BIT_CD_MIN_TILE_PX}x{BIT_CD_MIN_TILE_PX}px "
                  f"crop to work properly (BIT_CD's training resolution) — at Sentinel-2's "
                  f"~{SENTINEL2_RES_M}m/pixel, that means >= {min_km}km. Your "
                  f"--buffer-km {args.buffer_km} would produce a "
                  f"~{int(args.buffer_km*1000/SENTINEL2_RES_M)}x{int(args.buffer_km*1000/SENTINEL2_RES_M)}px "
                  f"crop, which gets stretched with border-replicated padding and won't give BIT_CD "
                  f"anything real to look at. Bumping --buffer-km to {min_km} for this run.")
            args.buffer_km = min_km
        else:
            # Round UP to a clean multiple of the tile size so every tile —
            # including the ones at the edge — is full real content, not
            # just "big enough for one tile somewhere in the middle."
            n_tiles = math.ceil(args.buffer_km / min_km)
            rounded_km = round(n_tiles * min_km, 2)
            if rounded_km != args.buffer_km:
                print(f"Rounding --buffer-km {args.buffer_km} up to {rounded_km} "
                      f"({n_tiles}x{n_tiles} clean {BIT_CD_MIN_TILE_PX}px tiles for BIT_CD, "
                      f"no partial-padding edge tile).")
                args.buffer_km = rounded_km

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

    # --- DINOv2 semantic change map (zero-shot, resolution-agnostic) ---
    dino_map = None
    dino_mask = None
    dino_layer_full = None
    if args.use_dino:
        print(f"Running DINOv2 ({args.dino_model}) semantic change map...")
        dino_model, dino_device, dino_patch_size = dsc.load_dino(args.dino_model)
        if dino_model is not None:
            dino_map = dsc.dino_change_map(before, after, dino_model, dino_device, dino_patch_size)
            dino_mask = (dino_map > args.dino_thresh).astype(np.uint8) * 255
            dino_layer_full = dsc.dino_heatmap_full(dino_map)
            cv2.imwrite(os.path.join(args.out, "dino_change.png"), dino_layer_full)
    else:
        print("DINO disabled")
    # OR every available detector's mask together for region proposal: generic
    # SSIM/color (any visual change), spectral index (vegetation/built-up
    # change invisible in RGB), BIT_CD (buildings, including ones the other
    # two miss), and DINO (semantic change invisible to pixel-level diffs).
    # A region only needs ONE detector to fire.
    combined_mask = mask
    if index_mask is not None:
        combined_mask = cv2.bitwise_or(combined_mask, index_mask)
    if bit_cd_mask is not None:
        combined_mask = cv2.bitwise_or(combined_mask, bit_cd_mask)
    if dino_mask is not None:
        combined_mask = cv2.bitwise_or(combined_mask, dino_mask)

    if index_mask is not None or bit_cd_mask is not None or dino_mask is not None:
        mask = combined_mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= args.min_area]
        changed_pct = 100.0 * np.count_nonzero(mask) / mask.size

    generic_layer_full = compute_generic_heatmap_full(before, after)

    # --- Region proposal: SAM segments (object-shaped) or legacy contour boxes ---
    seg_masks = None
    sam_scores = None
    if args.use_sam:
        print(f"Running SAM ({args.sam_model_type}) region proposal...")
        mask_generator = dsc.load_sam(args.sam_checkpoint, model_type=args.sam_model_type,
                                       points_per_side=args.sam_points_per_side,
                                       min_mask_region_area=args.min_area)
        if mask_generator is None:
            print("Falling back to contour-based region proposal.", file=sys.stderr)
            boxes = select_top_regions(merge_overlapping_boxes(raw_boxes), args.max_regions)
        else:
            if dino_map is not None:
                score_map = dino_map
                score_source = "dino"
            else:
                print("Note: --use-sam without --use-dino falls back to the generic heatmap "
                      "(0-1 scale) for segment scoring — consider also passing --use-dino for "
                      "a more reliable semantic score.")
                score_map = compute_generic_change_score_full(before, after)
                score_source = "generic"
            sam_regions = dsc.sam_change_regions(
                after, score_map, mask_generator,
                min_change_score=args.sam_min_change_score,
                min_area=args.min_area, max_regions=args.max_regions,
            )
            boxes = [r[0] for r in sam_regions]
            seg_masks = [r[2] for r in sam_regions]
            sam_scores = [r[1] for r in sam_regions]
            print(f"SAM proposed {len(boxes)} region(s) above score {args.sam_min_change_score} "
                  f"(scored via {score_source}).")
    else:
        boxes = merge_overlapping_boxes(raw_boxes)
        boxes = select_top_regions(boxes, args.max_regions)

    print(f"Changed area: {changed_pct:.2f}% | {len(raw_boxes)} raw regions -> "
          f"{len(boxes)} after merge/cap")

    source_labels = label_box_sources(boxes, bit_cd_mask=bit_cd_mask, index_mask=index_mask,
                                       dino_mask=dino_mask, sam_scores=sam_scores)

    region_geo = None
    if stac_mode:
        region_geo = [pixel_box_to_geo(b, before.shape, args.lat, args.lon, args.buffer_km)
                      for b in boxes]
        print("\n--- Region locations ---")
        for i, g in enumerate(region_geo):
            maps_url = f"https://www.google.com/maps?q={g['center_lat']:.6f},{g['center_lon']:.6f}"
            print(f"#{i+1}: ({g['center_lat']:.5f}, {g['center_lon']:.5f})  {maps_url}")

    overlay = draw_overlay(after, boxes, source_labels, seg_masks=seg_masks)
    cv2.imwrite(os.path.join(args.out, "overlay.png"), overlay)
    cv2.imwrite(os.path.join(args.out, "mask.png"), mask)
    print(f"Saved overlay -> {os.path.join(args.out, 'overlay.png')}")

    if not boxes:
        print("No candidate regions found.")
        return

    bit_cd_layer_full = cv2.cvtColor(bit_cd_mask, cv2.COLOR_GRAY2BGR) if bit_cd_mask is not None else None
    layers = {
        "bit_cd": bit_cd_layer_full,
        "ndvi": ndvi_layer_full,
        "ndbi": ndbi_layer_full,
        "dino": dino_layer_full,
        "generic": generic_layer_full,
        "sam_overlay_enabled": args.use_sam and seg_masks is not None,
    }
    available_signals = {name for name in ("bit_cd", "ndvi", "ndbi", "dino")
                          if layers.get(name) is not None}
    if layers["sam_overlay_enabled"]:
        available_signals.add("sam_overlay")

    montage = build_montage(before, after, boxes, layers=layers, seg_masks=seg_masks)
    cv2.imwrite(os.path.join(args.out, "montage.png"), montage)

    if args.no_describe:
        return

    print(f"Asking {args.model} to classify + describe {len(boxes)} region(s)...")
    try:
        result = classify_and_describe(
            montage, len(boxes), source_labels=source_labels, model=args.model,
            ollama_url=args.ollama_url, available_signals=available_signals,
            timeout=args.ollama_timeout, keep_alive=args.ollama_keep_alive,
        )
        result = merge_location_into_result(result, boxes, region_geo)
        print("\n--- Per-region classification ---")
        print(result)
        with open(os.path.join(args.out, "regions.txt"), "w") as f:
            f.write(result)
    except requests.exceptions.Timeout:
        print(f"\nOllama request timed out after {args.ollama_timeout}s. The model may still be "
              f"generating -- try again with a larger --ollama-timeout, or check `ollama ps` to "
              f"see if {args.model} is still loaded/running.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\nCould not reach Ollama at {args.ollama_url}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()