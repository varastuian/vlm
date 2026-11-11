"""
visualize_change.py

Build a single self-contained HTML gallery comparing before/after Sentinel-2
imagery: before, after, NDVI before/after, NDBI before/after, a generic
band-difference heatmap, a DINOv2 patch-similarity change heatmap, and a
SAM (Segment Anything) segmentation overlay. Scalar panels (NDVI/NDBI/
generic-diff/DINO) get a labeled colorbar legend baked into the PNG.

You can restrict everything to an area of interest (AOI) instead of the
whole tile:

    # interactive: pops up a quicklook, drag a box, close the window
    python visualize_change.py --raw-dir raw/ --select-aoi

    # manual: pixel coords (x0 y0 x1 y1) on the native 10m red-band grid
    python visualize_change.py --raw-dir raw/ --aoi 4000 3000 6000 5000

With an AOI set, every band is read with a windowed rasterio read scoped to
just that region (via world-coordinate bounds, so it's correct even though
swir16 is a coarser 20m band) - the full 10980x10980 arrays are never
touched at all, which is both faster and lighter on memory than the
whole-tile decimated path.

Usage (whole tile, no AOI):
    python visualize_change.py --raw-dir raw/ --out-dir viz_out/ \
        --sam-checkpoint sam_vit_b_01ec64.pth --device cuda

Requirements:
    pip install rasterio numpy pillow matplotlib torch torchvision \
                transformers segment-anything opencv-python-headless
    # --select-aoi additionally needs a GUI-capable matplotlib backend
    # (Qt5Agg or TkAgg); if you're headless, use --aoi instead.

SAM checkpoint (download once, pick one size):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth   # smallest/fastest
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  # best/slowest

If --sam-checkpoint points at a file that doesn't exist, the SAM panel is
skipped (everything else still runs). DINOv2 downloads its weights from the
Hugging Face hub on first run (needs internet once, then it's cached).

Assumes the two dates for each band are already co-registered (same
Sentinel-2 tile / pixel grid), which is the case for your raw/ folder
(tile 39SXV, both dates).
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Band loading
# --------------------------------------------------------------------------

DATE_RE = re.compile(r"_(\d{8})_")


def find_band_files(raw_dir, band):
    """Return (date_str, path) list for a given band, sorted oldest -> newest."""
    pattern = os.path.join(raw_dir, f"*_L2A_{band}.tif")
    files = glob.glob(pattern)
    dated = []
    for f in files:
        m = DATE_RE.search(os.path.basename(f))
        if m:
            dated.append((m.group(1), f))
    dated.sort(key=lambda x: x[0])
    return dated


def format_date(yyyymmdd):
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def target_shape_for_dims(h, w, max_size):
    """Compute the (height, width) every band is resampled to on read.

    Always returns a concrete shape - even with max_size=None this is just
    (h, w) - because bands can have different native resolutions (e.g.
    swir16 is a 20m band, half the pixel dimensions of the 10m red/nir/
    visual bands for the same ground footprint). Passing this as out_shape
    on every read forces all bands onto one common pixel grid; skipping it
    for "no decimation needed" caused nir/red and swir16 to come out at
    different shapes for the same AOI.
    """
    if max_size is None or max(h, w) <= max_size:
        return (h, w)
    scale = max_size / max(h, w)
    return (max(1, int(round(h * scale))), max(1, int(round(w * scale))))


def read_band(path, scale=10000.0, target_shape=None, bounds=None):
    """Read a single-band S2 reflectance tif, scaled to ~0-1.

    bounds (minx, miny, maxx, maxy), if given, windows the read to that
    world-coordinate box (correct regardless of this band's native
    resolution). target_shape, if given, decimates/resamples on read.
    """
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform) if bounds else None
        read_kwargs = {}
        if target_shape is not None:
            read_kwargs["out_shape"] = target_shape
            read_kwargs["resampling"] = Resampling.bilinear
        arr = src.read(1, window=window, **read_kwargs).astype(np.float32)
    return arr / scale


def read_visual(path, target_shape=None, bounds=None):
    """Read the pre-made TCI visual.tif as an (H, W, 3) uint8 array."""
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform) if bounds else None
        read_kwargs = {}
        if target_shape is not None:
            read_kwargs["out_shape"] = (src.count,) + target_shape
            read_kwargs["resampling"] = Resampling.bilinear
        arr = src.read(window=window, **read_kwargs)
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.repeat(arr[0][:, :, None], 3, axis=2)
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    rgb = 255 * (rgb - rgb.min()) / (np.ptp(rgb) + 1e-6)
    return rgb.astype(np.uint8)


def load_pair(raw_dir, band, target_shape, bounds, read_fn=read_band):
    dated = find_band_files(raw_dir, band)
    if len(dated) < 2:
        raise FileNotFoundError(
            f"Need 2 dates for band '{band}', found {len(dated)} in {raw_dir}"
        )
    (d0, p0), (d1, p1) = dated[0], dated[-1]
    print(f"  {band}: before={d0} ({os.path.basename(p0)})  after={d1} ({os.path.basename(p1)})")
    return (
        read_fn(p0, target_shape=target_shape, bounds=bounds),
        read_fn(p1, target_shape=target_shape, bounds=bounds),
    )


# --------------------------------------------------------------------------
# AOI selection
# --------------------------------------------------------------------------

def pixel_box_to_bounds(ref_path, x0, y0, x1, y1):
    """Convert a pixel box (native grid of ref_path) to world (minx,miny,maxx,maxy)."""
    with rasterio.open(ref_path) as src:
        transform = src.transform
    px0, px1 = sorted((x0, x1))
    py0, py1 = sorted((y0, y1))
    minx, maxy = transform * (px0, py0)
    maxx, miny = transform * (px1, py1)
    return (minx, miny, maxx, maxy)


def select_aoi_gui(quicklook_rgb, native_shape, ref_path):
    """Show a quicklook, let the user drag a rectangle, return world bounds
    plus the equivalent native-pixel box (for logging/aoi.json)."""
    from matplotlib.widgets import RectangleSelector

    box = {}

    def onselect(eclick, erelease):
        box["x0"], box["y0"] = eclick.xdata, eclick.ydata
        box["x1"], box["y1"] = erelease.xdata, erelease.ydata

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(quicklook_rgb)
    ax.set_title("Drag a box over your AOI, then close this window to continue")
    _selector = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    plt.show()

    if "x0" not in box:
        raise RuntimeError("No AOI was selected (window closed without dragging a box).")

    qh, qw = quicklook_rgb.shape[:2]
    nh, nw = native_shape
    sx, sy = nw / qw, nh / qh
    x0 = max(0, int(round(min(box["x0"], box["x1"]) * sx)))
    x1 = min(nw, int(round(max(box["x0"], box["x1"]) * sx)))
    y0 = max(0, int(round(min(box["y0"], box["y1"]) * sy)))
    y1 = min(nh, int(round(max(box["y0"], box["y1"]) * sy)))
    bounds = pixel_box_to_bounds(ref_path, x0, y0, x1, y1)
    return bounds, (x0, y0, x1, y1)


# --------------------------------------------------------------------------
# Index computation
# --------------------------------------------------------------------------

def normalized_diff(a, b):
    return (a - b) / (a + b + 1e-6)


def signed_delta_range(diff, pct=99):
    """Symmetric (vmin, vmax) around 0 for a signed after-minus-before diff,
    sized to the 99th percentile of |diff| so a few extreme pixels don't
    wash out the color scale."""
    vmax = float(np.percentile(np.abs(diff), pct))
    if vmax < 1e-6:
        vmax = float(np.abs(diff).max()) if diff.size else 1e-3
    if vmax < 1e-6:
        vmax = 1e-3
    return -vmax, vmax


def rgb_to_gray(rgb_uint8):
    """Perceptual grayscale (0-1 float) from an (H, W, 3) uint8 RGB image."""
    arr = rgb_uint8.astype(np.float32) / 255.0
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def save_rgb(arr_uint8, path):
    Image.fromarray(arr_uint8).save(path)


def save_with_legend(arr, path, cmap_name, vmin, vmax, tick_labels, dpi=150):
    """Render a scalar array with a colorbar whose ticks are plain-language
    labels (e.g. 'Less vegetation' -> 'More vegetation') instead of raw
    numeric index values, so it reads without knowing the underlying formula."""
    h, w = arr.shape
    fig_w = w / dpi + 1.3  # extra room for the colorbar + text labels
    fig_h = h / dpi
    fig, ax = plt.subplots(figsize=(max(fig_w, 2), max(fig_h, 2)), dpi=dpi)
    im = ax.imshow(arr, cmap=cmap_name, vmin=vmin, vmax=vmax)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    ticks = np.linspace(vmin, vmax, len(tick_labels))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8, length=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# DINOv2-based change heatmap
# --------------------------------------------------------------------------

def dino_change_distance(rgb_before, rgb_after, device="cpu", model_name="facebook/dinov2-small"):
    """Returns a (H, W) float array of per-pixel patch cosine-distance,
    upsampled to the input image size. Caller colorizes/saves it."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def embed(rgb):
        img = Image.fromarray(rgb)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        patch_tokens = out.last_hidden_state[0, 1:]  # drop CLS token
        n = patch_tokens.shape[0]
        side = int(n ** 0.5)
        return patch_tokens[: side * side].reshape(side, side, -1)

    feat_b = embed(rgb_before)
    feat_a = embed(rgb_after)

    feat_b = torch.nn.functional.normalize(feat_b, dim=-1)
    feat_a = torch.nn.functional.normalize(feat_a, dim=-1)
    cos_sim = (feat_b * feat_a).sum(-1)
    dist = (1 - cos_sim).detach().cpu().numpy()

    dist_img = Image.fromarray(dist.astype(np.float32), mode="F")
    dist_img = dist_img.resize((rgb_before.shape[1], rgb_before.shape[0]), Image.BILINEAR)
    return np.array(dist_img)


# --------------------------------------------------------------------------
# SAM segmentation overlay
# --------------------------------------------------------------------------

def sam_overlay(rgb, checkpoint, model_type="vit_b", device="cpu"):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=16, pred_iou_thresh=0.86, min_mask_region_area=200
    )
    masks = generator.generate(rgb)

    overlay = rgb.copy().astype(np.float32)
    rng = np.random.default_rng(0)
    for m in masks:
        color = rng.integers(0, 255, size=3)
        seg = m["segmentation"]
        overlay[seg] = 0.5 * overlay[seg] + 0.5 * color
    return overlay.astype(np.uint8), len(masks)


# --------------------------------------------------------------------------
# HTML gallery
# --------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Bitemporal Change Visualization</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; margin:0; padding:24px; }}
  h1 {{ font-weight:600; margin-bottom:4px; }}
  h2 {{ font-weight:600; font-size:16px; color:#ccc; margin:28px 0 12px;
       border-top:1px solid #333; padding-top:20px; }}
  .pair-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }}
  .col-header {{ text-align:center; font-weight:600; color:#ccc; font-size:15px; }}
  .col-date {{ display:block; font-weight:400; color:#888; font-size:12px; margin-top:2px; }}
  .grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; }}
  figure {{ margin:0; background:#1b1b1b; border-radius:8px; overflow:hidden; }}
  figure img {{ width:100%; display:block; background:#fff; }}
  figcaption {{ padding:8px 12px; font-size:14px; color:#aaa; text-align:center; }}
</style>
</head>
<body>
<h1>Bitemporal Change Visualization</h1>
<h2>Before / After / Difference</h2>
<div class="pair-grid">
  <div class="col-header">Before<span class="col-date">{before_date}</span></div>
  <div class="col-header">After<span class="col-date">{after_date}</span></div>
  <div class="col-header">Difference<span class="col-date">After − Before</span></div>
{pair_rows}
</div>
<h2>Change Detection</h2>
<div class="grid">
{full_cards}
</div>
</body>
</html>
"""

CARD_TEMPLATE = """<figure>
  <img src="{src}" alt="{label}">
  <figcaption>{label}</figcaption>
</figure>"""


def build_html(out_dir, pairs, full_panels, before_date, after_date):
    """pairs: list of (label, before_fname, after_fname, delta_fname), rendered
    as aligned rows across the three Before/After/Difference columns.
    full_panels: list of (label, fname) for the standalone change-detection
    panels below."""
    pair_rows = "\n".join(
        CARD_TEMPLATE.format(src=before_f, label=label) + "\n"
        + CARD_TEMPLATE.format(src=after_f, label=label) + "\n"
        + CARD_TEMPLATE.format(src=delta_f, label=f"Δ {label}")
        for label, before_f, after_f, delta_f in pairs
    )
    full_cards = "\n".join(
        CARD_TEMPLATE.format(src=fname, label=label) for label, fname in full_panels
    )
    html = HTML_TEMPLATE.format(
        pair_rows=pair_rows, full_cards=full_cards,
        before_date=before_date, after_date=after_date,
    )
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--out-dir", default="viz_out")
    ap.add_argument("--sam-checkpoint", default="sam_vit_b_01ec64.pth", help="Path to SAM .pth checkpoint")
    ap.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument(
        "--dino-model", default="facebook/dinov2-small",
        choices=["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large"],
        help="DINOv2 has no plain 'facebook/dinov2' checkpoint - must pick a size.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--max-size", type=int, default=2048,
        help="Decimate on read so the longest side is at most this many pixels "
             "(applies to the whole tile, or to the AOI crop if one is set). "
             "Use 0 for full native resolution.",
    )
    ap.add_argument(
        "--select-aoi", default=True,action="store_true",
        help="Pop up a window to drag-select an AOI on a quicklook of the "
             "'before' image; only that region is processed. Needs a GUI "
             "matplotlib backend (skip this and use --aoi if headless).",
    )
    ap.add_argument(
        "--aoi", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
        help="Manually specify an AOI in native pixel coordinates of the 10m "
             "red-band grid, instead of using --select-aoi.",
    )
    ap.add_argument(
        "--aoi-quicklook-size", type=int, default=1024,
        help="Max dimension of the quicklook image shown for --select-aoi.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_size = args.max_size if args.max_size and args.max_size > 0 else None

    visual_dates = find_band_files(args.raw_dir, "visual")
    red_dates = find_band_files(args.raw_dir, "red")
    ref_path = red_dates[0][1]
    before_date_str = format_date(red_dates[0][0])
    after_date_str = format_date(red_dates[-1][0])
    with rasterio.open(ref_path) as ref:
        native_w, native_h = ref.width, ref.height

    # --- resolve AOI (if any) to world-coordinate bounds ------------------
    bounds = None
    aoi_native_box = None
    if args.select_aoi:
        print("Loading quicklook for AOI selection...")
        ql_shape = target_shape_for_dims(native_h, native_w, args.aoi_quicklook_size)
        quicklook = read_visual(visual_dates[0][1], target_shape=ql_shape)
        try:
            bounds, aoi_native_box = select_aoi_gui(quicklook, (native_h, native_w), ref_path)
        except Exception as e:
            raise SystemExit(
                f"AOI selection failed ({e}). If you're headless/over SSH, use "
                f"--aoi X0 Y0 X1 Y1 instead of --select-aoi."
            )
        print(f"  selected AOI (native pixel coords): {aoi_native_box}")
    elif args.aoi:
        aoi_native_box = tuple(args.aoi)
        bounds = pixel_box_to_bounds(ref_path, *aoi_native_box)
        print(f"Using manual AOI (native pixel coords): {aoi_native_box}")

    if aoi_native_box is not None:
        x0, y0, x1, y1 = aoi_native_box
        crop_h, crop_w = (y1 - y0), (x1 - x0)
        if crop_h <= 0 or crop_w <= 0:
            raise SystemExit(f"AOI box is empty/invalid: {aoi_native_box}")
        target_shape = target_shape_for_dims(crop_h, crop_w, max_size)
        with rasterio.open(ref_path) as ref:
            crs = str(ref.crs)
        with open(out_dir / "aoi.json", "w") as f:
            json.dump(
                {
                    "pixel_box_native_red_grid_x0y0x1y1": list(aoi_native_box),
                    "world_bounds_minx_miny_maxx_maxy": list(bounds),
                    "crs": crs,
                },
                f, indent=2,
            )
        print(f"  processing AOI at {target_shape[1]}x{target_shape[0]} px "
              f"(native crop is {crop_w}x{crop_h})")
    else:
        target_shape = target_shape_for_dims(native_h, native_w, max_size)
        print(f"Loading full tile...")
        print(f"  processing at {target_shape[1]}x{target_shape[0]} px "
              f"(native tile is {native_w}x{native_h})")

    rgb_before = read_visual(visual_dates[0][1], target_shape=target_shape, bounds=bounds)
    rgb_after = read_visual(visual_dates[-1][1], target_shape=target_shape, bounds=bounds)

    red_b, red_a = load_pair(args.raw_dir, "red", target_shape, bounds)
    nir_b, nir_a = load_pair(args.raw_dir, "nir", target_shape, bounds)
    swir_b, swir_a = load_pair(args.raw_dir, "swir16", target_shape, bounds)

    print("Computing NDVI / NDBI...")
    ndvi_b = normalized_diff(nir_b, red_b)
    ndvi_a = normalized_diff(nir_a, red_a)
    ndbi_b = normalized_diff(swir_b, nir_b)
    ndbi_a = normalized_diff(swir_a, nir_a)

    print("Computing generic difference...")
    stack_b = np.stack([red_b, nir_b, swir_b])
    stack_a = np.stack([red_a, nir_a, swir_a])
    generic_diff = np.abs(stack_a - stack_b).mean(axis=0)
    generic_diff_vmax = float(np.percentile(generic_diff, 99))

    pairs = []       # (label, before_fname, after_fname, delta_fname)
    full_panels = [] # (label, fname) -> standalone change-detection panels

    save_rgb(rgb_before, out_dir / "before.png")
    save_rgb(rgb_after, out_dir / "after.png")
    gray_before = rgb_to_gray(rgb_before)
    gray_after = rgb_to_gray(rgb_after)
    visual_delta = gray_after - gray_before
    vmin, vmax = signed_delta_range(visual_delta)
    save_with_legend(
        visual_delta, out_dir / "visual_delta.png", "RdBu_r", vmin, vmax,
        ["Darker in after", "No change", "Brighter in after"],
    )
    pairs.append(("RGB", "before.png", "after.png", "visual_delta.png"))

    save_with_legend(
        ndvi_b, out_dir / "ndvi_before.png", "RdYlGn", -1, 1,
        ["Water / built-up", "Bare soil", "Dense vegetation"],
    )
    save_with_legend(
        ndvi_a, out_dir / "ndvi_after.png", "RdYlGn", -1, 1,
        ["Water / built-up", "Bare soil", "Dense vegetation"],
    )
    ndvi_delta = ndvi_a - ndvi_b
    vmin, vmax = signed_delta_range(ndvi_delta)
    save_with_legend(
        ndvi_delta, out_dir / "ndvi_delta.png", "RdYlGn", vmin, vmax,
        ["Vegetation loss", "No change", "Vegetation gain"],
    )
    pairs.append(("NDVI", "ndvi_before.png", "ndvi_after.png", "ndvi_delta.png"))

    # NDBI runs the opposite way (high = built-up), so use a reversed colormap
    # to keep "green = vegetated / red = built-up" consistent with the NDVI panel.
    save_with_legend(
        ndbi_b, out_dir / "ndbi_before.png", "RdYlGn_r", -1, 1,
        ["Vegetation / water", "Mixed", "Built-up / bare soil"],
    )
    save_with_legend(
        ndbi_a, out_dir / "ndbi_after.png", "RdYlGn_r", -1, 1,
        ["Vegetation / water", "Mixed", "Built-up / bare soil"],
    )
    ndbi_delta = ndbi_a - ndbi_b
    vmin, vmax = signed_delta_range(ndbi_delta)
    save_with_legend(
        ndbi_delta, out_dir / "ndbi_delta.png", "RdYlGn_r", vmin, vmax,
        ["Less built-up", "No change", "More built-up"],
    )
    pairs.append(("NDBI", "ndbi_before.png", "ndbi_after.png", "ndbi_delta.png"))

    save_with_legend(
        generic_diff, out_dir / "generic_diff.png", "viridis", 0, generic_diff_vmax,
        ["No / little change", "Strong change"],
    )
    full_panels.append(("Generic Difference", "generic_diff.png"))

    print(f"Running DINOv2 ({args.dino_model}) change heatmap "
          f"(downloads model weights on first run)...")
    try:
        dist = dino_change_distance(rgb_before, rgb_after, device=args.device, model_name=args.dino_model)
        save_with_legend(
            dist, out_dir / "dino_change.png", "inferno", 0, float(dist.max()),
            ["Similar (unchanged)", "Very different (changed)"],
        )
        full_panels.append(("DINOv2 Change Heatmap", "dino_change.png"))
    except Exception as e:
        print(f"  [skipped] DINO heatmap failed: {e}")

    if args.sam_checkpoint and os.path.exists(args.sam_checkpoint):
        print("Running SAM segmentation on 'after' image...")
        try:
            sam_img, n_masks = sam_overlay(
                rgb_after, args.sam_checkpoint, args.sam_model_type, args.device
            )
            save_rgb(sam_img, out_dir / "sam_overlay.png")
            full_panels.append((f"SAM Segments ({n_masks})", "sam_overlay.png"))
        except Exception as e:
            print(f"  [skipped] SAM failed: {e}")
    else:
        print(f"  [skipped] SAM: checkpoint '{args.sam_checkpoint}' not found")

    build_html(out_dir, pairs, full_panels, before_date_str, after_date_str)
    print(f"\nDone. Open {out_dir / 'index.html'} in a browser.")


if __name__ == "__main__":
    main()