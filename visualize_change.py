"""
visualize_change.py

Build a single self-contained HTML gallery comparing before/after Sentinel-2
imagery: before, after, NDVI before/after, NDBI before/after, a generic
band-difference heatmap, a DINOv2 patch-similarity change heatmap, and a
SAM (Segment Anything) segmentation overlay.

Usage:
    python visualize_change.py --raw-dir raw/ --out-dir viz_out/ \
        --sam-checkpoint sam_vit_b_01ec64.pth

Requirements:
    pip install rasterio numpy pillow matplotlib torch torchvision \
                transformers segment-anything opencv-python-headless

SAM checkpoint (download once, pick one size):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth   # smallest/fastest
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  # best/slowest

If --sam-checkpoint is omitted, the SAM panel is skipped (everything else
still runs). DINOv2 downloads its weights from the Hugging Face hub on
first run (needs internet once, then it's cached).

Assumes the two dates for each band are already co-registered (same
Sentinel-2 tile / pixel grid), which is the case for your raw/ folder
(tile 39SXV, both dates).
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
import matplotlib
import matplotlib.cm as cm

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


def read_band(path, scale=10000.0):
    """Read a single-band S2 reflectance tif, scaled to ~0-1."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr / scale


def read_visual(path):
    """Read the pre-made TCI visual.tif as an (H, W, 3) uint8 array."""
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.repeat(arr[0][:, :, None], 3, axis=2)
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    rgb = 255 * (rgb - rgb.min()) / (np.ptp(rgb) + 1e-6)
    return rgb.astype(np.uint8)


def load_pair(raw_dir, band, read_fn=read_band):
    dated = find_band_files(raw_dir, band)
    if len(dated) < 2:
        raise FileNotFoundError(
            f"Need 2 dates for band '{band}', found {len(dated)} in {raw_dir}"
        )
    (d0, p0), (d1, p1) = dated[0], dated[-1]
    print(f"  {band}: before={d0} ({os.path.basename(p0)})  after={d1} ({os.path.basename(p1)})")
    return read_fn(p0), read_fn(p1)


# --------------------------------------------------------------------------
# Index computation + colorization
# --------------------------------------------------------------------------

def normalized_diff(a, b):
    return (a - b) / (a + b + 1e-6)


def get_cmap(name):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)


def colorize(arr, cmap_name="RdYlGn", vmin=-1, vmax=1):
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    rgba = get_cmap(cmap_name)(norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def save_png(arr_uint8, path):
    Image.fromarray(arr_uint8).save(path)


# --------------------------------------------------------------------------
# DINOv2-based change heatmap
# --------------------------------------------------------------------------

def dino_change_heatmap(rgb_before, rgb_after, device="cpu"):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    model_name = "facebook/dinov2-small"
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
    dist = (1 - cos_sim).detach().numpy()

    dist_img = Image.fromarray((dist / (dist.max() + 1e-6) * 255).astype(np.uint8))
    dist_img = dist_img.resize((rgb_before.shape[1], rgb_before.shape[0]), Image.BILINEAR)
    heat = get_cmap("inferno")(np.array(dist_img) / 255.0)
    return (heat[:, :, :3] * 255).astype(np.uint8)


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
  h1 {{ font-weight:600; }}
  .grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; }}
  figure {{ margin:0; background:#1b1b1b; border-radius:8px; overflow:hidden; }}
  figure img {{ width:100%; display:block; }}
  figcaption {{ padding:8px 12px; font-size:14px; color:#aaa; }}
</style>
</head>
<body>
<h1>Bitemporal Change Visualization</h1>
<div class="grid">
{cards}
</div>
</body>
</html>
"""

CARD_TEMPLATE = """<figure>
  <img src="{src}" alt="{label}">
  <figcaption>{label}</figcaption>
</figure>"""


def build_html(out_dir, panels):
    cards = "\n".join(
        CARD_TEMPLATE.format(src=fname, label=label) for label, fname in panels
    )
    html = HTML_TEMPLATE.format(cards=cards)
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
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading bands...")
    visual_dates = find_band_files(args.raw_dir, "visual")
    rgb_before = read_visual(visual_dates[0][1])
    rgb_after = read_visual(visual_dates[-1][1])

    red_b, red_a = load_pair(args.raw_dir, "red")
    nir_b, nir_a = load_pair(args.raw_dir, "nir")
    swir_b, swir_a = load_pair(args.raw_dir, "swir16")

    print("Computing NDVI / NDBI...")
    ndvi_b = normalized_diff(nir_b, red_b)
    ndvi_a = normalized_diff(nir_a, red_a)
    ndbi_b = normalized_diff(swir_b, nir_b)
    ndbi_a = normalized_diff(swir_a, nir_a)

    print("Computing generic difference...")
    stack_b = np.stack([red_b, nir_b, swir_b])
    stack_a = np.stack([red_a, nir_a, swir_a])
    generic_diff = np.abs(stack_a - stack_b).mean(axis=0)
    generic_diff_img = colorize(
        generic_diff, cmap_name="viridis", vmin=0, vmax=float(np.percentile(generic_diff, 99))
    )

    panels = []

    save_png(rgb_before, out_dir / "before.png"); panels.append(("Before", "before.png"))
    save_png(rgb_after, out_dir / "after.png"); panels.append(("After", "after.png"))

    save_png(colorize(ndvi_b), out_dir / "ndvi_before.png"); panels.append(("NDVI Before", "ndvi_before.png"))
    save_png(colorize(ndvi_a), out_dir / "ndvi_after.png"); panels.append(("NDVI After", "ndvi_after.png"))
    save_png(colorize(ndbi_b), out_dir / "ndbi_before.png"); panels.append(("NDBI Before", "ndbi_before.png"))
    save_png(colorize(ndbi_a), out_dir / "ndbi_after.png"); panels.append(("NDBI After", "ndbi_after.png"))

    save_png(generic_diff_img, out_dir / "generic_diff.png")
    panels.append(("Generic Difference", "generic_diff.png"))

    print("Running DINOv2 change heatmap (downloads model weights on first run)...")
    try:
        dino_heat = dino_change_heatmap(rgb_before, rgb_after, device=args.device)
        save_png(dino_heat, out_dir / "dino_change.png")
        panels.append(("DINOv2 Change Heatmap", "dino_change.png"))
    except Exception as e:
        print(f"  [skipped] DINO heatmap failed: {e}")

    if args.sam_checkpoint:
        print("Running SAM segmentation on 'after' image...")
        try:
            sam_img, n_masks = sam_overlay(
                rgb_after, args.sam_checkpoint, args.sam_model_type, args.device
            )
            save_png(sam_img, out_dir / "sam_overlay.png")
            panels.append((f"SAM Segments ({n_masks})", "sam_overlay.png"))
        except Exception as e:
            print(f"  [skipped] SAM failed: {e}")
    else:
        print("  [skipped] SAM: no --sam-checkpoint given")

    build_html(out_dir, panels)
    print(f"\nDone. Open {out_dir / 'index.html'} in a browser.")


if __name__ == "__main__":
    main()
