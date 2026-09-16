#!/usr/bin/env python3
"""
bit_cd_infer.py — BIT_CD (Bitemporal Image Transformer) model loading and
tiled inference, pretrained on LEVIR-CD.

This is a LIBRARY module: model loading + tiled prediction, used by
categorized_change.py (the actual pipeline entry point — STAC download,
generic diff, NDVI/NDBI index change, BIT_CD, change graph, Qwen
explanation, all in one script). Run categorized_change.py, not this file,
for the full workflow.

This module also has a minimal standalone CLI for quick BIT_CD-only runs on
local image files, useful for testing BIT_CD in isolation.

SETUP (one-time):
  This repo vendors the BIT_CD model code + pretrained LEVIR-CD checkpoint
  directly under bit_cd/ (next to this script) — no separate clone needed.
    pip install -r requirements.txt

STANDALONE USAGE (local files only — see categorized_change.py for STAC):
  python bit_cd_infer.py --before before.png --after after.png --out results
  python bit_cd_infer.py --before before.png --after after.png --no-describe
  python bit_cd_infer.py --before before.png --after after.png --bit-cd-repo /path/to/BIT_CD

Notes:
  - Images are processed in 256x256 TILES (not a single whole-image resize) —
    this matches the resolution BIT_CD/LEVIR-CD was actually trained on.
    Resizing a full-scene image (e.g. LEVIR-CD's raw 1024x1024 train/A/train_1.png)
    straight down to 256x256 shrinks small buildings past the point the model
    can detect them; tiling avoids that.
  - This is a *building* change detector (LEVIR-CD is a building-CD
    dataset) — it will not reliably flag vegetation, road, or vehicle
    changes; that's expected behavior of the pretrained weights, not a bug.
"""

import argparse
import base64
import os
import sys
import types
from argparse import Namespace

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

# Vendored BIT_CD model code + LEVIR-CD checkpoint ship inside this repo at
# bit_cd/ — no external clone needed. Resolved relative to this file so it
# works no matter what directory you run the script from.
DEFAULT_VENDOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bit_cd")

# Sentinel-2's "visual"/red/green/nir bands are 10m/pixel. A crop needs to be
# at least tile_size px per side for BIT_CD to see real content rather than
# mostly border-replicated padding. categorized_change.py imports this to
# size its STAC fetch correctly when --use-bit-cd is on.
SENTINEL2_RES_M = 10


def min_buffer_km_for_tiles(tile_size=256, n_tiles_per_side=1):
    """km needed for a crop that's a clean multiple of tile_size px per
    side, at Sentinel-2's ~10m/pixel — so tiling has no partially-padded
    tiles at all, not just "big enough for one tile"."""
    px = tile_size * n_tiles_per_side
    return round(px * SENTINEL2_RES_M / 1000, 2)


def load_pair(before_path: str, after_path: str):
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)
    if before is None:
        raise FileNotFoundError(f"Could not read image: {before_path}")
    if after is None:
        raise FileNotFoundError(f"Could not read image: {after_path}")
    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    return before, after


def to_tensor_normalized(img_bgr):
    """BGR uint8 HWC -> normalized float32 CHW tensor, matching BIT_CD's
    training preprocessing (ToTensor + normalize mean=0.5, std=0.5 per channel)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - 0.5) / 0.5
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def _patch_torchvision_compat():
    """BIT_CD's vendored resnet.py imports the old
    torchvision.models.utils.load_state_dict_from_url, which newer
    torchvision versions removed. Shim it so the import doesn't crash."""
    try:
        import torchvision.models.utils as tvmu  # noqa: F401
    except ModuleNotFoundError:
        tvmu = types.ModuleType("torchvision.models.utils")
        sys.modules["torchvision.models.utils"] = tvmu
        import torchvision.models as tvm
        tvm.utils = tvmu

    import torchvision.models.utils as tvmu
    if not hasattr(tvmu, "load_state_dict_from_url"):
        from torch.hub import load_state_dict_from_url
        tvmu.load_state_dict_from_url = load_state_dict_from_url


def load_bit_cd_model(repo_path: str, checkpoint_path: str, net_g: str, device):
    """Load the BIT_CD generator network and pretrained LEVIR-CD weights."""
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"BIT_CD repo not found at {repo_path}. Clone it first:\n"
            "  git clone https://github.com/justchenhao/BIT_CD.git"
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    _patch_torchvision_compat()

    import models as bit_models  # the repo's own top-level `models` package

    # The backbone's __init__ downloads ImageNet weights (pretrained=True)
    # before the full LEVIR-CD checkpoint overwrites everything anyway.
    # Skip the pointless download so this works offline / faster.
    def _no_download(orig_fn):
        def wrapped(pretrained=False, progress=True, **kwargs):
            return orig_fn(pretrained=False, progress=progress, **kwargs)
        return wrapped

    for name in ("resnet18", "resnet34", "resnet50"):
        if hasattr(bit_models, name):
            setattr(bit_models, name, _no_download(getattr(bit_models, name)))

    from models.networks import define_G

    args_ns = Namespace(net_G=net_g, n_class=2)
    net_g_model = define_G(args_ns, gpu_ids=[])
    net_g_model.to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. The official repo ships it at "
            "checkpoints/BIT_LEVIR/best_ckpt.pt after cloning."
        )
    # weights_only=False: PyTorch >=2.6 defaults to True, which rejects this
    # older checkpoint's pickled numpy scalars. Safe here since it's the
    # official BIT_CD repo's own published checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net_g_model.load_state_dict(checkpoint["model_G_state_dict"])
    net_g_model.eval()

    return net_g_model


@torch.no_grad()
def predict_change_mask(model, before, after, device, tile_size=256, min_area=100):
    """
    Tile the image pair into tile_size x tile_size patches (matching BIT_CD's
    training resolution), run the model per tile, and stitch the predicted
    masks back together at full resolution. This avoids the accuracy loss
    from resizing a large scene (e.g. LEVIR-CD's raw 1024x1024 images) down
    to 256x256 in one shot, which shrinks small buildings past the point the
    model can detect them.
    """
    h, w = before.shape[:2]
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    before_p = cv2.copyMakeBorder(before, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    after_p = cv2.copyMakeBorder(after, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    H, W = before_p.shape[:2]

    mask_p = np.zeros((H, W), dtype=np.uint8)
    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile1 = before_p[y:y + tile_size, x:x + tile_size]
            tile2 = after_p[y:y + tile_size, x:x + tile_size]

            t1 = to_tensor_normalized(tile1).to(device)
            t2 = to_tensor_normalized(tile2).to(device)
            logits = model(t1, t2)                        # (1, 2, tile, tile)
            pred = torch.argmax(logits, dim=1)             # (1, tile, tile)
            pred = pred.squeeze(0).byte().cpu().numpy() * 255
            mask_p[y:y + tile_size, x:x + tile_size] = pred

    mask = mask_p[:h, :w]  # drop the padding, back to original resolution

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]

    changed_pct = 100.0 * np.count_nonzero(mask) / mask.size
    return mask, boxes, changed_pct


def draw_overlay(after, boxes):
    overlay = after.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return overlay


def encode_image_b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def describe_with_qwen(before, after, overlay, boxes, changed_pct,
                        model="qwen3-vl", ollama_url="http://localhost:11434"):
    """Ask a local Qwen VL model (via Ollama) to describe the detected changes."""
    prompt = (
        "You are comparing a 'before' and an 'after' satellite/aerial image of the "
        "same location. A change detection model trained on the LEVIR-CD building "
        f"dataset found {len(boxes)} changed region(s), covering about "
        f"{changed_pct:.1f}% of the image area (highlighted with red boxes in the third image).\n"
        "Describe, in plain language, what actually changed — new construction, "
        "demolished buildings, structural additions, etc. Be concise and specific."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image_b64(before), encode_image_b64(after), encode_image_b64(overlay)],
        "stream": False,
    }
    resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def main():
    """Minimal standalone CLI for quick BIT_CD-only testing on local files.
    For the full pipeline (STAC download, generic diff, NDVI/NDBI, BIT_CD,
    change graph, Qwen), use categorized_change.py instead."""
    ap = argparse.ArgumentParser(description="Standalone BIT_CD-only change detection (local files)")
    ap.add_argument("--before", required=True, help="Path to the 'before' image")
    ap.add_argument("--after", required=True, help="Path to the 'after' image")
    ap.add_argument("--out", default="bit_cd_output", help="Output directory")
    ap.add_argument("--bit-cd-repo", default=None,
                     help="Path to the BIT_CD model code + checkpoint. Defaults to the "
                          "vendored copy shipped in this repo (./bit_cd).")
    ap.add_argument("--checkpoint", default=None,
                     help="Path to checkpoint (.pt). Defaults to "
                          "<bit-cd-repo>/checkpoints/BIT_LEVIR/best_ckpt.pt")
    ap.add_argument("--net-g", default="base_transformer_pos_s4_dd8_dedim8",
                     help="Generator architecture name (must match the checkpoint)")
    ap.add_argument("--tile-size", type=int, default=256,
                     help="Tile size for inference (matches BIT_CD's training resolution)")
    ap.add_argument("--min-area", type=int, default=100, help="Minimum changed-region area in px")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--model", default="qwen3-vl", help="Ollama VLM model name")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--no-describe", action="store_true", help="Skip the VLM description step")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    before, after = load_pair(args.before, args.after)
    if min(before.shape[:2]) < args.tile_size:
        print(f"Warning: your image is {before.shape[1]}x{before.shape[0]}px — smaller than "
              f"the {args.tile_size}x{args.tile_size}px BIT_CD was trained on. It will be "
              f"padded with replicated edge pixels rather than resized, so BIT_CD is mostly "
              f"looking at stretched padding and likely won't find real changes here.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = args.checkpoint or os.path.join(
        args.bit_cd_repo or DEFAULT_VENDOR_PATH, "checkpoints", "BIT_LEVIR", "best_ckpt.pt"
    )

    print(f"Loading BIT_CD ({args.net_g}) on {device}...")
    model = load_bit_cd_model(args.bit_cd_repo or DEFAULT_VENDOR_PATH, checkpoint_path, args.net_g, device)

    mask, boxes, changed_pct = predict_change_mask(
        model, before, after, device, tile_size=args.tile_size, min_area=args.min_area
    )
    overlay = draw_overlay(after, boxes)

    mask_path = os.path.join(args.out, "mask.png")
    overlay_path = os.path.join(args.out, "overlay.png")
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)

    print(f"Changed area: {changed_pct:.2f}%")
    print(f"Detected {len(boxes)} changed region(s):")
    for i, (x, y, w, h) in enumerate(boxes, 1):
        print(f"  [{i}] box=({x},{y},{w},{h})")
    print(f"Saved mask -> {mask_path}")
    print(f"Saved overlay -> {overlay_path}")

    if not args.no_describe:
        if not boxes:
            print("\nNo changes detected — skipping VLM description.")
            return
        print("\nAsking Qwen VLM to describe the changes...")
        try:
            description = describe_with_qwen(
                before, after, overlay, boxes, changed_pct,
                model=args.model, ollama_url=args.ollama_url,
            )
            print("\n--- Change Description ---")
            print(description)
            with open(os.path.join(args.out, "description.txt"), "w") as f:
                f.write(description)
        except requests.exceptions.RequestException as e:
            print(f"\nCould not reach Ollama at {args.ollama_url}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
