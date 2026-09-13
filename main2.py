#!/usr/bin/env python3
"""
change_detect.py — simple before/after change detection + VLM description.

Pipeline:
  1. Load "before" and "after" images (auto-resizes to match).
  2. Geometrically align "after" onto "before" (ECC / homography) to cancel
     out small camera shifts so they don't look like false "changes".
  3. Compute a change mask using SSIM (structural similarity) rather than
     raw pixel differencing — much more robust to lighting/exposure shifts.
  4. Find changed regions (bounding boxes) and draw an overlay.
  5. Send before/after/overlay images to a local Qwen VLM (via Ollama) and
     ask it to describe what changed, in plain language.

Usage:
  python change_detect.py --before before.jpg --after after.jpg --out out_dir
  python change_detect.py --before before.jpg --after after.jpg --model qwen3-vl --no-describe
  python change_detect.py --before before.jpg --after after.jpg --no-align   # skip alignment

Requirements:
  pip install opencv-python-headless numpy requests scikit-image
  Ollama running locally with a Qwen VL model pulled, e.g.:
    ollama pull qwen3-vl
    ollama serve   (usually already running as a service)
"""

import argparse
import base64
import json
import os
import sys

import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity as ssim


def load_and_align(before_path: str, after_path: str):
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)
    if before is None:
        raise FileNotFoundError(f"Could not read image: {before_path}")
    if after is None:
        raise FileNotFoundError(f"Could not read image: {after_path}")

    # Resize "after" to match "before" if dimensions differ.
    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))

    return before, after


def align_images(before, after, max_iterations=200, eps=1e-6):
    """
    Geometrically align 'after' onto 'before' using ECC with a homography
    warp model. Corrects small camera shifts/rotation/zoom between the two
    shots so that alignment error doesn't get mistaken for scene change.
    Falls back to the unaligned image if ECC fails to converge (e.g. scenes
    are too different for feature-based registration).
    """
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)

    try:
        _, warp_matrix = cv2.findTransformECC(
            g1, g2, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria
        )
        h, w = before.shape[:2]
        aligned = cv2.warpPerspective(
            after, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, True
    except cv2.error:
        return after, False


def compute_change_mask(before, after, blur_ksize=5, ssim_thresh=0.85, min_area=150):
    """
    SSIM-based change detection: compares local structure/contrast instead of
    raw pixel values, so it's far less sensitive to lighting, shadows, and
    minor exposure differences than a plain absdiff would be.
    ssim_thresh: pixels with local similarity BELOW this are flagged as changed
    (0-1 scale; lower = more sensitive to change).
    """
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    g1 = cv2.GaussianBlur(g1, (blur_ksize, blur_ksize), 0)
    g2 = cv2.GaussianBlur(g2, (blur_ksize, blur_ksize), 0)

    score, diff_map = ssim(g1, g2, full=True)
    # diff_map is in [-1, 1] (1 = identical); convert to a 0-255 dissimilarity map.
    dissimilarity = ((1.0 - diff_map) * 127.5).astype(np.uint8)
    thresh_val = int((1.0 - ssim_thresh) * 127.5)
    _, mask = cv2.threshold(dissimilarity, thresh_val, 255, cv2.THRESH_BINARY)

    # Clean up noise, close small gaps.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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
    """Ask a local Qwen VL model (via Ollama) to describe the changes."""
    prompt = (
        "You are comparing a 'before' and an 'after' image of the same scene.\n"
        f"An automated diff found {len(boxes)} changed region(s), covering about "
        f"{changed_pct:.1f}% of the image area (highlighted with red boxes in the third image).\n"
        "Describe, in plain language, what actually changed between the two images "
        "(objects added/removed/moved, structural changes, etc). "
        "Ignore minor lighting or noise differences. Be concise and specific."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [
            encode_image_b64(before),
            encode_image_b64(after),
            encode_image_b64(overlay),
        ],
        "stream": False,
    }

    resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def main():
    ap = argparse.ArgumentParser(description="Before/after change detection + VLM description")
    ap.add_argument("--before", required=True, help="Path to the 'before' image")
    ap.add_argument("--after", required=True, help="Path to the 'after' image")
    ap.add_argument("--out", default="change_output", help="Output directory")
    ap.add_argument("--ssim-thresh", type=float, default=0.85,
                     help="SSIM similarity threshold, 0-1 (lower = more sensitive to change)")
    ap.add_argument("--min-area", type=int, default=150, help="Minimum changed-region area in px")
    ap.add_argument("--no-align", action="store_true",
                     help="Skip ECC geometric alignment (use if images are already pixel-aligned)")
    ap.add_argument("--model", default="qwen3-vl", help="Ollama VLM model name")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--no-describe", action="store_true", help="Skip the VLM description step")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    before, after = load_and_align(args.before, args.after)

    if not args.no_align:
        after_aligned, aligned_ok = align_images(before, after)
        if aligned_ok:
            print("Alignment: converged, using warped 'after' image.")
            after = after_aligned
        else:
            print("Alignment: did not converge, continuing with unaligned images.")

    mask, boxes, changed_pct = compute_change_mask(
        before, after, ssim_thresh=args.ssim_thresh, min_area=args.min_area
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
            print("\nNo significant changes detected — skipping VLM description.")
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
            print("Make sure 'ollama serve' is running and the model is pulled "
                  f"(ollama pull {args.model}).", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
