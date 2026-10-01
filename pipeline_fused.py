#!/usr/bin/env python3
"""
UrbanSentinel fused change-detection pipeline.

    imagery pair -> [DINOv2 + BIT + spectral] -> fusion -> SAM/CC regions
                 -> VLM (Ollama qwen3-vl) final audit -> report + overlays

Signals
    dino      DINOv2 patch-embedding cosine distance (semantic change)
    bit       BIT_CD transformer, LEVIR-CD pretrained (building change)
    spectral  brightness/color proxy deltas (always available; pass real
              multispectral bands for physical NDVI/NDBI when you have them)
    sam       Segment Anything segments objects; each inherits the fused score
    vlm       final arbiter: confirms/rejects + categorizes each region

Usage
    python pipeline.py --before a.png --after b.png
    python pipeline.py --before a.png --after b.png --no-sam --no-vlm
    python pipeline.py --before a.png --after b.png --signals dino,spectral
    python pipeline.py --before a.tif --after b.tif --band-mode multispectral
"""
import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

# Allow running from inside UrbanSentinel/ or the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from UrbanSentinel import detectors, fusion, vlm
except ImportError:
    import detectors  # noqa: F401
    import fusion  # noqa: F401
    import vlm  # noqa: F401

import rasterio  # noqa: E402  (optional for multispectral mode)

from detectors import load_sam  # noqa: E402
from fusion import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, normalize01  # noqa: E402
from vlm import classify_regions  # noqa: E402


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_pair(before_path, after_path):
    """Load a before/after pair as BGR uint8.

    Plain images (png/jpg) are read directly. GeoTIFFs: RGB visual bands are
    used for the VLM and DINO/SAM; with --band-mode multispectral the
    spectral signal uses true NIR/SWIR-based NDVI/NDBI deltas instead of the
    RGB proxy.
    """
    def read_any(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            try:
                import rasterio
                with rasterio.open(path) as src:
                    data = src.read(out_shape=(min(3, src.count), src.height, src.width))
                    rgb = np.moveaxis(data[:3], 0, -1)
                    if rgb.dtype != np.uint8:
                        hi = np.percentile(rgb[np.isfinite(rgb)], 99) if np.isfinite(rgb).any() else 255
                        scale = 255.0 / max(hi, 1e-6) if hi <= 1.5 else 1.0
                        if hi > 255:
                            scale = 255.0 / max(hi, 1e-6)
                        rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), src.count
            except Exception:
                raise FileNotFoundError(f"Could not read image: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img, 3

    before, nch_b = read_any(before_path)
    after, nch_a = read_any(after_path)
    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    return before, after


def spectral_from_bands(before_path, after_path):
    """True dNDVI/dNDBI magnitude from multispectral GeoTIFFs (red/nir/swir16)."""
    def band(path, name):
        with rasterio.open(path) as src:
            for i in range(1, src.count + 1):
                if name in (src.descriptions[i - 1] or "").lower():
                    return src.read(i).astype(np.float32)
        return None

    def norm(x):
        finite = x[np.isfinite(x)]
        if finite.size and np.percentile(finite, 99) > 2.0:
            x = x / 10000.0
        return np.clip(x, 0, 1)

    out = None
    for a_name, b_name in (("nir", "red"), ("swir16", "nir")):
        a1, b1 = band(before_path, a_name), band(before_path, b_name)
        a2, b2 = band(after_path, a_name), band(after_path, b_name)
        if None in (a1, b1, a2, b2):
            return None
        idx1 = (a1 - b1) / np.maximum(a1 + b1, 0.02)
        idx2 = (a2 - b2) / np.maximum(a2 + b2, 0.02)
        d = np.abs(norm(idx2) - norm(idx1))
        out = d if out is None else np.maximum(out, d)
    if out is None:
        return None
    return normalize01(out)


# --------------------------------------------------------------------------- #
# Visualization + report
# --------------------------------------------------------------------------- #
def make_overlay(after, regions, answers=None, only_confirmed=False):
    vis = after.copy()
    for i, r in enumerate(regions, start=1):
        ans = (answers or {}).get(i, {})
        if only_confirmed and not ans.get("confirmed", True):
            continue
        color = (0, 200, 0) if ans.get("confirmed", True) else (0, 0, 255)
        cnts, _ = cv2.findContours(r["seg"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)
        x, y, w, h = r["box"]
        cv2.putText(vis, f"#{i}", (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def score_card_image(fused, per_signal):
    """Horizontal strip of all score maps for quick visual comparison."""
    tiles = []
    for name, m in [("fused", fused)] + sorted(per_signal.items()):
        m8 = (normalize01(m) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(m8, cv2.COLORMAP_INFERNO)
        cv2.putText(heat, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        tiles.append(heat)
    return np.hstack(tiles)


def write_report(out_dir, regions, answers, det_meta, weights, thresholds, paths):
    lines = [
        f"UrbanSentinel fused change report — {datetime.now().isoformat(timespec='seconds')}",
        f"before: {paths[0]}",
        f"after : {paths[1]}",
        f"signals: {sorted(weights.keys())} weights={weights}",
        f"thresholds={thresholds}",
        f"detector meta: {json.dumps({k: v for k, v in det_meta.items()}, default=str)}",
        "",
    ]
    for i, r in enumerate(regions, start=1):
        x, y, w, h = r["box"]
        ans = answers.get(i, {})
        status = ("CONFIRMED" if ans.get("confirmed")
                  else "REJECTED" if ans else "UNREVIEWED")
        lines.append(
            f"#{i} box=({x},{y},{w},{h}) px={int(r['seg'].sum())} "
            f"score={r['score']:.2f} fired={'+'.join(sorted(r['sources'])) or 'none'} "
            f"[{status}] {ans.get('text', '')}")
    path = os.path.join(out_dir, "report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="UrbanSentinel fused change detection")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", default="fused_output")
    ap.add_argument("--signals", default="dino,bit,spectral,structure",
                    help="Comma list from: dino,bit,spectral,structure")
    ap.add_argument("--band-mode", choices=["rgb", "multispectral"], default="rgb",
                    help="multispectral: use real NDVI/NDBI from GeoTIFF bands for "
                         "the spectral signal (needs red/nir/swir16 bands)")
    ap.add_argument("--dino-model", default="dinov2_vits14")
    ap.add_argument("--bit-checkpoint", default=None)
    ap.add_argument("--sam-checkpoint", default=None)
    ap.add_argument("--sam-points", type=int, default=32)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--max-regions", type=int, default=30)
    ap.add_argument("--min-score", type=float, default=0.2,
                    help="Min fused score for a region to be reported")
    ap.add_argument("--w-dino", type=float, default=DEFAULT_WEIGHTS["dino"])
    ap.add_argument("--w-bit", type=float, default=DEFAULT_WEIGHTS["bit"])
    ap.add_argument("--w-spectral", type=float, default=DEFAULT_WEIGHTS["spectral"])
    ap.add_argument("--w-structure", type=float, default=DEFAULT_WEIGHTS["structure"])
    ap.add_argument("--bare-soil-thresh", type=float, default=0.65,
                    help="Region flagged as bare-soil context when bare fraction >= this")
    ap.add_argument("--thr-dino", type=float, default=DEFAULT_THRESHOLDS["dino"])
    ap.add_argument("--thr-bit", type=float, default=DEFAULT_THRESHOLDS["bit"])
    ap.add_argument("--thr-spectral", type=float, default=DEFAULT_THRESHOLDS["spectral"])
    ap.add_argument("--thr-structure", type=float, default=DEFAULT_THRESHOLDS["structure"])
    ap.add_argument("--thr-fused", type=float, default=DEFAULT_THRESHOLDS["fused"])
    # VLM
    ap.add_argument("--vlm-model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--vlm-batch", type=int, default=4)
    ap.add_argument("--vlm-timeout", type=int, default=900)
    ap.add_argument("--vlm-max-tokens", type=int, default=1024)
    ap.add_argument("--no-vlm", action="store_true", help="Skip the VLM audit stage")
    ap.add_argument("--no-sam", action="store_true", help="Skip SAM (use CC regions only)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    enabled = [s.strip() for s in args.signals.split(",") if s.strip()]

    print(f"Loading pair: {args.before} | {args.after}")
    before, after = load_pair(args.before, args.after)
    print(f"  size: {before.shape[1]}x{before.shape[0]}")

    weights = {"dino": args.w_dino, "bit": args.w_bit, "spectral": args.w_spectral,
               "structure": args.w_structure}
    thresholds = {"dino": args.thr_dino, "bit": args.thr_bit,
                  "spectral": args.thr_spectral, "structure": args.thr_structure,
                  "fused": args.thr_fused}

    sam = None
    if not args.no_sam:
        print("Loading SAM...")
        sam = load_sam(args.sam_checkpoint, points_per_side=args.sam_points,
                       min_mask_region_area=args.min_area)
        if sam is None:
            print("  SAM unavailable — falling back to connected-component regions.")

    print(f"Running detectors: {enabled} ...")
    score_maps, det_meta = fusion.run_detectors(
        before, after, enabled, dino_model=args.dino_model,
        bit_checkpoint=args.bit_checkpoint)
    if args.band_mode == "multispectral":
        real = spectral_from_bands(args.before, args.after)
        if real is not None:
            score_maps["spectral"] = real
            det_meta["spectral"] = {"kind": "NDVI/NDBI from GeoTIFF bands"}
        else:
            print("  multispectral bands not found; keeping RGB proxy spectral signal.")

    if not score_maps:
        sys.exit("No detector produced a score map.")

    fused, per_signal = fusion.fuse_score_maps(score_maps, weights, before.shape[:2])
    cand = fusion.candidate_mask(fused, per_signal, thresholds)
    print(f"  fused map done; candidate pixels: {int(cand.sum())}")

    regions = []
    if sam is not None:
        # Free DINO/BIT weights first — SAM needs the GPU headroom.
        detectors.unload_detectors()
        found = detectors.sam_regions(after, fused, sam, min_score=args.min_score,
                                      min_area=args.min_area, candidate_mask=cand,
                                      min_overlap=0.10)
        regions += fusion.sam_to_regions(found, before.shape[:2])
        print(f"  SAM regions: {len(found)}")
    regions += fusion.connected_regions(cand, fused, min_area=args.min_area)
    for r in regions:
        r["sources"] = fusion.signal_support(r, per_signal, thresholds)
    regions = fusion.merge_regions(regions, args.max_regions)
    regions = [r for r in regions if r["score"] >= args.min_score * 0.5]
    print(f"{len(regions)} region(s) after merging.")

    cv2.imwrite(os.path.join(args.out, "fused_heatmap.png"),
                score_card_image(fused, per_signal))
    cv2.imwrite(os.path.join(args.out, "candidate_mask.png"), cand * 255)

    answers = {}
    if regions and not args.no_vlm:
        # Attach bare-soil context + per-signal means BEFORE the VLM sees the
        # regions, so its verdict rests on evidence, not just the heatmap.
        for i, r in enumerate(regions, start=1):
            frac, ctx = fusion.is_bare_soil(before, after, r["seg"])
            r["context"] = {"is_bare_soil": frac >= args.bare_soil_thresh, **ctx}
            r["signal_means"] = {
                name: round(float(m[r["seg"]].mean()), 2)
                for name, m in per_signal.items() if r["seg"].any()
            }
        n_bare = sum(1 for r in regions if r["context"]["is_bare_soil"])
        print(f"  bare-soil flagged regions: {n_bare}/{len(regions)}")
        print(f"Auditing {len(regions)} region(s) with {args.vlm_model} ...")
        try:
            answers = classify_regions(
                regions, fused, before, after, out_dir=args.out,
                url=args.ollama_url, model=args.vlm_model, batch=args.vlm_batch,
                timeout=args.vlm_timeout, max_tokens=args.vlm_max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"VLM stage failed ({e}) — keeping geometric results only.",
                  file=sys.stderr)
    else:
        print("VLM stage skipped.")

    overlay = make_overlay(after, regions, answers)
    cv2.imwrite(os.path.join(args.out, "overlay.png"), overlay)
    report = write_report(args.out, regions, answers, det_meta, weights,
                          thresholds, (args.before, args.after))
    print(f"Saved overlay + report -> {args.out}{os.sep} (report: {report})")
    print("\n--- Regions ---")
    print(open(report, encoding="utf-8").read())


if __name__ == "__main__":
    main()
