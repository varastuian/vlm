#!/usr/bin/env python3
"""
Unified change-detection pipeline for Sentinel-2 bi-temporal imagery.

Pipeline:
    Sentinel-2 T1 + Sentinel-2 T2
        │
        ├── IR-MAD statistical change
        │
        ├── Spectral change evidence
        │     ├── dNDVI
        │     ├── dNDBI
        │     ├── dMNDWI
        │     ├── brightness change
        │     └── spectral angle
        │
        └── DINOv2 semantic feature change
              T1 features vs T2 features
                   │
                   ▼
           candidate change fusion
                   │
                   ▼
          morphology / region cleanup
                   │
                   ▼
             connected components
                   │
                   ▼
          final change regions
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_raster

sys.path.insert(0, str(Path(__file__).resolve().parent))

import change_core as cc
from detectors import dino_detector, load_dino, unload_detectors


@dataclass
class PipelineParams:
    """Configuration parameters for the unified pipeline."""
    sensitivity: str = "balanced"
    min_area_px: int = 5
    sam_min: float = 0.05
    min_significance: float = 7.0
    reject_slivers: bool = True
    ignore_water_variability: bool = True
    smooth_sigma: float = 1.0
    calibrate: bool = True
    semantic_min_pct: float = 60.0
    
    # DINOv2 settings
    dino_model: str = "dinov2_vits14"
    dino_weight: float = 0.35
    dino_threshold: float = 0.35
    
    # Fusion weights
    irmad_weight: float = 0.30
    spectral_weight: float = 0.20
    dino_weight: float = 0.35
    structure_weight: float = 0.15
    
    # Thresholds
    fused_threshold: float = 0.45
    
    # Region filtering
    min_region_area: int = 100
    max_regions: int = 30
    min_region_score: float = 0.2
    
    # Structure gate (optional)
    target_structures: bool = True
    all_changes: bool = False
    min_rect: float = 0.80
    min_solidity: float = 0.88
    building_max_aspect: float = 4.0
    building_max_px: int = 150
    road_min_len_px: int = 12
    road_max_width_px: float = 6.0
    road_min_aspect: float = 3.5
    min_shape_px: int = 16
    tiny_min_significance: float = 12.0
    max_ndvi_after: float = 0.30
    
    # Output
    output_dir: str = "unified_output"


def parse_args():
    ap = argparse.ArgumentParser(description="Unified Sentinel-2 change detection pipeline")
    ap.add_argument("--before", default="data/S2A_39SXV_20200128_1_L2A_visual.tif",
                    help="Path to T1 (before) visual GeoTIFF or directory")
    ap.add_argument("--after", default="data/S2B_39SXV_20260128_0_L2A_visual.tif",
                    help="Path to T2 (after) visual GeoTIFF or directory")
    ap.add_argument("--aoi", nargs=4, type=int, default=[1487,10075, 225, 225],
                    metavar=("X", "Y", "W", "H"),
                    help="AOI window on T2 visual grid (x y w h)")
    ap.add_argument("--output-dir", default="unified_output")
    ap.add_argument("--sensitivity", choices=["strict", "balanced", "sensitive"], default="balanced")
    ap.add_argument("--min-area", type=int, default=5)
    ap.add_argument("--min-significance", type=float, default=7.0)
    ap.add_argument("--no-water-veto", action="store_true")
    ap.add_argument("--no-sliver-reject", action="store_true")
    ap.add_argument("--dino-model", default="dinov2_vits14")
    ap.add_argument("--dino-weight", type=float, default=0.35)
    ap.add_argument("--irmad-weight", type=float, default=0.30)
    ap.add_argument("--spectral-weight", type=float, default=0.20)
    ap.add_argument("--structure-weight", type=float, default=0.15)
    ap.add_argument("--fused-threshold", type=float, default=0.45)
    ap.add_argument("--min-region-area", type=int, default=100)
    ap.add_argument("--max-regions", type=int, default=30)
    ap.add_argument("--min-region-score", type=float, default=0.2)
    ap.add_argument("--no-structure-gate", action="store_true")
    ap.add_argument("--all-changes", action="store_true", help="Disable structure gate (detect all changes)")
    return ap.parse_args()


def load_scenes_from_paths(before_path: str, after_path: str, aoi: Optional[Tuple[int, int, int, int]] = None):
    """Load Sentinel-2 scenes from visual GeoTIFF paths and discover matching band files."""
    before_vis = Path(before_path)
    after_vis = Path(after_path)
    
    if before_vis.is_dir():
        scenes_before = cc.find_scenes(before_vis)
        before_vis = scenes_before[list(scenes_before.keys())[-1]]["visual"]
    
    if after_vis.is_dir():
        scenes_after = cc.find_scenes(after_vis)
        after_vis = scenes_after[list(scenes_after.keys())[-1]]["visual"]
    
    data_dir = before_vis.parent
    scenes = cc.find_scenes(data_dir)
    
    date_before = cc.DATE_RE.search(before_vis.name).group(1)
    date_after = cc.DATE_RE.search(after_vis.name).group(1)
    
    if date_before not in scenes or date_after not in scenes:
        raise ValueError(f"Could not find both dates in {data_dir}. Available: {list(scenes.keys())}")
    
    scene_before = scenes[date_before]
    scene_after = scenes[date_after]
    
    if aoi is None:
        with rasterio.open(after_vis) as ref:
            aoi = (0, 0, ref.width, ref.height)
    
    return cc.load_aoi_pair(scene_before, scene_after, *aoi)


def run_irmad_spectral(before: cc.Scene, after: cc.Scene, params: PipelineParams, meta: dict) -> cc.Result:
    """Run IR-MAD + spectral evidence pipeline from change_core."""
    cc_params = cc.Params(
        sensitivity=params.sensitivity,
        min_area_px=params.min_area_px,
        sam_min=params.sam_min,
        min_significance=params.min_significance,
        reject_slivers=params.reject_slivers,
        ignore_water_variability=params.ignore_water_variability,
        smooth_sigma=params.smooth_sigma,
        calibrate=params.calibrate,
        semantic_min_pct=params.semantic_min_pct,
        target="structures" if params.target_structures and not params.all_changes else "all",
        min_rect=params.min_rect,
        min_solidity=params.min_solidity,
        building_max_aspect=params.building_max_aspect,
        building_max_px=params.building_max_px,
        road_min_len_px=params.road_min_len_px,
        road_max_width_px=params.road_max_width_px,
        road_min_aspect=params.road_min_aspect,
        min_shape_px=params.min_shape_px,
        tiny_min_significance=params.tiny_min_significance,
        max_ndvi_after=params.max_ndvi_after,
    )
    
    px_area_m2 = abs(meta['res'][0] * meta['res'][1])
    return cc.detect_changes(before, after, cc_params, px_area_m2=px_area_m2)


def run_dinov2(before: cc.Scene, after: cc.Scene, params: PipelineParams) -> Tuple[np.ndarray, dict]:
    """Run DINOv2 semantic feature change detection."""
    before_rgb = cv2.cvtColor(before.rgb, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after.rgb, cv2.COLOR_BGR2RGB)
    
    score_map, meta = dino_detector(before_rgb, after_rgb, model_name=params.dino_model)
    if score_map is None:
        print("  DINOv2 unavailable, skipping semantic signal")
        return None, {}
    
    score_map = cc.normalize01(score_map) if hasattr(cc, 'normalize01') else normalize01(score_map)
    return score_map, meta


def normalize01(x, low=2, high=98):
    """Percentile stretch to 0-1, robust to outliers."""
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def fuse_signals(irmad_result: cc.Result, dino_map: Optional[np.ndarray], params: PipelineParams) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fuse IR-MAD, spectral, DINOv2, and structure signals into a single score map."""
    H, W = irmad_result.T.shape
    
    irmad_score = cc.heatmap(irmad_result.T, irmad_result.valid)[:, :, 0] / 255.0
    irmad_score = normalize01(irmad_score)
    
    spectral_score = irmad_result.indices.get("landcover_evidence", np.zeros((H, W), dtype=bool)).astype(np.float32)
    spectral_score = cv2.GaussianBlur(spectral_score, (0, 0), 1.0)
    spectral_score = normalize01(spectral_score)
    
    structure_score = normalize01(irmad_result.sam) if hasattr(irmad_result, 'sam') else np.zeros((H, W))
    
    weights = {
        "irmad": params.irmad_weight,
        "spectral": params.spectral_weight,
        "dino": params.dino_weight,
        "structure": params.structure_weight,
    }
    
    fused = np.zeros((H, W), dtype=np.float32)
    per_signal = {
        "irmad": irmad_score,
        "spectral": spectral_score,
        "structure": structure_score,
    }
    total_w = params.irmad_weight + params.spectral_weight + params.structure_weight
    
    fused += params.irmad_weight * irmad_score
    fused += params.spectral_weight * spectral_score
    fused += params.structure_weight * structure_score
    
    if dino_map is not None:
        if dino_map.shape != (H, W):
            dino_map = cv2.resize(dino_map, (W, H), interpolation=cv2.INTER_LINEAR)
        dino_map = normalize01(dino_map)
        per_signal["dino"] = dino_map
        fused += params.dino_weight * dino_map
        total_w += params.dino_weight
    
    if total_w > 0:
        fused /= total_w
    
    return fused, per_signal


def candidate_mask(fused: np.ndarray, per_signal: Dict[str, np.ndarray], 
                   thresholds: Dict[str, float]) -> np.ndarray:
    """Union rule: pixel is candidate if any signal crosses threshold OR fused crosses threshold."""
    h, w = fused.shape
    votes = np.zeros((h, w), dtype=np.uint8)
    
    for name, thr in thresholds.items():
        if name == "fused":
            continue
        m = per_signal.get(name)
        if m is not None:
            votes |= (m >= thr).astype(np.uint8)
    
    fused_thr = thresholds.get("fused", 0.45)
    votes |= (fused >= fused_thr).astype(np.uint8)
    
    votes = cv2.morphologyEx(votes * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (votes > 0).astype(np.uint8)


def connected_regions(mask: np.ndarray, score_map: np.ndarray, 
                      min_area: int = 100, max_regions: int = 30) -> List[Dict]:
    """Extract regions from connected components of candidate mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        seg = (labels == i)
        score = float(score_map[seg].mean())
        regions.append({
            "box": (int(x), int(y), int(w), int(h)),
            "score": score,
            "seg": seg,
            "area_px": int(area),
            "sources": set(),
        })
    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions[:max_regions]


def signal_support(region: Dict, per_signal: Dict[str, np.ndarray], 
                   thresholds: Dict[str, float]) -> set:
    """Which signals fire inside this region (mean >= threshold)?"""
    seg = region["seg"]
    if not seg.any():
        return set()
    fired = set()
    for name, m in per_signal.items():
        if name in thresholds and float(m[seg].mean()) >= thresholds[name]:
            fired.add(name)
    return fired


def merge_regions(regions: List[Dict], max_regions: int, overlap: float = 0.4) -> List[Dict]:
    """De-duplicate overlapping regions."""
    kept = []
    for r in sorted(regions, key=lambda r: r["score"], reverse=True):
        x, y, w, h = r["box"]
        dup = False
        for k in kept:
            kx, ky, kw, kh = k["box"]
            iw = min(x + w, kx + kw) - max(x, kx)
            ih = min(y + h, ky + kh) - max(y, ky)
            if iw > 0 and ih > 0:
                ov = iw * ih / min(w * h, kw * kh)
                if ov >= overlap:
                    k["sources"] |= r["sources"]
                    k["score"] = max(k["score"], r["score"])
                    dup = True
                    break
        if not dup:
            kept.append(r)
        if len(kept) >= max_regions:
            break
    return kept


def enrich_regions(regions: List[Dict], irmad_result: cc.Result, 
                   per_signal: Dict[str, np.ndarray], before: cc.Scene, after: cc.Scene) -> List[Dict]:
    """Enrich regions with spectral metrics and class info."""
    for r in regions:
        seg = r["seg"]
        ys, xs = np.nonzero(seg)
        
        r["sources"] = signal_support(r, per_signal, {
            "irmad": 0.35, "spectral": 0.35, "dino": 0.35, "structure": 0.25, "fused": 0.45
        })
        
        r["dNDVI"] = round(float(irmad_result.indices["dNDVI"][seg].mean()), 3)
        r["dNDBI"] = round(float(irmad_result.indices["dNDBI"][seg].mean()), 3)
        r["dMNDWI"] = round(float(irmad_result.indices["dMNDWI"][seg].mean()), 3)
        r["dBrightness"] = round(float(irmad_result.indices["dBright"][seg].mean()), 3)
        r["spectral_angle_deg"] = round(float(np.degrees(irmad_result.sam[seg].mean())), 1)
        r["significance"] = round(float(-np.log10(np.maximum(cc.chi2.sf(irmad_result.T[seg], cc.NB), 1e-300)).mean()), 1)
        
        cnt = np.bincount(irmad_result.class_map[seg], minlength=8)
        cnt[0] = 0
        code = int(cnt.argmax())
        r["class_code"] = code
        r["class_name"] = cc.CLASS_INFO[code][0]
        r["area_ha"] = round(r["area_px"] * irmad_result.px_area_m2 / 10000.0, 3)
        r["position"] = cc._position_word(xs.mean() / irmad_result.T.shape[1], ys.mean() / irmad_result.T.shape[0])
        
        shp = cc.shape_metrics(seg, cc.Params(target="structures" if hasattr(irmad_result, 'params') and irmad_result.params.target == "structures" else "all"))
        r["shape"] = shp["kind"]
        r["rectangularity"] = shp["rectangularity"]
        r["solidity"] = shp["solidity"]
        r["aspect"] = shp["aspect"]
        r["mean_width_px"] = shp["mean_width_px"]
        r["skeleton_px"] = shp["skeleton_px"]
    
    return regions


def make_overlay(after_rgb: np.ndarray, regions: List[Dict]) -> np.ndarray:
    """Create visualization overlay with numbered regions."""
    vis = after_rgb.copy()
    for i, r in enumerate(regions, start=1):
        color = (0, 200, 0)
        cnts, _ = cv2.findContours(r["seg"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)
        x, y, w, h = r["box"]
        cv2.putText(vis, f"#{i}", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def make_side_by_side(before_rgb: np.ndarray, after_rgb: np.ndarray, regions: List[Dict]) -> np.ndarray:
    """Create side-by-side before/after comparison with change overlay on after."""
    h, w = after_rgb.shape[:2]
    # Resize before to match after if needed
    if before_rgb.shape[:2] != (h, w):
        before_rgb = cv2.resize(before_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # After with overlay
    after_overlay = make_overlay(after_rgb, regions)
    
    # Labels
    before_label = np.zeros((40, w, 3), dtype=np.uint8)
    after_label = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(before_label, "BEFORE (T1)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(after_label, "AFTER (T2) + Changes", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    
    # Stack vertically: label + image
    top = np.hstack([before_label, after_label])
    before_stack = np.vstack([before_label, before_rgb])
    after_stack = np.vstack([after_label, after_overlay])
    side_by_side = np.hstack([before_stack, after_stack])
    
    return side_by_side


def score_card_image(fused: np.ndarray, per_signal: Dict[str, np.ndarray]) -> np.ndarray:
    """Horizontal strip of all score maps for visual comparison."""
    tiles = []
    for name, m in [("fused", fused)] + sorted(per_signal.items()):
        m8 = (normalize01(m) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(m8, cv2.COLORMAP_INFERNO)
        cv2.putText(heat, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(heat)
    return np.hstack(tiles)


def write_report(out_dir: str, regions: List[Dict], irmad_result: cc.Result, 
                 params: PipelineParams, paths: Tuple[str, str], per_signal: Dict):
    """Write text report with region details."""
    lines = [
        f"Unified Sentinel-2 Change Detection Report — {datetime.now().isoformat(timespec='seconds')}",
        f"before: {paths[0]}",
        f"after : {paths[1]}",
        f"signals: irmad, spectral, dino, structure weights={{'irmad': {params.irmad_weight}, 'spectral': {params.spectral_weight}, 'dino': {params.dino_weight}, 'structure': {params.structure_weight}}}",
        f"thresholds: fused={params.fused_threshold}",
        f"IR-MAD: sensitivity={params.sensitivity}, min_significance={params.min_significance}",
        f"DINOv2: model={params.dino_model}",
        f"structure_gate: {'enabled' if params.target_structures and not params.all_changes else 'disabled'}",
        "",
    ]
    for i, r in enumerate(regions, start=1):
        lines.append(
            f"#{i} box={r['box']} px={r['area_px']} ha={r['area_ha']} "
            f"score={r['score']:.2f} fired={'+'.join(sorted(r['sources'])) or 'none'} "
            f"class={r['class_name']} shape={r['shape']} "
            f"dNDVI={r['dNDVI']} dNDBI={r['dNDBI']} dMNDWI={r['dMNDWI']} "
            f"dBright={r['dBrightness']} SAM={r['spectral_angle_deg']:.1f}deg "
            f"sig={r['significance']} rect={r['rectangularity']} solid={r['solidity']}"
        )
    path = os.path.join(out_dir, "report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def save_debug_masks(out_dir: str, irmad_result: cc.Result, fused: np.ndarray, 
                     per_signal: Dict[str, np.ndarray], cand: np.ndarray, after_rgb: np.ndarray):
    """Save debug visualizations."""
    cv2.imwrite(os.path.join(out_dir, "T_heatmap.png"), cc.heatmap(irmad_result.T, irmad_result.valid))
    cv2.imwrite(os.path.join(out_dir, "fused_heatmap.png"), score_card_image(fused, per_signal))
    cv2.imwrite(os.path.join(out_dir, "candidate_mask.png"), cand * 255)
    cv2.imwrite(os.path.join(out_dir, "irmad_candidate.png"), (irmad_result.indices["irmad_candidate"] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "landcover_evidence.png"), (irmad_result.indices["landcover_evidence"] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "built_candidate.png"), (irmad_result.indices["built_candidate"] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "veg_loss.png"), (irmad_result.indices["veg_loss"] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "veg_gain.png"), (irmad_result.indices["veg_gain"] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "water_gain.png"), (irmad_result.indices["w_gain"] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "water_loss.png"), (irmad_result.indices["w_loss"] * 255).astype(np.uint8))
    
    for name, m in per_signal.items():
        if name not in ("irmad", "spectral", "structure"):
            cv2.imwrite(os.path.join(out_dir, f"signal_{name}.png"), (normalize01(m) * 255).astype(np.uint8))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("UNIFIED SENTINEL-2 CHANGE DETECTION PIPELINE")
    print("=" * 60)
    
    # Load scenes
    print("\n[1/6] Loading Sentinel-2 scenes...")
    aoi = tuple(args.aoi) if args.aoi else None
    before, after, meta = load_scenes_from_paths(args.before, args.after, aoi)
    print(f"  AOI: {before.refl.shape[1]}x{before.refl.shape[0]} px, res={meta['res']}m")
    
    # Save AOI visual RGB images
    cv2.imwrite(os.path.join(args.output_dir, "aoi_before_rgb.png"), before.rgb)
    cv2.imwrite(os.path.join(args.output_dir, "aoi_after_rgb.png"), after.rgb)
    print(f"  Saved AOI visuals: aoi_before_rgb.png, aoi_after_rgb.png")
    
    # Convert args to PipelineParams
    params = PipelineParams(
        sensitivity=args.sensitivity,
        min_area_px=args.min_area,
        min_significance=args.min_significance,
        ignore_water_variability=not args.no_water_veto,
        reject_slivers=not args.no_sliver_reject,
        dino_model=args.dino_model,
        dino_weight=args.dino_weight,
        irmad_weight=args.irmad_weight,
        spectral_weight=args.spectral_weight,
        structure_weight=args.structure_weight,
        fused_threshold=args.fused_threshold,
        min_region_area=args.min_region_area,
        max_regions=args.max_regions,
        min_region_score=args.min_region_score,
        target_structures=not args.no_structure_gate,
        all_changes=args.all_changes,
        output_dir=args.output_dir,
    )
    
    # Stage 1: IR-MAD + Spectral Evidence
    print("\n[2/6] Running IR-MAD statistical change + spectral evidence...")
    irmad_result = run_irmad_spectral(before, after, params, meta)
    print(f"  IR-MAD candidates: {int(irmad_result.indices['irmad_candidate'].sum())} px")
    print(f"  Land-cover evidence: {int(irmad_result.indices['landcover_evidence'].sum())} px")
    print(f"  Combined candidates: {int(irmad_result.indices['cand'].sum())} px")
    print(f"  Funnel: {irmad_result.funnel[-1]}")
    
    # Stage 2: DINOv2 Semantic Feature Change
    print("\n[3/6] Running DINOv2 semantic feature change...")
    dino_map, dino_meta = run_dinov2(before, after, params)
    if dino_map is not None:
        print(f"  DINOv2 map computed: {dino_map.shape}")
    else:
        print("  DINOv2 skipped (unavailable)")
    
    # Stage 3: Candidate Fusion
    print("\n[4/6] Fusing candidate signals...")
    fused, per_signal = fuse_signals(irmad_result, dino_map, params)
    thresholds = {
        "irmad": 0.35, "spectral": 0.35, "dino": 0.35, "structure": 0.25,
        "fused": params.fused_threshold
    }
    cand = candidate_mask(fused, per_signal, thresholds)
    print(f"  Candidate pixels: {int(cand.sum())}")
    
    # Stage 4: Morphology / Region Cleanup (handled in connected_regions)
    print("\n[5/6] Extracting regions (connected components)...")
    regions = connected_regions(cand, fused, min_area=params.min_region_area, max_regions=params.max_regions)
    print(f"  Raw regions: {len(regions)}")
    
    # Stage 5: Enrich + Merge
    print("\n[6/6] Enriching and merging regions...")
    regions = enrich_regions(regions, irmad_result, per_signal, before, after)
    regions = merge_regions(regions, params.max_regions)
    regions = [r for r in regions if r["score"] >= params.min_region_score * 0.5]
    print(f"  Final regions: {len(regions)}")
    
    # Save outputs
    print("\nSaving outputs...")
    save_debug_masks(args.output_dir, irmad_result, fused, per_signal, cand, after.rgb)
    
    overlay = make_overlay(after.rgb, regions)
    cv2.imwrite(os.path.join(args.output_dir, "overlay.png"), overlay)
    
    # Side-by-side before/after comparison
    side_by_side = make_side_by_side(before.rgb, after.rgb, regions)
    cv2.imwrite(os.path.join(args.output_dir, "side_by_side.png"), side_by_side)
    print(f"  Saved side_by_side.png")
    
    report_path = write_report(args.output_dir, regions, irmad_result, params, 
                               (args.before, args.after), per_signal)
    
    # Save fused map as GeoTIFF
    with rasterio.open(Path(args.after)) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    
    with rasterio.open(os.path.join(args.output_dir, "fused_score.tif"), 'w', **profile) as dst:
        dst.write(fused.astype(rasterio.float32), 1)
    
    print(f"\nDone! Outputs in: {args.output_dir}")
    print(f"Report: {report_path}")
    print("\n--- Regions Summary ---")
    print(open(report_path, encoding="utf-8").read())


if __name__ == "__main__":
    main()