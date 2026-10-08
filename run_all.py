#!/usr/bin/env python3
"""
Complete pipeline + interactive VLM Q&A in one script.
Run: python run_all.py [--no-vlm] [--output-dir DIR]
"""
import argparse
import cv2
import numpy as np
import os

import change_core as cc
from pipeline_unified import (
    load_scenes_from_paths, run_irmad_spectral, PipelineParams,
    connected_regions, merge_regions, enrich_regions,
    fuse_signals, candidate_mask, run_dinov2,
    normalize01, make_overlay_with_vlm, make_side_by_side,
    score_card_image, write_report, save_debug_masks
)
from vlm_qa import build_montage, evidence_table, stream_answer, audit


def run_pipeline(args):
    """Run the full detection pipeline."""
    print("=" * 60)
    print("UNIFIED SENTINEL-2 CHANGE DETECTION PIPELINE")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load scenes
    print("\n[1/7] Loading Sentinel-2 scenes...")
    aoi = tuple(args.aoi) if args.aoi else None
    before, after, meta = load_scenes_from_paths(args.before, args.after, aoi)
    print(f"  AOI: {before.refl.shape[1]}x{before.refl.shape[0]} px, res={meta['res']}m")
    
    cv2.imwrite(os.path.join(args.output_dir, "aoi_before_rgb.png"), before.rgb)
    cv2.imwrite(os.path.join(args.output_dir, "aoi_after_rgb.png"), after.rgb)
    
    # Pipeline params
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
        building_max_px=args.building_max_px,
        output_dir=args.output_dir,
    )
    
    # Stage 1: IR-MAD + Spectral
    print("\n[2/7] Running IR-MAD + spectral evidence...")
    irmad_result = run_irmad_spectral(before, after, params, meta)
    print(f"  IR-MAD candidates: {int(irmad_result.indices['irmad_candidate'].sum())} px")
    print(f"  Land-cover evidence: {int(irmad_result.indices['landcover_evidence'].sum())} px")
    print(f"  Combined candidates: {int(irmad_result.indices['cand'].sum())} px")
    
    # Stage 2: DINOv2
    print("\n[3/7] Running DINOv2 semantic feature change...")
    dino_map, dino_meta = run_dinov2(before, after, params)
    if dino_map is not None:
        print(f"  DINOv2 map: {dino_map.shape}")
    else:
        print("  DINOv2 skipped")
    
    # Stage 3: Fusion
    print("\n[4/7] Fusing candidate signals...")
    fused, per_signal = fuse_signals(irmad_result, dino_map, params)
    thresholds = {
        "irmad": 0.35, "spectral": 0.35, "dino": 0.35,
        "structure": 0.25, "fused": params.fused_threshold
    }
    cand = candidate_mask(fused, per_signal, thresholds)
    print(f"  Candidate pixels: {int(cand.sum())}")
    
    # Stage 4: Regions
    print("\n[5/7] Extracting regions...")
    regions = connected_regions(cand, fused, min_area=params.min_region_area, max_regions=params.max_regions)
    print(f"  Raw regions: {len(regions)}")
    
    # Stage 5: Enrich + Merge
    print("\n[6/7] Enriching and merging regions...")
    regions = enrich_regions(regions, irmad_result, per_signal, before, after)
    regions = merge_regions(regions, params.max_regions)
    regions = [r for r in regions if r["score"] >= params.min_region_score * 0.5]
    for i, r in enumerate(regions, start=1):
        r["id"] = i
    print(f"  Final regions: {len(regions)}")
    
    # Stage 6: VLM Audit (optional)
    vlm_answers = {}
    if regions and not args.no_vlm:
        print("\n[7/7] Running VLM audit...")
        try:
            before_rgb = cv2.cvtColor(before.rgb, cv2.COLOR_BGR2RGB)
            after_rgb = cv2.cvtColor(after.rgb, cv2.COLOR_BGR2RGB)
            
            labels = np.zeros_like(before_rgb[:,:,0], dtype=np.int32)
            for i, r in enumerate(regions, start=1):
                labels[r["seg"]] = i
            
            batch_size = args.vlm_max_regions if args.vlm_max_regions > 0 else len(regions)
            for batch_start in range(0, len(regions), batch_size):
                batch_regions = regions[batch_start:batch_start + batch_size]
                print(f"  VLM batch {batch_start//batch_size + 1}: regions {batch_start+1}-{min(batch_start+batch_size, len(regions))}")
                
                montage = build_montage(before_rgb, after_rgb, labels, batch_regions, max_regions=len(batch_regions))
                table = evidence_table(batch_regions, 
                    cc.DATE_RE.search(args.before).group(1) if cc.DATE_RE.search(args.before) else "T1",
                    cc.DATE_RE.search(args.after).group(1) if cc.DATE_RE.search(args.after) else "T2",
                    f"AOI {before.refl.shape[1]}x{before.refl.shape[0]}px",
                    max_regions=len(batch_regions))
                
                batch_ids = [r["id"] for r in batch_regions]
                batch_answers = audit(args.ollama_url, args.vlm_model, montage, table,
                                       batch_ids, timeout=args.vlm_timeout, max_tokens=args.vlm_max_tokens)
                vlm_answers.update(batch_answers)
            
            for r in regions:
                if r["id"] in vlm_answers:
                    r["vlm"] = vlm_answers[r["id"]]
                    r["vlm_confirmed"] = vlm_answers[r["id"]]["real_change"]
                    r["vlm_category"] = vlm_answers[r["id"]]["category"]
                    r["vlm_reason"] = vlm_answers[r["id"]]["reason"]
            
            n_confirmed = sum(1 for r in regions if r.get("vlm_confirmed", False))
            print(f"  VLM confirmed: {n_confirmed}/{len(regions)} regions")
            
        except Exception as e:
            print(f"  VLM audit failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[7/7] VLM audit skipped")
    
    # Save outputs
    print("\nSaving outputs...")
    # Skip debug masks - only save essential outputs
    
    # Overlay: just region numbers on after image (RGB -> BGR for saving)
    overlay = make_overlay_with_vlm(after.rgb, regions, vlm_answers)
    cv2.imwrite(os.path.join(args.output_dir, "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Side-by-side: before | after with overlay (RGB -> BGR for saving)
    side_by_side = make_side_by_side(before.rgb, after.rgb, regions)
    cv2.imwrite(os.path.join(args.output_dir, "side_by_side.png"), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    
    report_path = write_report(args.output_dir, regions, irmad_result, params,
                               (args.before, args.after), per_signal, vlm_answers)
    
    # Save fused map as GeoTIFF
    import rasterio
    from pathlib import Path
    with rasterio.open(Path(args.after)) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(os.path.join(args.output_dir, "fused_score.tif"), 'w', **profile) as dst:
        dst.write(fused.astype(rasterio.float32), 1)
    
    print(f"\nDone! Outputs in: {args.output_dir}")
    print(f"Report: {report_path}")
    print("\n--- Regions Summary ---")
    print(open(report_path, encoding="utf-8").read())
    
    return before, after, meta, regions, fused, per_signal, vlm_answers, irmad_result


def interactive_qa(before, after, regions, fused, per_signal, vlm_answers, args):
    """Interactive VLM Q&A session."""
    if args.no_vlm:
        print("\nVLM not available for Q&A (--no-vlm was used)")
        return
    
    # Prepare montage and table
    before_rgb = cv2.cvtColor(before.rgb, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after.rgb, cv2.COLOR_BGR2RGB)
    labels = np.zeros_like(before_rgb[:,:,0], dtype=np.int32)
    for i, r in enumerate(regions, start=1):
        labels[r["seg"]] = i
    
    montage = build_montage(before_rgb, after_rgb, labels, regions, max_regions=len(regions))
    table = evidence_table(regions,
        cc.DATE_RE.search(args.before).group(1) if cc.DATE_RE.search(args.before) else "T1",
        cc.DATE_RE.search(args.after).group(1) if cc.DATE_RE.search(args.after) else "T2",
        f"AOI {before.refl.shape[1]}x{before.refl.shape[0]}px",
        max_regions=len(regions))
    
    # Save montage
    cv2.imwrite(os.path.join(args.output_dir, "vlm_montage.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"\nSaved montage to {args.output_dir}/vlm_montage.png")
    
    print("\n" + "=" * 60)
    print("INTERACTIVE VLM Q&A")
    print("=" * 60)
    print("Ask questions about the detected regions.")
    print("The VLM sees: BEFORE | AFTER | CHANGE OUTLINE for each region")
    print("Plus the evidence table with spectral metrics.")
    print("\nExamples:")
    print("  Which regions show new buildings?")
    print("  Show me all construction sites")
    print("  Is region 5 a real building or false positive?")
    print("  Describe region 1 and 2 in detail")
    print("  Are there any new roads?")
    print("\nType 'quit' to exit.\n")
    
    history = []
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if not question:
            continue
        
        print("VLM: ", end='', flush=True)
        try:
            for chunk in stream_answer(
                args.ollama_url, args.vlm_model,
                question, history, montage, table,
                timeout=args.vlm_timeout, max_tokens=args.vlm_max_tokens
            ):
                print(chunk, end='', flush=True)
            print()
            history.append({"role": "user", "content": question})
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\nGoodbye!")


def parse_args():
    ap = argparse.ArgumentParser(description="Sentinel-2 Change Detection + VLM Q&A")
    ap.add_argument("--before", default="data/S2A_39SXV_20200128_1_L2A_visual.tif")
    ap.add_argument("--after", default="data/S2B_39SXV_20260128_0_L2A_visual.tif")
    ap.add_argument("--aoi", nargs=4, type=int, default=[1487,10075, 225, 225],
                    metavar=("X", "Y", "W", "H"))
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
    ap.add_argument("--all-changes", action="store_true")
    ap.add_argument("--building-max-px", type=int, default=500)
    
    # VLM options
    ap.add_argument("--vlm-model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--vlm-timeout", type=int, default=900)
    ap.add_argument("--no-vlm", action="store_true", help="Skip VLM audit stage")
    ap.add_argument("--vlm-max-regions", type=int, default=6, help="Max regions per VLM batch (0 = all)")
    ap.add_argument("--vlm-max-tokens", type=int, default=3000)
    ap.add_argument("--no-qa", action="store_true", help="Skip interactive Q&A after pipeline")
    
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Run pipeline
    before, after, meta, regions, fused, per_signal, vlm_answers, irmad_result = run_pipeline(args)
    
    # Interactive Q&A
    if not args.no_qa:
        interactive_qa(before, after, regions, fused, per_signal, vlm_answers, args)


if __name__ == "__main__":
    main()