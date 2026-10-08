#!/usr/bin/env python3
"""
Interactive VLM Q&A for change detection regions.
Run: python ask_vlm.py
"""
import cv2
import numpy as np

import change_core as cc
from pipeline_unified import (
    load_scenes_from_paths, run_irmad_spectral, PipelineParams,
    connected_regions, merge_regions, enrich_regions,
    fuse_signals, candidate_mask, run_dinov2
)
from vlm_qa import build_montage, evidence_table, stream_answer


def main():
    # Load data and run detection (same as pipeline)
    print("Loading data and running detection...")
    before, after, meta = load_scenes_from_paths(
        'data/S2A_39SXV_20200128_1_L2A_visual.tif',
        'data/S2B_39SXV_20260128_0_L2A_visual.tif',
        (1487, 10075, 225, 225)
    )
    
    params = PipelineParams(building_max_px=500)
    irmad_result = run_irmad_spectral(before, after, params, meta)
    dino_map, _ = run_dinov2(before, after, params)
    fused, per_signal = fuse_signals(irmad_result, dino_map, params)
    
    thresholds = {
        "irmad": 0.35, "spectral": 0.35, "dino": 0.35,
        "structure": 0.25, "fused": params.fused_threshold
    }
    cand = candidate_mask(fused, per_signal, thresholds)
    
    regions = connected_regions(cand, fused, min_area=params.min_region_area, max_regions=params.max_regions)
    regions = enrich_regions(regions, irmad_result, per_signal, before, after)
    regions = merge_regions(regions, params.max_regions)
    
    # Assign IDs
    for i, r in enumerate(regions, start=1):
        r["id"] = i
    
    print(f"Found {len(regions)} regions")
    
    # Prepare montage and table for VLM
    before_rgb = cv2.cvtColor(before.rgb, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after.rgb, cv2.COLOR_BGR2RGB)
    labels = np.zeros_like(before_rgb[:,:,0], dtype=np.int32)
    for i, r in enumerate(regions, start=1):
        labels[r["seg"]] = i
    
    montage = build_montage(before_rgb, after_rgb, labels, regions, max_regions=len(regions))
    table = evidence_table(regions, '20200128', '20260128', f'AOI {before.refl.shape[1]}x{before.refl.shape[0]}px', max_regions=len(regions))
    
    # Save montage for reference
    cv2.imwrite("vlm_montage.png", cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print("Saved vlm_montage.png")
    print("\n" + table)
    print("\n--- Ready for questions ---")
    print("Type your question (or 'quit' to exit)")
    print("Examples:")
    print("  - Which regions show new buildings?")
    print("  - Show me all construction sites")
    print("  - Is region 5 a real building?")
    print("  - Describe region 1 and 2")
    
    history = []
    while True:
        question = input("\n> ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if not question:
            continue
        
        print("\nVLM: ", end='', flush=True)
        try:
            for chunk in stream_answer(
                'http://localhost:11434', 'qwen3-vl:4b-instruct',
                question, history, montage, table, timeout=120, max_tokens=1000
            ):
                print(chunk, end='', flush=True)
            print()
            # Add to history
            history.append({"role": "user", "content": question})
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()