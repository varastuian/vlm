"""
Fusion of multiple change-detection signals into one score map and one
consensus region set.

Signals come from UrbanSentinel.models (DINOv2, BIT, spectral proxy) and are
combined in two stages:

  1. Pixel level  — each score map is percentile-normalized to 0-1, then a
                    weighted sum produces the fused score. A pixel is a
                    "candidate" if either the deep signals or the fast
                    spectral signal fires (union, not intersection, so a
                    single weak model cannot hide a real change).

  2. Region level — SAM segments the after-image; each segment inherits the
                    mean fused score. Independently, connected components of
                    the candidate mask give detector-free regions. Regions
                    supported by >= min_detectors signals are consensus
                    regions; the rest are marked "single-source" so the VLM
                    can treat them more skeptically.
"""
import cv2
import numpy as np

try:
    from . import bit_detector, dino_detector, spectral_detector, structure_detector
except ImportError:  # run as a plain script / from inside the package dir
    from detectors import (bit_detector, dino_detector, spectral_detector,
                           structure_detector)


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #
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


def match_size(score, shape):
    if score.shape != shape:
        score = cv2.resize(score, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return score


# --------------------------------------------------------------------------- #
# Stage 1: pixel-level fusion
# --------------------------------------------------------------------------- #
def fuse_score_maps(score_maps, weights, shape):
    """Weighted sum of percentile-normalized score maps. Returns (fused, per_signal)."""
    fused = np.zeros(shape, dtype=np.float32)
    total_w = 0.0
    per_signal = {}
    for name, raw in score_maps.items():
        if raw is None:
            continue
        n = normalize01(match_size(raw, shape))
        w = float(weights.get(name, 0.0))
        fused += w * n
        total_w += w
        per_signal[name] = n
    if total_w > 0:
        fused /= total_w
    return fused, per_signal


# --------------------------------------------------------------------------- #
# Stage 2: region proposals + consensus
# --------------------------------------------------------------------------- #
def candidate_mask(fused, per_signal, thresholds, min_frac=0.0):
    """Union rule: a pixel is candidate if any individual signal crosses its
    threshold OR the fused score crosses the fused threshold."""
    h, w = fused.shape
    votes = np.zeros((h, w), dtype=np.uint8)
    for name, thr in thresholds.items():
        m = per_signal.get(name)
        if m is not None:
            votes |= (m >= thr).astype(np.uint8)
    fused_thr = thresholds.get("fused", 0.45)
    votes |= (fused >= fused_thr).astype(np.uint8)
    votes = cv2.morphologyEx(votes * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (votes > 0).astype(np.uint8)


def connected_regions(mask, score_map, min_area=100, max_regions=30):
    """Fallback region proposals from connected components (no SAM needed)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        seg = (labels == i)
        score = float(score_map[seg].mean())
        regions.append({"box": (int(x), int(y), int(w), int(h)),
                        "score": score,
                        "seg": seg,
                        "sources": set()})
    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions[:max_regions]


def sam_to_regions(found, shape):
    """Convert models.sam_regions output to the common region dict format."""
    regions = []
    for box, score, seg in found:
        regions.append({"box": box, "score": float(score),
                        "seg": np.asarray(seg).astype(bool),
                        "sources": set()})
    return regions


def merge_regions(regions, max_regions, overlap=0.4):
    """De-duplicate overlapping regions (SAM + CC often find the same object)."""
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


def is_bare_soil(before, after, seg, ndvi_thresh=0.30):
    """Fraction of the region that looks like bare/low-vegetation land.

    Uses the green-vs-red vegetation proxy: bare soil keeps (green - red) low
    on BOTH dates. Returns the max bare fraction across the two dates plus a
    context dict for the VLM prompt.
    """
    def bare_frac(img, mask):
        g = img[:, :, 1].astype(np.float32)[mask] / 255.0
        r = img[:, :, 2].astype(np.float32)[mask] / 255.0
        # low green relative to red => sparse vegetation / bare ground
        return float(((g - r) < 0.02).astype(np.float32).mean())
    seg2d = seg if seg.ndim == 2 else seg[:, :, 0]
    b = bare_frac(before, seg2d)
    a = bare_frac(after, seg2d)
    # Bare-soil context requires BOTH dates to look bare: if vegetation
    # appeared or disappeared, that itself is a real change, not context.
    frac = min(b, a)
    return frac, {"is_bare_soil": frac, "bare_frac_before": round(b, 3),
                  "bare_frac_after": round(a, 3)}


def signal_support(region, per_signal, thresholds):
    """Which signals fire inside this region (mean >= threshold)?"""
    seg = region["seg"]
    if not seg.any():
        return set()
    fired = set()
    for name, m in per_signal.items():
        if name in thresholds and float(m[seg].mean()) >= thresholds[name]:
            fired.add(name)
    return fired


# --------------------------------------------------------------------------- #
# Public entry: run all detectors + fuse
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {"dino": 0.35, "bit": 0.30, "spectral": 0.15, "structure": 0.20}
DEFAULT_THRESHOLDS = {"dino": 0.35, "bit": 0.30, "spectral": 0.35, "structure": 0.25,
                      "fused": 0.45}


def run_detectors(before, after, enabled=("dino", "bit", "spectral", "structure"), **kwargs):
    """Run each enabled detector. Returns ({name: score_map}, {name: meta})."""
    score_maps, meta = {}, {}
    if "dino" in enabled:
        s, m = dino_detector(before, after, model_name=kwargs.get("dino_model", "dinov2_vits14"))
        if s is not None:
            score_maps["dino"], meta["dino"] = s, m
    if "bit" in enabled:
        s, m = bit_detector(before, after,
                            repo_path=kwargs.get("bit_repo"),
                            checkpoint_path=kwargs.get("bit_checkpoint"))
        if s is not None:
            score_maps["bit"], meta["bit"] = s, m
    if "spectral" in enabled:
        s, m = spectral_detector(before, after)
        score_maps["spectral"], meta["spectral"] = s, m  # always available
    if "structure" in enabled:
        s, m = structure_detector(before, after)
        score_maps["structure"], meta["structure"] = s, m  # always available
    return score_maps, meta


def fuse(before, after, enabled=("dino", "bit", "spectral"),
         weights=None, thresholds=None, min_area=100, max_regions=30,
         sam_generator=None, sam_min_score=0.2, sam_min_overlap=0.10, **kwargs):
    """Full fusion. Returns dict with fused map, mask, regions and metadata."""
    weights = weights or DEFAULT_WEIGHTS
    thresholds = thresholds or DEFAULT_THRESHOLDS

    score_maps, det_meta = run_detectors(before, after, enabled, **kwargs)
    if not score_maps:
        raise RuntimeError("No detector produced a score map — cannot fuse.")

    fused, per_signal = fuse_score_maps(score_maps, weights, before.shape[:2])
    cand = candidate_mask(fused, per_signal, thresholds)

    # Region proposals: SAM when available, connected components always.
    regions = []
    if sam_generator is not None:
        found = __import__("UrbanSentinel.models", fromlist=["sam_regions"]).sam_regions(
            after, fused, sam_generator,
            min_score=sam_min_score, min_area=min_area,
            candidate_mask=cand, min_overlap=sam_min_overlap)
        regions += sam_to_regions(found, before.shape[:2])
    regions += connected_regions(cand, fused, min_area=min_area)
    for r in regions:
        r["sources"] = signal_support(r, per_signal, thresholds)

    regions = merge_regions(regions, max_regions)
    return {
        "fused": fused,
        "candidate_mask": cand,
        "regions": regions,
        "per_signal": per_signal,
        "score_maps": score_maps,
        "det_meta": det_meta,
        "weights": weights,
        "thresholds": thresholds,
    }
