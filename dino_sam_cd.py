"""
DINOv2 dense-feature change detection + SAM region proposals.

Adds two zero-shot, no-training-required signals to the change-detection
pipeline:

  1. dino_change_map()   - per-patch semantic dissimilarity between the
     before/after scene, upsampled to full resolution. More robust than
     raw SSIM/Lab-color to sensor calibration drift, seasonal color shift,
     and other "same content, different pixels" cases, because it compares
     learned semantic embeddings rather than raw pixel statistics.

  2. sam_change_regions() - runs Segment Anything on the 'after' scene and
     scores each resulting segment by its mean change score (DINO or the
     generic heatmap), giving region proposals that follow real object
     boundaries instead of rectangular bounding boxes from contour-finding.

Both are frozen, off-the-shelf models -- no fine-tuning, no labeled
Sentinel-2 data required. This sidesteps the BIT_CD resolution mismatch
(BIT_CD expects ~0.5m/px LEVIR-CD-style imagery; DINO/SAM don't care what
resolution you give them).

Optional deps (only imported lazily, at call time):
    pip install torch torchvision --break-system-packages
    pip install git+https://github.com/facebookresearch/segment-anything.git --break-system-packages

    SAM checkpoint (pick one -- vit_b is the lightest / most CPU-friendly):
        vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
        vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
        vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

    DINOv2 weights download automatically via torch.hub on first use
    (~90MB for dinov2_vits14).
"""
import sys

import cv2
import numpy as np

_DINO_CACHE = {}
_SAM_CACHE = {}


def load_dino(model_name="dinov2_vits14", device=None):
    """
    Load (and cache) a frozen DINOv2 backbone.
    Returns (model, device, patch_size), or (None, None, None) if torch /
    the model weights aren't available -- callers should check for None
    and skip the DINO signal rather than crash.
    """
    key = (model_name, device)
    if key in _DINO_CACHE:
        return _DINO_CACHE[key]
    try:
        import torch
    except ImportError:
        print("Warning: torch not available, skipping --use-dino", file=sys.stderr)
        return None, None, None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:  # noqa: BLE001 - report and degrade gracefully
        print(f"Warning: could not load DINOv2 ({e}), skipping --use-dino", file=sys.stderr)
        return None, None, None
    model.eval().to(device)
    patch_size = 14  # all dinov2_* variants use 14x14 patches
    result = (model, device, patch_size)
    _DINO_CACHE[key] = result
    return result


def _preprocess_for_dino(img_bgr, patch_size):
    """BGR uint8 -> normalized CHW tensor, resized to a multiple of patch_size."""
    import torch

    H, W = img_bgr.shape[:2]
    new_h = max(patch_size, (H // patch_size) * patch_size)
    new_w = max(patch_size, (W // patch_size) * patch_size)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, (new_h, new_w)


def dino_patch_features(img_bgr, dino_model, device, patch_size=14):
    """Return an (h_patches, w_patches, dim) grid of L2-normalized patch embeddings."""
    import torch

    tensor, (new_h, new_w) = _preprocess_for_dino(img_bgr, patch_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        out = dino_model.forward_features(tensor)
        tokens = out["x_norm_patchtokens"]  # (1, num_patches, dim)
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
    h_p, w_p = new_h // patch_size, new_w // patch_size
    feats = tokens[0].reshape(h_p, w_p, -1)
    return feats.cpu().numpy()


def dino_change_map(before, after, dino_model, device, patch_size=14):
    """
    Full-resolution (H x W, matches `before`) semantic-dissimilarity map in
    [0, 2] (0 = identical embedding, 2 = opposite). Upsampled from the patch
    grid with bicubic interpolation for a smoother map than the blocky
    nearest-neighbor artifacts a raw patch-grid resize would give.
    """
    f1 = dino_patch_features(before, dino_model, device, patch_size)
    f2 = dino_patch_features(after, dino_model, device, patch_size)
    # If before/after produced slightly different patch-grid shapes (can
    # happen if their pixel dimensions differ), resize f2's grid onto f1's.
    if f1.shape[:2] != f2.shape[:2]:
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]), interpolation=cv2.INTER_LINEAR)
    cos_sim = np.sum(f1 * f2, axis=-1)
    cos_dist = 1.0 - cos_sim  # (h_p, w_p), range 0..2

    H, W = before.shape[:2]
    full = cv2.resize(cos_dist.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(full, 0, 2)


def dino_heatmap_full(dino_map, vmax=1.0):
    """JET-colorized version of dino_change_map(), for the montage / debug output."""
    norm = np.clip(dino_map / vmax, 0, 1)
    heat = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)


def load_sam(checkpoint_path, model_type="vit_b", device=None,
             points_per_side=24, pred_iou_thresh=0.86, min_mask_region_area=100):
    """
    Load (and cache) a SAM automatic mask generator.
    Returns None if segment-anything / torch / the checkpoint aren't
    available -- callers should check for None and skip --use-sam.
    """
    key = (checkpoint_path, model_type, device)
    if key in _SAM_CACHE:
        return _SAM_CACHE[key]
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        print("Warning: segment-anything / torch not available, skipping --use-sam",
              file=sys.stderr)
        return None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not load SAM checkpoint ({e}), skipping --use-sam",
              file=sys.stderr)
        return None
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    _SAM_CACHE[key] = generator
    return generator


def sam_change_regions(after, change_map, mask_generator, min_change_score=0.2,
                        min_area=100, max_regions=10, candidate_mask=None,
                        min_change_overlap=0.10):
    """
    Run SAM on `after`, score each resulting segment by its mean value in
    `change_map` (pass dino_change_map()'s output, or any other full-res
    float change-intensity map -- e.g. the generic SSIM/color heatmap -- as
    a fallback when DINO isn't available), and return the top-scoring
    segments as (box_xywh, score, seg_mask_bool) tuples, sorted by score
    descending. `change_map` must be full-resolution, same H x W as `after`.
    If `candidate_mask` is given, a segment must overlap it by at least
    `min_change_overlap`; this keeps unrelated SAM objects out of the results.
    """
    if candidate_mask is not None and candidate_mask.shape != change_map.shape:
        raise ValueError("candidate_mask and change_map must have the same shape")
    candidate = candidate_mask.astype(bool) if candidate_mask is not None else None

    masks = mask_generator.generate(after)
    scored = []
    for m in masks:
        seg = m["segmentation"]
        if seg.sum() < min_area:
            continue
        if candidate is not None and float(candidate[seg].mean()) < min_change_overlap:
            continue
        score = float(change_map[seg].mean())
        if score < min_change_score:
            continue
        x, y, w, h = m["bbox"]
        scored.append(((int(x), int(y), int(w), int(h)), score, seg))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:max_regions]


def sam_mask_overlay(after_crop, seg_crop, color=(0, 255, 255), alpha=0.5):
    """
    after_crop with the SAM segment fill + outline drawn on it -- used as a
    change-graph quadrant so you can see exactly what SAM segmented, not
    just its bounding box.
    """
    overlay = after_crop.copy()
    colored = np.zeros_like(after_crop)
    colored[seg_crop] = color
    blended = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)
    contours, _ = cv2.findContours(seg_crop.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, 2)
    return blended
