"""
Detector wrappers for the UrbanSentinel fused change-detection pipeline.

Every detector here is optional and loads lazily: if a model / checkpoint /
dependency is missing, that signal is skipped with a warning and the fusion
simply runs on the remaining signals. All wrappers share the same contract:

    detector(before_bgr, after_bgr) -> (score_map, meta_dict)

score_map : float32 (H, W), any positive range (fusion normalizes it)
meta      : dict with extra info (checkpoint name, probabilities, ...)
"""
import gc
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Repo root (UrbanSentinel/ sits next to bit_cd/ at the repo top level)
REPO_ROOT = Path(__file__).resolve().parents[2]
BIT_CD_PATH = REPO_ROOT / "bit_cd"
SAM_CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "sam_vit_b_01ec64.pth",
    REPO_ROOT / "UrbanSentinel" / "sam_vit_b_01ec64.pth",
]

_DINO_CACHE = {}
_BIT_CACHE = {}
_SAM_CACHE = {}


def unload_detectors():
    """Drop cached models and release GPU memory (call between pipeline stages)."""
    _DINO_CACHE.clear()
    _BIT_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# DINOv2 — semantic change (zero-shot, robust to seasonal color shifts)
# --------------------------------------------------------------------------- #
def load_dino(model_name="dinov2_vits14", device=None):
    """(model, device, patch_size) or (None, None, None) on failure."""
    try:
        import torch
    except ImportError:
        logger.warning("torch not available — DINOv2 signal disabled.")
        return None, None, None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (model_name, device)
    if key in _DINO_CACHE:
        return _DINO_CACHE[key]
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load DINOv2 (%s) — signal disabled.", e)
        return None, None, None
    model.eval().to(device)
    result = (model, device, 14)
    _DINO_CACHE[key] = result
    return result


def _dino_features(img_rgb, model, device, patch_size):
    import torch

    h, w = img_rgb.shape[:2]
    new_h = max(patch_size, (h // patch_size) * patch_size)
    new_w = max(patch_size, (w // patch_size) * patch_size)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.inference_mode():
        tokens = model.forward_features(tensor)["x_norm_patchtokens"]
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
    return tokens[0].reshape(new_h // patch_size, new_w // patch_size, -1).cpu().numpy()


def dino_detector(before, after, model_name="dinov2_vits14"):
    """Cosine-distance map between DINOv2 patch embeddings (0 = identical)."""
    model, device, patch_size = load_dino(model_name)
    if model is None:
        return None, {}
    before_rgb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)
    f1 = _dino_features(before_rgb, model, device, patch_size)
    f2 = _dino_features(after_rgb, model, device, patch_size)
    if f1.shape[:2] != f2.shape[:2]:
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
    change = 1.0 - np.sum(f1 * f2, axis=-1)
    score = cv2.resize(change.astype(np.float32), (before.shape[1], before.shape[0]),
                       interpolation=cv2.INTER_CUBIC)
    return np.clip(score, 0.0, 2.0), {"model": model_name}


# --------------------------------------------------------------------------- #
# BIT (Bitemporal Image Transformer, LEVIR-CD pretrained) — building change
# --------------------------------------------------------------------------- #
def load_bit(repo_path=None, checkpoint_path=None, device=None):
    """Loads the vendored BIT_CD model; returns (model, device) or (None, None)."""
    try:
        import torch
    except ImportError:
        logger.warning("torch not available — BIT signal disabled.")
        return None, None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(repo_path), str(checkpoint_path), device)
    if key in _BIT_CACHE:
        return _BIT_CACHE[key]

    repo = Path(repo_path or BIT_CD_PATH)
    ckpt = Path(checkpoint_path or repo / "checkpoints" / "BIT_LEVIR" / "best_ckpt.pt")
    if not ckpt.is_file():
        logger.warning("BIT checkpoint not found at %s — signal disabled.", ckpt)
        return None, None
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from models.networks import BASE_Transformer
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not import BIT_CD model code (%s) — signal disabled.", e)
        return None, None

    model = BASE_Transformer(
        input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
        with_pos="learned", enc_depth=1, dec_depth=8, decoder_dim_head=8,
        backbone="resnet18", if_upsample_2x=True,
    )
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        state = state.get("model_G_state_dict") or state.get("state_dict") or state
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        model.load_state_dict(state, strict=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load BIT weights (%s) — signal disabled.", e)
        return None, None
    model.eval().to(device)
    result = (model, device)
    _BIT_CACHE[key] = result
    return result


def bit_detector(before, after, tile_size=256, repo_path=None, checkpoint_path=None):
    """Tiled BIT change probability (0-1). Trained on buildings (LEVIR-CD)."""
    import torch

    model, device = load_bit(repo_path, checkpoint_path)
    if model is None:
        return None, {}

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def to_tensor(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)).unsqueeze(0)

    h, w = before.shape[:2]
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    b = cv2.copyMakeBorder(before, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    a = cv2.copyMakeBorder(after, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    prob = np.zeros((b.shape[0], b.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for y in range(0, b.shape[0], tile_size):
            for x in range(0, b.shape[1], tile_size):
                t1 = to_tensor(b[y:y + tile_size, x:x + tile_size]).to(device)
                t2 = to_tensor(a[y:y + tile_size, x:x + tile_size]).to(device)
                logits = model(t1, t2)  # (1, 2, tile, tile)
                p = torch.softmax(logits, dim=1)[0, 1]
                prob[y:y + tile_size, x:x + tile_size] = p.cpu().numpy()
    return prob[:h, :w], {"checkpoint": os.path.basename(str(checkpoint_path or "BIT_LEVIR"))}


# --------------------------------------------------------------------------- #
# SAM — object-shaped region proposals ranked by any score map
# --------------------------------------------------------------------------- #
def load_sam(checkpoint_path=None, model_type="vit_b", device=None,
             points_per_side=32, pred_iou_thresh=0.75, min_mask_region_area=60):
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        logger.warning("segment-anything not installed — SAM signal disabled.")
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(checkpoint_path), model_type, device, points_per_side)
    if key in _SAM_CACHE:
        return _SAM_CACHE[key]

    ckpt = checkpoint_path
    if ckpt is None:
        for cand in SAM_CHECKPOINT_CANDIDATES:
            if cand.is_file():
                ckpt = cand
                break
    if ckpt is None or not Path(ckpt).is_file():
        logger.warning("SAM checkpoint not found (looked in %s) — signal disabled.",
                       [str(c) for c in SAM_CHECKPOINT_CANDIDATES])
        return None
    try:
        sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load SAM checkpoint (%s) — signal disabled.", e)
        return None
    sam.to(device)
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    _SAM_CACHE[key] = gen
    return gen


def sam_regions(after, score_map, mask_generator, min_score=0.2, min_area=100,
                max_regions=15, candidate_mask=None, min_overlap=0.10):
    """Score SAM segments by mean fused score; return [(box, score, seg_mask)].

    Falls back to CPU if the GPU runs out of memory mid-generation.
    """
    if mask_generator is None:
        return []
    candidate = candidate_mask.astype(bool) if candidate_mask is not None else None
    try:
        masks = mask_generator.generate(after)
    except Exception as e:  # noqa: BLE001
        if "out of memory" not in str(e).lower():
            raise
        logger.warning("SAM CUDA OOM — retrying on CPU...")
        import torch
        try:
            mask_generator.predictor.model.to("cpu")
        except AttributeError:
            pass
        torch.cuda.empty_cache()
        masks = mask_generator.generate(after)
    found = []
    for m in masks:
        seg = m["segmentation"]
        if int(seg.sum()) < min_area:
            continue
        if candidate is not None and float(candidate[seg].mean()) < min_overlap:
            continue
        score = float(score_map[seg].mean())
        if score < min_score:
            continue
        x, y, w, h = m["bbox"]
        found.append(((int(x), int(y), int(w), int(h)), score, seg))
    found.sort(key=lambda t: t[1], reverse=True)
    return found[:max_regions]


# --------------------------------------------------------------------------- #
# Structure (edge-density gain) — kills bare-soil false positives
# --------------------------------------------------------------------------- #
def _edge_density(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = (cv2.magnitude(sx, sy) > 60).astype(np.float32)
    return cv2.boxFilter(edges, -1, (15, 15), normalize=True)


def structure_detector(before, after):
    """Edge-density GAIN from before to after.

    New construction/roads add persistent straight structure (strong edges);
    bare-soil moisture, plowing and illumination drift change brightness and
    fine texture without adding persistent edges. Use as a veto-style
    supporting signal in bare-soil areas.
    """
    gain = _edge_density(after) - _edge_density(before)
    return np.clip(gain * 3.0, 0.0, 1.0), {
        "kind": "edge-density gain (new structures add edges)"}


# --------------------------------------------------------------------------- #
# Spectral indices (NDVI/NDBI deltas) — cheap supporting evidence
# --------------------------------------------------------------------------- #
def spectral_detector(before, after):
    """|dNDVI| + |dNDBI| magnitude from RGB(BGR) input only (green proxy for NIR).

    With true multi-band input pass bands explicitly instead. Returns a
    0-1 magnitude map plus per-band means so the VLM prompt can cite them.
    """
    def index(img, ch_a, ch_b):
        a = img[:, :, ch_a].astype(np.float32) / 255.0
        b = img[:, :, ch_b].astype(np.float32) / 255.0
        denom = np.maximum(a + b, 0.02)
        return np.clip((a - b) / denom, -1.0, 1.0)

    # BGR: "pseudo-NDVI" from blue/nir-like channels is not physical; instead
    # use a simple brightness/vegetation proxy: green vs blue+red.
    d_bright = np.abs(
        before.astype(np.float32).mean(axis=2) - after.astype(np.float32).mean(axis=2)
    ) / 255.0
    d_green = np.abs(index(before, 1, 2) - index(after, 1, 2))  # green vs red proxy
    score = np.clip(0.5 * d_bright + 0.5 * d_green, 0.0, 1.0)
    return score, {"kind": "rgb-proxy (pass real bands for NDVI/NDBI)"}
