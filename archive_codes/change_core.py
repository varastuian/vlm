"""
change_core.py - false-positive-aware bi-temporal change detection for
Sentinel-2 L2A band files (green / red / nir / swir16 / visual GeoTIFFs).

Pipeline (who is responsible for what)
--------------------------------------
  Sentinel-2 T1/T2 (STAC assets, already on the same grid)
      -> alignment VALIDATION (no co-registration is performed here)
      -> IR-MAD ........................ STATISTICAL change evidence (T, p-values)
      -> irmad_candidate ............... broad, high-recall statistical candidates
      -> radiometric normalisation on IR-MAD-stable pixels
      -> spectral land-cover transition evidence (NDVI / NDBI / MNDWI / brightness / SAM)
      -> cand = irmad_candidate AND landcover_evidence
      -> false-positive vetoes (brightness-only, water variability)
      -> morphology / minimum-area / sliver filtering
      -> optional semantic filter (DINOv2 map, interface unchanged)
      -> region extraction (+ per-region significance gate)
      -> optional structure gate (shape + spectral, buildings/roads)

Downstream stages that are NOT in this module:
      semantic verification ........... DINO feature change between T1 and T2
      object classification ........... Qwen2.5-VL (vlm_qa.py)
      precise footprints .............. SAM2

This module only produces "high-recall but selective" candidates.  Nothing here is a
building classifier.

Why IR-MAD alone is not the definition of change
------------------------------------------------
IR-MAD flags any pixel whose joint spectral relationship between T1 and T2 is unusual.
On arid scenes that includes ploughing, tillage, soil moisture and crop residue on bare
soil - statistically significant, but not a land-cover transition.  So IR-MAD only
nominates candidates; a candidate must ALSO show plausible transition evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from scipy import ndimage as ndi
from scipy.linalg import eigh
from scipy.stats import chi2
from skimage.filters import apply_hysteresis_threshold, threshold_otsu
from skimage.morphology import skeletonize

DATE_RE = re.compile(r"_(\d{8})_")
# Band order is load-bearing: index 0..3 = green, red, nir, swir16.
#   NDVI  = nd(nir, red)      NDBI = nd(swir, nir)      MNDWI = nd(green, swir)
BANDS = ("green", "red", "nir", "swir16")
NB = len(BANDS)

# (high alpha for seeds, low alpha for growth)
PRESETS = {
    "strict": (1e-7, 1e-5),
    "balanced": (1e-5, 1e-3),
    "sensitive": (1e-4, 1e-2),
}

CLASS_INFO = {
    1: ("bright non-vegetated surface appeared (building / pavement / bare)", (230, 30, 30)),
    2: ("vegetation loss / ground exposed", (255, 150, 0)),
    3: ("vegetation gain", (40, 200, 60)),
    4: ("water appeared", (40, 110, 255)),
    5: ("water disappeared", (0, 220, 220)),
    6: ("dark surface appeared (asphalt / burn scar / wet soil)", (150, 60, 200)),
    7: ("other material change", (200, 200, 200)),
    8: ("construction site / active earthworks", (255, 100, 100)),
}

# --- land-cover transition thresholds (fixed on purpose; do not tune per screenshot) --- #
VEG_LOSS_NDVI_BEFORE_MIN = 0.20
VEG_LOSS_NDVI_AFTER_MAX = 0.15
VEG_LOSS_DNDVI_MAX = -0.12
VEG_GAIN_NDVI_AFTER_MIN = 0.25
VEG_GAIN_DNDVI_MIN = 0.12
WATER_MNDWI_MIN = 0.10
WATER_DMNDWI = 0.20

# Built surface (finished buildings/pavement) - strict thresholds
BUILT_NDVI_AFTER_MAX = 0.25
BUILT_DNDBI_MIN = 0.08
BUILT_DBRIGHT_MIN = 0.02

# Construction site / active earthworks - relaxed spectral, distinct texture
# Key: soil disturbance (dBright can be negative initially), NDBI rise, low NDVI, high spectral angle (shape change)
CONSTR_NDVI_AFTER_MAX = 0.40        # allow partial vegetation during construction
CONSTR_DNDBI_MIN = 0.04             # NDBI rises as soil exposed
CONSTR_DBRIGHT_MIN = -0.03          # can be darker (excavation) or brighter (materials)
CONSTR_SAM_MIN = 0.08               # spectral shape MUST change (disturbed soil != veg)
CONSTR_TEXTURE_MIN = 0.15           # edge density increase (structure detector)


# --------------------------------------------------------------------------- #
# Scene discovery
# --------------------------------------------------------------------------- #
def find_scenes(data_dir):
    """{'YYYYMMDD': {band: path}} for every date that has all bands + visual."""
    data_dir = Path(data_dir)
    scenes = {}
    for vis in data_dir.glob("*_visual.tif"):
        m = DATE_RE.search(vis.name)
        if not m:
            continue
        prefix = vis.name[: -len("_visual.tif")]
        files = {b: data_dir / f"{prefix}_{b}.tif" for b in BANDS}
        files["visual"] = vis
        if all(p.exists() for p in files.values()):
            scenes[m.group(1)] = files
    return dict(sorted(scenes.items()))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    refl: np.ndarray          # (H, W, 4) float32 reflectance 0-1
    rgb: np.ndarray           # (H, W, 3) uint8
    nodata: np.ndarray        # (H, W) bool
    bad: np.ndarray           # (H, W) bool  cloud | snow (dilated)
    cloud: np.ndarray
    snow: np.ndarray
    calib: dict = field(default_factory=dict)
    # ---- grid metadata of the SOURCE raster (used by validate_alignment) ---- #
    crs: object = None            # source CRS
    transform: object = None      # affine of the AOI window in the source raster
    bounds: tuple = None          # AOI bounds in the source CRS
    native_shape: tuple = None    # AOI window size in SOURCE pixels (rows, cols)
    src_transform: object = None  # affine of the whole source raster (grid phase check)


def _read_reflectance(path, bounds, shape):
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        raw = src.read(1, window=win, out_shape=shape, resampling=Resampling.bilinear,
                       boundless=True, fill_value=0).astype(np.float32)
        scale = float(src.scales[0]) if src.scales else 1.0
        offset = float(src.offsets[0]) if src.offsets else 0.0
    nodata = raw <= 0
    if scale != 1.0 or offset != 0.0:
        refl, how = raw * scale + offset, f"tags scale={scale} offset={offset}"
    else:
        finite = raw[np.isfinite(raw) & ~nodata]
        if finite.size and np.percentile(finite, 99) > 2.0:
            refl, how = raw / 10000.0, "DN/10000 (no offset tag)"
        else:
            refl, how = raw, "already reflectance"
    return np.clip(refl, 0.0, 1.0), nodata, how


def _read_rgb(path, bounds, shape):
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        n = min(3, src.count)
        data = src.read(list(range(1, n + 1)), window=win, out_shape=(n, *shape),
                        resampling=Resampling.bilinear, boundless=True, fill_value=0)
    if n == 1:
        data = np.repeat(data, 3, axis=0)
    img = np.moveaxis(data[:3], 0, -1).astype(np.float32)
    if img.max() > 255 or img.max() <= 1.0:
        hi = np.percentile(img, 99) if img.max() > 0 else 1.0
        img = img / max(hi, 1e-6) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def nd(a, b):
    """Normalised difference (a-b)/(a+b), safe."""
    s = a + b
    return np.divide(a - b, s, out=np.zeros_like(a, dtype=np.float32),
                     where=np.abs(s) > 1e-6).clip(-1, 1)


def cloud_snow_masks(refl, nodata=None, min_cloud_px=50):
    """Heuristic masks. Dark-object subtraction first, so a processing-baseline offset
    (+0.1 on newer L2A products) cannot make ordinary bright surfaces look like cloud.
    Cloud blobs smaller than `min_cloud_px` are ignored: bright new roofs are exactly
    what we want to detect, clouds are large."""
    ok = ~nodata if nodata is not None else np.ones(refl.shape[:2], bool)
    dos = np.array([np.percentile(refl[..., i][ok], 0.5) if ok.any() else 0.0 for i in range(NB)])
    r_ = np.clip(refl - dos, 0.0, 1.0)
    g, n, s = r_[..., 0], r_[..., 2], r_[..., 3]
    ndsi = nd(g, s)
    snow = (ndsi > 0.40) & (g > 0.25) & (n > 0.11)
    cloud = (g > 0.30) & (n > 0.28) & (s > 0.22) & ~snow
    lab, k = ndi.label(cloud, structure=np.ones((3, 3)))
    if k:
        sizes = ndi.sum(cloud, lab, index=np.arange(1, k + 1))
        cloud = np.isin(lab, np.flatnonzero(sizes >= min_cloud_px) + 1)
    return cloud, snow


def load_aoi_pair(scene_before, scene_after, x, y, w, h):
    """Read one AOI (native px window on the AFTER visual grid) for both dates.
    Each Scene records the grid of its own SOURCE raster so detect_changes() can verify
    that the two dates really share a grid instead of assuming it."""
    with rasterio.open(scene_after["visual"]) as ref:
        bounds = ref.window_bounds(Window(x, y, w, h))
        crs, res = ref.crs, (abs(ref.res[0]), abs(ref.res[1]))
    shape = (h, w)
    out = []
    for files in (scene_before, scene_after):
        with rasterio.open(files["visual"]) as s:
            b = bounds if s.crs == crs else transform_bounds(crs, s.crs, *bounds)
            win = from_bounds(*b, transform=s.transform)
            grid = dict(crs=s.crs, transform=window_transform(win, s.transform), bounds=tuple(b),
                        native_shape=(int(round(win.height)), int(round(win.width))),
                        src_transform=s.transform)
        bands, nod, hows = [], np.zeros(shape, bool), {}
        for name in BANDS:
            refl, nd_, how = _read_reflectance(files[name], b, shape)
            bands.append(refl)
            nod |= nd_
            hows[name] = how
        refl = np.stack(bands, axis=-1)
        cloud, snow = cloud_snow_masks(refl, nod)
        bad = ndi.binary_dilation(cloud | snow, structure=np.ones((3, 3)), iterations=4)
        out.append(Scene(refl=refl, rgb=_read_rgb(files["visual"], b, shape), nodata=nod,
                         bad=bad, cloud=cloud, snow=snow, calib=hows, **grid))
    meta = {"bounds": bounds, "crs": crs, "res": res}
    return out[0], out[1], meta


# --------------------------------------------------------------------------- #
# Alignment validation (replaces co-registration)
# --------------------------------------------------------------------------- #
def validate_alignment(before, after):
    """Both dates must sit on the same grid. Nothing is resampled or shifted here.
    Raises ValueError listing every mismatch."""
    for name in ("crs", "transform", "bounds", "native_shape", "src_transform"):
        if getattr(before, name, None) is None or getattr(after, name, None) is None:
            raise ValueError(
                f"Cannot verify alignment: Scene.{name} is missing. Build scenes with "
                f"load_aoi_pair() (or fill crs/transform/bounds/native_shape/src_transform yourself).")
    problems = []
    if before.crs != after.crs:
        problems.append(f"CRS differs: before={before.crs} after={after.crs}")
    if before.refl.shape != after.refl.shape:
        problems.append(f"array shape differs: before={before.refl.shape} after={after.refl.shape}")
    if tuple(before.native_shape) != tuple(after.native_shape):
        problems.append(f"source window size differs: before={before.native_shape} "
                        f"after={after.native_shape} (different native resolution?)")
    px = max(abs(after.transform.a), abs(after.transform.e))
    tol = 1e-3 * px                                   # 1/1000 of a pixel
    ta, tb = np.array(tuple(before.transform)[:6]), np.array(tuple(after.transform)[:6])
    if not np.allclose(ta, tb, rtol=0.0, atol=tol):
        problems.append(f"affine transform differs: before={tuple(before.transform)[:6]} "
                        f"after={tuple(after.transform)[:6]}")
    if not np.allclose(before.bounds, after.bounds, rtol=0.0, atol=tol):
        problems.append(f"bounds differ: before={tuple(before.bounds)} after={tuple(after.bounds)}")
    # The AOI-window transform inherits its origin from the requested bounds, so it cannot
    # see a sub-pixel grid offset between the two source rasters. Check the grid phase of the
    # full rasters: same pixel size, and origin offset must be a whole number of pixels.
    sa, sb_ = before.src_transform, after.src_transform
    if not np.allclose([sa.a, sa.b, sa.d, sa.e], [sb_.a, sb_.b, sb_.d, sb_.e], rtol=0.0, atol=1e-6 * px):
        problems.append("source pixel size / rotation differs between dates")
    else:
        off_x, off_y = (sa.c - sb_.c) / sb_.a, (sa.f - sb_.f) / sb_.e
        if abs(off_x - round(off_x)) > 1e-3 or abs(off_y - round(off_y)) > 1e-3:
            problems.append(f"pixel grids are offset by a fractional pixel ({off_x:.3f}, {off_y:.3f} px)")
    if problems:
        raise ValueError(
            "Before/after are NOT on the same grid, so they cannot be differenced pixel-by-pixel. "
            "The STAC assets must be reprojected/resampled onto one grid BEFORE detection "
            "(detect_changes() deliberately does not co-register or resample). Problems: "
            + "; ".join(problems))


# --------------------------------------------------------------------------- #
# Co-registration (NOT used by detect_changes any more; kept for other callers)
# --------------------------------------------------------------------------- #
def _grad(img):
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return cv2.magnitude(sx, sy)


def coregister(refl_b, refl_a, max_shift=1.5, min_response=0.05):
    """Sub-pixel shift of AFTER onto BEFORE. Returns (refl_a_aligned, dx, dy, response)."""
    h, w = refl_b.shape[:2]
    if min(h, w) < 96:
        return refl_a, 0.0, 0.0, 0.0
    gb = _grad(refl_b[..., 2] + refl_b[..., 1])
    ga = _grad(refl_a[..., 2] + refl_a[..., 1])
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(gb, ga, win)
    if resp < min_response or max(abs(dx), abs(dy)) > max_shift or max(abs(dx), abs(dy)) < 0.15:
        return refl_a, float(dx), float(dy), float(resp)
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = np.stack([cv2.warpAffine(refl_a[..., i], m, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT) for i in range(NB)], -1)
    return aligned, float(dx), float(dy), float(resp)


# --------------------------------------------------------------------------- #
# IR-MAD
# --------------------------------------------------------------------------- #
def _cca(X, Y, w):
    sw = w.sum()
    mx, my = (w[:, None] * X).sum(0) / sw, (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    eps = 1e-9 * np.eye(X.shape[1])
    Sxx = (Xc * w[:, None]).T @ Xc / sw + eps
    Syy = (Yc * w[:, None]).T @ Yc / sw + eps
    Sxy = (Xc * w[:, None]).T @ Yc / sw
    M = Sxy @ np.linalg.solve(Syy, Sxy.T)
    lam, A = eigh((M + M.T) / 2, Sxx)
    order = np.argsort(lam)[::-1]
    lam, A = np.clip(lam[order], 1e-8, 1 - 1e-8), A[:, order]
    rho = np.sqrt(lam)
    B = np.linalg.solve(Syy, Sxy.T @ A) / rho
    return A, B, rho, mx, my


def irmad(X, Y, valid, max_iter=30, tol=1e-4, subsample=250_000, seed=0):
    """Returns (T, info). T ~ chi2(p) for unchanged pixels (p = number of bands)."""
    n, p = X.shape
    idx = np.flatnonzero(valid)
    if idx.size < 50:
        raise ValueError("Too few valid pixels for IR-MAD.")
    rng = np.random.default_rng(seed)
    fit = idx if idx.size <= subsample else rng.choice(idx, subsample, replace=False)
    Xf, Yf = X[fit].astype(np.float64), Y[fit].astype(np.float64)
    w = np.ones(len(fit))
    rho_old = np.zeros(p)
    iters = 0
    for iters in range(1, max_iter + 1):
        A, B, rho, mx, my = _cca(Xf, Yf, w)
        sig2 = np.maximum(2 * (1 - rho), 1e-6)
        mad = (Xf - mx) @ A - (Yf - my) @ B
        w = chi2.sf(np.sum(mad ** 2 / sig2, axis=1), p)
        if np.max(np.abs(rho - rho_old)) < tol:
            break
        rho_old = rho
    mad_all = (X.astype(np.float64) - mx) @ A - (Y.astype(np.float64) - my) @ B
    T = np.sum(mad_all ** 2 / sig2, axis=1)
    return T.astype(np.float32), {"rho": rho.tolist(), "iterations": iters}


def normalise_to(refl_b, refl_a, stable):
    """Per-band weighted linear fit AFTER -> BEFORE on stable pixels."""
    out = np.empty_like(refl_a)
    fits = {}
    idx = np.flatnonzero(stable.ravel())
    if idx.size > 200_000:
        idx = np.random.default_rng(0).choice(idx, 200_000, replace=False)
    for i, name in enumerate(BANDS):
        b = refl_b[..., i].ravel()[idx]
        a = refl_a[..., i].ravel()[idx]
        if idx.size < 50 or np.std(a) < 1e-4:
            slope, icpt = 1.0, float(np.mean(b) - np.mean(a)) if idx.size else 0.0
        else:
            slope = float(np.cov(a, b)[0, 1] / np.var(a, ddof=1))
            icpt = float(np.mean(b) - slope * np.mean(a))
        out[..., i] = np.clip(refl_a[..., i] * slope + icpt, 0, 1)
        fits[name] = (float(slope), float(icpt))
    return out, fits


# --------------------------------------------------------------------------- #
# Parameters / result
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    sensitivity: str = "balanced"
    min_area_px: int = 5
    sam_min: float = 0.05             # spectral angle in RADIANS; below this a change is brightness-only
    min_significance: float = 7.0     # region mean -log10(p) required (7 => p < 1e-7)
    reject_slivers: bool = True       # drop <=2 px wide, short segments (edge artefacts)
    ignore_water_variability: bool = True
    coregister: bool = False          # DEPRECATED / ignored: detect_changes() validates alignment instead
    smooth_sigma: float = 1.0         # px; Gaussian blur of both dates before IR-MAD
    calibrate: bool = True            # rescale T by the scene's measured overdispersion (never more sensitive)
    semantic_min_pct: float = 60.0    # only used when a semantic map is supplied
    # ---- shape / structure gate -------------------------------------------- #
    target: str = "structures"        # "structures" = only building-/road-like regions; "all" = every change
    min_rect: float = 0.80            # area / rotated-bounding-rectangle area (rectangle 1.0, circle 0.78)
    min_solidity: float = 0.88        # area / convex-hull area (irregular blobs are low)
    building_max_aspect: float = 4.0
    building_max_px: int = 150        # 150 px = 1.5 ha; larger rectangles are usually farm plots
    road_min_len_px: int = 12         # skeleton length (120 m)
    road_max_width_px: float = 6.0
    road_min_aspect: float = 3.5
    min_shape_px: int = 16            # below this a shape cannot be judged at 10 m
    tiny_min_significance: float = 12.0
    max_ndvi_after: float = 0.30      # a building/road is not green afterwards


@dataclass
class Result:
    T: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    class_map: np.ndarray
    regions: list
    funnel: list
    diag: dict
    valid: np.ndarray
    sam: np.ndarray
    indices: dict
    px_area_m2: float
    rejected: list = field(default_factory=list)   # regions removed by the structure gate (with reason)


def refine_footprint(seg, cmag, pad=3):
    """IR-MAD regions are blurred (smoothing + hysteresis growth), so a 5x5 building comes out
    8x9 and a 2 px road ~8 px wide. Re-threshold the UNSMOOTHED change magnitude (Otsu) inside the
    region's neighbourhood to recover the real footprint before judging its shape."""
    H, W = seg.shape
    ys, xs = np.nonzero(seg)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    sub = seg[y0:y1, x0:x1]
    near = ndi.binary_dilation(sub, iterations=2)
    vals = cmag[y0:y1, x0:x1][near]
    if vals.size < 12 or np.ptp(vals) < 1e-4:
        return seg
    foot = near & (cmag[y0:y1, x0:x1] >= threshold_otsu(vals))
    lab, k = ndi.label(foot, structure=np.ones((3, 3)))
    if k == 0:
        return seg
    sizes = ndi.sum(foot, lab, index=np.arange(1, k + 1))
    keep = np.flatnonzero(sizes >= max(6, 0.08 * sizes.max())) + 1
    foot = np.isin(lab, keep)
    if foot.sum() < 4:
        return seg
    out = np.zeros_like(seg)
    out[y0:y1, x0:x1] = foot
    return out


def shape_metrics(seg, p):
    """Geometry of one region (boolean mask) and a kind: building-like / road-like / irregular / tiny."""
    ys, xs = np.nonzero(seg)
    crop = np.pad(seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 3).astype(np.uint8)
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    crop = ndi.binary_fill_holes(crop).astype(np.uint8)
    area = int(crop.sum())
    out = {"rectangularity": 0.0, "solidity": 0.0, "aspect": 1.0, "skeleton_px": 0,
           "mean_width_px": 0.0, "kind": "irregular"}
    if area < p.min_shape_px:
        out["kind"] = "tiny"
        return out
    cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=len)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    w, h = w + 1.0, h + 1.0                      # contour runs through pixel centres
    short, long_ = min(w, h), max(w, h)
    hull = cv2.convexHull(c)
    hm = np.zeros_like(crop)
    cv2.fillConvexPoly(hm, hull, 1)
    skel = int(skeletonize(crop > 0).sum())
    mean_w = area / max(skel, 1)
    out.update(rectangularity=round(min(area / (w * h), 1.0), 2),
               solidity=round(min(area / max(int(hm.sum()), 1), 1.0), 2),
               aspect=round(long_ / max(short, 1e-6), 1), skeleton_px=skel,
               mean_width_px=round(mean_w, 1))
    if (skel >= p.road_min_len_px and mean_w <= p.road_max_width_px
            and out["aspect"] >= p.road_min_aspect):
        out["kind"] = "road-like"
    elif (out["rectangularity"] >= p.min_rect and out["solidity"] >= p.min_solidity
          and out["aspect"] <= p.building_max_aspect):
        out["kind"] = "building-like" if area <= p.building_max_px else "large rectangle (field-like)"
    return out


def structure_verdict(reg, ndvi_after, p):
    """(accept, reason). Only meaningful when p.target == 'structures'."""
    kind, code = reg["shape"], reg["class_code"]
    # thin roads are mixed with their surroundings at 10 m -> judge them by the greenest-free quartile
    nv = reg.get("ndvi_after_p25", ndvi_after) if reg["shape"] == "road-like" else ndvi_after
    if nv > p.max_ndvi_after:
        return False, f"still vegetated afterwards (NDVI {nv:.2f})"
    if kind == "tiny":
        ok = code in (1, 7, 8) and reg["significance"] >= p.tiny_min_significance
        return ok, "" if ok else "too small to confirm a structure"
    if kind == "road-like":
        return (True, "") if code in (1, 2, 6, 7, 8) else (False, "thin line but spectral type is not road-like")
    if kind == "building-like":
        if code == 6:
            return False, "dark compact patch - wet ground / shadow, not a roof (dark = road only if linear)"
        return (True, "") if code in (1, 2, 7, 8) else (False, "rectangular but water/vegetation change")
    if kind.startswith("large rectangle"):
        # Allow large rectangles if they're construction sites (code 8)
        if code == 8:
            return (True, "")
        return False, f"rectangle larger than {p.building_max_px} px - looks like a farm plot"
    # Allow irregular shapes for construction sites (code 8) - they're inherently irregular
    if code == 8:
        # Construction sites: require significance and spectral evidence, but not rectangular shape
        if reg["significance"] >= p.min_significance:
            return (True, "")
        return False, "construction candidate but low statistical significance"
    return False, (f"irregular outline (rectangularity {reg['rectangularity']}, "
                   f"solidity {reg['solidity']}) - not a building or road")


def _position_word(cx, cy):
    v = "N" if cy < 0.33 else "S" if cy > 0.66 else ""
    h = "W" if cx < 0.33 else "E" if cx > 0.66 else ""
    return (v + h) or "center"


def detect_changes(before: Scene, after: Scene, params: Params = None, px_area_m2=100.0,
                   semantic_map=None):
    params = params or Params()

    # ======================================================================= #
    # STAGE 0 - alignment VALIDATION. No co-registration, no resampling, no shifting.
    # ======================================================================= #
    validate_alignment(before, after)

    H, W = before.refl.shape[:2]
    valid = ~(before.nodata | after.nodata | before.bad | after.bad)
    if valid.sum() < 100:
        raise ValueError("Almost the whole AOI is nodata/cloud/snow - pick another AOI or date.")
    funnel = [("AOI pixels", H * W), ("valid (no nodata / cloud / snow)", int(valid.sum()))]
    diag = {"cloud_pct": float(100 * (before.cloud | after.cloud).mean()),
            "snow_pct": float(100 * (before.snow | after.snow).mean()),
            "calibration_before": before.calib, "calibration_after": after.calib,
            # kept so existing UI code reading diag["shift_px"] keeps working
            "shift_px": (0.0, 0.0), "shift_response": 0.0,
            "alignment": "validated identical grid (no co-registration performed)"}

    Xb, Xa = before.refl, after.refl

    # ======================================================================= #
    # STAGE 1 - STATISTICAL change evidence (IR-MAD).
    #   T = statistical evidence that the spectral relationship between T1 and T2 is
    #   unusual. It says "something is odd here", NOT "land cover changed".
    # ======================================================================= #
    def _smooth(x):
        if params.smooth_sigma <= 0:
            return x
        return np.stack([cv2.GaussianBlur(x[..., i], (0, 0), params.smooth_sigma)
                         for i in range(NB)], -1)

    T, info = irmad(_smooth(Xb).reshape(-1, NB), _smooth(Xa).reshape(-1, NB), valid.ravel())
    T = T.reshape(H, W)
    overdisp = 1.0
    if params.calibrate:
        overdisp = max(1.0, float(np.median(T[valid])) / float(chi2.median(NB)))
        T = T / overdisp
    T[~valid] = 0.0
    info["overdispersion"] = overdisp
    diag["irmad"] = info
    a_hi, a_lo = PRESETS[params.sensitivity]
    t_hi, t_lo = chi2.isf(a_hi, NB), chi2.isf(a_lo, NB)
    diag["thresholds_T"] = (float(t_lo), float(t_hi))

    # Statistical CANDIDATES only - not "confirmed change".
    irmad_candidate = apply_hysteresis_threshold(T, t_lo, t_hi) & valid
    funnel.append(("IR-MAD candidate (statistical only)", int(irmad_candidate.sum())))

    # ======================================================================= #
    # STAGE 2 - radiometric normalisation on IR-MAD-stable pixels, then spectral features.
    # ======================================================================= #
    stable = (chi2.sf(T, NB) > 0.5) & valid
    Xa_n, fits = normalise_to(Xb, Xa, stable)
    diag["normalisation_fit"] = fits            # slope, intercept per band (after -> before)
    cmag = np.linalg.norm(np.stack([cv2.GaussianBlur((Xa_n - Xb)[..., i], (0, 0), 0.6)
                                    for i in range(NB)], -1), axis=-1)
    gb, rb, nb_, sb = (Xb[..., i] for i in range(4))        # green, red, nir, swir16
    ga, ra, na, sa = (Xa_n[..., i] for i in range(4))
    ndvi_b, ndvi_a = nd(nb_, rb), nd(na, ra)
    ndbi_b, ndbi_a = nd(sb, nb_), nd(sa, na)
    mndwi_b, mndwi_a = nd(gb, sb), nd(ga, sa)
    bright_b, bright_a = Xb.mean(-1), Xa_n.mean(-1)
    d = {"dNDVI": ndvi_a - ndvi_b, "dNDBI": ndbi_a - ndbi_b,
         "dMNDWI": mndwi_a - mndwi_b, "dBright": bright_a - bright_b}
    Sb, Sa = _smooth(Xb), _smooth(Xa_n)
    dot = np.sum(Sb * Sa, -1)
    sam = np.arccos(np.clip(dot / (np.linalg.norm(Sb, axis=-1) * np.linalg.norm(Sa, axis=-1) + 1e-9),
                            -1, 1)).astype(np.float32)

    # ======================================================================= #
    # STAGE 3 - SPECTRAL LAND-COVER TRANSITION EVIDENCE (explicit, fixed thresholds).
    # ======================================================================= #
    veg_loss = ((ndvi_b > VEG_LOSS_NDVI_BEFORE_MIN) & (ndvi_a < VEG_LOSS_NDVI_AFTER_MAX)
                & (d["dNDVI"] < VEG_LOSS_DNDVI_MAX))
    veg_gain = (ndvi_a > VEG_GAIN_NDVI_AFTER_MIN) & (d["dNDVI"] > VEG_GAIN_DNDVI_MIN)
    w_gain = (mndwi_a > WATER_MNDWI_MIN) & (d["dMNDWI"] > WATER_DMNDWI)
    w_loss = (mndwi_b > WATER_MNDWI_MIN) & (d["dMNDWI"] < -WATER_DMNDWI)

    # built_candidate is NOT a building classifier. It is only spectral evidence that a
    # non-vegetated, brighter, SWIR-richer surface with a changed spectral SHAPE appeared.
    # NDBI alone cannot identify buildings (bare soil also has high NDBI), which is why
    # it is combined with low NDVI, a brightness increase and a real spectral-angle change.
    # Telling roofs from soil is the job of the later semantic / VLM / SAM2 stages.
    built_candidate = ((ndvi_a < BUILT_NDVI_AFTER_MAX) & (d["dNDBI"] > BUILT_DNDBI_MIN)
                       & (d["dBright"] > BUILT_DBRIGHT_MIN) & (sam > params.sam_min))

    # construction_candidate: active earthworks / construction sites
    # Distinct from finished buildings: can be darker (excavation), partial vegetation,
    # but MUST show spectral shape change (sam) and NDBI rise from soil exposure.
    # CRITICAL: stricter thresholds to distinguish from agricultural changes
    # (plowing, harvesting) which change spectral but don't add persistent structure.
    # Construction: dNDBI rise + dBright positive (material added) + high SAM (structural change)
    construction_candidate = ((ndvi_a < CONSTR_NDVI_AFTER_MAX) & (d["dNDBI"] > CONSTR_DNDBI_MIN)
                              & (d["dBright"] > 0.01) & (sam > CONSTR_SAM_MIN))

    landcover_evidence = veg_loss | veg_gain | w_gain | w_loss | built_candidate | construction_candidate
    funnel.append(("land-cover evidence (all valid px, before IR-MAD gate)",
                   int((landcover_evidence & valid).sum())))

    # THE KEY GATE: a pixel must be statistically odd AND show a plausible transition.
    cand = irmad_candidate & landcover_evidence & valid
    funnel.append(("combined candidate (IR-MAD AND land-cover evidence)", int(cand.sum())))

    rules = {"veg_loss": veg_loss, "veg_gain": veg_gain, "w_gain": w_gain, "w_loss": w_loss,
             "built_candidate": built_candidate, "construction_candidate": construction_candidate,
             "landcover_evidence": landcover_evidence}
    diag["evidence_px_valid"] = {k: int((v & valid).sum()) for k, v in rules.items()}
    diag["evidence_px_in_irmad"] = {k: int((v & irmad_candidate).sum()) for k, v in rules.items()}
    diag["irmad_candidates_without_evidence_px"] = int((irmad_candidate & ~landcover_evidence).sum())

    # Pixel class map (used to label regions). Later = higher priority. dark_new only labels
    # pixels that carry no other evidence (e.g. morphology fill) - it is not a candidate rule.
    dark_new = (ndvi_a < 0.20) & (ndvi_b < 0.20) & (d["dBright"] < -0.02)
    cls = np.full((H, W), 7, np.uint8)
    for code, m_ in ((6, dark_new), (3, veg_gain), (2, veg_loss),
                     (1, built_candidate), (5, w_loss), (4, w_gain),
                     (8, construction_candidate)):
        cls[m_] = code

    # ======================================================================= #
    # STAGE 4 - false-positive vetoes
    # ======================================================================= #
    m = cand.copy()
    brightness_only = ((sam < params.sam_min) & (np.abs(d["dNDVI"]) < 0.05)
                       & (np.abs(d["dNDBI"]) < 0.05) & (np.abs(d["dMNDWI"]) < 0.05))
    m &= ~brightness_only
    funnel.append(("after brightness-only veto (moisture / sun angle)", int(m.sum())))
    if params.ignore_water_variability:
        both_water = (mndwi_b > 0.10) & (mndwi_a > 0.10) & ~(w_gain | w_loss)
        m &= ~both_water
        funnel.append(("after water-surface variability veto", int(m.sum())))
    else:
        funnel.append(("water-surface variability veto (disabled)", int(m.sum())))

    # ======================================================================= #
    # STAGE 5 - morphology / minimum-area / sliver filtering
    # ======================================================================= #
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m8 &= valid.astype(np.uint8)
    labels, n = ndi.label(m8, structure=np.ones((3, 3)))
    if n:
        areas = ndi.sum(m8, labels, index=np.arange(1, n + 1))
        keep = areas >= params.min_area_px
        if params.reject_slivers:
            edt = ndi.distance_transform_edt(np.pad(m8, 1))[1:-1, 1:-1]
            maxdt = ndi.maximum(edt, labels, index=np.arange(1, n + 1))
            keep &= ~((maxdt <= 1.0) & (areas < 12))
        m8 = np.isin(labels, np.flatnonzero(keep) + 1).astype(np.uint8)
    funnel.append((f"morphology / size filter (min {params.min_area_px} px)", int(m8.sum())))

    # ======================================================================= #
    # STAGE 6 - optional semantic filter. Kept ONLY for interface compatibility.
    #   This keeps regions whose mean semantic score is above a percentile of the AOI; it is
    #   a score filter on ONE map, NOT a semantic change detector. A proper one compares DINO
    #   features of T1 and T2 per location. It is deliberately separate from the spectral stages.
    # ======================================================================= #
    if semantic_map is not None:
        thr = np.percentile(semantic_map[valid], params.semantic_min_pct)
        labels, n = ndi.label(m8, structure=np.ones((3, 3)))
        if n:
            means = ndi.mean(semantic_map, labels, index=np.arange(1, n + 1))
            m8 = np.isin(labels, np.flatnonzero(means >= thr) + 1).astype(np.uint8)
        funnel.append(("DINO filter (percentile score map)", int(m8.sum())))
    else:
        funnel.append(("DINO filter (not used)", int(m8.sum())))

    # ======================================================================= #
    # STAGE 7 - region extraction (+ significance gate, + optional structure gate)
    # ======================================================================= #
    labels, n = ndi.label(m8, structure=np.ones((3, 3)))
    neglogp = np.minimum(-np.log10(np.maximum(chi2.sf(T, NB), 1e-300)), 20.0)
    regions, final = [], np.zeros((H, W), np.uint8)
    kept_labels = np.zeros((H, W), np.int32)
    dropped_sig = dropped_sam = 0
    rejected = []
    for lab in range(1, n + 1):
        seg = labels == lab
        ys, xs = np.nonzero(seg)
        area = int(seg.sum())
        sig = float(neglogp[seg].mean())
        cnt = np.bincount(cls[seg], minlength=8)
        cnt[0] = 0
        code = int(cnt.argmax())
        mean_sam = float(sam[seg].mean())
        dv, db = float(d["dNDVI"][seg].mean()), float(d["dNDBI"][seg].mean())
        if sig < params.min_significance:
            dropped_sig += area
            continue
        if (mean_sam < params.sam_min and abs(dv) < 0.05 and abs(db) < 0.05
                and code not in (4, 5)):
            dropped_sam += area                  # brightness-only region (moisture / sun angle)
            continue
        reg_seg = refine_footprint(seg, cmag) if params.target == "structures" else seg
        ys, xs = np.nonzero(reg_seg)
        area = int(reg_seg.sum())
        shp = shape_metrics(reg_seg, params)
        rid = len(regions) + 1
        reg = {
            "id": rid,
            "box": (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1)),
            "area_px": area,
            "area_ha": round(area * px_area_m2 / 10000.0, 3),
            "position": _position_word(xs.mean() / W, ys.mean() / H),
            "class_code": code,
            "class_name": CLASS_INFO[code][0],
            "significance": round(sig, 1),
            "dNDVI": round(float(d["dNDVI"][reg_seg].mean()), 3),
            "dNDBI": round(float(d["dNDBI"][reg_seg].mean()), 3),
            "dMNDWI": round(float(d["dMNDWI"][reg_seg].mean()), 3),
            "dBrightness": round(float(d["dBright"][reg_seg].mean()), 3),
            "spectral_angle_deg": round(float(np.degrees(sam[reg_seg].mean())), 1),
            "ndvi_after": round(float(np.median(ndvi_a[reg_seg])), 2),
            "ndvi_after_p25": round(float(np.percentile(ndvi_a[reg_seg], 25)), 2),
            "shape": shp["kind"], "rectangularity": shp["rectangularity"],
            "solidity": shp["solidity"], "aspect": shp["aspect"],
            "mean_width_px": shp["mean_width_px"], "skeleton_px": shp["skeleton_px"],
        }
        seg = reg_seg
        if params.target == "structures":
            ok, why = structure_verdict(reg, reg["ndvi_after"], params)
            if not ok:
                reg["reject_reason"] = why
                rejected.append(reg)
                continue
        final[seg] = 1
        kept_labels[seg] = rid
        regions.append(reg)
    diag["dropped_low_significance_px"] = dropped_sig
    diag["dropped_brightness_only_region_px"] = dropped_sam
    funnel.append((f"after region significance (>= {params.min_significance:g}) + brightness-only", int(final.sum())))
    regions.sort(key=lambda r: r["area_px"] * r["significance"], reverse=True)
    remap = {r["id"]: i for i, r in enumerate(regions, start=1)}
    new_labels = np.zeros_like(kept_labels)
    for old, new in remap.items():
        new_labels[kept_labels == old] = new
    for r in regions:
        r["id"] = remap[r["id"]]
    if params.target == "structures":
        funnel.append((f"structure gate (shape + spectral): rejected {len(rejected)} regions", int(final.sum())))
    funnel.append(("final regions", len(regions)))

    # Debug masks (stored in `indices` so the Result API is unchanged). See debug_masks().
    dbg = {"irmad_candidate": irmad_candidate, "veg_loss": veg_loss, "veg_gain": veg_gain,
           "w_gain": w_gain, "w_loss": w_loss, "built_candidate": built_candidate,
           "construction_candidate": construction_candidate,
           "landcover_evidence": landcover_evidence, "cand": cand}
    return Result(T=T, mask=final, labels=new_labels, class_map=cls, regions=regions,
                  funnel=funnel, diag=diag, valid=valid, sam=sam, px_area_m2=px_area_m2,
                  indices={"ndvi_b": ndvi_b, "ndvi_a": ndvi_a, **d, **dbg}, rejected=rejected)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def class_overlay(rgb, res: Result, keep_ids=None, alpha=0.6):
    out = rgb.copy().astype(np.float32)
    for code, (_, color) in CLASS_INFO.items():
        sel = (res.class_map == code) & (res.mask == 1)
        if keep_ids is not None:
            sel &= np.isin(res.labels, list(keep_ids))
        out[sel] = (1 - alpha) * out[sel] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


def heatmap(T, valid):
    p = np.minimum(-np.log10(np.maximum(chi2.sf(T, NB), 1e-300)), 12.0) / 12.0
    p[~valid] = 0
    return cv2.cvtColor(cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB)


DEBUG_MASK_ORDER = ("irmad_candidate", "veg_loss", "veg_gain", "w_gain", "w_loss",
                    "built_candidate", "construction_candidate", "landcover_evidence", "cand")


def debug_masks(res: Result):
    """name -> array for every stage: T (continuous), the boolean stage masks, and the final mask.
    Use it to see whether soil false positives come from IR-MAD or from the spectral rules."""
    out = {"T": res.T}
    out.update({k: res.indices[k] for k in DEBUG_MASK_ORDER})
    out["final_mask"] = res.mask.astype(bool)
    return out


def debug_panel(res: Result, rgb=None):
    """name -> RGB uint8 image for each debug mask (T as the usual heatmap; masks white-on-dark,
    optionally blended over `rgb`)."""
    panel = {"T": heatmap(res.T, res.valid)}
    for name, m in debug_masks(res).items():
        if name == "T":
            continue
        img = np.full((*m.shape, 3), 25, np.uint8) if rgb is None else (rgb * 0.45).astype(np.uint8)
        img[m.astype(bool)] = (255, 255, 255) if rgb is None else (255, 60, 0)
        panel[name] = img
    return panel


def draw_regions(rgb, regions, verdicts=None, only=None):
    """Numbered boxes: green = VLM confirmed, red = VLM rejected, yellow = unreviewed."""
    out = rgb.copy()
    for r in regions:
        if only is not None and r["id"] not in only:
            continue
        v = (verdicts or {}).get(r["id"])
        color = (255, 220, 0) if v is None else ((0, 220, 0) if v["real_change"] else (255, 0, 0))
        x, y, w, h = r["box"]
        cv2.rectangle(out, (max(0, x - 2), max(0, y - 2)), (x + w + 1, y + h + 1), color, 1)
        cv2.putText(out, f"#{r['id']}", (x, max(10, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)
    return out