import os
import csv
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.warp import reproject

from skimage import measure, morphology
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")
RAW = ROOT / "raw"
OUT = ROOT / "phase456_output"

OUT.mkdir(exist_ok=True)

BEFORE = "S2A_39SXV_20200128_1_L2A"
AFTER = "S2B_39SXV_20260128_0_L2A"

# ---------------- DINO ----------------

DINO_MODEL = "dinov2_vits14"

DINO_INPUT_SIZE = 448
DINO_STRIDE = 224

# RTX 4080 16 GB
# Start with 8. You can later try 16.
DINO_BATCH_SIZE = 8

# ---------------- Candidate detection ----------------

CHANGE_PERCENTILE = 97.0
MIN_OBJECT_PIXELS = 25
MAX_CANDIDATES = 100

# ---------------- Fusion ----------------

W_DINO = 0.50
W_NDBI = 0.25
W_NDVI = 0.15
W_NDMI = 0.10


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("PHASE 4-6")
print("DINOv2 + SPECTRAL CHANGE + CHANGE CANDIDATES")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA runtime: {torch.version.cuda}")

    torch.backends.cudnn.benchmark = True


# ============================================================
# FILE PATHS
# ============================================================

def tif(product, suffix):
    return RAW / f"{product}_{suffix}.tif"


RGB_BEFORE = tif(BEFORE, "visual")
RGB_AFTER = tif(AFTER, "visual")

RED_BEFORE = tif(BEFORE, "red")
RED_AFTER = tif(AFTER, "red")

NIR_BEFORE = tif(BEFORE, "nir")
NIR_AFTER = tif(AFTER, "nir")

SWIR_BEFORE = tif(BEFORE, "swir16")
SWIR_AFTER = tif(AFTER, "swir16")


# ============================================================
# RASTER HELPERS
# ============================================================

def read_rgb(path):

    with rasterio.open(path) as src:

        arr = src.read()

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    if arr.shape[0] != 3:
        raise ValueError(
            f"Expected RGB image with 3 bands: {path}"
        )

    return arr, profile, transform, crs


def read_rgb_to_reference(path, reference_path):

    """Read an RGB raster on exactly the reference raster grid."""

    with rasterio.open(reference_path) as ref:

        ref_height = ref.height
        ref_width = ref.width
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_profile = ref.profile.copy()

    with rasterio.open(path) as src:

        if src.count != 3:
            raise ValueError(f"Expected RGB image with 3 bands: {path}")

        same_grid = (
            src.height == ref_height
            and src.width == ref_width
            and src.transform == ref_transform
            and src.crs == ref_crs
        )

        if same_grid:
            return (
                src.read(),
                ref_profile,
                ref_transform,
                ref_crs
            )

        destination = np.full(
            (3, ref_height, ref_width),
            np.nan,
            dtype=np.float32
        )

        for band_index in range(3):

            reproject(
                source=rasterio.band(src, band_index + 1),
                destination=destination[band_index],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan
            )

    print(
        f"Aligned RGB: {path.name} -> "
        f"{reference_path.name} grid"
    )

    return (
        destination,
        ref_profile,
        ref_transform,
        ref_crs
    )


def save_float_tif(path, array, reference_profile):

    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2,
        nodata=None
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)

    print(f"Saved: {path}")


def save_uint8_tif(path, array, reference_profile):

    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        compress="deflate",
        nodata=0
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.uint8), 1)

    print(f"Saved: {path}")


def percentile_normalize(x, low=2, high=98):

    x = np.asarray(x, dtype=np.float32)

    valid = np.isfinite(x)

    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)

    lo = np.percentile(x[valid], low)
    hi = np.percentile(x[valid], high)

    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)

    return np.clip(y, 0.0, 1.0).astype(np.float32)


def robust_abs_change(x):

    x = np.abs(x).astype(np.float32)

    return percentile_normalize(
        x,
        2,
        98
    )


def normalize_rgb_pair(before, after):

    """Normalize both RGB dates with shared per-band limits."""

    before = before.astype(np.float32)
    after = after.astype(np.float32)

    if before.shape[0] != 3 or after.shape[0] != 3:
        raise ValueError("RGB rasters must contain exactly three bands.")

    before_out = np.zeros_like(before, dtype=np.float32)
    after_out = np.zeros_like(after, dtype=np.float32)

    for band_index in range(3):

        values = np.concatenate([
            before[band_index].ravel(),
            after[band_index].ravel()
        ])
        valid = np.isfinite(values)

        if not np.any(valid):
            continue

        lo, hi = np.percentile(values[valid], [2, 98])

        if hi <= lo:
            continue

        before_out[band_index] = np.clip(
            (before[band_index] - lo) / (hi - lo),
            0.0,
            1.0
        )
        after_out[band_index] = np.clip(
            (after[band_index] - lo) / (hi - lo),
            0.0,
            1.0
        )

    before_out = np.nan_to_num(
        before_out,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    after_out = np.nan_to_num(
        after_out,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return before_out, after_out


# ============================================================
# RESAMPLE BAND TO REFERENCE GRID
# ============================================================

def read_band_to_reference(path, reference_path):

    """
    Reads a single band and resamples it to exactly match
    the reference raster.

    This is important for Sentinel-2 because:

        Red   = 10 m
        NIR   = 10 m
        Green = 10 m
        SWIR  = 20 m

    Therefore SWIR is typically:

        5490 x 5490

    while NIR is:

        10980 x 10980
    """

    with rasterio.open(reference_path) as ref:

        dst_height = ref.height
        dst_width = ref.width

        dst_transform = ref.transform
        dst_crs = ref.crs

    with rasterio.open(path) as src:

        source = src.read(1).astype(np.float32)

        destination = np.full(
            (dst_height, dst_width),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=source,
            destination=destination,

            src_transform=src.transform,
            src_crs=src.crs,

            dst_transform=dst_transform,
            dst_crs=dst_crs,

            resampling=Resampling.bilinear,

            src_nodata=src.nodata,
            dst_nodata=np.nan
        )

    print(
        f"{path.name}: "
        f"{source.shape} -> {destination.shape}"
    )

    return destination


# ============================================================
# REFLECTANCE NORMALIZATION
# ============================================================

def normalize_reflectance(x):

    x = x.astype(np.float32)

    finite = np.isfinite(x)

    if np.any(finite):

        p99 = np.percentile(
            x[finite],
            99
        )

        # Sentinel-2 L2A products are commonly
        # scaled by 10000.
        if p99 > 2.0:
            x = x / 10000.0

    return x


# ============================================================
# SPECTRAL INDEX
# ============================================================

def safe_index(a, b):

    denominator = a + b

    result = np.zeros_like(
        a,
        dtype=np.float32
    )

    valid = (
        np.isfinite(a)
        &
        np.isfinite(b)
        &
        (np.abs(denominator) > 1e-6)
    )

    result[valid] = (
        (a[valid] - b[valid])
        /
        denominator[valid]
    )

    return np.clip(
        result,
        -1.0,
        1.0
    )


# ============================================================
# LOAD RGB
# ============================================================

print("\nLoading RGB...")

rgb_before, ref_profile, ref_transform, ref_crs = \
    read_rgb(RGB_BEFORE)

rgb_after, profile_after, transform_after, crs_after = \
    read_rgb_to_reference(RGB_AFTER, RGB_BEFORE)


if rgb_before.shape != rgb_after.shape:

    raise ValueError(
        "Before/after RGB dimensions differ:\n"
        f"Before: {rgb_before.shape}\n"
        f"After : {rgb_after.shape}"
    )

print("\nNormalizing RGB images...")

rgb_valid_before = np.all(np.isfinite(rgb_before), axis=0)
rgb_valid_after = np.all(np.isfinite(rgb_after), axis=0)

rgb_before, rgb_after = normalize_rgb_pair(
    rgb_before,
    rgb_after
)
print(
    "Normalized BEFORE:",
    rgb_before.dtype,
    rgb_before.min(),
    rgb_before.max(),
    np.percentile(rgb_before, [1, 50, 99])
)

print(
    "Normalized AFTER:",
    rgb_after.dtype,
    rgb_after.min(),
    rgb_after.max(),
    np.percentile(rgb_after, [1, 50, 99])
)
H = rgb_before.shape[1]
W = rgb_before.shape[2]

print(f"RGB shape: {rgb_before.shape}")
print(f"Raster: {W} x {H}")


# ============================================================
# LOAD DINO
# ============================================================

print("\nLoading DINOv2...")

dino = torch.hub.load(
    "facebookresearch/dinov2",
    DINO_MODEL
)

dino = dino.to(DEVICE)

dino.eval()

print("DINO loaded.")


# ============================================================
# DINO TRANSFORM
# ============================================================

imagenet_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize(
        (
            DINO_INPUT_SIZE,
            DINO_INPUT_SIZE
        ),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# EXTRACT IMAGE PATCH
# ============================================================

def make_patch(rgb, y, x, size):

    patch = rgb[
        :,
        y:y + size,
        x:x + size
    ]

    # Pad at edges if necessary.
    if (
        patch.shape[1] != size
        or
        patch.shape[2] != size
    ):

        padded = np.zeros(
            (3, size, size),
            dtype=rgb.dtype
        )

        padded[
            :,
            :patch.shape[1],
            :patch.shape[2]
        ] = patch

        patch = padded

    patch = np.transpose(
        patch,
        (1, 2, 0)
    )

    if patch.dtype != np.uint8:
        patch = np.nan_to_num(
            patch.astype(np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )
        return np.clip(
            patch * 255.0,
            0,
            255
        ).astype(np.uint8)

    return patch


# ============================================================
# BATCH DINO FEATURES
# ============================================================

@torch.inference_mode()
def dino_batch_features(images):

    tensors = torch.stack(
        [
            imagenet_transform(img)
            for img in images
        ]
    )

    tensors = tensors.to(
        DEVICE,
        non_blocking=True
    )

    if DEVICE == "cuda":

        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16
        ):

            features = dino.forward_features(
                tensors
            )

    else:

        features = dino.forward_features(
            tensors
        )

    # DINOv2 spatial patch tokens
    #
    # Shape:
    # B x N x C
    #
    # For ViT-S/14 with 448x448:
    #
    # 448 / 14 = 32
    #
    # therefore:
    #
    # N = 32 x 32 = 1024

    tokens = features[
        "x_norm_patchtokens"
    ]

    n_tokens = tokens.shape[1]

    grid = int(
        math.sqrt(n_tokens)
    )

    if grid * grid != n_tokens:

        raise RuntimeError(
            f"Unexpected token count: "
            f"{n_tokens}"
        )

    tokens = tokens.reshape(
        tokens.shape[0],
        grid,
        grid,
        tokens.shape[2]
    )

    return tokens.float().cpu()


# ============================================================
# BITEMPORAL DINO CHANGE
# ============================================================

print("\nRunning bitemporal DINO...")

if H < DINO_INPUT_SIZE or W < DINO_INPUT_SIZE:
    raise ValueError(
        f"RGB raster ({W} x {H}) is smaller than the "
        f"DINO patch size ({DINO_INPUT_SIZE})."
    )

ys = list(
    range(
        0,
        H - DINO_INPUT_SIZE + 1,
        DINO_STRIDE
    )
)

xs = list(
    range(
        0,
        W - DINO_INPUT_SIZE + 1,
        DINO_STRIDE
    )
)


# Make sure final row/column reaches the image edge.
if ys[-1] != H - DINO_INPUT_SIZE:
    ys.append(
        H - DINO_INPUT_SIZE
    )

if xs[-1] != W - DINO_INPUT_SIZE:
    xs.append(
        W - DINO_INPUT_SIZE
    )


patch_locations = [
    (y, x)
    for y in ys
    for x in xs
]


print(
    f"Number of DINO patches: "
    f"{len(patch_locations)}"
)

print(
    f"DINO batch size: "
    f"{DINO_BATCH_SIZE}"
)


# Full-resolution spatial DINO change map.
dino_change = np.zeros(
    (H, W),
    dtype=np.float32
)

dino_weight = np.zeros(
    (H, W),
    dtype=np.float32
)


for start in range(
    0,
    len(patch_locations),
    DINO_BATCH_SIZE
):

    batch_locations = patch_locations[
        start:
        start + DINO_BATCH_SIZE
    ]


    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    before_batch = [
        make_patch(
            rgb_before,
            y,
            x,
            DINO_INPUT_SIZE
        )
        for y, x in batch_locations
    ]


    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    after_batch = [
        make_patch(
            rgb_after,
            y,
            x,
            DINO_INPUT_SIZE
        )
        for y, x in batch_locations
    ]


    # --------------------------------------------------------
    # DINO FEATURES
    # --------------------------------------------------------

    before_features = \
        dino_batch_features(
            before_batch
        )

    after_features = \
        dino_batch_features(
            after_batch
        )


    # --------------------------------------------------------
    # COSINE DISTANCE
    # --------------------------------------------------------

    before_features = F.normalize(
        before_features,
        dim=-1
    )

    after_features = F.normalize(
        after_features,
        dim=-1
    )


    # Shape:
    #
    # B x 32 x 32

    change = 1.0 - (
        before_features
        *
        after_features
    ).sum(dim=-1)


    change = change.numpy()


    # --------------------------------------------------------
    # UPSAMPLE SPATIAL DINO MAP
    # --------------------------------------------------------

    change_tensor = torch.from_numpy(
        change
    ).unsqueeze(1)


    change_tensor = F.interpolate(
        change_tensor,

        size=(
            DINO_INPUT_SIZE,
            DINO_INPUT_SIZE
        ),

        mode="bilinear",

        align_corners=False
    )


    change = change_tensor.squeeze(
        1
    ).numpy()


    # --------------------------------------------------------
    # ACCUMULATE OVERLAPPING WINDOWS
    # --------------------------------------------------------

    for i, (y, x) in enumerate(
        batch_locations
    ):

        dino_change[
            y:y + DINO_INPUT_SIZE,
            x:x + DINO_INPUT_SIZE
        ] += change[i]

        dino_weight[
            y:y + DINO_INPUT_SIZE,
            x:x + DINO_INPUT_SIZE
        ] += 1.0


    if (
        start == 0
        or
        start % (
            DINO_BATCH_SIZE * 50
        ) == 0
    ):

        progress = (
            100.0
            *
            start
            /
            len(patch_locations)
        )

        print(
            f"DINO patch "
            f"{start}/"
            f"{len(patch_locations)} "
            f"({progress:.1f}%)"
        )


# Avoid division by zero.
dino_weight[
    dino_weight == 0
] = 1.0


dino_change /= dino_weight


dino_change = percentile_normalize(
    dino_change,
    2,
    98
)


save_float_tif(
    OUT / "01_dino_change.tif",
    dino_change,
    ref_profile
)


# ============================================================
# LOAD SPECTRAL BANDS
# ============================================================

print("\nLoading spectral bands...")

print(
    "\nImportant: all spectral bands are resampled to the "
    "before-image 10 m reference grid."
)


# ------------------------------------------------------------
# RED
# ------------------------------------------------------------

red_b = normalize_reflectance(
    read_band_to_reference(
        RED_BEFORE,
        RGB_BEFORE
    )
)

red_a = normalize_reflectance(
    read_band_to_reference(
        RED_AFTER,
        RGB_BEFORE
    )
)


# ------------------------------------------------------------
# NIR
# ------------------------------------------------------------

nir_b = normalize_reflectance(
    read_band_to_reference(
        NIR_BEFORE,
        RGB_BEFORE
    )
)

nir_a = normalize_reflectance(
    read_band_to_reference(
        NIR_AFTER,
        RGB_BEFORE
    )
)


# ------------------------------------------------------------
# SWIR
# ------------------------------------------------------------

swir_b = normalize_reflectance(
    read_band_to_reference(
        SWIR_BEFORE,
        RGB_BEFORE
    )
)

swir_a = normalize_reflectance(
    read_band_to_reference(
        SWIR_AFTER,
        RGB_BEFORE
    )
)


# ============================================================
# VERIFY DIMENSIONS
# ============================================================

expected_shape = nir_b.shape

print("\nFinal spectral dimensions:")

for name, arr in [
    ("red_b", red_b),
    ("red_a", red_a),
    ("nir_b", nir_b),
    ("nir_a", nir_a),
    ("swir_b", swir_b),
    ("swir_a", swir_a)
]:

    print(
        f"{name}: {arr.shape}"
    )

    if arr.shape != expected_shape:

        raise RuntimeError(
            f"{name} has wrong shape: "
            f"{arr.shape} != "
            f"{expected_shape}"
        )


valid_mask = (
    rgb_valid_before
    & rgb_valid_after
    & np.isfinite(red_b)
    & np.isfinite(red_a)
    & np.isfinite(nir_b)
    & np.isfinite(nir_a)
    & np.isfinite(swir_b)
    & np.isfinite(swir_a)
    & ((np.abs(red_b) + np.abs(nir_b) + np.abs(swir_b)) > 0)
    & ((np.abs(red_a) + np.abs(nir_a) + np.abs(swir_a)) > 0)
)

if not np.any(valid_mask):
    raise RuntimeError("No valid overlapping pixels were found.")


# ============================================================
# SPECTRAL INDICES
# ============================================================

print("\nCalculating spectral indices...")


# NDVI
#
# (NIR - RED)
# -----------
# (NIR + RED)

ndvi_b = safe_index(
    nir_b,
    red_b
)

ndvi_a = safe_index(
    nir_a,
    red_a
)


# NDBI
#
# (SWIR - NIR)
# ------------
# (SWIR + NIR)

ndbi_b = safe_index(
    swir_b,
    nir_b
)

ndbi_a = safe_index(
    swir_a,
    nir_a
)


# NDMI
#
# (NIR - SWIR)
# ------------
# (NIR + SWIR)

ndmi_b = safe_index(
    nir_b,
    swir_b
)

ndmi_a = safe_index(
    nir_a,
    swir_a
)


# Absolute temporal change

delta_ndvi = robust_abs_change(
    ndvi_a - ndvi_b
)

delta_ndbi = robust_abs_change(
    ndbi_a - ndbi_b
)

delta_ndmi = robust_abs_change(
    ndmi_a - ndmi_b
)


# ============================================================
# SAVE SPECTRAL CHANGE
# ============================================================

save_float_tif(
    OUT / "02_delta_NDVI.tif",
    delta_ndvi,
    ref_profile
)

save_float_tif(
    OUT / "03_delta_NDBI.tif",
    delta_ndbi,
    ref_profile
)

save_float_tif(
    OUT / "04_delta_NDMI.tif",
    delta_ndmi,
    ref_profile
)


# ============================================================
# FUSION
# ============================================================

print("\nFusing DINO + spectral evidence...")


fused = (
    W_DINO * dino_change
    +
    W_NDBI * delta_ndbi
    +
    W_NDVI * delta_ndvi
    +
    W_NDMI * delta_ndmi
)


fused = np.nan_to_num(
    fused,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

fused[~valid_mask] = np.nan


fused = percentile_normalize(
    fused,
    2,
    98
)

fused = np.nan_to_num(
    fused,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


save_float_tif(
    OUT / "05_fused_change.tif",
    fused,
    ref_profile
)


# ============================================================
# CHANGE CANDIDATES
# ============================================================

print("\nExtracting candidate change regions...")


valid_scores = fused[valid_mask]

if np.max(valid_scores) <= 0:
    threshold = np.inf
else:
    threshold = np.percentile(
        valid_scores,
        CHANGE_PERCENTILE
    )


print(
    f"Threshold "
    f"({CHANGE_PERCENTILE} percentile): "
    f"{threshold:.4f}"
)


mask = fused >= threshold
mask &= valid_mask


# Remove tiny isolated regions.

mask = morphology.remove_small_objects(
    mask,
    min_size=MIN_OBJECT_PIXELS
)


# Morphological cleanup.

mask = morphology.opening(
    mask,
    morphology.disk(2)
)

mask = morphology.closing(
    mask,
    morphology.disk(3)
)


mask_uint8 = (
    mask.astype(np.uint8)
    *
    255
)


save_uint8_tif(
    OUT / "06_change_candidates.tif",
    mask_uint8,
    ref_profile
)


Image.fromarray(
    mask_uint8
).save(
    OUT / "06_change_candidates.png"
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

print("\nFinding connected candidate regions...")


labels = measure.label(
    mask,
    connectivity=2
)


regions = measure.regionprops(
    labels,
    intensity_image=fused
)


regions = sorted(
    regions,
    key=lambda r: r.area,
    reverse=True
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUT /
    "candidate_regions.csv"
)

pixel_area_m2 = abs(
    ref_transform.a * ref_transform.e
    - ref_transform.b * ref_transform.d
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "id",
        "area_pixels",
        "min_row",
        "min_col",
        "max_row",
        "max_col",
        "centroid_row",
        "centroid_col",
        "area_m2",
        "centroid_x",
        "centroid_y",
        "mean_change",
        "max_change"
    ])


    for idx, region in enumerate(
        regions
    ):

        minr, minc, maxr, maxc = \
            region.bbox

        values = region.intensity_image[
            region.image
        ]

        centroid_y, centroid_x = xy(
            ref_transform,
            region.centroid[0],
            region.centroid[1]
        )

        writer.writerow([
            idx + 1,
            region.area,
            minr,
            minc,
            maxr,
            maxc,
            region.centroid[0],
            region.centroid[1],
            region.area * pixel_area_m2,
            centroid_x,
            centroid_y,
            float(np.mean(values)),
            float(np.max(values))
        ])


print(
    f"Saved: {csv_path}"
)

print(
    f"Detected regions: "
    f"{len(regions)}"
)


# ============================================================
# SAVE CANDIDATE CROPS
# ============================================================

print("\nSaving candidate crops...")


crop_dir = (
    OUT /
    "candidates"
)

crop_dir.mkdir(
    exist_ok=True
)


def rgb_to_uint8(arr):

    arr = np.transpose(
        arr,
        (1, 2, 0)
    )

    if arr.dtype != np.uint8:

        lo = np.percentile(
            arr,
            2
        )

        hi = np.percentile(
            arr,
            98
        )

        if hi > lo:

            arr = (
                (
                    arr.astype(
                        np.float32
                    )
                    -
                    lo
                )
                /
                (hi - lo)
                *
                255.0
            )

        arr = np.clip(
            arr,
            0,
            255
        ).astype(np.uint8)

    return arr


for idx, region in enumerate(
    regions[:MAX_CANDIDATES]
):

    minr, minc, maxr, maxc = \
        region.bbox


    # Context around candidate.

    margin = 128


    y1 = max(
        0,
        minr - margin
    )

    x1 = max(
        0,
        minc - margin
    )

    y2 = min(
        H,
        maxr + margin
    )

    x2 = min(
        W,
        maxc + margin
    )


    before_crop = \
        rgb_before[
            :,
            y1:y2,
            x1:x2
        ]


    after_crop = \
        rgb_after[
            :,
            y1:y2,
            x1:x2
        ]


    candidate_mask = \
        mask_uint8[
            y1:y2,
            x1:x2
        ]


    Image.fromarray(
        rgb_to_uint8(
            before_crop
        )
    ).save(
        crop_dir /
        f"{idx+1:03d}_before.jpg",
        quality=95
    )


    Image.fromarray(
        rgb_to_uint8(
            after_crop
        )
    ).save(
        crop_dir /
        f"{idx+1:03d}_after.jpg",
        quality=95
    )


    Image.fromarray(
        candidate_mask
    ).save(
        crop_dir /
        f"{idx+1:03d}_mask.png"
    )


print(
    f"Saved candidate crops to: "
    f"{crop_dir}"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Output directory: {OUT}"
)
