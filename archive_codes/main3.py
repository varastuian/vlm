import os
import csv
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
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
OUT = ROOT / "phase456_output_png"

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

# ---------------- Area of interest ----------------
# Bottom-center quarter of the source scene by default.
AOI_WIDTH_FRACTION = 0.25
AOI_HEIGHT_FRACTION = 0.25

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


def save_change_png(path, array):

    """Save a change score as a grayscale PNG for visual inspection."""

    image = (
        percentile_normalize(array, 2, 98) * 255.0
    ).astype(np.uint8)

    save_png_image(
        Image.fromarray(image, mode="L"),
        path
    )
    print(f"Saved: {path}")


def save_mask_png(path, array):

    save_png_image(
        Image.fromarray(
            np.asarray(array, dtype=np.uint8),
            mode="L"
        ),
        path
    )

    print(f"Saved: {path}")


def save_png_image(image, path):

    """Write PNG explicitly and retry with a safe alternate filename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        image.save(str(path), format="PNG")
    except OSError as error:
        retry_path = path.with_name(
            f"{path.stem}_retry{path.suffix}"
        )
        print(
            f"PNG write failed for {path}: {error}. "
            f"Retrying as {retry_path}."
        )
        image.save(str(retry_path), format="PNG")


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

    """Create temporary DINO inputs; source RGB arrays stay untouched."""

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


def choose_png_rgb_scale(*arrays):

    """Choose one fixed display scale for the complete image pair."""

    values = np.concatenate([
        np.asarray(array, dtype=np.float32).ravel()
        for array in arrays
    ])
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 1.0

    p99 = np.percentile(values, 99)

    if p99 <= 1.0:
        return 1.0
    if p99 <= 255.0:
        return 255.0
    if p99 <= 10000.0:
        return 10000.0
    return 65535.0


def rgb_to_png(rgb, scale):

    """Convert RGB to 8-bit PNG using a fixed, non-local scale."""

    image = np.transpose(
        rgb,
        (1, 2, 0)
    ).astype(np.float32)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=scale,
        neginf=0.0
    )

    return np.clip(
        image / scale * 255.0,
        0.0,
        255.0
    ).round().astype(np.uint8)


# ============================================================
# RESAMPLE BAND TO REFERENCE GRID
# ============================================================

def read_band_to_reference(path, reference_path, window=None):

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

        if window is None:
            dst_height = ref.height
            dst_width = ref.width
            dst_transform = ref.transform
        else:
            dst_height = int(window.height)
            dst_width = int(window.width)
            dst_transform = window_transform(
                window,
                ref.transform
            )

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

rgb_after_raw, profile_after_raw, transform_after_raw, crs_after_raw = \
    read_rgb(RGB_AFTER)

rgb_after_aligned, profile_after, transform_after, crs_after = \
    read_rgb_to_reference(RGB_AFTER, RGB_BEFORE)


if rgb_before.shape != rgb_after_aligned.shape:

    raise ValueError(
        "Before/after RGB dimensions differ:\n"
        f"Before: {rgb_before.shape}\n"
        f"After : {rgb_after_aligned.shape}"
    )

same_after_grid = (
    rgb_after_raw.shape == rgb_before.shape
    and transform_after_raw == ref_transform
    and crs_after_raw == ref_crs
)

full_height = rgb_before.shape[1]
full_width = rgb_before.shape[2]

aoi_width = max(
    1,
    int(round(full_width * AOI_WIDTH_FRACTION))
)
aoi_height = max(
    1,
    int(round(full_height * AOI_HEIGHT_FRACTION))
)

aoi_x1 = (full_width - aoi_width) // 2
aoi_y1 = full_height - aoi_height
aoi_x2 = aoi_x1 + aoi_width
aoi_y2 = aoi_y1 + aoi_height

print(
    "\nUsing bottom-center AOI: "
    f"x={aoi_x1}:{aoi_x2}, "
    f"y={aoi_y1}:{aoi_y2} "
    f"({aoi_width} x {aoi_height} pixels)"
)

rgb_before = rgb_before[:, aoi_y1:aoi_y2, aoi_x1:aoi_x2]
rgb_after_aligned = rgb_after_aligned[
    :, aoi_y1:aoi_y2, aoi_x1:aoi_x2
]

if same_after_grid:
    rgb_after_raw = rgb_after_raw[
        :, aoi_y1:aoi_y2, aoi_x1:aoi_x2
    ]

ref_transform = window_transform(
    Window(aoi_x1, aoi_y1, aoi_width, aoi_height),
    ref_transform
)

aoi_window = Window(
    aoi_x1,
    aoi_y1,
    aoi_width,
    aoi_height
)

rgb_valid_before = np.all(np.isfinite(rgb_before), axis=0)
rgb_valid_after = np.all(np.isfinite(rgb_after_aligned), axis=0)

print("\nPreparing temporary DINO inputs...")

dino_rgb_before, dino_rgb_after = normalize_rgb_pair(
    rgb_before,
    rgb_after_aligned
)
print(
    "DINO BEFORE:",
    dino_rgb_before.dtype,
    dino_rgb_before.min(),
    dino_rgb_before.max()
)

print(
    "DINO AFTER:",
    dino_rgb_after.dtype,
    dino_rgb_after.min(),
    dino_rgb_after.max()
)

if same_after_grid:
    rgb_after_output = rgb_after_raw
else:
    print(
        "Warning: AFTER RGB crop output uses the aligned grid because "
        "the source grids differ."
    )
    rgb_after_output = rgb_after_aligned

png_rgb_scale = choose_png_rgb_scale(
    rgb_before,
    rgb_after_output
)

print(
    f"PNG RGB scale: 0-{png_rgb_scale:g} -> 0-255 "
    "(fixed for all crops)"
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
            dino_rgb_before,
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
            dino_rgb_after,
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


save_change_png(
    OUT / "01_dino_change.png",
    dino_change
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
        RGB_BEFORE,
        aoi_window
    )
)

red_a = normalize_reflectance(
    read_band_to_reference(
        RED_AFTER,
        RGB_BEFORE,
        aoi_window
    )
)


# ------------------------------------------------------------
# NIR
# ------------------------------------------------------------

nir_b = normalize_reflectance(
    read_band_to_reference(
        NIR_BEFORE,
        RGB_BEFORE,
        aoi_window
    )
)

nir_a = normalize_reflectance(
    read_band_to_reference(
        NIR_AFTER,
        RGB_BEFORE,
        aoi_window
    )
)


# ------------------------------------------------------------
# SWIR
# ------------------------------------------------------------

swir_b = normalize_reflectance(
    read_band_to_reference(
        SWIR_BEFORE,
        RGB_BEFORE,
        aoi_window
    )
)

swir_a = normalize_reflectance(
    read_band_to_reference(
        SWIR_AFTER,
        RGB_BEFORE,
        aoi_window
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


if expected_shape != (aoi_height, aoi_width):
    raise RuntimeError(
        "Spectral AOI grid does not match the RGB AOI: "
        f"{expected_shape} != {(aoi_height, aoi_width)}"
    )

print(
    f"Spectral AOI shape: {nir_b.shape}"
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

save_change_png(
    OUT / "02_delta_NDVI.png",
    delta_ndvi
)

save_change_png(
    OUT / "03_delta_NDBI.png",
    delta_ndbi
)

save_change_png(
    OUT / "04_delta_NDMI.png",
    delta_ndmi
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


save_change_png(
    OUT / "05_fused_change.png",
    fused
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


save_mask_png(
    OUT / "06_change_candidates.png",
    mask_uint8
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


def save_rgb_png_crop(
    path,
    rgb,
    x1,
    y1,
    x2,
    y2,
    scale
):

    """Save a consistently scaled RGB crop as PNG."""

    crop = rgb[:, y1:y2, x1:x2]
    save_png_image(
        Image.fromarray(
            rgb_to_png(crop, scale),
            mode="RGB"
        ),
        path
    )


def save_change_overlay(path, rgb, mask, x1, y1, x2, y2, scale):

    """Save the RGB crop with detected changes highlighted in red."""

    base = Image.fromarray(
        rgb_to_png(rgb[:, y1:y2, x1:x2], scale),
        mode="RGB"
    ).convert("RGBA")

    alpha = np.where(
        mask[y1:y2, x1:x2] > 0,
        180,
        0
    ).astype(np.uint8)

    overlay = np.zeros(
        (alpha.shape[0], alpha.shape[1], 4),
        dtype=np.uint8
    )
    overlay[..., 0] = 255
    overlay[..., 1] = 40
    overlay[..., 2] = 40
    overlay[..., 3] = alpha

    result = Image.alpha_composite(
        base,
        Image.fromarray(overlay, mode="RGBA")
    ).convert("RGB")

    save_png_image(result, path)


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

    if x2 <= x1 or y2 <= y1:
        print(
            f"Skipping empty candidate crop {idx + 1}: "
            f"x={x1}:{x2}, y={y1}:{y2}"
        )
        continue

    candidate_mask = \
        mask_uint8[
            y1:y2,
            x1:x2
        ]


    save_rgb_png_crop(
        crop_dir /
        f"{idx+1:03d}_before.png",
        rgb_before,
        x1,
        y1,
        x2,
        y2,
        png_rgb_scale
    )


    save_rgb_png_crop(
        crop_dir /
        f"{idx+1:03d}_after.png",
        rgb_after_output,
        x1,
        y1,
        x2,
        y2,
        png_rgb_scale
    )


    save_change_overlay(
        crop_dir /
        f"{idx+1:03d}_changes.png",
        rgb_after_output,
        mask_uint8,
        x1,
        y1,
        x2,
        y2,
        png_rgb_scale
    )


    save_png_image(
        Image.fromarray(candidate_mask),
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
