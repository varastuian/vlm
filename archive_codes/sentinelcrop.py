
from pathlib import Path
import json
import sys
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

# ---------------- CONFIG ----------------
INPUT_DIR = Path("/media/varas/Data/Code/paper/remoteSensingChangeDetectionVLM/data")
OUTPUT_DIR = Path("/media/varas/Data/Code/paper/remoteSensingChangeDetectionVLM/data/aoi_products")

BEFORE_IMAGE = INPUT_DIR / "S2A_39SXV_20200128_1_L2A_visual.tif"

# Full-image permanent preview size
PREVIEW_MAX_SIZE = 2048

# AOI crop quicklook size
QUICKLOOK_MAX_SIZE = 2048

# For your [Red, Green, NIR, SWIR1] TIFF, display NIR/Red/Green.
FALSE_COLOR_4_BAND = True
# ----------------------------------------


def stretch(data):
    out = np.zeros_like(data, dtype=np.uint8)
    for i in range(data.shape[0]):
        b = data[i].astype(np.float32)
        valid = np.isfinite(b)
        if not valid.any():
            continue
        lo, hi = np.percentile(b[valid], [2, 98])
        if hi <= lo:
            out[i] = np.clip(b, 0, 255).astype(np.uint8)
        else:
            out[i] = np.clip((b - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return out


def visual_rgb(path, max_size=2048):
    with rasterio.open(path) as src:
        scale = min(1.0, max_size / max(src.width, src.height))
        h = max(1, int(src.height * scale))
        w = max(1, int(src.width * scale))
        data = src.read(
            out_shape=(src.count, h, w),
            resampling=rasterio.enums.Resampling.bilinear,
            out_dtype="float32",
        )

    if data.shape[0] >= 4 and FALSE_COLOR_4_BAND:
        # [Red, Green, NIR, SWIR1] -> [NIR, Red, Green]
        data = data[[2, 0, 1]]
    elif data.shape[0] >= 3:
        data = data[:3]
    else:
        data = np.repeat(data[:1], 3, axis=0)

    return np.moveaxis(stretch(data), 0, -1)


def draw_aoi(image):
    result = {"coords": None}

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image)
    ax.set_title(
        "Draw AOI with the LEFT mouse button.\n"
        "Release to finish, then close this window."
    )
    ax.set_axis_off()

    def selected(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        result["coords"] = (x1, y1, x2, y2)
        print(f"Selected display AOI: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")

    selector = RectangleSelector(
    ax,
    selected,
    useblit=False,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=True,
    props=dict(
        facecolor="red",
        edgecolor="yellow",
        alpha=0.25,
        fill=True,
        linewidth=2,
    ),
)

    # IMPORTANT: keep the selector alive
    fig._aoi_selector = selector

    fig.canvas.draw_idle()
    plt.show()
    return result["coords"]


def display_to_geo(path, coords, display_w, display_h):
    x1, y1, x2, y2 = coords
    with rasterio.open(path) as src:
        sx1 = x1 * src.width / display_w
        sx2 = x2 * src.width / display_w
        sy1 = y1 * src.height / display_h
        sy2 = y2 * src.height / display_h

        # Pixel coordinates -> source CRS coordinates
        corners = [
            src.transform * (sx1, sy1),
            src.transform * (sx2, sy1),
            src.transform * (sx2, sy2),
            src.transform * (sx1, sy2),
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return (min(xs), min(ys), max(xs), max(ys)), src.crs


def save_preview(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = visual_rgb(src_path, PREVIEW_MAX_SIZE)
    Image.fromarray(rgb).save(dst_path, quality=92)


def crop_geotiff(src_path, dst_path, aoi_bounds, aoi_crs):
    with rasterio.open(src_path) as src:
        if src.crs is None:
            print(f"SKIP {src_path.name}: no CRS")
            return False

        bounds = aoi_bounds
        if src.crs != aoi_crs:
            bounds = transform_bounds(aoi_crs, src.crs, *aoi_bounds, densify_pts=21)

        left = max(bounds[0], src.bounds.left)
        bottom = max(bounds[1], src.bounds.bottom)
        right = min(bounds[2], src.bounds.right)
        top = min(bounds[3], src.bounds.top)

        if left >= right or bottom >= top:
            print(f"SKIP {src_path.name}: AOI does not overlap")
            return False

        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(window=window)

        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=src.window_transform(window),
            driver="GTiff",
            compress="deflate",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

        return True


def main():
    if not INPUT_DIR.exists():
        sys.exit(f"Input directory does not exist: {INPUT_DIR}")
    if not BEFORE_IMAGE.exists():
        sys.exit(f"BEFORE_IMAGE does not exist: {BEFORE_IMAGE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previews = OUTPUT_DIR / "previews"
    crops = OUTPUT_DIR / "crops"
    quicklooks = OUTPUT_DIR / "quicklooks"
    previews.mkdir(exist_ok=True)
    crops.mkdir(exist_ok=True)
    quicklooks.mkdir(exist_ok=True)

    tif_files = sorted(list(INPUT_DIR.rglob("*.tif")) + list(INPUT_DIR.rglob("*.tiff")))
    print(f"Found {len(tif_files)} GeoTIFF products.")

    # 1. Show BEFORE image and draw AOI
    sample = visual_rgb(BEFORE_IMAGE, PREVIEW_MAX_SIZE)
    coords = draw_aoi(sample)
    if coords is None:
        sys.exit("No AOI selected.")

    h, w = sample.shape[:2]
    aoi_bounds, aoi_crs = display_to_geo(BEFORE_IMAGE, coords, w, h)

    print("\nAOI:")
    print("CRS:", aoi_crs)
    print("Bounds:", aoi_bounds)

    # Save reusable AOI metadata
    with open(OUTPUT_DIR / "aoi.json", "w") as f:
        json.dump(
            {"crs": str(aoi_crs),
             "bounds": dict(zip(["left", "bottom", "right", "top"], aoi_bounds))},
            f, indent=2
        )

    # 2. Permanent small full-image previews
    for src in tif_files:
        rel = src.relative_to(INPUT_DIR).with_suffix(".jpg")
        try:
            save_preview(src, previews / rel)
            print("Preview:", src.name)
        except Exception as e:
            print("Preview ERROR:", src.name, e)

    # 3. AOI crop of every product
    for src in tif_files:
        rel = src.relative_to(INPUT_DIR)
        crop_path = crops / rel
        try:
            if crop_geotiff(src, crop_path, aoi_bounds, aoi_crs):
                # Visual quicklook of crop
                rgb = visual_rgb(crop_path, QUICKLOOK_MAX_SIZE)
                ql = quicklooks / rel.with_suffix(".jpg")
                ql.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(ql, quality=92)
                print("Crop:", src.name)
        except Exception as e:
            print("Crop ERROR:", src.name, e)

    print("\nDONE")
    print("Output:", OUTPUT_DIR)
    print("  previews/   = small versions of all raw images")
    print("  crops/      = georeferenced AOI GeoTIFFs")
    print("  quicklooks/ = small visual versions of AOI crops")
    print("  aoi.json    = reusable AOI coordinates")
    print("Raw files were not modified.")


if __name__ == "__main__":
    main()
