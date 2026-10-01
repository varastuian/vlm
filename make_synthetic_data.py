"""Synthetic Sentinel-2-like pair (same file naming as the real data) with planted
true changes AND common false-positive traps. Returns ground-truth masks."""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage as ndi

SIZE = 900
SOIL = np.array([.12, .16, .22, .30])      # green, red, nir, swir16
VEG = np.array([.06, .04, .35, .18])
ROOF = np.array([.30, .31, .32, .36])
ASPH = np.array([.07, .07, .08, .09])


def _field(rng, size):
    base = np.zeros((size, size, 4), np.float32) + SOIL
    def unit(x): return (x - x.mean()) / x.std()
    veg = unit(ndi.gaussian_filter(rng.standard_normal((size, size)), 25)) > 0.4
    base[veg] = VEG
    tex = unit(ndi.gaussian_filter(rng.standard_normal((size, size)), 3)) * 0.12 + 1.0
    return base * tex[..., None], veg


def make(out_dir, seed=0):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ref, veg = _field(rng, SIZE)
    a = ref.copy()
    truth = np.zeros((SIZE, SIZE), bool)

    def paint(y, x, h, w, spec):
        a[y:y + h, x:x + w] = spec
        truth[y:y + h, x:x + w] = True

    paint(100, 100, 5, 5, ROOF); paint(300, 500, 4, 6, ROOF)          # buildings
    paint(600, 200, 2, 90, ASPH)                                      # new road (2 px wide)
    a[400:430, 300:330] = SOIL; truth[400:430, 300:330] = True        # field -> bare (veg loss)
    ref[400:430, 300:330] = VEG                                       # (make sure it WAS vegetated)
    ref[700:725, 600:625] = SOIL; a[700:725, 600:625] = VEG; truth[700:725, 600:625] = True  # veg gain

    # false-positive traps applied to the 'after' image only
    a[200:260, 650:710] *= 0.78                                        # soil-moisture darkening
    cloud = np.zeros((SIZE, SIZE), bool); cloud[40:80, 700:750] = True
    a[cloud] = [.5, .5, .5, .4]
    a = ndi.shift(a, (0.5, 0.4, 0), order=1, mode="reflect")           # co-registration error
    a = a * 1.06 + 0.005                                               # illumination / atmosphere
    for _ in range(40):                                                # salt speckle
        y, x = rng.integers(0, SIZE, 2); a[y, x] *= 1.8

    def noisy(x): return np.clip(x + rng.normal(0, 0.008, x.shape), 0.001, 1)
    ref, a = noisy(ref), noisy(a)

    tf = from_origin(500000, 4000000, 10, -10)
    def write(path, arr, dtype, res=10):
        h, w = arr.shape[-2:]
        arr3 = arr if arr.ndim == 3 else arr[None]
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=arr3.shape[0],
                           dtype=dtype, crs="EPSG:32639",
                           transform=from_origin(500000, 4000000, res, res)) as dst:
            dst.write(arr3.astype(dtype))

    for tag, img, offset in (("S2A_39SXV_20200128_1_L2A", ref, 0), ("S2B_39SXV_20260128_0_L2A", a, 1000)):
        for i, b in enumerate(["green", "red", "nir", "swir16"]):
            band = img[..., i]
            if b == "swir16":       # 20 m band like the real product
                band = band.reshape(SIZE // 2, 2, SIZE // 2, 2).mean((1, 3))
                write(out_dir / f"{tag}_{b}.tif", band * 10000 + offset, "uint16", res=20)
            else:
                write(out_dir / f"{tag}_{b}.tif", band * 10000 + offset, "uint16")
        rgb = np.stack([img[..., 1], img[..., 0], img[..., 0] * .8], 0)  # fake TCI
        write(out_dir / f"{tag}_visual.tif", np.clip(rgb * 255 * 3, 0, 255), "uint8")
    return truth & ~cloud


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
