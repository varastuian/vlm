#!/usr/bin/env python3
"""
Local urban-area ("city") detection with the pretrained UNet from
https://github.com/lleviraz/urban_segmentation  (NB0 + NB3 + NB6 merged, no Colab / Drive).
python urban_local.py run --lat 35.236864 --lon 52.336017 --size-km 12 --year 2023 --gee-project "festive-folio-425310-r8"
Pipeline
  1. download : Sentinel-2 11-band 300x300 GeoTIFF tiles (3 km x 3 km each) straight to disk via Google Earth Engine
  2. prep     : add NDVI/NDTI/NDVIre/MNDWI -> 15-band tiles (same maths as NB3)
  3. predict  : UNet -> 1-band masks (0 = Other, 1 = Non-residential, 2 = Residential), mosaic, PNG overview, stats

Examples
  python urban_local.py run --city "Bend, Oregon" --size-km 9 --year 2023 --gee-project my-ee-project
  python urban_local.py run --lat 44.058 --lon -121.315 --size-km 12 --year 2023 --gee-project my-ee-project
  # already have the 11-band tiles (e.g. downloaded from Drive)?
  python urban_local.py prep    --raw my_tiles/ --out work/prep
  python urban_local.py predict --prep work/prep --out work/pred
"""
import argparse
import glob
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.merge import merge
from rasterio.warp import transform as warp_transform

# Same 11 bands, same order, as NB0 (the model was trained on exactly this order)
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
TILE_M = 3000          # 300 px * 10 m
TILE_PX = 300
CLASS_NAMES = {0: "Other", 1: "Non-residential", 2: "Residential"}
CLASS_COLORS = {0: "#00ff00", 1: "#fe0001", 2: "#0000ff"}   # same colours as the notebook
MODEL_URL = "https://github.com/lleviraz/urban_segmentation/raw/main/models/final_model.pth"


# --------------------------------------------------------------------------- #
# 1. download                                                                  #
# --------------------------------------------------------------------------- #
def geocode(city):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": city, "format": "json", "limit": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "urban-local-script/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)
    if not res:
        sys.exit(f"Could not geocode '{city}'. Use --lat/--lon instead.")
    return float(res[0]["lat"]), float(res[0]["lon"])


def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_tiles(lat, lon, size_km):
    """Grid of 3 km tiles in the local UTM zone, centred on (lat, lon)."""
    epsg = utm_epsg(lat, lon)
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    cx, cy = xs[0], ys[0]
    n = max(1, math.ceil(size_km * 1000 / TILE_M))
    x0 = round((cx - n * TILE_M / 2) / TILE_M) * TILE_M      # snap to the 3 km grid
    y0 = round((cy - n * TILE_M / 2) / TILE_M) * TILE_M
    tiles = []
    for r in range(n):
        for c in range(n):
            tiles.append((f"r{r:02d}_c{c:02d}", x0 + c * TILE_M, y0 + r * TILE_M))
    return epsg, tiles


def build_s2_image(ee, region, year, max_cloud):
    """Same recipe as NB0: S2 SR, low cloud %, cloud-probability mask (<20), /10000, mean composite."""
    start, end = f"{year}-01-01", f"{year}-12-31"
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(start, end).filterBounds(region)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)))
    clouds = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
              .filterDate(start, end).filterBounds(region))
    joined = ee.ImageCollection(ee.Join.saveFirst("cloud_mask").apply(
        s2, clouds, ee.Filter.equals(leftField="system:index", rightField="system:index")))

    def mask_clouds(img):
        prob = ee.Image(img.get("cloud_mask")).select("probability")
        return img.mask(prob.lt(20)).divide(10000)

    n = s2.size().getInfo()
    if n == 0:
        sys.exit(f"No Sentinel-2 scenes with <{max_cloud}% cloud for that area/year. "
                 f"Try --max-cloud 20 or another --year.")
    print(f"{n} Sentinel-2 scenes in the composite")
    return joined.map(mask_clouds).select(BANDS).mean().toFloat()


def download_tile(image, ee, epsg, name, x0, y0, out_dir, retries=4):
    dst = Path(out_dir) / f"s2_{name}.tif"
    if dst.exists():
        return name, "cached"
    geom = ee.Geometry.Rectangle([x0, y0, x0 + TILE_M, y0 + TILE_M],
                                 proj=f"EPSG:{epsg}", geodesic=False)
    for attempt in range(retries):
        try:
            url = image.getDownloadURL({
                "bands": BANDS, "region": geom, "dimensions": f"{TILE_PX}x{TILE_PX}",
                "crs": f"EPSG:{epsg}", "format": "GEO_TIFF"})
            data = urllib.request.urlopen(url, timeout=180).read()
            if data[:2] == b"PK":                      # sometimes delivered zipped
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    tif = next(n for n in z.namelist() if n.lower().endswith(".tif"))
                    data = z.read(tif)
            dst.write_bytes(data)
            return name, "ok"
        except Exception as e:                          # noqa: BLE001
            err = e
            time.sleep(2 ** attempt)
    return name, f"FAILED: {err}"


def cmd_download(a):
    import ee
    lat, lon = (a.lat, a.lon) if a.lat is not None else geocode(a.city)
    print(f"Centre: {lat:.5f}, {lon:.5f}")
    try:
        ee.Initialize(project=a.gee_project)
    except Exception:                                   # first run: log in via browser
        ee.Authenticate()
        ee.Initialize(project=a.gee_project)

    epsg, tiles = make_tiles(lat, lon, a.size_km)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps({"epsg": epsg, "lat": lat, "lon": lon, "tiles": tiles}))

    # bounding box of the AOI (lon/lat) just to restrict the scene search
    xs = [t[1] for t in tiles] + [t[1] + TILE_M for t in tiles]
    ys = [t[2] for t in tiles] + [t[2] + TILE_M for t in tiles]
    region = ee.Geometry.Rectangle([min(xs), min(ys), max(xs), max(ys)],
                                   proj=f"EPSG:{epsg}", geodesic=False)
    image = build_s2_image(ee, region, a.year, a.max_cloud)

    print(f"Downloading {len(tiles)} tiles to {out} ...")
    fails = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(download_tile, image, ee, epsg, n, x, y, out) for n, x, y in tiles]
        for i, f in enumerate(as_completed(futs), 1):
            name, status = f.result()
            fails += status.startswith("FAILED")
            print(f"  [{i}/{len(tiles)}] {name}: {status}")
    print(f"Done, {len(tiles) - fails} ok, {fails} failed.")


# --------------------------------------------------------------------------- #
# 2. prep (NB3 logic: 11 bands -> 15 bands)                                    #
# --------------------------------------------------------------------------- #
def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def cmd_prep(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.raw, "*.tif")))
    if not files:
        sys.exit(f"No .tif files in {a.raw}")
    skipped = 0
    for f in files:
        with rio.open(f) as src:
            x = src.read().astype("float32")
            prof = src.profile
        if x.shape[0] != 11:
            sys.exit(f"{f}: expected 11 bands, found {x.shape[0]}")
        if not np.isfinite(x[:3]).any() or np.nansum(np.abs(x[:3])) == 0:
            print(f"skip (no data / fully clouded): {Path(f).name}")
            skipped += 1
            continue
        b = lambda i: x[i - 1]                         # noqa: E731  1-based like the notebook
        extra = np.stack([nd(b(7), b(3)),               # NDVI   (B8  vs B4)
                          nd(b(10), b(11)),             # NDTI   (B11 vs B12)
                          nd(b(4), b(3)),               # NDVIre (B5  vs B4)
                          nd(b(2), b(10))])             # MNDWI  (B3  vs B11)
        prof.update(count=15, dtype="float32", nodata=0, compress="lzw")
        with rio.open(out / Path(f).name, "w", **prof) as dst:
            dst.write(np.concatenate([x, extra.astype("float32")]))
    print(f"Prepared {len(files) - skipped} tiles -> {out} ({skipped} skipped)")


# --------------------------------------------------------------------------- #
# 3. predict                                                                   #
# --------------------------------------------------------------------------- #
def build_unet():
    """UNet copied from the repo (libs/training.py); only change: dropout flag so eval mode can be deterministic."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DoubleConv(nn.Module):
        def __init__(self, i, o, mid=None):
            super().__init__()
            mid = mid or o
            self.double_conv = nn.Sequential(
                nn.Conv2d(i, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Dropout2d(0.1))

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(i, o))

        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        def __init__(self, i, o, bilinear=True):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(i, o, i // 2)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            return self.conv(torch.cat([x2, x1], dim=1))

    class OutConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv = nn.Conv2d(i, o, 1)
            self.keep_dropout = True       # original always applies dropout(0.3), even in eval()

        def forward(self, x):
            x = F.dropout2d(x, 0.3, training=self.keep_dropout)
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, n_channels=15, n_classes=3, base=64):
            super().__init__()
            f = 2
            self.inc = DoubleConv(n_channels, base)
            self.down1, self.down2 = Down(base, base * 2), Down(base * 2, base * 4)
            self.down3, self.down4 = Down(base * 4, base * 8), Down(base * 8, base * 16 // f)
            self.up1, self.up2 = Up(base * 16, base * 8 // f), Up(base * 8, base * 4 // f)
            self.up3, self.up4 = Up(base * 4, base * 2 // f), Up(base * 2, base)
            self.outc = OutConv(base, n_classes)

        def forward(self, x):
            x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
            x4 = self.down3(x3); x5 = self.down4(x4)
            x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
            return self.outc(x)

    return UNet()


def scale_bands(arr):
    """Per-image, per-band min-max to 0..1; NaNs -> band mean (what S2OnlyDataset does in the notebook)."""
    arr = arr.astype("float32").copy()
    arr[~np.isfinite(arr)] = np.nan
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        band = arr[i]
        mean = np.nanmean(band) if np.isfinite(band).any() else 0.0
        band = np.nan_to_num(band, nan=mean)
        lo, hi = band.min(), band.max()
        out[i] = (band - lo) / (hi - lo) if hi > lo else 0.0
    return out


def ensure_model(path):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model weights (~69 MB) to {p} ...")
        urllib.request.urlretrieve(MODEL_URL, p)
    return p


def cmd_predict(a):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_unet()
    sd = torch.load(ensure_model(a.model), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit(f"Checkpoint does not match architecture. missing={missing[:3]} unexpected={unexpected[:3]}")
    model.to(device)
    if a.deterministic:
        model.eval()
        model.outc.keep_dropout = False
        print("Mode: deterministic (eval, no dropout)")
    else:
        model.train()      # the notebook never calls model.eval() for the UNet; this reproduces its behaviour
        print("Mode: notebook-faithful (BatchNorm batch stats + dropout active, tiny run-to-run noise)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.prep, "*.tif")))
    if not files:
        sys.exit(f"No .tif files in {a.prep}")

    pred_files, counts = [], np.zeros(3, dtype=np.int64)
    with torch.no_grad():
        for i, f in enumerate(files, 1):
            with rio.open(f) as src:
                arr, prof = src.read(), src.profile
            x = torch.from_numpy(scale_bands(arr))[None].to(device)     # batch of 1, like the notebook
            pred = model(x).argmax(1)[0].cpu().numpy().astype("uint8")
            prof.update(count=1, dtype="uint8", nodata=None, compress="lzw")
            dst = out / ("pred_" + Path(f).name)
            with rio.open(dst, "w", **prof) as d:
                d.write(pred, 1)
            pred_files.append(dst)
            counts += np.bincount(pred.ravel(), minlength=3)[:3]
            print(f"  [{i}/{len(files)}] {Path(f).name}", end="\r")
    print()

    # ---- mosaic + overview -------------------------------------------------
    srcs = [rio.open(p) for p in pred_files]
    mosaic, tr = merge(srcs, nodata=255, dtype="uint8")
    meta = srcs[0].profile
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=tr, nodata=255, count=1)
    with rio.open(out / "city_mask_mosaic.tif", "w", **meta) as d:
        d.write(mosaic)
    for s in srcs:
        s.close()

    rgb_srcs = [rio.open(f) for f in files]
    rgb, _ = merge(rgb_srcs, indexes=[3, 2, 1], nodata=0)
    for s in rgb_srcs:
        s.close()

    stats = {CLASS_NAMES[k]: {"pixels": int(counts[k]),
                              "percent": round(100 * counts[k] / counts.sum(), 2)} for k in range(3)}
    stats["built_up_total_percent"] = round(100 * (counts[1] + counts[2]) / counts.sum(), 2)
    stats["area_km2_per_class"] = {CLASS_NAMES[k]: round(counts[k] * 100 / 1e6, 2) for k in range(3)}
    (out / "city_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from matplotlib.colors import ListedColormap

    img = np.clip(np.moveaxis(rgb, 0, -1) / 0.3, 0, 1)
    m = np.ma.masked_equal(mosaic[0], 255)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(img); ax[0].set_title("Sentinel-2 RGB")
    ax[1].imshow(img); ax[1].imshow(m, cmap=ListedColormap([CLASS_COLORS[k] for k in range(3)]),
                                    vmin=0, vmax=2, alpha=0.55, interpolation="nearest")
    ax[1].set_title("Predicted land use")
    ax[1].legend(handles=[mp.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k]) for k in range(3)],
                 loc="lower right")
    for x in ax:
        x.axis("off")
    plt.tight_layout()
    plt.savefig(out / "city_overview.png", dpi=150)
    print(f"\nSaved: {out/'city_overview.png'}, {out/'city_mask_mosaic.tif'}, {out/'city_stats.json'}")


# --------------------------------------------------------------------------- #
def cmd_run(a):
    work = Path(a.work)
    a.out = work / "raw"
    cmd_download(a)
    a.raw, a.out = work / "raw", work / "prep"
    cmd_prep(a)
    a.prep, a.out = work / "prep", work / "pred"
    cmd_predict(a)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_area(s):
        s.add_argument("--city", help='e.g. "Bend, Oregon" (geocoded with OpenStreetMap Nominatim)')
        s.add_argument("--lat", type=float); s.add_argument("--lon", type=float)
        s.add_argument("--size-km", type=float, default=9, help="side of the square area, rounded up to multiples of 3 km")
        s.add_argument("--year", type=int, default=2023)
        s.add_argument("--max-cloud", type=float, default=5, help="max CLOUDY_PIXEL_PERCENTAGE per scene (NB0 uses 5)")
        s.add_argument("--gee-project", required=True, help="your Google Cloud / Earth Engine project id")
        s.add_argument("--workers", type=int, default=4)

    def add_model(s):
        s.add_argument("--model", default="models/final_model.pth")
        s.add_argument("--deterministic", action="store_true",
                       help="eval mode without dropout (repeatable). Default reproduces the notebook exactly.")

    d = sub.add_parser("download"); add_area(d); d.add_argument("--out", default="work/raw"); d.set_defaults(f=cmd_download)
    q = sub.add_parser("prep"); q.add_argument("--raw", required=True); q.add_argument("--out", default="work/prep"); q.set_defaults(f=cmd_prep)
    r = sub.add_parser("predict"); r.add_argument("--prep", required=True); r.add_argument("--out", default="work/pred"); add_model(r); r.set_defaults(f=cmd_predict)
    u = sub.add_parser("run"); add_area(u); add_model(u); u.add_argument("--work", default="work"); u.set_defaults(f=cmd_run)

    a = p.parse_args()
    if a.cmd in ("download", "run") and a.lat is None and not a.city:
        p.error("give --city or --lat/--lon")
    a.f(a)


if __name__ == "__main__":
    main()
