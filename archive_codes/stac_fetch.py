#!/usr/bin/env python3
"""
stac_fetch.py — fetch a Sentinel-2 crop for a lat/lon + date range from a
public STAC catalog, with local caching so repeat queries for the same
location/date-range/area don't hit the network again.

Uses Element84's Earth Search STAC API (public, no API key / signing
required) against the sentinel-2-l2a collection's "visual" (true-color)
asset.

NOTE: I could not test live network calls to the STAC API from this sandbox
(its egress allowlist doesn't include earth-search.aws.element84.com) — the
windowed-read math (lat/lon -> pixel window) was validated separately
against a synthetic GeoTIFF. Run this on your own machine, which has normal
internet access.

Usage as a library:
    from stac_fetch import fetch_scene
    img, meta = fetch_scene(lat=40.6, lon=15.05,
                             date_start="2020-01-01", date_end="2020-03-01")

Usage from the CLI (fetch + save one crop):
    python stac_fetch.py --lat 40.6 --lon 15.05 \
        --date-start 2020-01-01 --date-end 2020-03-01 --out scene.png
"""

import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"

# Earth Search v1 sentinel-2-l2a asset keys for the bands needed to compute
# NDVI/NDBI/NDWI. red/green/nir are native 10m; swir16 is native 20m and
# gets resampled up to match during the windowed read.
INDEX_BAND_ASSETS = {"red": "red", "green": "green", "nir": "nir", "swir16": "swir16"}


def _cache_key(lat, lon, date_start, date_end, buffer_km, collection, max_cloud):
    raw = (f"{round(lat, 5)}_{round(lon, 5)}_{date_start}_{date_end}_"
           f"{buffer_km}_{collection}_{max_cloud}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _download_with_progress(url, dest_path, chunk_size=1024 * 1024):
    """Stream url to dest_path, printing a progress bar. Downloads to a
    .part file first and renames on success, so a crashed/interrupted
    download is never mistaken for a complete cached file."""
    part_path = dest_path + ".part"
    t0 = time.time()
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.time() - t0, 1e-6)
                speed_mb_s = (downloaded / (1024 * 1024)) / elapsed
                if total:
                    pct = 100 * downloaded / total
                    bar_len = 30
                    filled = int(bar_len * downloaded / total)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(
                        f"\r      [{bar}] {pct:5.1f}%  "
                        f"{downloaded/1024/1024:6.1f}/{total/1024/1024:6.1f} MB  "
                        f"{speed_mb_s:5.1f} MB/s"
                    )
                else:
                    sys.stdout.write(
                        f"\r      downloaded {downloaded/1024/1024:6.1f} MB  {speed_mb_s:5.1f} MB/s"
                    )
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(part_path, dest_path)


def _get_raw_asset(item, asset_key, cache_dir):
    """
    Download (with progress) the full asset for a STAC item, or reuse it if
    already cached. Cached per (item.id, asset_key) rather than per query,
    so re-cropping the same scene at a different --buffer-km, or fetching a
    different index band for a scene you already pulled the visual asset
    for, doesn't re-download anything.
    """
    raw_dir = os.path.join(cache_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"{item.id}_{asset_key}.tif")

    if os.path.isfile(raw_path):
        print(f"      [raw cache hit] {asset_key} for {item.id} already on disk")
        return raw_path

    href = item.assets[asset_key].href
    print(f"      downloading {asset_key} asset for {item.id}...")
    _download_with_progress(href, raw_path)
    return raw_path


def fetch_scene(lat, lon, date_start, date_end, buffer_km=1.0,
                cache_dir="stac_cache", collection=DEFAULT_COLLECTION,
                stac_url=DEFAULT_STAC_URL, max_cloud=30, asset_key="visual"):
    """
    Returns (image_bgr, meta_dict). On a repeat call with the same
    (lat, lon, date_start, date_end, buffer_km, collection, max_cloud), this
    reads straight from disk cache and does no network I/O at all.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(lat, lon, date_start, date_end, buffer_km, collection, max_cloud)
    png_path = os.path.join(cache_dir, f"{key}.png")
    meta_path = os.path.join(cache_dir, f"{key}.json")

    if os.path.isfile(png_path) and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[cache hit] {meta['item_id']} ({meta['datetime']}, "
              f"cloud={meta['cloud_cover']}%) — no download, reading from disk")
        return cv2.imread(png_path), meta

    t0 = time.time()
    print(f"[1/4] searching {collection} for {date_start}..{date_end} near ({lat}, {lon})...")

    import pystac_client  # imported lazily so the rest of the app works without it

    catalog = pystac_client.Client.open(stac_url)
    search = catalog.search(
        collections=[collection],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=1,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No {collection} scenes found for {date_start}..{date_end} near "
            f"({lat}, {lon}) with cloud cover < {max_cloud}%. Try widening the "
            f"date range or raising --max-cloud."
        )
    item = items[0]
    print(f"[2/4] found {item.id} (cloud={item.properties.get('eo:cloud_cover')}%) "
          f"in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("[3/4] fetching asset (cached per-scene, so a different --buffer-km on "
          "this same scene later won't re-download)...")
    raw_path = _get_raw_asset(item, asset_key, cache_dir)

    with rasterio.open(raw_path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        res_m = src.res[0]  # meters/pixel (10m for Sentinel-2 visual)
        half_px = int((buffer_km * 1000) / res_m / 2)
        window = Window(col - half_px, row - half_px, half_px * 2, half_px * 2)
        window = window.intersection(Window(0, 0, src.width, src.height))
        arr = src.read([1, 2, 3], window=window)
        img_rgb = np.transpose(arr, (1, 2, 0))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    print(f"      cropped {int(window.width)}x{int(window.height)}px window "
          f"in {time.time() - t1:.1f}s")

    cv2.imwrite(png_path, img_bgr)
    meta = {
        "item_id": item.id,
        "datetime": str(item.datetime),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "lat": lat,
        "lon": lon,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[4/4] cached crop at {png_path} (total {time.time() - t0:.1f}s)")
    return img_bgr, meta


def _compute_indices(bands):
    """bands: dict of float32 reflectance arrays (red, green, nir, swir16),
    all same shape. Returns NDVI, NDBI, NDWI — the standard normalized-
    difference indices used for vegetation, built-up area, and water."""
    eps = 1e-6
    red, green, nir, swir16 = bands["red"], bands["green"], bands["nir"], bands["swir16"]
    ndvi = (nir - red) / (nir + red + eps)        # vegetation: high = healthy vegetation
    ndbi = (swir16 - nir) / (swir16 + nir + eps)  # built-up: high = built-up/bare surface
    ndwi = (green - nir) / (green + nir + eps)    # water: high = water
    return {"ndvi": ndvi, "ndbi": ndbi, "ndwi": ndwi}


def fetch_scene_with_indices(lat, lon, date_start, date_end, buffer_km=1.0,
                              cache_dir="stac_cache", collection=DEFAULT_COLLECTION,
                              stac_url=DEFAULT_STAC_URL, max_cloud=30):
    """
    Like fetch_scene, but also reads the red/green/nir/swir16 bands and
    computes NDVI/NDBI/NDWI for the same crop. Cached as a single .npz
    (visual RGB + indices) so a repeat query for the same (lat, lon, date
    range, area) is instant with no network calls.

    Returns (visual_bgr, indices_dict, meta_dict) where indices_dict has
    keys "ndvi", "ndbi", "ndwi", each a float32 array the same H x W as
    visual_bgr.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(lat, lon, date_start, date_end, buffer_km, collection, max_cloud) + "_idx"
    npz_path = os.path.join(cache_dir, f"{key}.npz")
    meta_path = os.path.join(cache_dir, f"{key}.json")

    if os.path.isfile(npz_path) and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[cache hit] {meta['item_id']} ({meta['datetime']}, "
              f"cloud={meta['cloud_cover']}%) — no download, reading from disk")
        data = np.load(npz_path)
        indices = {"ndvi": data["ndvi"], "ndbi": data["ndbi"], "ndwi": data["ndwi"]}
        return data["visual"], indices, meta

    t0 = time.time()
    print(f"[1/4] searching {collection} for {date_start}..{date_end} "
          f"near ({lat}, {lon}) (fetching bands for indices)...")

    import pystac_client

    catalog = pystac_client.Client.open(stac_url)
    search = catalog.search(
        collections=[collection],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=1,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No {collection} scenes found for {date_start}..{date_end} near "
            f"({lat}, {lon}) with cloud cover < {max_cloud}%. Try widening the "
            f"date range or raising --max-cloud."
        )
    item = items[0]
    print(f"[2/4] found {item.id} (cloud={item.properties.get('eo:cloud_cover')}%) "
          f"in {time.time() - t0:.1f}s")

    t1 = time.time()
    n_assets = 1 + len(INDEX_BAND_ASSETS)
    print(f"[3/4] fetching {n_assets} assets (visual + {', '.join(INDEX_BAND_ASSETS)}) — "
          f"each is cached per-scene, so re-cropping this same scene later won't re-download...")

    # Window against the 10m "visual" asset first, so every other band
    # (including native-20m swir16) gets resampled to match this exact shape.
    visual_path = _get_raw_asset(item, "visual", cache_dir)
    with rasterio.open(visual_path) as vsrc:
        xs, ys = warp_transform("EPSG:4326", vsrc.crs, [lon], [lat])
        row, col = vsrc.index(xs[0], ys[0])
        res_m = vsrc.res[0]
        half_px = int((buffer_km * 1000) / res_m / 2)
        window = Window(col - half_px, row - half_px, half_px * 2, half_px * 2)
        window = window.intersection(Window(0, 0, vsrc.width, vsrc.height))
        out_shape = (int(window.height), int(window.width))
        arr = vsrc.read([1, 2, 3], window=window)
        visual = cv2.cvtColor(np.transpose(arr, (1, 2, 0)), cv2.COLOR_RGB2BGR)

    bands = {}
    for name, asset_key in INDEX_BAND_ASSETS.items():
        band_path = _get_raw_asset(item, asset_key, cache_dir)
        with rasterio.open(band_path) as bsrc:
            bxs, bys = warp_transform("EPSG:4326", bsrc.crs, [lon], [lat])
            brow, bcol = bsrc.index(bxs[0], bys[0])
            bres = bsrc.res[0]
            bhalf_px = int((buffer_km * 1000) / bres / 2)
            bwindow = Window(bcol - bhalf_px, brow - bhalf_px, bhalf_px * 2, bhalf_px * 2)
            bwindow = bwindow.intersection(Window(0, 0, bsrc.width, bsrc.height))
            raw = bsrc.read(1, window=bwindow, out_shape=out_shape,
                             resampling=Resampling.bilinear)
            bands[name] = raw.astype(np.float32) / 10000.0  # L2A reflectance scale factor
    print(f"      all assets ready in {time.time() - t1:.1f}s")

    indices = _compute_indices(bands)

    np.savez_compressed(npz_path, visual=visual, **indices)
    meta = {
        "item_id": item.id,
        "datetime": str(item.datetime),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "lat": lat,
        "lon": lon,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[4/4] cached crop + indices at {npz_path} (total {time.time() - t0:.1f}s)")
    return visual, indices, meta


def fetch_before_after_with_indices(lat, lon, before_start, before_end,
                                     after_start, after_end, buffer_km=1.0,
                                     cache_dir="stac_cache", collection=DEFAULT_COLLECTION,
                                     stac_url=DEFAULT_STAC_URL, max_cloud=30):
    """Convenience wrapper: fetch (and cache) a before/after pair, each with
    its visual RGB crop and NDVI/NDBI/NDWI indices, in one call."""
    b_visual, b_idx, b_meta = fetch_scene_with_indices(
        lat, lon, before_start, before_end, buffer_km, cache_dir,
        collection, stac_url, max_cloud,
    )
    a_visual, a_idx, a_meta = fetch_scene_with_indices(
        lat, lon, after_start, after_end, buffer_km, cache_dir,
        collection, stac_url, max_cloud,
    )
    return b_visual, a_visual, b_idx, a_idx, b_meta, a_meta


def fetch_before_after(lat, lon, before_start, before_end, after_start, after_end,
                        buffer_km=1.0, cache_dir="stac_cache",
                        collection=DEFAULT_COLLECTION, stac_url=DEFAULT_STAC_URL,
                        max_cloud=30, asset_key="visual"):
    """Convenience wrapper: fetch (and cache) a before/after pair in one call."""
    before_img, before_meta = fetch_scene(
        lat, lon, before_start, before_end, buffer_km, cache_dir,
        collection, stac_url, max_cloud, asset_key,
    )
    after_img, after_meta = fetch_scene(
        lat, lon, after_start, after_end, buffer_km, cache_dir,
        collection, stac_url, max_cloud, asset_key,
    )
    return before_img, after_img, before_meta, after_meta


def main():
    ap = argparse.ArgumentParser(description="Fetch a cached Sentinel-2 crop via STAC")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--date-start", required=True)
    ap.add_argument("--date-end", required=True)
    ap.add_argument("--buffer-km", type=float, default=1.0)
    ap.add_argument("--cache-dir", default="stac_cache")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--stac-url", default=DEFAULT_STAC_URL)
    ap.add_argument("--max-cloud", type=float, default=30)
    ap.add_argument("--out", default="scene.png")
    args = ap.parse_args()

    img, meta = fetch_scene(
        args.lat, args.lon, args.date_start, args.date_end, args.buffer_km,
        args.cache_dir, args.collection, args.stac_url, args.max_cloud,
    )
    cv2.imwrite(args.out, img)
    print(f"Saved -> {args.out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()