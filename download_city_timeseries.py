#!/usr/bin/env python3
"""Download aligned annual Sentinel-2 composites for city-growth analysis.

Example (defaults: 10 x 10 km, 2020--2025, June--September):
  python download_city_timeseries.py --gee-project YOUR_PROJECT
  python download_city_timeseries.py --dry-run

Outputs per year: reflectance.tif (11 bands, unit reflectance, nodata=-9999),
clear_observations.tif (0 means no clear data), RGB preview and provenance JSON.
These are seasonal composites, not individual acquisition-date images.
Dependencies: earthengine-api, numpy, rasterio, matplotlib.
"""

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np
import rasterio as rio
from rasterio.transform import Affine
from rasterio.warp import transform
from rasterio.windows import Window

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
SOURCE = 'COPERNICUS/S2_SR_HARMONIZED'
QA_SOURCE = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
NODATA = -9999


def make_grid(lat, lon, size_km):
    zone = min(60, int((lon + 180) // 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + zone
    x, y = transform('EPSG:4326', f'EPSG:{epsg}', [lon], [lat])
    pixels = round(size_km * 1000 / 10)
    if pixels < 1 or not math.isclose(pixels * 10, size_km * 1000, abs_tol=1e-6):
        raise ValueError('--size-km must be a positive multiple of 0.01 km')
    left = round((x[0] - pixels * 5) / 10) * 10
    top = round((y[0] + pixels * 5) / 10) * 10
    return {'epsg': epsg, 'width': pixels, 'height': pixels,
            'transform': [10, 0, left, 0, -10, top],
            'bounds': [left, top - pixels * 10, left + pixels * 10, top]}


def season(year, first, last):
    start = dt.date(year, first, 1)
    end = dt.date(year + 1, 1, 1) if last == 12 else dt.date(year, last + 1, 1)
    return start.isoformat(), end.isoformat()


def build_composite(ee, region, year, args):
    start, end = season(year, args.start_month, args.end_month)
    scenes = (ee.ImageCollection(SOURCE).filterBounds(region).filterDate(start, end)
              .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', args.max_cloud)))
    # An inner join excludes scenes without QA rather than silently accepting them.
    pairs = ee.Join.inner().apply(
        scenes, ee.ImageCollection(QA_SOURCE).filterBounds(region).filterDate(start, end),
        ee.Filter.equals(leftField='system:index', rightField='system:index'))

    def prepare(pair):
        pair = ee.Feature(pair)
        image = ee.Image(pair.get('primary'))
        score = ee.Image(pair.get('secondary')).select('cs_cdf')
        scl = image.select('SCL')
        valid = score.gte(args.clear_threshold)
        # Reject no-data, defective pixels, cloud shadows, clouds/cirrus and snow.
        for label in [0, 1, 3, 8, 9, 10, 11]:
            valid = valid.And(scl.neq(label))
        bands = image.select(BANDS)
        valid = valid.And(bands.mask().reduce(ee.Reducer.min()))
        return bands.multiply(0.0001).updateMask(valid).copyProperties(image, ['system:time_start'])

    clear = ee.ImageCollection(pairs.map(prepare))
    total, matched = scenes.size().getInfo(), clear.size().getInfo()
    if matched == 0:
        raise RuntimeError(f'{year}: no scenes with matching cloud QA; change the season or scene cloud limit')
    count = clear.select('B2').count().unmask(0).rename('clear_observations')
    median = clear.median().updateMask(count.gte(args.min_observations)).unmask(NODATA)
    image = median.addBands(count).toFloat()
    info = {'year': year, 'start_inclusive': start, 'end_exclusive': end,
            'scene_count': total, 'qa_matched_scene_count': matched,
            'scene_ids': scenes.aggregate_array('system:index').getInfo()}
    return image, info


def download_chunk(image, grid, row, col, height, width):
    affine = Affine(*grid['transform']) * Affine.translation(col, row)
    request = {'crs': f"EPSG:{grid['epsg']}", 'crs_transform': list(affine)[:6],
               'dimensions': [width, height], 'format': 'GEO_TIFF', 'filePerBand': False}
    for attempt in range(4):
        try:
            url = image.getDownloadURL(request)
            with urllib.request.urlopen(url, timeout=240) as response:
                data = response.read()
            with rio.MemoryFile(data) as mem, mem.open() as src:
                if (src.count, src.height, src.width) != (12, height, width):
                    raise RuntimeError('Unexpected download dimensions or bands')
                if src.crs != rio.crs.CRS.from_epsg(grid['epsg']) or not src.transform.almost_equals(affine):
                    raise RuntimeError('Downloaded raster does not match requested grid')
                return src.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def save_year(image, grid, folder):
    profile = dict(driver='GTiff', width=grid['width'], height=grid['height'],
                   crs=f"EPSG:{grid['epsg']}", transform=Affine(*grid['transform']),
                   compress='deflate', tiled=True)
    ref_tmp, qa_tmp = folder / 'reflectance.partial.tif', folder / 'clear_observations.partial.tif'
    valid_pixels = 0
    with rio.open(ref_tmp, 'w', **profile, count=11, dtype='float32', nodata=NODATA) as ref, \
            rio.open(qa_tmp, 'w', **profile, count=1, dtype='uint16') as qa:
        for i, band in enumerate(BANDS, 1):
            ref.set_band_description(i, band)
        qa.set_band_description(1, 'clear_observations')
        # 500 x 500 x 12 float32 = 12 MB, below EE's 32 MB request limit.
        for row in range(0, grid['height'], 500):
            for col in range(0, grid['width'], 500):
                h, w = min(500, grid['height'] - row), min(500, grid['width'] - col)
                arr = download_chunk(image, grid, row, col, h, w)
                window = Window(col, row, w, h)
                ref.write(arr[:11], window=window)
                qa.write(arr[11:12].astype('uint16'), window=window)
                valid_pixels += int(np.count_nonzero(arr[0] != NODATA))
                print(f'  downloaded row={row}, col={col}, size={w}x{h}', flush=True)
    ref_tmp.replace(folder / 'reflectance.tif')
    qa_tmp.replace(folder / 'clear_observations.tif')
    return round(100 * valid_pixels / (grid['width'] * grid['height']), 2)


def preview(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with rio.open(folder / 'reflectance.tif') as src:
        rgb = src.read([3, 2, 1], masked=True)
    # Fixed display stretch for all years, not independent min-max normalization.
    rgb = np.clip(np.moveaxis(rgb.filled(0), 0, -1) / 0.3, 0, 1)
    plt.imsave(folder / 'rgb.png', rgb)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lat', type=float, default=35.236864)
    p.add_argument('--lon', type=float, default=52.336017)
    p.add_argument('--size-km', type=float, default=10, help='square side length, not radius')
    p.add_argument('--start-year', type=int, default=2017)
    p.add_argument('--end-year', type=int, default=2025, help='inclusive')
    p.add_argument('--start-month', type=int, default=6)
    p.add_argument('--end-month', type=int, default=9, help='inclusive; same-year season')
    p.add_argument('--max-cloud', type=float, default=80, help='scene cloud percentage limit; pixel QA still applied')
    p.add_argument('--clear-threshold', type=float, default=0.6, help='Cloud Score+ cs_cdf minimum')
    p.add_argument('--min-observations', type=int, default=5)
    p.add_argument('--gee-project')
    p.add_argument('--out', type=Path, default=Path('work/city_timeseries'))
    p.add_argument('--dry-run', action='store_true', help='print plan without authentication, downloads or writes')
    args = p.parse_args()
    if not (-80 <= args.lat <= 84 and -180 <= args.lon <= 180):
        p.error('coordinates must be within UTM coverage (-80..84 latitude, -180..180 longitude)')
    if not (2017 <= args.start_year <= args.end_year <= dt.date.today().year):
        p.error('years must be ordered and between 2017 and the current year')
    if not (1 <= args.start_month <= args.end_month <= 12):
        p.error('months must satisfy 1 <= start-month <= end-month <= 12')
    if not (0 <= args.clear_threshold <= 1 and 0 <= args.max_cloud <= 100 and args.min_observations >= 1):
        p.error('invalid QA settings')
    try:
        grid = make_grid(args.lat, args.lon, args.size_km)
    except ValueError as exc:
        p.error(str(exc))
    config = {'version': 1, 'source': SOURCE, 'qa_source': QA_SOURCE,
              'lat': args.lat, 'lon': args.lon, 'grid': grid, 'bands': BANDS,
              'start_month': args.start_month, 'end_month': args.end_month,
              'max_cloud': args.max_cloud, 'clear_threshold': args.clear_threshold,
              'min_observations': args.min_observations, 'composite': 'median'}
    print(json.dumps({'config': config, 'years': list(range(args.start_year, args.end_year + 1)),
                      'out': str(args.out)}, indent=2))
    if args.dry_run:
        return
    if not args.gee_project:
        p.error('--gee-project is required unless --dry-run is used')
    import ee
    try:
        ee.Initialize(project=args.gee_project)
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project=args.gee_project)
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        p.error('output directory contains different settings; choose a new --out directory')
    config_path.write_text(json.dumps(config, indent=2))
    region = ee.Geometry.Rectangle(grid['bounds'], proj=f"EPSG:{grid['epsg']}", geodesic=False)
    for year in range(args.start_year, args.end_year + 1):
        folder = args.out / str(year)
        folder.mkdir(exist_ok=True)
        marker = folder / 'metadata.json'
        if marker.exists() and all((folder / name).exists() for name in ['reflectance.tif', 'clear_observations.tif', 'rgb.png']):
            print(f'{year}: cached completed year', flush=True)
            continue
        print(f'{year}: building seasonal composite ...', flush=True)
        marker.unlink(missing_ok=True)
        image, info = build_composite(ee, region, year, args)
        info['valid_coverage_percent'] = save_year(image, grid, folder)
        preview(folder)
        info['config'] = config
        marker.write_text(json.dumps(info, indent=2))
        print(f"{year}: complete; usable coverage {info['valid_coverage_percent']}%", flush=True)


if __name__ == '__main__':
    main()
