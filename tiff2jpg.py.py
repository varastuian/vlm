#!/usr/bin/env python3

from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import numpy as np
from PIL import Image
import html


INPUT_DIR = Path("stac_cache/raw")
OUTPUT_DIR = Path("stac_cache/previews")

MAX_SIZE = 1600

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_preview(tif_path, jpg_path):
    with rasterio.open(tif_path) as src:

        scale = min(1.0, MAX_SIZE / max(src.width, src.height))

        width = max(1, int(src.width * scale))
        height = max(1, int(src.height * scale))

        # Read first 3 bands
        count = min(src.count, 3)

        data = src.read(
            indexes=list(range(1, count + 1)),
            out_shape=(count, height, width),
            resampling=Resampling.bilinear
        )

        data = np.moveaxis(data, 0, -1)

        # If grayscale, make RGB
        if data.shape[2] == 1:
            data = np.repeat(data, 3, axis=2)

        # Percentile stretch
        data = data.astype(np.float32)

        for i in range(3):
            band = data[:, :, i]

            valid = band[np.isfinite(band)]

            if len(valid) > 0:
                lo, hi = np.percentile(valid, [2, 98])

                if hi > lo:
                    data[:, :, i] = np.clip(
                        (band - lo) / (hi - lo),
                        0,
                        1
                    )

        data = (data * 255).astype(np.uint8)

        Image.fromarray(data).save(
            jpg_path,
            "JPEG",
            quality=90
        )

        # Geographic footprint
        bounds = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds
        )

        west, south, east, north = bounds

        center_lon = (west + east) / 2
        center_lat = (south + north) / 2

        return {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "lat": center_lat,
            "lon": center_lon,
        }


def main():

    tif_files = sorted(INPUT_DIR.glob("*.tif"))

    if not tif_files:
        print("No TIFF files found.")
        return

    cards = []

    for tif in tif_files:

        print(f"\nProcessing: {tif.name}")

        jpg = OUTPUT_DIR / f"{tif.stem}.jpg"

        try:
            info = create_preview(tif, jpg)

        except Exception as e:
            print(f"ERROR: {e}")
            continue

        google_maps = (
            f"https://www.google.com/maps/"
            f"@{info['lat']:.6f},{info['lon']:.6f},13z"
        )

        print(f"  CRS:    {info['crs']}")
        print(f"  Size:   {info['width']} x {info['height']}")
        print(
            f"  Center: "
            f"{info['lat']:.6f}, {info['lon']:.6f}"
        )
        print(f"  Google: {google_maps}")

        cards.append({
            "name": tif.name,
            "jpg": jpg.name,
            "info": info,
            "google": google_maps
        })

    # Create HTML viewer
    html_file = OUTPUT_DIR / "index.html"

    with open(html_file, "w", encoding="utf-8") as f:

        f.write("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Sentinel-2 Preview</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f5f5f5;
    margin: 30px;
}

.card {
    background: white;
    padding: 20px;
    margin-bottom: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 8px #ccc;
}

img {
    max-width: 900px;
    width: 100%;
    height: auto;
    display: block;
    margin-bottom: 15px;
}

h2 {
    font-size: 20px;
}

a {
    display: inline-block;
    padding: 10px 15px;
    background: #4285f4;
    color: white;
    text-decoration: none;
    border-radius: 5px;
}

a:hover {
    background: #3367d6;
}

.info {
    font-family: monospace;
    margin-bottom: 15px;
}

</style>
</head>

<body>

<h1>Sentinel-2 GeoTIFF Previews</h1>
""")

        for card in cards:

            info = card["info"]

            f.write(f"""
<div class="card">

<h2>{html.escape(card["name"])}</h2>

<img src="{html.escape(card["jpg"])}">

<div class="info">
CRS: {html.escape(info["crs"])}<br>
Original size: {info["width"]} × {info["height"]}<br>
Center: {info["lat"]:.6f}, {info["lon"]:.6f}<br>
West: {info["west"]:.6f}<br>
South: {info["south"]:.6f}<br>
East: {info["east"]:.6f}<br>
North: {info["north"]:.6f}
</div>

<a href="{card["google"]}" target="_blank">
Open center in Google Maps
</a>

</div>
""")

        f.write("""
</body>
</html>
""")

    print("\n======================================")
    print("DONE")
    print("======================================")
    print(f"Previews: {OUTPUT_DIR}")
    print(f"HTML:     {html_file}")
    print()
    print("Open the viewer with:")
    print(f"  xdg-open {html_file}")


if __name__ == "__main__":
    main()