#!/usr/bin/env python3

from pathlib import Path
import re
import html
import json

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from PIL import Image


INPUT_DIR = Path("stac_cache/raw")
OUTPUT_DIR = Path("stac_cache/previews")

MAX_SIZE = 1600

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_visual(path):
    name = path.stem

    match = re.match(
        r"S2[AB]_([^_]+)_(\d{8})_[^_]+_L2A_visual$",
        name
    )

    if not match:
        return None

    return {
        "tile": match.group(1),
        "date": match.group(2),
    }


def create_preview(tif_path, jpg_path):

    with rasterio.open(tif_path) as src:

        scale = min(
            1.0,
            MAX_SIZE / max(src.width, src.height)
        )

        width = max(1, int(src.width * scale))
        height = max(1, int(src.height * scale))

        print(
            f"    {src.count} bands, "
            f"{src.width} × {src.height}"
        )

        if src.count >= 3:

            data = src.read(
                [1, 2, 3],
                out_shape=(3, height, width),
                resampling=Resampling.bilinear
            )

            data = np.moveaxis(data, 0, -1)

        elif src.count == 1:

            data = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.bilinear
            )

            data = np.stack(
                [data, data, data],
                axis=-1
            )

        else:
            raise ValueError(
                f"Unexpected band count: {src.count}"
            )

        data = data.astype(np.float32)

        for i in range(3):

            band = data[:, :, i]

            valid = band[np.isfinite(band)]

            if len(valid) == 0:
                continue

            lo, hi = np.percentile(
                valid,
                [2, 98]
            )

            if hi > lo:

                data[:, :, i] = np.clip(
                    (band - lo) / (hi - lo),
                    0,
                    1
                )

        data = np.nan_to_num(data)

        data = (
            data * 255
        ).astype(np.uint8)

        Image.fromarray(
            data,
            "RGB"
        ).save(
            jpg_path,
            "JPEG",
            quality=92
        )

        bounds = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds
        )

        west, south, east, north = bounds

        return {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "lat": (south + north) / 2,
            "lon": (west + east) / 2,
            # src.transform + src.crs are needed later to project
            # per-region pixel polygons to lat/lon.
            "transform": tuple(src.transform)[:6],
            "crs_obj": src.crs,
        }


# -----------------------------------------------------------------
# NEW: pixel-space polygon -> lat/lon polygon
# -----------------------------------------------------------------
def polygon_to_latlon(polygon_px, transform, crs):
    """
    polygon_px : Nx2 array of (x_pixel, y_pixel) in raster pixel coords
    transform  : affine transform of the raster (6-tuple or Affine)
    crs        : rasterio CRS

    Returns list of [lon, lat] pairs (GeoJSON ring order).
    """
    import rasterio.transform as rt
    from rasterio.warp import transform as warp_transform

    if not isinstance(transform, rt.Affine):
        # Affine(*transform) accepts a 6- or 9-tuple
        transform = rt.Affine(*transform)

    xs, ys = [], []
    for x, y in polygon_px:
        gx, gy = transform * (x, y)
        xs.append(gx)
        ys.append(gy)

    lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
    return [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]


def main():

    # Only visual products
    tif_files = sorted(
        INPUT_DIR.glob("*_visual.tif")
    )

    if not tif_files:

        print(
            "No *_visual.tif files found."
        )

        print("\nFiles found:")

        for f in INPUT_DIR.glob("*.tif"):
            print(" ", f.name)

        return

    scenes = {}

    for tif in tif_files:

        info = parse_visual(tif)

        if info is None:
            print(
                f"Skipping unrecognized: {tif.name}"
            )
            continue

        key = info["tile"]

        scenes.setdefault(
            key,
            []
        ).append(
            (info["date"], tif)
        )

    cards = []

    for tile, files in scenes.items():

        files.sort(
            key=lambda x: x[0]
        )

        if len(files) < 2:

            print(
                f"\nSkipping {tile}: "
                f"only one visual product."
            )

            continue

        before_date, before = files[0]
        after_date, after = files[-1]

        print()
        print("=" * 60)
        print(f"Scene: {tile}")
        print(f"BEFORE: {before.name}")
        print(f"AFTER:  {after.name}")

        before_jpg = (
            OUTPUT_DIR /
            f"{tile}_{before_date}_BEFORE.jpg"
        )

        after_jpg = (
            OUTPUT_DIR /
            f"{tile}_{after_date}_AFTER.jpg"
        )

        print("\nCreating BEFORE...")

        before_info = create_preview(
            before,
            before_jpg
        )

        print("Creating AFTER...")

        after_info = create_preview(
            after,
            after_jpg
        )

        cards.append({
            "tile": tile,
            "before_date": before_date,
            "after_date": after_date,
            "before_jpg": before_jpg.name,
            "after_jpg": after_jpg.name,
            "info": before_info,
        })

    # -----------------------------------------------------------------
    # NEW: load change regions (regions.geojson) if the detection
    # script has already produced one. Also infer the "which scene"
    # assignment by matching tile name in the geojson's properties.
    # -----------------------------------------------------------------
    regions_geojson = OUTPUT_DIR / "regions.geojson"
    change_features = []

    if regions_geojson.exists():
        try:
            with open(regions_geojson, "r", encoding="utf-8") as f:
                gj = json.load(f)
            change_features = gj.get("features", [])
            print(f"\nLoaded {len(change_features)} change feature(s) "
                  f"from {regions_geojson.name}")
        except Exception as e:
            print(f"Failed to load {regions_geojson}: {e}")

    # -----------------------------------------------------------------
    # Leaflet viewer with Google satellite tiles + change polygons
    # -----------------------------------------------------------------

    html_file = OUTPUT_DIR / "index.html"

    with open(html_file, "w", encoding="utf-8") as f:

        f.write("""
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Sentinel-2 Visual Before / After</title>

<link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<style>

html, body {
    margin: 0; padding: 0; height: 100%;
    font-family: Arial, sans-serif;
    background: #f4f4f4;
}

#map {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 1;
}

#sidebar {
    position: absolute;
    top: 10px; right: 10px;
    width: 380px;
    max-height: calc(100% - 20px);
    overflow-y: auto;
    background: rgba(255,255,255,0.96);
    border-radius: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    padding: 15px;
    z-index: 1000;
    font-size: 13px;
}

#sidebar h1 {
    font-size: 16px;
    margin: 0 0 12px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #ddd;
}

.card {
    background: white;
    padding: 10px;
    margin-bottom: 12px;
    border-radius: 8px;
    box-shadow: 0 1px 4px #ccc;
    cursor: pointer;
    transition: box-shadow 0.2s;
}

.card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.25); }
.card.active { outline: 3px solid #4285f4; }

.card h2 {
    margin: 0 0 8px 0;
    font-size: 14px;
    color: #333;
}

.images { display: flex; gap: 8px; }
.image-box { flex: 1; text-align: center; }
.image-box h3 { margin: 0 0 4px 0; font-size: 11px; color: #666; }
.image-box img {
    width: 100%; display: block;
    border-radius: 4px; border: 1px solid #ccc;
}

.info {
    margin-top: 8px;
    font-family: monospace;
    font-size: 11px;
    color: #555;
    line-height: 1.5;
}

.legend {
    background: #fafafa;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 12px;
    font-size: 11px;
    font-family: monospace;
}

.legend .swatch {
    display: inline-block;
    width: 12px; height: 12px;
    margin-right: 6px;
    vertical-align: middle;
    border: 1px solid #333;
}

@media (max-width: 900px) {
    #sidebar {
        width: calc(100% - 20px);
        max-height: 45%;
        top: auto; bottom: 10px;
    }
}

.leaflet-popup-content { margin: 8px; }
.popup-img { width: 260px; display: block; border-radius: 4px; margin-bottom: 4px; }
.popup-label { font-size: 12px; font-weight: bold; margin-bottom: 6px; text-align: center; }

</style>
</head>
<body>

<div id="map"></div>

<div id="sidebar">

<h1>Sentinel-2 Visual Before / After</h1>

<div class="legend">
    <div><span class="swatch" style="background:#4285f4;"></span>scene footprint</div>
    <div><span class="swatch" style="background:#ff3b30;"></span>change region (from detection)</div>
</div>
""")

        # Sidebar cards
        for idx, card in enumerate(cards):
            info = card["info"]
            f.write(f"""

<div class="card" id="card-{idx}" onclick="focusScene({idx})">
<h2>{html.escape(card["tile"])}</h2>
<div class="images">
<div class="image-box">
<h3>BEFORE — {card["before_date"]}</h3>
<img src="{html.escape(card["before_jpg"])}">
</div>
<div class="image-box">
<h3>AFTER — {card["after_date"]}</h3>
<img src="{html.escape(card["after_jpg"])}">
</div>
</div>
<div class="info">
CRS: {html.escape(info["crs"])}<br>
Size: {info["width"]} × {info["height"]}<br>
Center: {info["lat"]:.6f}, {info["lon"]:.6f}
</div>
</div>
""")

        # Build JS scenes array
        js_scenes = []
        for idx, card in enumerate(cards):
            info = card["info"]
            js_scenes.append(
                "{"
                f"id:{idx},"
                f'tile:"{card["tile"]}",'
                f'before:"{card["before_jpg"]}",'
                f'after:"{card["after_jpg"]}",'
                f'beforeDate:"{card["before_date"]}",'
                f'afterDate:"{card["after_date"]}",'
                f'west:{info["west"]},'
                f'south:{info["south"]},'
                f'east:{info["east"]},'
                f'north:{info["north"]},'
                f'center:[{info["lat"]},{info["lon"]}]'
                "}"
            )
        js_scenes_str = "[\n" + ",\n".join(js_scenes) + "\n]"

        # Inject change features (geojson) directly as a JS literal.
        # This avoids fetch() problems when opening the file via file://
        gj_features = change_features
        js_change_str = json.dumps(
            {"type": "FeatureCollection", "features": gj_features},
            ensure_ascii=False,
            indent=2,
        )

        f.write(f"""
</div>

<script>

// --- Embedded data ------------------------------------------------
const scenes = {js_scenes_str};

// Change regions produced by detect_changes.py / main.py detection.
// Each feature is a GeoJSON polygon with properties:
//   tile, index, score, dino, dndvi, dndbi, label (optional)
const changeGeoJSON = {js_change_str};

// --- Map init ------------------------------------------------------
const map = L.map('map', {{
    zoomControl: true,
    worldCopyJump: true,
    preferCanvas: true,
}}).setView([0, 0], 3);

L.tileLayer(
    'https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}',
    {{ maxZoom: 21, attribution: '&copy; Google' }}
).addTo(map);

// --- Scene rectangles ---------------------------------------------
const rectangles = {{}};
const allBounds = [];

function popupHtml(scene) {{
    return `
        <div class="popup-label">${{scene.tile}}</div>
        <img class="popup-img" src="${{scene.before}}">
        <div class="popup-label">BEFORE — ${{scene.beforeDate}}</div>
        <img class="popup-img" src="${{scene.after}}">
        <div class="popup-label">AFTER — ${{scene.afterDate}}</div>
    `;
}}

scenes.forEach(scene => {{
    const bounds = [
        [scene.south, scene.west],
        [scene.north, scene.east]
    ];
    allBounds.push(bounds);

    const rect = L.rectangle(bounds, {{
        color: '#4285f4',
        weight: 2,
        fillOpacity: 0.04,
        dashArray: '6 6',
    }}).addTo(map);

    rect.bindPopup(popupHtml(scene), {{ maxWidth: 300, minWidth: 280 }});
    rect.on('click', () => setActiveCard(scene.id));
    rectangles[scene.id] = rect;
}});

if (allBounds.length > 0) {{
    map.fitBounds(allBounds, {{ padding: [40, 40] }});
}}

// --- Change regions ------------------------------------------------
const changeLayer = L.geoJSON(changeGeoJSON, {{
    style: function(feature) {{
        return {{
            color: '#ff3b30',
            weight: 2,
            fillColor: '#ff3b30',
            fillOpacity: 0.35,
            dashArray: '3 3',
        }};
    }},
    onEachFeature: function(feature, layer) {{
        const p = feature.properties || {{}};
        const title = p.label
            ? p.label
            : (p.tile ? `${{p.tile}} #${{p.index}}` : 'change region');
        const stats = [
            p.score !== undefined ? `score=${{(+p.score).toFixed(2)}}` : null,
            p.dino !== undefined ? `DINO=${{(+p.dino).toFixed(2)}}` : null,
            p.dndvi !== undefined ? `dNDVI=${{(+p.dndvi).toFixed(2)}}` : null,
            p.dndbi !== undefined ? `dNDBI=${{(+p.dndbi).toFixed(2)}}` : null,
        ].filter(Boolean).join('  ');

        layer.bindPopup(
            `<div class="popup-label">${{title}}</div>` +
            (stats ? `<div style="font-family:monospace;font-size:12px;">${{stats}}</div>` : '')
        );

        layer.on('mouseover', () => layer.setStyle({{ fillOpacity: 0.6, weight: 3 }}));
        layer.on('mouseout',  () => layer.setStyle({{ fillOpacity: 0.35, weight: 2 }}));
    }}
}});

// Only add the change layer if we actually have features
if (changeGeoJSON.features && changeGeoJSON.features.length > 0) {{
    changeLayer.addTo(map);

    // Zoom to change regions if any exist, otherwise scene bounds
    try {{
        map.fitBounds(changeLayer.getBounds(), {{ padding: [40, 40], maxZoom: 16 }});
    }} catch (e) {{ /* ignore */ }}
}} else {{
    // Still expose the (empty) layer so the checkbox code below works
    console.log('No change regions in embedded GeoJSON.');
}}

// Toggle control for change layer
const overlayControl = L.control.layers(
    null,
    {{ 'Change regions': changeLayer }},
    {{ collapsed: false, position: 'topleft' }}
).addTo(map);

// --- Sidebar <-> Map interaction ----------------------------------
function setActiveCard(id) {{
    document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
    const el = document.getElementById('card-' + id);
    if (el) {{
        el.classList.add('active');
        el.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}
}}

function focusScene(id) {{
    const scene = scenes.find(s => s.id === id);
    if (!scene) return;
    map.fitBounds([
        [scene.south, scene.west],
        [scene.north, scene.east]
    ], {{ padding: [40, 40] }});
    rectangles[id].openPopup();
    setActiveCard(id);
}}

</script>
</body>
</html>
""")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"HTML: {html_file}")
    if change_features:
        print(f"Change features embedded: {len(change_features)}")
    print()
    print(f"xdg-open {html_file}")


if __name__ == "__main__":
    main()