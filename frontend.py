"""
Sentinel-2 change detection (simplified - no VLM, no AOI selection).
Run: streamlit run frontend.py
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

import change_core as cc

st.set_page_config(page_title="Sentinel Change Detection", layout="wide")

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path(os.environ.get("RS_DATA_DIR", HERE / "data/aoi_products/crops"))
REFERENCE_TIF = "change_detection_classified_result.tif"
DEFAULT_AOI = (228, 984, 225, 225)
MAX_AOI = 1200


@st.cache_data(show_spinner=False)
def read_preview(path, max_size=900):
    with rasterio.open(path) as src:
        scale = min(1.0, max_size / max(src.height, src.width))
        shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
        n = min(3, src.count)
        data = src.read(list(range(1, n + 1)), out_shape=(n, *shape), resampling=Resampling.bilinear)
        native = (src.height, src.width)
    if n == 1:
        data = np.repeat(data, 3, axis=0)
    img = np.moveaxis(data[:3], 0, -1).astype(np.float32)
    if img.max() > 255 or img.max() <= 1.0:
        img = img / max(np.percentile(img, 99), 1e-6) * 255
    return np.clip(img, 0, 255).astype(np.uint8), native


def fmt_date(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def mask_geotiff(mask, bounds, crs):
    h, w = mask.shape
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=h, width=w, count=1, dtype="uint8", crs=crs,
                      transform=transform_from_bounds(*bounds, w, h)) as dst:
            dst.write(mask[None].astype("uint8"))
        return mem.read()


PALETTE = [(190, 190, 190), (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (213, 94, 0),
           (204, 121, 167), (0, 114, 178), (120, 60, 20), (255, 255, 255), (60, 60, 60), (150, 0, 0)]


def colorize_categorical(arr, nodata=None):
    valid = np.ones(arr.shape, bool) if nodata is None else (arr != nodata)
    vals, counts = np.unique(arr[valid], return_counts=True)
    out = np.full((*arr.shape, 3), 40, np.uint8)
    legend = []
    if len(vals) == 0:
        return out, legend
    if len(vals) <= 12:
        for i, (v, n) in enumerate(zip(vals, counts)):
            out[(arr == v) & valid] = PALETTE[i]
            legend.append((PALETTE[i], f"{v:g}", int(n)))
    else:
        lo, hi = np.percentile(arr[valid], [2, 98])
        norm = np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)
        col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        out = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
        out[~valid] = 40
        legend.append(((255, 255, 255), f"continuous {lo:.3g} .. {hi:.3g} (2-98 %)", int(valid.sum())))
    return out, legend


def describe_reference(path, bounds, crs, aoi_shape):
    with rasterio.open(path) as src:
        meta_txt = (f"{src.width} x {src.height} px, {src.count} band(s), dtype {src.dtypes[0]}, "
                    f"CRS {src.crs}, nodata={src.nodata}")
        scale = min(1.0, 600 / max(src.width, src.height))
        ov = src.read(1, out_shape=(max(1, int(src.height * scale)), max(1, int(src.width * scale))),
                      resampling=Resampling.nearest)
        b = bounds if src.crs == crs else transform_bounds(crs, src.crs, *bounds)
        sb = src.bounds
        inter = not (b[2] <= sb.left or b[0] >= sb.right or b[3] <= sb.bottom or b[1] >= sb.top)
        win = window_from_bounds(*b, transform=src.transform)
        aoi = src.read(1, window=win, out_shape=aoi_shape, resampling=Resampling.nearest,
                       boundless=True, fill_value=src.nodata if src.nodata is not None else 0)
        nodata = src.nodata
        x0 = (b[0] - sb.left) / (sb.right - sb.left) * ov.shape[1]
        x1 = (b[2] - sb.left) / (sb.right - sb.left) * ov.shape[1]
        y0 = (sb.top - b[3]) / (sb.top - sb.bottom) * ov.shape[0]
        y1 = (sb.top - b[1]) / (sb.top - sb.bottom) * ov.shape[0]
    return meta_txt, ov, aoi, nodata, inter, (x0, y0, x1, y1)


st.title("Sentinel-2 Change Detection")
st.caption("IR-MAD statistical change detection with false-positive vetoes")

sb = st.sidebar
sb.header("Data")
data_dir = Path(sb.text_input("Data folder", str(DEFAULT_DATA)))
scenes = cc.find_scenes(data_dir) if data_dir.is_dir() else {}
if len(scenes) < 2:
    st.error(f"Need at least two complete dated scenes in `{data_dir}`. Found: {list(scenes) or 'none'}.")
    st.stop()
dates = list(scenes)
before_date = sb.selectbox("Before", dates, index=0, format_func=fmt_date)
after_date = sb.selectbox("After", dates, index=len(dates) - 1, format_func=fmt_date)
if before_date >= after_date:
    sb.warning("'Before' should be earlier than 'After'.")

sb.header("Detection")
target_label = sb.radio("What to detect", ["Buildings & roads only (shape-checked)", "All land-surface changes"],
                        help="Buildings & roads: region must be a compact near-rectangle (building) or thin line (road), not vegetated afterwards.")
target = "structures" if target_label.startswith("Buildings") else "all"
sensitivity = sb.select_slider("Sensitivity", ["strict", "balanced", "sensitive"], value="balanced",
                               help="strict = fewest false positives, sensitive = catches weaker/smaller changes.")
min_area = sb.slider("Minimum region size (pixels, 10 m each)", 2, 60, 5)
min_sig = sb.slider("Minimum region significance (-log10 p)", 3.0, 15.0, 7.0, 0.5,
                    help="Mean p-value of the region must be below 10^-x. Higher = fewer false positives.")
with sb.expander("Advanced false-positive controls"):
    sam_deg = st.slider("Brightness-only veto: spectral angle below (deg)", 0.0, 8.0, 2.9, 0.1,
                        help="Changes that only make pixels brighter/darker without changing spectral shape are rejected.")
    smooth = st.slider("Pre-smoothing sigma (px)", 0.0, 2.0, 1.0, 0.1,
                       help="Lower catches tiny objects but is more sensitive to misregistration.")
    coreg = st.checkbox("Sub-pixel co-registration", True)
    calib = st.checkbox("Calibrate statistic to scene noise", True)
    slivers = st.checkbox("Reject 1-2 px wide short slivers", True)
    water_var = st.checkbox("Ignore water-surface variability", True)
    st.markdown("**Shape gate (buildings & roads mode)**")
    min_rect = st.slider("Min rectangularity (area / rotated bounding box)", 0.5, 1.0, 0.80, 0.01,
                         help="Rectangle = 1.0, circle = 0.78, irregular patch < 0.7.")
    min_solid = st.slider("Min solidity (area / convex hull)", 0.5, 1.0, 0.88, 0.01)
    bmax = st.slider("Max building size (pixels; 100 px = 1 ha)", 20, 600, 150, 10,
                     help="Larger rectangles are treated as farm plots.")
    rmax_w = st.slider("Max road width (pixels)", 2.0, 10.0, 6.0, 0.5)
    max_ndvi = st.slider("Max NDVI afterwards", 0.10, 0.60, 0.30, 0.01,
                         help="A building/road is not green after the change.")
    use_dino = st.checkbox("Require DINOv2 semantic agreement (experimental)", False)


x, y, w, h = DEFAULT_AOI

params = cc.Params(sensitivity=sensitivity, min_area_px=min_area, sam_min=float(np.radians(sam_deg)),
                   min_significance=min_sig, smooth_sigma=smooth, coregister=coreg,
                   calibrate=calib, reject_slivers=slivers, ignore_water_variability=water_var,
                   target=target, min_rect=min_rect, min_solidity=min_solid, building_max_px=int(bmax),
                   road_max_width_px=float(rmax_w), max_ndvi_after=max_ndvi)

with st.spinner("Reading bands and running IR-MAD..."):
    B, A, meta = cc.load_aoi_pair(scenes[before_date], scenes[after_date], x, y, w, h)
    semantic = None
    if use_dino:
        try:
            from detectors import dino_detector
            semantic, _ = dino_detector(cv2.cvtColor(B.rgb, cv2.COLOR_RGB2BGR),
                                        cv2.cvtColor(A.rgb, cv2.COLOR_RGB2BGR))
            if semantic is None:
                st.warning("DINOv2 unavailable - continuing without semantic agreement.")
        except Exception as e:
            st.warning(f"DINOv2 failed ({e}) - continuing without it.")
    res = cc.detect_changes(B, A, params, px_area_m2=meta["res"][0] * meta["res"][1],
                            semantic_map=semantic)

before_rgb, after_rgb = B.rgb, A.rgb
bdate, adate = before_date, after_date
overlay = cc.class_overlay(after_rgb, res)

total_ha = sum(r["area_ha"] for r in res.regions)
m = st.columns(5)
m[0].metric("Regions", len(res.regions))
m[1].metric("Changed area", f"{total_ha:.2f} ha")
m[2].metric("Of valid AOI", f"{100 * sum(r['area_px'] for r in res.regions) / max(res.valid.sum(), 1):.2f}%")
m[3].metric("Cloud / snow masked", f"{res.diag['cloud_pct'] + res.diag['snow_pct']:.1f}%")
m[4].metric("Co-registration shift", "%.2f, %.2f px" % res.diag["shift_px"])

st.subheader(f"Results: {fmt_date(bdate)} \u2192 {fmt_date(adate)}")
c = st.columns(4)
c[0].image(before_rgb, caption=f"Before {fmt_date(bdate)}", width="stretch")
c[1].image(after_rgb, caption=f"After {fmt_date(adate)}", width="stretch")
c[2].image(cc.draw_regions(after_rgb, res.regions), caption="Detected changes (yellow boxes)", width="stretch")
c[3].image(cc.heatmap(res.T, res.valid), caption="Change significance (-log10 p)", width="stretch")

st.markdown(" &nbsp; ".join(
    f"<span style='color:rgb{col}'>&#9632;</span> {name}" for _, (name, col) in cc.CLASS_INFO.items()),
    unsafe_allow_html=True)

with st.expander("How false positives were removed (funnel) and radiometric checks"):
    st.dataframe(pd.DataFrame(res.funnel, columns=["stage", "pixels / regions"]), hide_index=True)
    if res.rejected:
        st.markdown(f"**Regions removed by the shape / spectral gate ({len(res.rejected)}):**")
        st.dataframe(pd.DataFrame([{
            "where": r["position"], "px": r["area_px"], "shape": r["shape"], "rect": r["rectangularity"],
            "solidity": r["solidity"], "aspect": r["aspect"], "NDVI after": r["ndvi_after"],
            "reason": r["reject_reason"]} for r in res.rejected]), hide_index=True)
    fits = res.diag["normalisation_fit"]
    st.write("After->before radiometric fit on unchanged pixels (slope, intercept). An intercept near "
             "-0.1 means the newer scene carries the +1000 DN processing offset; it was corrected.")
    st.dataframe(pd.DataFrame(fits, index=["slope", "intercept"]).round(3))
    st.caption(f"IR-MAD iterations: {res.diag['irmad']['iterations']}, canonical correlations: "
               f"{[round(v, 3) for v in res.diag['irmad']['rho']]}, noise overdispersion factor: "
               f"{res.diag['irmad']['overdispersion']:.2f}. Reflectance scaling read as: "
               f"{res.diag['calibration_after']['nir']} (after), {res.diag['calibration_before']['nir']} (before).")

ref_path = data_dir / REFERENCE_TIF
if ref_path.exists():
    with st.expander(f"Reference raster {REFERENCE_TIF} (not used by the detector)"):
        try:
            txt, ov, aoi_arr, nod, inter, (rx0, ry0, rx1, ry1) = describe_reference(
                ref_path, meta["bounds"], meta["crs"], res.mask.shape)
            st.caption(txt)
            if not inter:
                st.warning("Your AOI lies OUTSIDE this raster, so the AOI crop below is just fill value.")
            ov_rgb, ov_leg = colorize_categorical(ov, nod)
            cv2.rectangle(ov_rgb, (int(rx0), int(ry0)), (int(rx1), int(ry1)), (255, 0, 0), 2)
            aoi_rgb, aoi_leg = colorize_categorical(aoi_arr, nod)
            r1, r2 = st.columns(2)
            r1.image(ov_rgb, caption="Whole raster (red box = your AOI)", width="stretch")
            r2.image(aoi_rgb, caption="Your AOI", width="stretch")
            st.write("**Values in whole raster:** " + ", ".join(f"{n} (x{c})" for _, n, c in ov_leg[:12]))
            st.write("**Values in your AOI:** " + ", ".join(f"{n} (x{c})" for _, n, c in aoi_leg[:12]))
            if len(aoi_leg) == 1 and not aoi_leg[0][1].startswith("continuous"):
                st.info("The AOI contains a single value, so it is drawn as one flat colour. "
                        "For a classified change map that normally means 'one class only' (often 'no change').")
        except Exception as e:
            st.error(f"Could not read reference raster: {e}")

rows = []
for r in res.regions:
    rows.append({"#": r["id"], "where": r["position"], "area (ha)": r["area_ha"], "px": r["area_px"],
                 "shape": r["shape"], "rect": r["rectangularity"], "solidity": r["solidity"],
                 "spectral hint (rule-based)": r["class_name"], "signif.": r["significance"],
                 "dNDVI": r["dNDVI"], "dBright": r["dBrightness"], "angle deg": r["spectral_angle_deg"]})
df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True, width="stretch")
d1, d2 = st.columns(2)
d1.download_button("Regions CSV", df.to_csv(index=False).encode(), "changes.csv", "text/csv",
                   disabled=df.empty)
d2.download_button("Change mask GeoTIFF",
                   mask_geotiff(res.mask.astype("uint8"), meta["bounds"], meta["crs"]),
                   "change_mask.tif", "image/tiff", disabled=not res.regions)