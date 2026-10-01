"""
Sentinel-2 change detection + VLM Q&A.

    streamlit run frontend.py

Reads every complete dated scene (green/red/nir/swir16/visual .tif) from ./data
(override in the sidebar or with RS_DATA_DIR), detects changes between two dates with
IR-MAD + false-positive vetoes (change_core.py), then lets a local Ollama VLM audit
the regions and answer questions about them (vlm_qa.py).
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
import vlm_qa

# --- optional drawable canvas (needs a compat shim on newer Streamlit) ------- #
try:
    from streamlit.elements import image as _st_image
    from streamlit.elements.lib import image_utils as _image_utils
    from streamlit.elements.lib.layout_utils import LayoutConfig

    if not hasattr(_st_image, "image_to_url"):
        def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
            return _image_utils.image_to_url(image, LayoutConfig(width=width), clamp, channels,
                                             output_format, image_id)
        _st_image.image_to_url = _image_to_url_compat
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except Exception:  # noqa: BLE001
    CANVAS_OK = False

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path(os.environ.get("RS_DATA_DIR", HERE / "data"))
REFERENCE_TIF = "change_detection_classified_result.tif"
DEFAULT_AOI = (328, 784, 225, 225)      # x, y, w, h in native pixels
MAX_AOI = 1200                          # px per side, keeps laptop memory in check

st.set_page_config(page_title="Sentinel-2 change detection + VLM", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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


def choose_aoi(preview, native):
    """Returns (x, y, w, h) in native pixels of the AFTER visual grid."""
    nh, nw = native
    ph, pw = preview.shape[:2]
    modes = (["Draw on image"] if CANVAS_OK else []) + ["Enter pixel coordinates"]
    mode = st.radio("AOI selection", modes, horizontal=True, label_visibility="collapsed")
    dx, dy, dw, dh = DEFAULT_AOI
    dx, dy = min(dx, max(nw - 32, 0)), min(dy, max(nh - 32, 0))
    dw, dh = min(dw, nw - dx), min(dh, nh - dy)

    if mode == "Draw on image":
        init = {"version": "4.4.0", "objects": [{
            "type": "rect", "left": dx / nw * pw, "top": dy / nh * ph,
            "width": dw / nw * pw, "height": dh / nh * ph,
            "fill": "rgba(255,80,0,0.18)", "stroke": "#ff5000", "strokeWidth": 2}]}
        canvas = st_canvas(fill_color="rgba(255,80,0,0.18)", stroke_width=2, stroke_color="#ff5000",
                           background_image=Image.fromarray(preview), height=ph, width=pw,
                           drawing_mode="rect", initial_drawing=init, display_toolbar=True,
                           key="aoi_canvas")
        obj = init["objects"][0]
        if canvas.json_data and canvas.json_data.get("objects"):
            obj = canvas.json_data["objects"][-1]
        x = round(obj.get("left", 0) / pw * nw)
        y = round(obj.get("top", 0) / ph * nh)
        w = round(obj.get("width", 1) * obj.get("scaleX", 1) / pw * nw)
        h = round(obj.get("height", 1) * obj.get("scaleY", 1) / ph * nh)
    else:
        c = st.columns(4)
        x = c[0].number_input("x", 0, max(nw - 8, 0), dx, 8, key="aoi_x")
        y = c[1].number_input("y", 0, max(nh - 8, 0), dy, 8, key="aoi_y")
        w = c[2].number_input("width (px)", 8, min(MAX_AOI, nw), max(8, dw), 8, key="aoi_w")
        h = c[3].number_input("height (px)", 8, min(MAX_AOI, nh), max(8, dh), 8, key="aoi_h")
        shown = preview.copy()
        cv2.rectangle(shown, (int(x / nw * pw), int(y / nh * ph)),
                      (int((x + w) / nw * pw), int((y + h) / nh * ph)), (255, 80, 0), 2)
        st.image(shown, caption="Scene preview (orange = selected AOI)", width="stretch")

    x = int(max(0, min(nw - 8, x)))
    y = int(max(0, min(nh - 8, y)))
    w = int(max(8, min(nw - x, w, MAX_AOI)))
    h = int(max(8, min(nh - y, h, MAX_AOI)))
    st.caption(f"AOI: x={x}:{x + w}, y={y}:{y + h}  ({w}x{h} px, about {w * 10 / 1000:.1f} x "
               f"{h * 10 / 1000:.1f} km at 10 m). Max {MAX_AOI} px per side.")
    return x, y, w, h


def crop_triplet(before, after, overlay, region, shape):
    x1, y1, x2, y2 = vlm_qa._crop_box(region["box"], shape)
    return before[y1:y2, x1:x2], after[y1:y2, x1:x2], overlay[y1:y2, x1:x2]


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
    """Distinct colour per unique value (<=12) - or a 2-98 % stretch for continuous rasters.
    Never min-max scales a constant array (that is what produced the all-purple panel)."""
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
        # AOI rectangle on the overview
        x0 = (b[0] - sb.left) / (sb.right - sb.left) * ov.shape[1]
        x1 = (b[2] - sb.left) / (sb.right - sb.left) * ov.shape[1]
        y0 = (sb.top - b[3]) / (sb.top - sb.bottom) * ov.shape[0]
        y1 = (sb.top - b[1]) / (sb.top - sb.bottom) * ov.shape[0]
    return meta_txt, ov, aoi, nodata, inter, (x0, y0, x1, y1)


def models_for(url, force=False):
    cache = st.session_state.setdefault("_models", {})
    if force or url not in cache:
        try:
            cache[url] = (vlm_qa.list_models(url), None)
        except Exception as e:  # noqa: BLE001
            cache[url] = ([], str(e))
    return cache[url]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.title("Sentinel-2 change detection + VLM")
st.caption("IR-MAD statistical change detection with false-positive vetoes, then a local VLM "
           "audits the regions and answers your questions.")

sb = st.sidebar
sb.header("Data")
data_dir = Path(sb.text_input("Data folder", str(DEFAULT_DATA)))
scenes = cc.find_scenes(data_dir) if data_dir.is_dir() else {}
if len(scenes) < 2:
    st.error(f"Need at least two complete dated scenes (green, red, nir, swir16, visual) in "
             f"`{data_dir}`. Found: {list(scenes) or 'none'}.")
    st.stop()
dates = list(scenes)
before_date = sb.selectbox("Before", dates, index=0, format_func=fmt_date)
after_date = sb.selectbox("After", dates, index=len(dates) - 1, format_func=fmt_date)
if before_date >= after_date:
    sb.warning("'Before' should be earlier than 'After'.")

sb.header("Detection")
target_label = sb.radio("What to detect", ["Buildings & roads only (shape-checked)", "All land-surface changes"],
                        help="Buildings & roads: a region must be a compact near-rectangle (building) or a thin "
                             "line (road), not vegetated afterwards, and not a dark/wet or colour-only patch.")
target = "structures" if target_label.startswith("Buildings") else "all"
sensitivity = sb.select_slider("Sensitivity", ["strict", "balanced", "sensitive"], value="balanced",
                               help="strict = fewest false positives, sensitive = catches weaker/smaller changes.")
min_area = sb.slider("Minimum region size (pixels, 10 m each)", 2, 60, 5)
min_sig = sb.slider("Minimum region significance (-log10 p)", 3.0, 15.0, 7.0, 0.5,
                    help="Mean p-value of the region must be below 10^-x. Higher = fewer false positives.")
with sb.expander("Advanced false-positive controls"):
    sam_deg = st.slider("Brightness-only veto: spectral angle below (deg)", 0.0, 8.0, 2.9, 0.1,
                        help="Changes that only make pixels brighter/darker without changing the "
                             "spectral shape (soil moisture, sun angle) are rejected.")
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
    use_dino = st.checkbox("Require DINOv2 semantic agreement (experimental, downloads weights)", False)

sb.header("VLM (Ollama)")
ollama_url = sb.text_input("Ollama URL", "http://localhost:11434")
model_list, model_err = models_for(ollama_url)
if sb.button("Refresh models"):
    model_list, model_err = models_for(ollama_url, force=True)
if model_list:
    pref = next((i for i, m in enumerate(model_list) if "vl" in m.lower()), 0)
    vlm_model = sb.selectbox("Model", model_list, index=pref)
else:
    vlm_model = sb.text_input("Model", "qwen3-vl:4b-instruct")
    sb.caption(f"Ollama not reachable yet ({model_err[:80] if model_err else 'no models'}).")
max_regions = sb.slider("Regions shown to the VLM", 1, 8, 5,
                        help="Small local models get confused with many rows; keep this low.")

# --------------------------------------------------------------------------- #
# AOI + run
# --------------------------------------------------------------------------- #
preview, native = read_preview(str(scenes[after_date]["visual"]))
st.subheader(f"1. Choose the area (shown: {fmt_date(after_date)})")
aoi = choose_aoi(preview, native)

if st.button("2. Detect changes", type="primary"):
    x, y, w, h = aoi
    params = cc.Params(sensitivity=sensitivity, min_area_px=min_area, sam_min=float(np.radians(sam_deg)),
                       min_significance=min_sig, smooth_sigma=smooth, coregister=coreg,
                       calibrate=calib, reject_slivers=slivers, ignore_water_variability=water_var,
                       target=target, min_rect=min_rect, min_solidity=min_solid, building_max_px=int(bmax),
                       road_max_width_px=float(rmax_w), max_ndvi_after=max_ndvi)
    try:
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
                except Exception as e:  # noqa: BLE001
                    st.warning(f"DINOv2 failed ({e}) - continuing without it.")
            res = cc.detect_changes(B, A, params, px_area_m2=meta["res"][0] * meta["res"][1],
                                    semantic_map=semantic)
    except Exception as e:  # noqa: BLE001
        st.error(f"Detection failed: {e}")
        st.stop()
    st.session_state["run"] = {"res": res, "before": B.rgb, "after": A.rgb, "meta": meta,
                               "dates": (before_date, after_date), "aoi": aoi}
    st.session_state["audit"] = {}
    st.session_state["chat"] = []

run = st.session_state.get("run")
if not run:
    st.info("Pick an area and press **Detect changes**.")
    st.stop()

res, before_rgb, after_rgb, meta = run["res"], run["before"], run["after"], run["meta"]
bdate, adate = run["dates"]
verdicts = st.session_state.get("audit", {})

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
st.subheader(f"3. Changes {fmt_date(bdate)} -> {fmt_date(adate)}")
hide_rejected = st.checkbox("Hide regions the VLM rejected", True, disabled=not verdicts)
visible = [r for r in res.regions
           if not (hide_rejected and verdicts.get(r["id"]) and not verdicts[r["id"]]["real_change"])]
visible_ids = {r["id"] for r in visible}
overlay = cc.class_overlay(after_rgb, res, keep_ids=visible_ids)

total_ha = sum(r["area_ha"] for r in visible)
m = st.columns(5)
m[0].metric("Regions", len(visible))
m[1].metric("Changed area", f"{total_ha:.2f} ha")
m[2].metric("Of valid AOI", f"{100 * sum(r['area_px'] for r in visible) / max(res.valid.sum(), 1):.2f}%")
m[3].metric("Cloud / snow masked", f"{res.diag['cloud_pct'] + res.diag['snow_pct']:.1f}%")
m[4].metric("Co-registration shift", "%.2f, %.2f px" % res.diag["shift_px"])

c = st.columns(4)
c[0].image(before_rgb, caption=f"Before {fmt_date(bdate)}", width="stretch")
c[1].image(after_rgb, caption=f"After {fmt_date(adate)}", width="stretch")
c[2].image(cc.draw_regions(overlay, res.regions, verdicts, only=visible_ids),
           caption="Detected changes (box: yellow unreviewed, green VLM-confirmed, red rejected)",
           width="stretch")
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
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read reference raster: {e}")

rows = []
for r in visible:
    v = verdicts.get(r["id"], {})
    rows.append({"#": r["id"], "where": r["position"], "area (ha)": r["area_ha"], "px": r["area_px"],
                 "shape": r["shape"], "rect": r["rectangularity"], "solidity": r["solidity"],
                 "spectral hint (rule-based)": r["class_name"], "signif.": r["significance"],
                 "dNDVI": r["dNDVI"], "dBright": r["dBrightness"], "angle deg": r["spectral_angle_deg"],
                 "VLM": ("" if not v else "real" if v["real_change"] else "false pos."),
                 "VLM category": v.get("category", ""), "VLM reason": v.get("reason", "")})
df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True, width="stretch")
d1, d2 = st.columns(2)
d1.download_button("Regions CSV", df.to_csv(index=False).encode(), "changes.csv", "text/csv",
                   disabled=df.empty)
d2.download_button("Change mask GeoTIFF",
                   mask_geotiff(np.isin(res.labels, list(visible_ids)).astype("uint8"),
                                meta["bounds"], meta["crs"]),
                   "change_mask.tif", "image/tiff", disabled=not visible)

if visible:
    with st.expander(f"Region gallery (top {min(8, len(visible))})"):
        for r in visible[:8]:
            b, a, o = crop_triplet(before_rgb, after_rgb, overlay, r, before_rgb.shape)
            g = st.columns(3)
            for col, img, cap in zip(g, (b, a, o), ("before", "after", "change")):
                col.image(cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                          caption=f"#{r['id']} {cap}", width="stretch")

# --------------------------------------------------------------------------- #
# VLM
# --------------------------------------------------------------------------- #
st.subheader("4. VLM: audit and ask")
if not res.regions:
    st.info("No regions passed the filters, so the honest answer is: no building- or road-like change was "
            "found in this area. Switch 'What to detect' to 'All land-surface changes' to see what was rejected.")
    st.stop()

aoi_desc = (f"Area is {run['aoi'][2]}x{run['aoi'][3]} px at 10 m; "
            f"{len(res.regions)} regions passed the statistical and spectral filters.")


def evidence(regs):
    montage = vlm_qa.build_montage(before_rgb, after_rgb, res.labels, regs, max_regions)
    table = vlm_qa.evidence_table(regs, bdate, adate, aoi_desc, max_regions)
    return montage, table


a1, a2 = st.columns([1, 3])
if a1.button("Audit regions with VLM", help="Second-stage false-positive filter"):
    try:
        with st.spinner(f"{vlm_model} is reviewing {min(len(res.regions), max_regions)} regions..."):
            montage, table = evidence(res.regions)
            out = vlm_qa.audit(ollama_url, vlm_model, montage, table,
                               {r["id"] for r in res.regions[:max_regions]})
        st.session_state["audit"] = out
        st.rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"VLM audit failed: {e}")
a2.caption("The audit reviews the largest regions only. Verdicts from a 4B model are advisory: "
           "check the gallery before discarding anything.")

st.markdown("**Ask about the changes**")
q_cols = st.columns(3)
examples = ["Which of these changes look like new buildings or roads?",
            "Is the largest change real, or could it be soil moisture or seasonal?",
            "Summarize what changed between the two dates."]
for col, ex in zip(q_cols, examples):
    if col.button(ex, key=f"ex_{ex[:12]}"):
        st.session_state["pending_q"] = ex

chat = st.session_state.setdefault("chat", [])
for msg in chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask the VLM a question about the detected changes...") \
    or st.session_state.pop("pending_q", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        if not visible:
            raise RuntimeError("All regions are hidden - untick 'Hide regions the VLM rejected'.")
        montage, table = evidence(visible)
        with st.chat_message("assistant"):
            answer = st.write_stream(vlm_qa.stream_answer(ollama_url, vlm_model, question, chat,
                                                          montage, table))
        chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        with st.expander("What the VLM saw"):
            st.image(montage, width="stretch")
            st.text(table)
    except Exception as e:  # noqa: BLE001
        st.error(f"VLM request failed: {e}")