import base64
import glob
import itertools
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np
import requests

import dino_sam_cd as dsc

INPUT_FILE = "stac_cache/raw/S2A_39SXV_20200128_1_L2A_red.tif"
CENTER = None
SIZE_PX = 2048
PATCH_PX = 512
OVERLAP_PX = 64
DINO_THRESH = 0.3
INDEX_THRESH = 0.2
MIN_SCORE = 0.25
MIN_AREA = 100
MAX_REGIONS = 12
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
VLM_MODEL = "qwen3-vl:4b-instruct"
DESCRIBE = True
DEBUG = True
OUT_DIR = "categorized_output"

NEEDED = ("red", "nir", "swir16", "visual")
SCL_INVALID = (3, 8, 9, 10)
RAW_RE = re.compile(r"^(S2[ABC]_\w{5}_(\d{8})_\d+)_L2A_(\w+)\.tif$")
BOA_OFFSET_DATE = "20220125"
DINO_MODEL = "dinov2_vits14"
SAM_TYPE, SAM_POINTS = "vit_b", 24
DINO_FULL = 0.6
INDEX_FULL = 0.4
W_DINO = 0.7
MIN_OVERLAP = 0.10
MAX_SEG_FRAC = 0.35
TILE = 200
PANELS = ("BEFORE", "AFTER", "DINO", "INDEX")
VLM_URL, VLM_TIMEOUT, VLM_TOKENS, VLM_BATCH = "http://localhost:11434", 900, 1024, 4


def find_pair(path):
    folder, name = os.path.split(path)
    m = RAW_RE.match(name)
    if not m:
        sys.exit(f"Unrecognised file name: {name}")
    prefix, date, band = m.groups()
    tile = prefix.split("_")[1]

    others = {}
    for p in glob.glob(os.path.join(folder, f"S2?_{tile}_*_L2A_{band}.tif")):
        pm = RAW_RE.match(os.path.basename(p))
        if pm and pm.group(1) != prefix:
            others[pm.group(2)] = pm.group(1)
    if not others:
        sys.exit(f"No other date found for tile {tile} in {folder}")

    def days_apart(d):
        return abs((datetime.strptime(d, "%Y%m%d") - datetime.strptime(date, "%Y%m%d")).days)

    other_date = max(others, key=days_apart)
    before, after = sorted([(date, prefix), (other_date, others[other_date])])

    scenes = []
    for d, pre in (before, after):
        files = {b: os.path.join(folder, f"{pre}_L2A_{b}.tif") for b in NEEDED}
        scl = os.path.join(folder, f"{pre}_L2A_scl.tif")
        if os.path.exists(scl):
            files["scl"] = scl
        missing = [os.path.basename(f) for f in files.values() if not os.path.exists(f)]
        if missing:
            sys.exit(f"Missing files: {missing}")
        scenes.append({"id": pre, "date": d, **files})
    print(f"before={scenes[0]['id']}  after={scenes[1]['id']}")
    return scenes


def radiometry(path, date):
    import rasterio
    with rasterio.open(path) as src:
        scale, offset = src.scales[0], src.offsets[0]
    if scale == 1.0 and offset == 0.0:
        return 1e-4, (-0.1 if date >= BOA_OFFSET_DATE else 0.0)
    return scale, offset


def load_scene():
    import rasterio
    import rasterio.windows as rw
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform

    before, after = find_pair(INPUT_FILE)

    with rasterio.open(before["red"]) as src:
        ref_transform, ref_crs = src.transform, src.crs
        if CENTER is None:
            col, row = src.width / 2, src.height / 2
        else:
            x, y = warp_transform("EPSG:4326", src.crs, [CENTER[1]], [CENTER[0]])
            col, row = ~src.transform * (x[0], y[0])
        half = SIZE_PX // 2
        win = rw.Window(int(col) - half, int(row) - half, SIZE_PX, SIZE_PX)
        win = win.intersection(rw.Window(0, 0, src.width, src.height))
    win = rw.Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
    H, W = int(win.height), int(win.width)
    print(f"Window: {W}x{H}px")

    def read(path, resampling):
        with rasterio.open(path) as src:
            if src.crs != ref_crs:
                sys.exit(f"{os.path.basename(path)}: different CRS")
            if src.transform == ref_transform:
                w = win
            elif src.res[0] > 10:
                w = rw.from_bounds(*rw.bounds(win, ref_transform), transform=src.transform)
            else:
                sys.exit(f"{os.path.basename(path)}: 10 m grid differs from the before scene")
            return src.read(window=w, out_shape=(src.count, H, W), resampling=resampling)

    arrays = {}
    meta = {"crs": ref_crs.to_string(), "transform": tuple(rw.transform(win, ref_transform))[:6]}
    for when, sc in (("before", before), ("after", after)):
        for band in ("red", "nir", "swir16"):
            arrays[(when, band)] = read(sc[band], Resampling.bilinear)[0]
        arrays[(when, "visual")] = np.ascontiguousarray(
            read(sc["visual"], Resampling.bilinear).transpose(1, 2, 0)[..., ::-1])
        if "scl" in sc:
            arrays[(when, "scl")] = read(sc["scl"], Resampling.nearest)[0]
        scale, offset = radiometry(sc["red"], sc["date"])
        meta[when] = {"id": sc["id"], "date": sc["date"], "scale": scale, "offset": offset}
    return Scene(arrays, meta)


class Scene:
    def __init__(self, arrays, meta):
        self.raw, self.meta = arrays, meta
        self.h, self.w = arrays[("before", "red")].shape
        self.has_scl = ("before", "scl") in arrays and ("after", "scl") in arrays
        self.shift = (0.0, 0.0)
        self.shift = self._global_shift()
        for when in ("before", "after"):
            m = meta[when]
            red = self.refl(when, "red", (slice(None, None, 8), slice(None, None, 8)))
            print(f"{when:6s} scale={m['scale']} offset={m['offset']} | "
                  f"red reflectance p1={np.percentile(red, 1):.3f} median={np.median(red):.3f}")
        print(f"cloud mask: {'SCL' if self.has_scl else 'OFF'} | "
              f"median shift removed: dNDVI={self.shift[0]:+.3f} dNDBI={self.shift[1]:+.3f}")

    def refl(self, when, band, sl):
        m = self.meta[when]
        return np.clip(self.raw[(when, band)][sl].astype(np.float32) * m["scale"] + m["offset"], 0, 1)

    def rgb(self, when, sl):
        return self.raw[(when, "visual")][sl].copy()

    def indices(self, when, sl):
        red, nir, swir = (self.refl(when, n, sl) for n in ("red", "nir", "swir16"))
        return ((nir - red) / np.maximum(nir + red, 0.01),
                (swir - nir) / np.maximum(swir + nir, 0.01))

    def valid(self, sl):
        bad = np.zeros(self.raw[("before", "red")][sl].shape, bool)
        for when in ("before", "after"):
            bad |= self.raw[(when, "red")][sl] == 0
            if self.has_scl:
                bad |= np.isin(self.raw[(when, "scl")][sl], SCL_INVALID)
        return cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0

    def _global_shift(self):
        s = (slice(None, None, 8), slice(None, None, 8))
        nb, bb = self.indices("before", s)
        na, ba = self.indices("after", s)
        ok = self.valid(s)
        return float(np.median((na - nb)[ok])), float(np.median((ba - bb)[ok]))

    def latlon(self, col, row):
        try:
            from rasterio.transform import Affine
            from rasterio.warp import transform
        except ImportError:
            return None
        x, y = Affine(*self.meta["transform"]) * (col, row)
        lon, lat = transform(self.meta["crs"], "EPSG:4326", [x], [y])
        return lat[0], lon[0]


def patch_starts(n, size, step):
    if n <= size:
        return [0]
    s = list(range(0, n - size + 1, step))
    if s[-1] + size < n:
        s.append(n - size)
    return s


def index_composite(dndvi, dndbi):
    return (np.dstack([np.clip(dndbi / 0.4, 0, 1), np.clip(dndvi / 0.4, 0, 1),
                       np.clip(-dndvi / 0.4, 0, 1)]) * 255).astype(np.uint8)


def evidence_row(before, after, heat, idx_rgb, box, seg, pad=0.3):
    x, y, w, h = box
    H, W = before.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x1, y1, x2, y2 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    seg_t = cv2.resize(seg[y1:y2, x1:x2].astype(np.uint8), (TILE, TILE),
                       interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(seg_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # exact pixel the reported lat/lon refers to (box center), marked on AFTER so you can
    # check directly whether the reported coordinate sits on the object you expect
    cx = int((x + w / 2 - x1) / max(x2 - x1, 1) * TILE)
    cy = int((y + h / 2 - y1) / max(y2 - y1, 1) * TILE)
    panels = []
    for img, outline in ((before, True), (after, True), (heat, False), (idx_rgb, False)):
        p = cv2.resize(img[y1:y2, x1:x2], (TILE, TILE), interpolation=cv2.INTER_LINEAR)
        if outline:
            cv2.drawContours(p, cnts, -1, (0, 255, 255), 1)
        panels.append(p)
    marker = panels[1]
    cv2.drawMarker(marker, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    return np.hstack(panels)


def save_debug(path, before, after, valid, heat, idx_rgb, cand, found_masks):
    def gray(m):
        return cv2.cvtColor((m > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    after_o = after.copy()
    for seg in found_masks:
        cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(after_o, cnts, -1, (0, 255, 255), 1)
    tiles = [(before, "before"), (after_o, "after + SAM segments"), (gray(valid), "valid (white=clear)"),
             (heat, "DINO"), (idx_rgb, "INDEX (r=veg loss g=gain b=built)"), (gray(cand), "candidates")]
    for img, label in tiles:
        cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(path, np.vstack([np.hstack([t[0] for t in tiles[:3]]),
                                 np.hstack([t[0] for t in tiles[3:]])]))


def detect_patch(scene, y0, x0, n, models):
    sl = (slice(y0, min(y0 + PATCH_PX, scene.h)), slice(x0, min(x0 + PATCH_PX, scene.w)))

    before, after = scene.rgb("before", sl), scene.rgb("after", sl)
    valid = scene.valid(sl)
    nb, bb = scene.indices("before", sl)
    na, ba = scene.indices("after", sl)
    dndvi = cv2.GaussianBlur(na - nb - scene.shift[0], (5, 5), 0) * valid
    dndbi = cv2.GaussianBlur(ba - bb - scene.shift[1], (5, 5), 0) * valid
    idx_mag = np.maximum(np.abs(dndvi), np.abs(dndbi))

    dino_map = dsc.dino_change_map(before, after, *models["dino"])
    if dino_map.shape != valid.shape:
        dino_map = cv2.resize(dino_map, (valid.shape[1], valid.shape[0]))
    dino_map = dino_map * valid

    cand = ((dino_map > DINO_THRESH) | (idx_mag > INDEX_THRESH)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    found = []
    if np.count_nonzero(cand) >= MIN_AREA:
        score_map = (W_DINO * np.clip(dino_map / DINO_FULL, 0, 1)
                     + (1 - W_DINO) * np.clip(idx_mag / INDEX_FULL, 0, 1)).astype(np.float32)
        found = dsc.sam_change_regions(
            after, score_map, models["sam"], min_change_score=MIN_SCORE,
            min_area=MIN_AREA, max_regions=MAX_REGIONS, candidate_mask=cand,
            min_change_overlap=MIN_OVERLAP)
    found = [(b, s, np.asarray(m).astype(bool)) for b, s, m in found]
    too_big = [f for f in found if f[2].mean() > MAX_SEG_FRAC]
    found = [f for f in found if f[2].any() and f[2].mean() <= MAX_SEG_FRAC]

    print(f"  valid={valid.mean():.0%}  DINO p99={np.percentile(dino_map, 99):.2f}  "
          f"index p99={np.percentile(idx_mag, 99):.2f}  candidates={np.count_nonzero(cand)}px  "
          f"SAM kept={len(found)} dropped_too_big={len(too_big)}")

    heat = cv2.applyColorMap((np.clip(dino_map / 0.8, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    idx_rgb = index_composite(dndvi, dndbi)
    if DEBUG:
        save_debug(os.path.join(OUT_DIR, "debug", f"patch_{n:02d}.png"),
                   before, after, valid, heat, idx_rgb, cand, [f[2] for f in found])

    regions = []
    for box, score, seg in found:
        cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
        regions.append({
            "box": (box[0] + x0, box[1] + y0, box[2], box[3]),
            "score": float(score),
            "dino": float(dino_map[seg].mean()),
            "dndvi": float(dndvi[seg].mean()),
            "dndbi": float(dndbi[seg].mean()),
            "row": evidence_row(before, after, heat, idx_rgb, box, seg),
            "contours": list(cnts),
        })
    return regions


def merge_regions(regions, overlap=0.3):
    def overlap_of_smaller(a, b):
        iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        return 0.0 if iw <= 0 or ih <= 0 else iw * ih / min(a[2] * a[3], b[2] * b[3])

    groups = []
    for r in sorted(regions, key=lambda r: r["box"][2] * r["box"][3], reverse=True):
        for g in groups:
            if overlap_of_smaller(r["box"], g["box"]) >= overlap:
                g["score"] = max(g["score"], r["score"])
                g["contours"] = g["contours"] + r["contours"]
                break
        else:
            groups.append(dict(r))
    return sorted(groups, key=lambda g: g["score"], reverse=True)[:MAX_REGIONS]


def build_montage(regions, first):
    rows = []
    for i, r in enumerate(regions, start=first):
        bar = np.full((22, TILE * len(PANELS), 3), 255, np.uint8)
        for c, name in enumerate(PANELS):
            cv2.putText(bar, f"#{i} {name}", (c * TILE + 5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows += [bar, r["row"]]
    return np.vstack(rows)


def make_prompt(regions, first):
    stats = "\n".join(f"  #{i}: DINO={r['dino']:.2f}  dNDVI={r['dndvi']:+.2f}  dNDBI={r['dndbi']:+.2f}"
                      for i, r in enumerate(regions, start=first))
    return (
        f"Sentinel-2 imagery (10 m/pixel). The image has {len(regions)} rows, regions "
        f"#{first}-#{first + len(regions) - 1}. Each row has 4 panels: BEFORE, AFTER, DINO, INDEX. "
        "A yellow outline on BEFORE/AFTER marks the candidate segment.\n"
        "DINO = semantic change heatmap (red/yellow = content differs, blue = same).\n"
        "INDEX = spectral change: red = vegetation loss, green = vegetation gain, "
        "blue = more built-up/bare surface, magenta = vegetation replaced by built-up/bare "
        "(typical construction or clearing), black = no index change.\n"
        "Mean values inside each segment (DINO > 0.4 is strong; negative dNDVI = vegetation "
        "loss; positive dNDBI = more built-up/bare):\n" + stats + "\n"
        "For EACH region give one line:\n"
        "  #<number>: <category> — <one-sentence description>\n"
        "Category must be one of: building/construction, vegetation/agriculture, "
        "road/infrastructure, water/flooding, bare soil/land-use, vehicle/object, "
        "damage/disaster, no significant change, other.\n"
        "Judge from BEFORE vs AFTER first; DINO and INDEX are supporting evidence only. "
        "If before and after look the same, answer 'no significant change'.")


def ask_ollama(prompt, image_bgr):
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode montage")
    r = requests.post(f"{VLM_URL}/api/generate", timeout=VLM_TIMEOUT, json={
        "model": VLM_MODEL, "prompt": prompt, "stream": False, "think": False,
        "images": [base64.b64encode(buf.tobytes()).decode()],
        "keep_alive": "10m", "options": {"num_predict": VLM_TOKENS}})
    r.raise_for_status()
    body = r.json()
    text = (body.get("response") or "").strip()
    if not text:
        hint = " (thinking-only output: use the non-thinking `qwen3-vl:4b-instruct` tag)" if body.get("thinking") else ""
        raise RuntimeError(f"Ollama returned no text, done_reason={body.get('done_reason')}{hint}")
    return text


def classify(regions):
    answers = {}
    for start in range(0, len(regions), VLM_BATCH):
        chunk, first = regions[start:start + VLM_BATCH], start + 1
        montage = build_montage(chunk, first)
        cv2.imwrite(os.path.join(OUT_DIR, f"montage_{first:02d}.png"), montage)
        if not DESCRIBE:
            continue
        print(f"Asking {VLM_MODEL} about regions #{first}-#{first + len(chunk) - 1}...")
        try:
            text = ask_ollama(make_prompt(chunk, first), montage)
        except (requests.exceptions.RequestException, RuntimeError) as e:
            print(f"VLM step failed: {e}", file=sys.stderr)
            break
        for line in text.splitlines():
            m = re.match(r"^\W*#?(\d+)\s*[:.)]\s*(.*)$", line.strip())
            if m:
                answers[int(m.group(1))] = m.group(2)
    return answers


def sanity_check(scene):
    corners = {"top-left": (0, 0), "top-right": (scene.w, 0),
               "bottom-left": (0, scene.h), "bottom-right": (scene.w, scene.h)}
    print("Scene corners (check these fall where you expect on a map):")
    for name, (c, r) in corners.items():
        lat, lon = scene.latlon(c, r)
        print(f"  {name:12s}: {lat:.5f}, {lon:.5f}  https://www.google.com/maps?q={lat:.6f},{lon:.6f}")
    if CENTER is not None:
        lat, lon = scene.latlon(scene.w / 2, scene.h / 2)
        d = ((lat - CENTER[0]) * 111_320) ** 2 + ((lon - CENTER[1]) * 111_320 * np.cos(np.radians(lat))) ** 2
        print(f"CENTER round-trip: requested {CENTER}, window center is {lat:.5f}, {lon:.5f} "
              f"({d ** 0.5:.0f}m away, should be near 0)")
    else:
        print("CENTER is None: the window is centered on the raster, not on a place you chose. "
              "Set CENTER = (lat, lon) to look at a specific location.")


def main():
    os.makedirs(os.path.join(OUT_DIR, "debug"), exist_ok=True)

    scene = load_scene()
    sanity_check(scene)

    print("Loading DINOv2 and SAM...")
    dino = dsc.load_dino(DINO_MODEL)
    if dino[0] is None:
        sys.exit("DINOv2 failed to load.")
    sam = dsc.load_sam(SAM_CHECKPOINT, model_type=SAM_TYPE, points_per_side=SAM_POINTS,
                       min_mask_region_area=MIN_AREA)
    if sam is None:
        sys.exit(f"SAM failed to load (checkpoint: {SAM_CHECKPOINT}).")
    models = {"dino": dino, "sam": sam}

    step = max(PATCH_PX - OVERLAP_PX, 1)
    grid = list(enumerate(itertools.product(patch_starts(scene.h, PATCH_PX, step),
                                            patch_starts(scene.w, PATCH_PX, step)), start=1))

    regions = []
    for n, (y0, x0) in grid:
        print(f"[patch {n}/{len(grid)}] x={x0} y={y0}")
        regions += detect_patch(scene, y0, x0, n, models)

    regions = merge_regions(regions)
    print(f"\n{len(regions)} region(s) after merging patches.")

    overlay = scene.rgb("after", (slice(None), slice(None)))
    for i, r in enumerate(regions, start=1):
        cv2.drawContours(overlay, r["contours"], -1, (0, 0, 255), 2)
        cv2.putText(overlay, f"#{i}", (r["box"][0], max(14, r["box"][1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "overlay.png"), overlay)

    if not regions:
        print("No regions. Check the debug sheets or lower DINO_THRESH / MIN_SCORE.")
        return

    answers = classify(regions)

    lines = []
    for i, r in enumerate(regions, start=1):
        x, y, w, h = r["box"]
        ll = scene.latlon(x + w / 2, y + h / 2)
        where = (f"({ll[0]:.5f}, {ll[1]:.5f}) [https://www.google.com/maps?q={ll[0]:.6f},{ll[1]:.6f}]"
                 if ll else f"pixel bbox x={x} y={y} w={w} h={h}")
        line = (f"#{i} @ {where} score={r['score']:.2f} DINO={r['dino']:.2f} "
                f"dNDVI={r['dndvi']:+.2f} dNDBI={r['dndbi']:+.2f}")
        if i in answers:
            line += f" | {answers[i]}"
        lines.append(line)
    print("\n--- Regions ---\n" + "\n".join(lines))
    with open(os.path.join(OUT_DIR, "regions.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()