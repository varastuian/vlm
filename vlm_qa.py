"""
vlm_qa.py - local VLM (Ollama) layer for the change-detection frontend.

Two jobs:
  * audit()  - second-stage false-positive filter. The VLM looks at BEFORE | AFTER |
               CHANGE crops for each candidate region and returns strict JSON
               {id, real_change, category, reason}.
  * stream_answer() - answers the user's free-form question about the changes,
               grounded in (a) a montage image and (b) a numeric evidence table
               computed by change_core, with earlier Q&A turns kept as context.

Small local VLMs (qwen3-vl 4B) are unreliable when given many images or long
prompts, so: one montage image, at most `max_regions` rows, terse prompts.
"""
from __future__ import annotations

import base64
import json
import re

import cv2
import numpy as np
import requests

TILE = 224
CATEGORIES = (
    "new building / structure", "road / pavement", "construction site / bare ground",
    "agriculture / vegetation change", "water change", "cloud / shadow / haze",
    "soil moisture / illumination / seasonal", "image misalignment / noise", "other",
)

SYSTEM_PROMPT = (
    "You are a cautious remote-sensing analyst comparing two Sentinel-2 images of the same place "
    "(BEFORE = earlier, AFTER = later). One pixel is 10 m x 10 m: a house is a few pixels, a road "
    "is a thin line 1-3 pixels wide. You get a montage (one row per numbered region: BEFORE | AFTER | "
    "CHANGE outline) and a table of measurements. Rules:\n"
    "1. A NEW BUILDING is a small, compact, near-rectangular bright or distinct object in AFTER that is "
    "absent in BEFORE. A NEW ROAD is a thin, long, mostly straight line in AFTER that is absent in BEFORE.\n"
    "2. A CONSTRUCTION SITE is an area of disturbed ground / bare soil in AFTER that was vegetated or "
    "different in BEFORE. It appears as irregular patches, soil exposure, material piles, or excavation. "
    "It does NOT need to be rectangular - irregular outlines are EXPECTED for construction.\n"
    "3. Colour, brightness, moisture, vegetation-vigour, ploughing/harvest or shadow differences of "
    "fields or natural ground are NOT buildings or roads, even if the outlined area is rectangular. "
    "Irregular or blob-like outlines are not buildings or roads (but CAN be construction sites).\n"
    "4. Judge from the images first. The table's 'spectral hint' is an unreliable rule of thumb - never "
    "repeat it as a fact. 'shape' is a geometric measurement and is more trustworthy.\n"
    "5. If no region clearly satisfies rule 1 or 2, answer that none of the regions look like new buildings, "
    "roads, or construction sites. Saying 'none' or 'unclear' is a good answer.\n"
    "6. Refer to regions as #<id>. Never invent regions, dates or objects. Be short and concrete."
)


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def list_models(url):
    """Installed Ollama model tags (raises requests.RequestException if unreachable)."""
    r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
    r.raise_for_status()
    return sorted(m["name"] for m in r.json().get("models", []))


def _encode(img_rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Could not encode image for the VLM.")
    return base64.b64encode(buf.tobytes()).decode()


def _chat_stream(url, model, messages, fmt=None, max_tokens=900, timeout=900):
    payload = {"model": model, "messages": messages, "stream": True, "think": False,
               "keep_alive": "10m", "options": {"num_predict": max_tokens, "temperature": 0.2}}
    if fmt:
        payload["format"] = fmt
    endpoint = f"{url.rstrip('/')}/api/chat"
    for attempt in range(2):
        try:
            resp = requests.post(endpoint, json=payload, stream=True, timeout=timeout)
        except requests.ConnectionError as e:
            raise RuntimeError(f"Cannot reach Ollama at {url}. Start it with `ollama serve`.") from e
        if resp.status_code == 400 and "think" in resp.text.lower() and attempt == 0:
            payload.pop("think", None)      # model / Ollama version without thinking support
            continue
        if resp.status_code == 404:
            raise RuntimeError(f"Model '{model}' not found. Run `ollama pull {model}`.")
        resp.raise_for_status()
        break
    got = False
    for line in resp.iter_lines():
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("error"):
            raise RuntimeError(obj["error"])
        piece = (obj.get("message") or {}).get("content", "")
        if piece:
            got = True
            yield piece
    if not got:
        raise RuntimeError("The VLM returned no text (thinking-only model? use an '-instruct' tag).")


# --------------------------------------------------------------------------- #
# Evidence: montage + table
# --------------------------------------------------------------------------- #
def _crop_box(box, shape, pad_frac=0.8, min_side=24):
    x, y, w, h = box
    H, W = shape[:2]
    side = max(w, h, min_side // 2)
    px = int(max(min_side - w, 0) / 2 + pad_frac * side)
    py = int(max(min_side - h, 0) / 2 + pad_frac * side)
    return max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)


def build_montage(before_rgb, after_rgb, labels, regions, max_regions=6):
    """Rows: BEFORE | AFTER | CHANGE overlay for the first `max_regions` regions."""
    rows = []
    for r in regions[:max_regions]:
        x1, y1, x2, y2 = _crop_box(r["box"], before_rgb.shape)
        seg = (labels[y1:y2, x1:x2] == r["id"]).astype(np.uint8)
        seg_t = cv2.resize(seg, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        contours = cv2.findContours(seg_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        panels = []
        for img in (before_rgb, after_rgb):
            p = cv2.resize(img[y1:y2, x1:x2], (TILE, TILE), interpolation=cv2.INTER_CUBIC)
            panels.append(p)
        over = panels[1].copy()
        cv2.drawContours(over, contours, -1, (255, 255, 0), 1)
        panels.append(over)
        bar = np.full((22, TILE * 3, 3), 255, np.uint8)
        for c, name in enumerate(("BEFORE", "AFTER", "CHANGE (yellow outline)")):
            cv2.putText(bar, f"#{r['id']} {name}", (c * TILE + 5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1, cv2.LINE_AA)
        rows += [bar, np.hstack(panels)]
    if not rows:
        raise ValueError("No regions to show the VLM.")
    return np.vstack(rows)


def evidence_table(regions, before_date, after_date, aoi_desc, max_regions=6):
    lines = [f"BEFORE date {_fmt_date(before_date)}, AFTER date {_fmt_date(after_date)}. {aoi_desc}",
             f"Showing {min(len(regions), max_regions)} of {len(regions)} detected regions "
             "(sorted by size x significance):"]
    for r in regions[:max_regions]:
        shape = r.get("shape", "unknown")
        geo = ""
        if "rectangularity" in r:
            geo = (f"shape={shape} (rectangularity {r['rectangularity']}, solidity {r['solidity']}, "
                   f"elongation {r['aspect']}), ")
        lines.append(
            f"#{r['id']}: position={r['position']}, size={r['area_px']} px ({r['area_ha']} ha), {geo}"
            f"NDVI after={r.get('ndvi_after', 'n/a')}, dNDVI={r['dNDVI']:+.2f}, "
            f"dBrightness={r['dBrightness']:+.3f}, spectral angle={r['spectral_angle_deg']} deg, "
            f"spectral hint (unreliable)='{r['class_name']}'")
    lines.append("dNDVI<0 = less vegetation; dBrightness>0 = brighter; small spectral angle = mere "
                 "brightness shift. rectangularity 1.0 = perfect rectangle; elongation>3.5 = long thin line.")
    return "\n".join(lines)


def _fmt_date(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else str(d)


# --------------------------------------------------------------------------- #
# Audit (false-positive filter)
# --------------------------------------------------------------------------- #
def audit(url, model, montage_rgb, table, region_ids, timeout=900, max_tokens=2048):
    """Returns {id: {'real_change': bool, 'category': str, 'reason': str}}."""
    prompt = (
        f"{table}\n\nFor EACH region in the montage decide whether the BEFORE and AFTER panels show "
        "a REAL change (new building, new road, construction site) or a FALSE POSITIVE (cloud, shadow, haze, "
        "soil moisture, lighting, season, crop or vegetation colour change, ploughing, misalignment, noise). "
        "Set real_change=true for:\n"
        "  - A distinct compact rectangular object (building) absent in BEFORE\n"
        "  - A thin straight line (road) absent in BEFORE\n"
        "  - An area of disturbed ground / bare soil / excavation / material piles (construction site) "
        "that was different in BEFORE - irregular outline is EXPECTED for construction\n"
        "If the panels look alike or the change is only colour/brightness without structural change, it is a "
        "false positive.\n"
        'Reply with JSON only: {"regions":[{"id":<number>,"real_change":true|false,'
        f'"category":"<one of: {"; ".join(CATEGORIES)}>","reason":"<max 15 words>"}}]}}')
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": [_encode(montage_rgb)]}]
    text = "".join(_chat_stream(url, model, msgs, fmt="json", max_tokens=max_tokens, timeout=timeout))
    return parse_audit(text, region_ids)


def parse_audit(text, region_ids):
    # Try direct JSON parse first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks or prose
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            json_str = m.group(0)
            # Fix common JSON issues: trailing commas, missing quotes, etc.
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)  # remove trailing commas
            json_str = re.sub(r"([{,])\s*(\w+)\s*:", r'\1"\2":', json_str)  # quote unquoted keys
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
    items = data.get("regions", data if isinstance(data, list) else [])
    out = {}
    for it in items:
        try:
            rid = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if rid in region_ids:
            real = it.get("real_change")
            if isinstance(real, str):
                real = real.strip().lower() in ("true", "yes", "1")
            out[rid] = {"real_change": bool(real), "category": str(it.get("category", "")),
                        "reason": str(it.get("reason", ""))}
    return out


# --------------------------------------------------------------------------- #
# Free-form Q&A
# --------------------------------------------------------------------------- #
def stream_answer(url, model, question, history, montage_rgb, table, timeout=900, max_tokens=700):
    """Generator of text chunks. `history` = [{'role','content'}, ...] (text only)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append({"role": "user",
                 "content": f"EVIDENCE TABLE\n{table}\n\nQUESTION: {question}",
                 "images": [_encode(montage_rgb)]})
    yield from _chat_stream(url, model, msgs, max_tokens=max_tokens, timeout=timeout)