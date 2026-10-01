"""
VLM stage: after geometric fusion, every candidate region is shown to a
vision-language model (local Ollama, e.g. qwen3-vl) as a montage row:

    BEFORE | AFTER | FUSED heatmap | per-signal votes

The VLM is the *final arbiter*: it confirms or rejects each region, assigns a
category (construction, vegetation loss, flooding, ...) and writes a one-line
description. The numeric evidence (per-signal scores inside the segment) is
included in the prompt so the model can reason over it, but the pixel panels
come first.

Batched: several rows go into one montage to keep a small local VLM happy.
"""
import base64
import re

import cv2
import numpy as np
import requests

TILE = 200          # px per panel
PANELS = ("BEFORE", "AFTER", "FUSED", "VOTES")

CATEGORIES = (
    "building/construction", "vegetation/agriculture", "road/infrastructure",
    "water/flooding", "bare soil/land-use", "vehicle/object",
    "damage/disaster", "no significant change", "other",
)

VOTE_COLORS = {
    "dino": (255, 80, 80),      # BGR red
    "bit": (80, 80, 255),       # BGR orange-red
    "spectral": (80, 255, 80),  # BGR green
    "structure": (255, 200, 80),  # BGR light blue
}


# --------------------------------------------------------------------------- #
# Montage
# --------------------------------------------------------------------------- #
def votes_panel(seg_shape, seg, fired, size=(TILE, TILE)):
    """Colored overlays showing which detectors fired inside the segment."""
    panel = np.full((seg_shape[0], seg_shape[1], 3), 255, np.uint8)
    panel[:] = 30
    for name in ("spectral", "bit", "dino"):  # later draws win
        if name in fired:
            panel[seg] = VOTE_COLORS[name]
    label = " ".join(sorted(fired)) if fired else "none"
    cv2.putText(panel, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.resize(panel, size, interpolation=cv2.INTER_NEAREST)


def fused_panel(fused_crop, size=(TILE, TILE)):
    heat = cv2.applyColorMap((np.clip(fused_crop, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)
    return cv2.resize(heat, size, interpolation=cv2.INTER_LINEAR)


def evidence_row(before, after, fused, region, pad=0.3):
    """One montage row for a region: BEFORE | AFTER | FUSED | VOTES."""
    x, y, w, h = region["box"]
    H, W = before.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(W, x + w + px), min(H, y + h + py)

    seg = region["seg"]
    seg_crop = seg[y1:y2, x1:x2].astype(np.uint8)
    seg_t = cv2.resize(seg_crop, (TILE, TILE), interpolation=cv2.INTER_NEAREST)

    panels = []
    outline = cv2.findContours(seg_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for img, is_map in ((before, False), (after, False)):
        p = cv2.resize(img[y1:y2, x1:x2], (TILE, TILE), interpolation=cv2.INTER_LINEAR)
        cv2.drawContours(p, outline, -1, (0, 255, 255), 1)
        panels.append(p)
    panels.append(fused_panel(fused[y1:y2, x1:x2]))
    panels.append(votes_panel(seg_crop.shape, seg_crop > 0, region.get("sources", set())))
    return np.hstack(panels)


def build_montage(regions, first, fused, before, after):
    rows = []
    for i, r in enumerate(regions, start=first):
        bar = np.full((22, TILE * len(PANELS), 3), 255, np.uint8)
        for c, name in enumerate(PANELS):
            cv2.putText(bar, f"#{i} {name}", (c * TILE + 5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows += [bar, evidence_row(before, after, fused, r)]
    return np.vstack(rows)


# --------------------------------------------------------------------------- #
# Prompt + client
# --------------------------------------------------------------------------- #
def make_prompt(regions, first):
    lines = []
    for i, r in enumerate(regions, start=first):
        src = "+".join(sorted(r.get("sources", []))) or "none"
        means = r.get("signal_means") or {}
        mean_txt = " ".join(f"{k}={v:.2f}" for k, v in sorted(means.items()))
        ctx_txt = " [bare-soil region]" if (r.get("context") or {}).get("is_bare_soil") else ""
        lines.append(
            f"  #{i}: fused={r['score']:.2f}  detectors_fired={src}  {mean_txt}{ctx_txt}")
    stats = "\n".join(lines)
    return (
        "You are auditing candidate change regions between two Sentinel-2 satellite "
        f"images of the same location (BEFORE = earlier date, AFTER = later date). "
        f"The montage has {len(regions)} numbered rows (#{first}-#{first + len(regions) - 1}). "
        "Each row: BEFORE panel, AFTER panel, FUSED heatmap (bright = the fusion model "
        "believes change), VOTES panel (which detectors agree; colors = detector names). "
        "A yellow outline marks the candidate segment on BEFORE/AFTER.\n"
        "Numeric evidence per region:\n" + stats + "\n"
        "A region should be confirmed only if you can SEE a difference between the "
        "BEFORE and AFTER panels in that row. Ignore seasonal color drift; look for "
        "new/removed structures, roads, water, bare ground or vehicles.\n"
        "In bare-soil/arid land (tagged [bare-soil region]), brightness or texture "
        "shifts from soil moisture, plowing, tilling or sun angle are NOT real "
        "change — but a genuinely new building, road or wall still is.\n"
        "For EACH region output exactly one line:\n"
        "  #<number>: <category> — <one-sentence description>; confirm or reject\n"
        "Category must be one of: " + ", ".join(CATEGORIES) + ".\n"
        "If the panels look identical, answer 'no significant change' (a rejection). "
        "End confirmed descriptions with 'CONFIRMED' and rejected ones with 'REJECTED'.")


def ask_ollama(prompt, image_bgr, url="http://localhost:11434",
               model="qwen3-vl:4b-instruct", timeout=600, max_tokens=1024,
               keep_alive="10m"):
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode montage as JPEG")
    resp = requests.post(
        f"{url}/api/generate", timeout=timeout,
        json={
            "model": model, "prompt": prompt, "stream": False, "think": False,
            "images": [base64.b64encode(buf.tobytes()).decode()],
            "keep_alive": keep_alive,
            "options": {"num_predict": max_tokens},
        })
    resp.raise_for_status()
    body = resp.json()
    text = (body.get("response") or "").strip()
    if not text:
        hint = (" The model produced only 'thinking' — use the non-thinking "
                "`qwen3-vl:4b-instruct` tag or raise max_tokens."
                if body.get("thinking") else "")
        raise RuntimeError(f"Ollama returned no text (done_reason={body.get('done_reason')}).{hint}")
    return text


LINE_RE = re.compile(r"^\W*#?(\d+)\s*[:.)\-]\s*(.+)$", re.IGNORECASE)


def parse_answers(text):
    """{'#<n>': '<category> — <description> ... CONFIRMED/REJECTED'}"""
    answers = {}
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            answers[int(m.group(1))] = m.group(2).strip()
    return answers


def is_confirmed(answer_text):
    a = answer_text.upper()
    if "REJECTED" in a and "CONFIRMED" not in a.replace("REJECTED", ""):
        return False
    if "NO SIGNIFICANT CHANGE" in a:
        return False
    return "CONFIRMED" in a


# --------------------------------------------------------------------------- #
# Public entry: classify regions in batches
# --------------------------------------------------------------------------- #
def classify_regions(regions, fused, before, after, out_dir=None,
                     url="http://localhost:11434", model="qwen3-vl:4b-instruct",
                     batch=4, timeout=600, max_tokens=1024, keep_alive="10m",
                     save_montages=True):
    """Returns {region_index(1-based): {'text':..., 'confirmed': bool}}."""
    answers = {}
    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
    for start in range(0, len(regions), batch):
        chunk = regions[start:start + batch]
        first = start + 1
        montage = build_montage(chunk, first, fused, before, after)
        if out_dir and save_montages:
            cv2.imwrite(os.path.join(out_dir, f"montage_{first:02d}.png"), montage)
        prompt = make_prompt(chunk, first)
        text = ask_ollama(prompt, montage, url=url, model=model,
                          timeout=timeout, max_tokens=max_tokens, keep_alive=keep_alive)
        for idx, answer in parse_answers(text).items():
            answers[idx] = {"text": answer, "confirmed": is_confirmed(answer)}
    return answers
