"""Sanity test on synthetic data: precision/recall of new core vs the OLD percentile method."""
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import change_core as cc
from make_synthetic_data import make, SIZE
from scipy import ndimage as ndi

def old_method(b, a):
    """Faithful copy of the previous calculate_change() maths (DN/10000, no offset handling)."""
    nd = cc.nd
    ndvi_b, ndvi_a = nd(b["nir"], b["red"]), nd(a["nir"], a["red"])
    ndwi_b, ndwi_a = nd(b["green"], b["nir"]), nd(a["green"], a["nir"])
    ndbi_b, ndbi_a = nd(b["swir16"], b["nir"]), nd(a["swir16"], a["nir"])
    cand = (ndvi_b < .35) & (ndvi_a < .35) & (ndwi_b < .2) & (ndwi_a < .2)
    idx = (abs(ndbi_a - ndbi_b) + abs(ndvi_a - ndvi_b) + abs(ndwi_a - ndwi_b)) / 3
    vis = np.abs(a["vis"].astype(np.float32) - b["vis"].astype(np.float32)).mean(-1) / 255
    score = .5 * idx + .25 * vis
    thr = max(.12, np.percentile(score[cand], 90))
    return cand & (score >= thr)

def prf(pred, truth):
    tp = (pred & truth).sum(); fp = (pred & ~truth).sum(); fn = (~pred & truth).sum()
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), int(fp)

if __name__ == "__main__":
    d = Path(tempfile.mkdtemp()); truth = make(d)
    scenes = cc.find_scenes(d); (db, fb), (da, fa) = list(scenes.items())
    print("scenes:", db, da)
    B, A, meta = cc.load_aoi_pair(fb, fa, 0, 0, SIZE, SIZE)
    print("calibration after:", A.calib)
    truth_d = ndi.binary_dilation(truth, iterations=2)
    for sens in ("strict", "balanced", "sensitive"):
        res = cc.detect_changes(B, A, cc.Params(sensitivity=sens), px_area_m2=100.0)
        p, r, fp = prf(res.mask.astype(bool), truth_d)
        print(f"NEW [{sens:9s}] precision={p:.2f} recall={r:.2f} false-pos px={fp} regions={len(res.regions)}")
    print("shift px:", res.diag["shift_px"], " norm fit (slope,icpt):", {k: tuple(round(x, 3) for x in v) for k, v in res.diag["normalisation_fit"].items()})
    objs = {"building1": (100,100,5,5), "building2": (300,500,4,6), "road": (600,200,2,90),
            "veg loss": (400,300,30,30), "veg gain": (700,600,25,25)}
    for name,(y,x,h,w) in objs.items():
        ov = res.mask[y:y+h, x:x+w].mean()
        print(f"   truth object {name:10s} covered: {ov:.0%}")
    print("   moisture-patch FP px:", int(res.mask[200:260,650:710].sum()), " cloud-area px:", int(res.mask[40:80,700:750].sum()))
    print("funnel:"); [print("   ", n, v) for n, v in res.funnel]
    for r in res.regions[:12]:
        print("   ", r["id"], r["box"], r["area_px"], r["class_name"][:30], r["significance"], r["spectral_angle_deg"])
    b = {k: B.refl[..., i] for i, k in enumerate(cc.BANDS)}; b["vis"] = B.rgb
    a = {k: A.refl[..., i] for i, k in enumerate(cc.BANDS)}; a["vis"] = A.rgb
    old = old_method(b, a)
    p, r, fp = prf(old, truth_d)
    print(f"OLD percentile method: precision={p:.2f} recall={r:.2f} false-pos px={fp}")
    # unchanged-scene test: same image twice -> should report ~nothing
    res0 = cc.detect_changes(B, B, cc.Params(), px_area_m2=100.0)
    print("identical pair -> regions:", len(res0.regions), " (old method would flag ~10% of candidates)")
