"""
fvtools: helpers for a small, theory-motivated analysis of COCO-FreeView fixations.

Question: does what predicts gaze change over the course of free viewing?
For every fixation index k we compare three maps:
  * center    - an empirical center-bias map (two-fold: built from the *other* half of images)
  * contrast  - a simple low-level feature-contrast map (intensity, colour-opponent, orientation)
  * objects   - COCO object masks (any annotated object)
and ask how often fixation k lands on an object.

Every effect is reported next to a shuffled-image control: the same map scored at fixations
recorded on a *different* image. Whatever survives that control is image-specific, not just
"people look at the middle of the screen".

Coordinate frames
  stimulus : the 1680x1050 display image (COCO image resized + padded to fit)
  grid     : the stimulus downsampled by GRID_DS (default 5 -> 336x210), where maps live
  original : the COCO image as annotated in instances_*.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import fftconvolve

STIM_W, STIM_H = 1680, 1050
GRID_DS = 5
GRID_W, GRID_H = STIM_W // GRID_DS, STIM_H // GRID_DS

# ----------------------------------------------------------------------------- loading


def load_fixations(path: str) -> pd.DataFrame:
    """Load the COCO-FreeView JSON (list of scanpath dicts) into one row per fixation.

    Expected keys per scanpath (COCO-Search18 convention): name, subject, X, Y, T, and
    optionally split. Unknown keys are ignored.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):  # tolerate {"scanpaths": [...]} style wrappers
        data = next(v for v in data.values() if isinstance(v, list))
    rows = []
    for sid, sp in enumerate(data):
        X = np.asarray(sp["X"], float)
        Y = np.asarray(sp["Y"], float)
        T = np.asarray(sp.get("T", [np.nan] * len(X)), float)
        for k in range(len(X)):
            rows.append((sid, sp["name"], sp.get("subject"), sp.get("split"), k, X[k], Y[k], T[k],
                         k == len(X) - 1))
    df = pd.DataFrame(rows, columns=["scan", "image", "subject", "split", "idx", "x", "y", "dur",
                                     "is_last"])
    return df


def to_stimulus_frame(df: pd.DataFrame, frame: str = "auto") -> tuple[pd.DataFrame, str]:
    """Make sure x/y are in 1680x1050 stimulus pixels.

    Some releases ship fixations rescaled to 512x320. 'auto' infers this from the range.
    """
    if frame == "auto":
        frame = "512x320" if (df.x.max() <= 512 * 1.05 and df.y.max() <= 320 * 1.05) else "1680x1050"
    out = df.copy()
    if frame == "512x320":
        out["x"] *= STIM_W / 512
        out["y"] *= STIM_H / 320
    return out, frame


def index_stimuli(stim_dir: str) -> dict[str, str]:
    """Map image file name -> path, searching stim_dir recursively (category sub-folders are fine)."""
    idx = {}
    for root, _, files in os.walk(stim_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                idx.setdefault(fn, os.path.join(root, fn))
    return idx


class CocoIndex:
    """Thin wrapper over pycocotools that looks images up by file name across splits."""

    def __init__(self, ann_files: list[str]):
        from pycocotools.coco import COCO  # imported lazily so the module loads without it

        self.cocos = []
        self.by_name = {}
        for fpath in ann_files:
            c = COCO(fpath)
            self.cocos.append(c)
            for img in c.dataset["images"]:
                self.by_name[img["file_name"]] = (len(self.cocos) - 1, img)
        c0 = self.cocos[0]
        self.cat_name = {c["id"]: c["name"] for c in c0.dataset["categories"]}
        self.cat_super = {c["id"]: c["supercategory"] for c in c0.dataset["categories"]}

    def __contains__(self, name):
        return name in self.by_name

    def image(self, name):
        return self.by_name[name][1]

    def label_map(self, name):
        """Per-pixel index of the *smallest* annotation covering it (-1 = background), plus ann list.

        Painting large objects first and small ones last means a fixation on a cup on a table
        counts as the cup, which matches how gaze usually relates to objects.
        """
        ci, img = self.by_name[name]
        c = self.cocos[ci]
        anns = c.loadAnns(c.getAnnIds(imgIds=img["id"], iscrowd=None))
        anns = sorted(anns, key=lambda a: -a["area"])
        lab = np.full((img["height"], img["width"]), -1, np.int32)
        for i, a in enumerate(anns):
            lab[c.annToMask(a) > 0] = i
        return lab, anns


# ------------------------------------------------------------------- stimulus alignment


def content_box(stim: np.ndarray, tol: int = 8) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the non-padding region of a stimulus image.

    Padding is assumed to be a uniform colour; we take the most common corner colour as padding.
    """
    a = stim.astype(np.int16)
    h, w = a.shape[:2]
    corners = [tuple(a[0, 0]), tuple(a[0, -1]), tuple(a[-1, 0]), tuple(a[-1, -1])]
    pad = np.array(max(set(corners), key=corners.count))
    differs = np.abs(a - pad).max(axis=2) > tol
    rows = np.where(differs.mean(axis=1) > 0.02)[0]
    cols = np.where(differs.mean(axis=0) > 0.02)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)


@dataclass
class Transform:
    """original (COCO) pixel -> stimulus pixel:  xs = sx * xo + ox,  ys = sy * yo + oy"""

    sx: float
    sy: float
    ox: float
    oy: float
    mode: str
    iou: float  # agreement between the predicted and the detected content box

    def box(self, w, h):
        return self.ox, self.oy, self.ox + self.sx * w, self.oy + self.sy * h


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def fit_transform(orig_w: int, orig_h: int, stim: np.ndarray | None) -> Transform:
    """Work out how the COCO image was placed on the 1680x1050 display.

    The dataset papers say images were 'resized and padded'; rather than assume where the
    padding went we try three placements and keep the one that matches the stimulus best.
    """
    s = min(STIM_W / orig_w, STIM_H / orig_h)
    cands = {
        "fit_center": (s, s, (STIM_W - s * orig_w) / 2, (STIM_H - s * orig_h) / 2),
        "fit_topleft": (s, s, 0.0, 0.0),
        "stretch": (STIM_W / orig_w, STIM_H / orig_h, 0.0, 0.0),
    }
    if stim is None:
        sx, sy, ox, oy = cands["fit_center"]
        return Transform(sx, sy, ox, oy, "fit_center(assumed)", np.nan)
    detected = content_box(stim)
    best = None
    for mode, (sx, sy, ox, oy) in cands.items():
        iou = _iou((ox, oy, ox + sx * orig_w, oy + sy * orig_h), detected)
        if best is None or iou > best.iou:
            best = Transform(sx, sy, ox, oy, mode, iou)
    return best


def grid_labels(lab: np.ndarray, tf: Transform) -> np.ndarray:
    """Resample an original-resolution label map onto the stimulus grid.

    Returns int32 (GRID_H, GRID_W): annotation index, -1 background, -2 padding.
    """
    gy, gx = np.mgrid[0:GRID_H, 0:GRID_W]
    xs = (gx + 0.5) * GRID_DS
    ys = (gy + 0.5) * GRID_DS
    xo = np.floor((xs - tf.ox) / tf.sx).astype(int)
    yo = np.floor((ys - tf.oy) / tf.sy).astype(int)
    h, w = lab.shape
    inside = (xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)
    out = np.full((GRID_H, GRID_W), -2, np.int32)
    out[inside] = lab[yo[inside], xo[inside]]
    return out


def to_grid(x, y):
    gx = np.clip((np.asarray(x) / GRID_DS).astype(int), 0, GRID_W - 1)
    gy = np.clip((np.asarray(y) / GRID_DS).astype(int), 0, GRID_H - 1)
    return gx, gy


# ------------------------------------------------------------------------------- maps


def _gabor(theta, freq=0.18, sigma=2.5, size=13):
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1].astype(float)
    xr = x * np.cos(theta) + y * np.sin(theta)
    env = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    k = env * np.exp(1j * 2 * np.pi * freq * xr)
    return k - k.mean()


_GABORS = [_gabor(t) for t in np.deg2rad([0, 45, 90, 135])]


def feature_contrast_map(rgb: np.ndarray, combine: str = "max",
                         scales=((1.0, 4.0), (2.0, 8.0))) -> np.ndarray:
    """A deliberately simple low-level saliency baseline on an RGB crop (float, 0..1).

    Channels: intensity, red-green and blue-yellow opponency, and oriented energy at 4
    orientations. Contrast = |centre - surround| (difference of Gaussians); for orientation,
    centre energy minus surround energy *of the same orientation*, i.e. iso-feature suppression.

    combine='max' takes the strongest channel at each location (the V1-saliency-hypothesis
    style of combination, Li 2002); 'mean' averages channels (closer to Itti & Koch 1998).
    This is NOT a model of V1 - it is a transparent low-level baseline.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    inten = (r + g + b) / 3
    feats = {"intensity": inten, "rg": r - g, "by": b - (r + g) / 2}
    chans = []
    for f in feats.values():
        acc = np.zeros_like(f)
        for sc, ss in scales:
            acc += np.abs(ndimage.gaussian_filter(f, sc) - ndimage.gaussian_filter(f, ss))
        chans.append(acc)
    for k in _GABORS:
        e = np.abs(fftconvolve(inten, k, mode="same"))
        acc = np.zeros_like(e)
        for sc, ss in scales:
            acc += np.clip(ndimage.gaussian_filter(e, sc) - ndimage.gaussian_filter(e, ss), 0, None)
        chans.append(acc)
    norm = []
    for c in chans:
        m = c.max()
        norm.append(c / m if m > 0 else c)
    stack = np.stack(norm)
    sal = stack.max(0) if combine == "max" else stack.mean(0)
    return ndimage.gaussian_filter(sal, 1.5)


def contrast_map_on_grid(stim: np.ndarray, gbox: tuple[int, int, int, int], combine="max") -> np.ndarray:
    """Feature-contrast map on the stimulus grid, computed on the content crop only.

    Computing on the crop matters: the edge between image and padding is the strongest
    'contrast' on the screen and would otherwise dominate the map. Padding is NaN.
    """
    small = stim[::GRID_DS, ::GRID_DS][:GRID_H, :GRID_W].astype(float) / 255.0
    x0, y0, x1, y1 = gbox
    out = np.full((GRID_H, GRID_W), np.nan)
    out[y0:y1, x0:x1] = feature_contrast_map(small[y0:y1, x0:x1], combine=combine)
    return out


def center_maps(fix: pd.DataFrame, images: list[str], sigma_grid: float = 8.0, seed: int = 0):
    """Two-fold empirical centre-bias maps: images in fold A are scored with the density from B."""
    rng = np.random.default_rng(seed)
    imgs = np.array(sorted(images))
    fold = dict(zip(imgs, rng.permutation(len(imgs)) % 2))
    dens = []
    for f in (0, 1):
        sub = fix[fix.image.map(fold) == f]
        gx, gy = to_grid(sub.x, sub.y)
        d = np.zeros((GRID_H, GRID_W))
        np.add.at(d, (gy, gx), 1)
        d = ndimage.gaussian_filter(d, sigma_grid)
        dens.append(d / d.sum())
    # image in fold f gets the density estimated from the other fold
    return {im: dens[1 - fold[im]] for im in imgs}


def zscore_in(m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """z-score a map using only valid (non-padding) pixels, so NSS isn't inflated by padding."""
    v = m[valid]
    sd = v.std()
    z = np.full_like(m, np.nan, dtype=float)
    z[valid] = (v - v.mean()) / (sd if sd > 0 else 1.0)
    return z


# ---------------------------------------------------------------------------- analysis


def score_image(name, stim, coco: CocoIndex, fix_own: pd.DataFrame, fix_donor: pd.DataFrame,
                cmap: np.ndarray, combine="max", keep_maps=False):
    """Score one image. Returns (own_rows, shuffled_rows, info dict).

    fix_own   : fixations recorded on this image
    fix_donor : fixations recorded on a different image (shuffled control), same columns
    """
    img = coco.image(name)
    tf = fit_transform(img["width"], img["height"], stim)
    lab, anns = coco.label_map(name)
    glab = grid_labels(lab, tf)
    valid = glab != -2
    ys, xs = np.where(valid)
    gbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    obj = (glab >= 0).astype(float)
    maps = {
        "center": zscore_in(cmap, valid),
        "contrast": zscore_in(np.nan_to_num(contrast_map_on_grid(stim, gbox, combine)), valid),
        "objects": zscore_in(ndimage.gaussian_filter(obj, 1.0), valid),
    }
    area_frac = obj[valid].mean()
    n_valid = valid.sum()
    # share of the image covered by the single largest object (largest = first painted, may be
    # partly overpainted by smaller ones, so use the annotation's own area scaled to the grid)
    max_obj_frac = max((a["area"] * tf.sx * tf.sy / GRID_DS**2 for a in anns), default=0.0) / n_valid
    cat =np.array([coco.cat_name[a["category_id"]] for a in anns] + ["background"])
    sup = np.array([coco.cat_super[a["category_id"]] for a in anns] + ["background"])

    def rows(f, kind):
        gx, gy = to_grid(f.x.values, f.y.values)
        L = glab[gy, gx]
        r = f[["scan", "subject", "idx", "x", "y", "dur"]].copy()
        r["map_image"] = name
        r["kind"] = kind
        r["in_content"] = L != -2
        r["on_object"] = L >= 0
        r["object"] = cat[np.where(L >= 0, L, -1)]
        r["supercat"] = sup[np.where(L >= 0, L, -1)]
        for k, m in maps.items():
            r[f"nss_{k}"] = m[gy, gx]
        r["obj_area_frac"] = area_frac
        return r[r.in_content]

    info = {"image": name, "mode": tf.mode, "iou": tf.iou, "obj_area_frac": area_frac,
            "max_obj_frac": min(max_obj_frac, 1.0), "n_objects": len(anns)}
    if keep_maps:
        info["maps"] = {k: v.copy() for k, v in maps.items()}
        info["glab"] = glab
    return rows(fix_own, "own"), rows(fix_donor, "shuffled"), info


def bootstrap_by_index(df: pd.DataFrame, value: str, max_idx: int, n_boot: int = 1000,
                       seed: int = 0, unit: str = "map_image") -> pd.DataFrame:
    """Mean of `value` per fixation index with a 95% CI from resampling *images* (not fixations).

    Fixations on the same image are not independent, so images are the resampling unit.
    """
    d = df[df.idx <= max_idx]
    per = d.groupby([unit, "idx"])[value].mean().unstack("idx")
    M = per.to_numpy(float)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, M.shape[1]))
    for b in range(n_boot):
        boots[b] = np.nanmean(M[rng.integers(0, M.shape[0], M.shape[0])], axis=0)
    return pd.DataFrame({
        "idx": per.columns.values,
        "mean": np.nanmean(M, axis=0),
        "lo": np.nanpercentile(boots, 2.5, axis=0),
        "hi": np.nanpercentile(boots, 97.5, axis=0),
        "n_images": np.sum(~np.isnan(M), axis=0),
    })


def _logit_fit(X: np.ndarray, y: np.ndarray, sw: np.ndarray | None = None,
               l2: float = 1e-4, iters: int = 50) -> np.ndarray:
    """Weighted logistic regression by Newton/IRLS. X must already include an intercept column.

    sw are row weights (a cluster bootstrap = weight each row by how often its image was drawn).
    A tiny L2 penalty (not on the intercept) keeps it stable in small bootstrap samples.
    """
    sw = np.ones(len(y)) if sw is None else sw
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (sw * (y - p)) - pen * w
        H = (X * (sw * p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def joint_by_index(own: pd.DataFrame, shuf: pd.DataFrame, max_idx: int,
                   predictors=("nss_contrast", "nss_objects"), n_boot: int = 300,
                   seed: int = 0) -> pd.DataFrame:
    """For each fixation index: does each map still predict *fixated vs. control* locations
    once the other map is in the model?

    Positives = own fixations on an image; negatives = the shuffled-control fixations scored on
    the same image (other people's gaze from another image, so they share the centre bias but
    not this image's content). The model is fitted per index:

        logit P(fixated) = b0 + b_contrast * contrast + b_objects * objects

    Predictors are the z-scored map values (as in NSS), so coefficients are on the same scale:
    log-odds change per 1 SD of the map. 95% CIs resample images.
    """
    cols = list(predictors)
    d = pd.concat([own.assign(y=1.0), shuf.assign(y=0.0)])
    d = d[(d.idx <= max_idx)].dropna(subset=cols)
    rng = np.random.default_rng(seed)
    out = []
    for k, dk in d.groupby("idx"):
        codes, names = pd.factorize(dk.map_image)
        X = np.column_stack([np.ones(len(dk))] + [dk[c].to_numpy(float) for c in cols])
        y = dk.y.to_numpy(float)
        est = _logit_fit(X, y)[1:]
        boots = np.empty((n_boot, len(cols)))
        for b in range(n_boot):
            draws = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
            boots[b] = _logit_fit(X, y, sw=draws[codes].astype(float))[1:]
        for j, c in enumerate(cols):
            out.append({"idx": k, "map": c.replace("nss_", ""), "coef": est[j],
                        "lo": np.percentile(boots[:, j], 2.5), "hi": np.percentile(boots[:, j], 97.5),
                        "n": len(dk)})
    return pd.DataFrame(out)


def add_saccades(fix: pd.DataFrame, px_per_deg: float | None = None) -> pd.DataFrame:
    """Amplitude of the saccade *into* fixation k (k >= 1), in pixels and optionally degrees."""
    f = fix.sort_values(["scan", "idx"]).copy()
    dx = f.groupby("scan").x.diff()
    dy = f.groupby("scan").y.diff()
    f["amp_px"] = np.hypot(dx, dy)
    if px_per_deg:
        f["amp_deg"] = f.amp_px / px_per_deg
    return f
