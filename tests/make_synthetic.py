"""
Build a tiny fake dataset in the same formats as the real one, so the notebook can be
smoke-tested end to end without downloading anything.

  python tests/make_synthetic.py data_synth

Ground truth baked in: early fixations go to the single high-contrast "pop-out" object,
later fixations spread over all objects, and a fifth land anywhere (center-biased).
The notebook should recover "contrast early, objects later". Numbers mean nothing else.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

STIM_W, STIM_H = 1680, 1050
CATS = [(1, "person", "person"), (18, "dog", "animal"), (3, "car", "vehicle"),
        (47, "cup", "kitchen"), (62, "chair", "furniture")]


def main(out, n_images=40, n_subj=10, seed=1):
    rng = np.random.default_rng(seed)
    os.makedirs(f"{out}/stimuli/sub", exist_ok=True)
    os.makedirs(f"{out}/annotations", exist_ok=True)
    images, anns, scans = [], [], []
    aid = 1
    for i in range(n_images):
        name = f"{i:012d}.jpg"
        w, h = [(640, 480), (480, 640), (640, 427)][i % 3]
        bg = (rng.normal(0.45, 0.04, (h, w, 1)) * np.ones(3) * 255).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(bg)
        dr = ImageDraw.Draw(im)
        objs = []
        for j in range(rng.integers(3, 6)):
            bw, bh = rng.integers(40, 140, 2)
            x0, y0 = rng.integers(0, w - bw), rng.integers(0, h - bh)
            popout = j == 0
            col = (230, 30, 30) if popout else tuple(int(v) for v in rng.integers(95, 140, 3))
            dr.rectangle([x0, y0, x0 + bw, y0 + bh], fill=col)
            cid = CATS[rng.integers(len(CATS))][0]
            poly = [x0, y0, x0 + bw, y0, x0 + bw, y0 + bh, x0, y0 + bh]
            anns.append(dict(id=aid, image_id=i + 1, category_id=int(cid), iscrowd=0,
                             area=float(bw * bh), bbox=[float(x0), float(y0), float(bw), float(bh)],
                             segmentation=[[float(v) for v in poly]]))
            aid += 1
            objs.append((x0 + bw / 2, y0 + bh / 2, bw, bh, popout))
        images.append(dict(id=i + 1, file_name=name, width=w, height=h))

        s = min(STIM_W / w, STIM_H / h)
        nw, nh = round(w * s), round(h * s)
        ox, oy = (STIM_W - nw) // 2, (STIM_H - nh) // 2
        stim = Image.new("RGB", (STIM_W, STIM_H), (0, 0, 0))
        stim.paste(im.resize((nw, nh)), (ox, oy))
        stim.save(f"{out}/stimuli/sub/{name}", quality=92)

        for subj in range(1, n_subj + 1):
            X, Y, T = [STIM_W / 2], [STIM_H / 2], [200]
            for k in range(1, int(rng.integers(10, 17))):
                u = rng.random()
                if u < 0.2:
                    x, y = rng.normal(STIM_W / 2, 250), rng.normal(STIM_H / 2, 170)
                else:
                    p_pop = 0.8 if k <= 2 else 0.1
                    o = objs[0] if rng.random() < p_pop else objs[rng.integers(1, len(objs))]
                    x = (o[0] + rng.uniform(-0.4, 0.4) * o[2]) * s + ox
                    y = (o[1] + rng.uniform(-0.4, 0.4) * o[3]) * s + oy
                X.append(float(np.clip(x, 0, STIM_W - 1)))
                Y.append(float(np.clip(y, 0, STIM_H - 1)))
                T.append(float(rng.integers(150, 400)))
            scans.append(dict(name=name, subject=subj, split="train" if i % 5 else "valid",
                              X=X, Y=Y, T=T, length=len(X)))

    cats = [dict(id=c, name=n, supercategory=s) for c, n, s in CATS]
    with open(f"{out}/annotations/instances_val2017.json", "w") as f:
        json.dump(dict(images=images, annotations=anns, categories=cats), f)
    with open(f"{out}/COCOFreeView_fixations_trainval.json", "w") as f:
        json.dump(scans, f)
    print(f"wrote {n_images} images, {len(scans)} scanpaths to {out}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_synth")
