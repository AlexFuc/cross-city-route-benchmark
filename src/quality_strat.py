"""Share of GT routes that equal the length-shortest path, stratified by shortest-path trip length.
H1 (map-matching gap filling) predicts higher shares for sparsely sampled data (BJ 60s, SF ~60s)
than for densely sampled data (CD/XA 2-4s, PT 15s) at the same trip length."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402

BINS = [0, 1.5, 3, 5, 8, 1e9]
LAB = ["<1.5km", "1.5-3", "3-5", "5-8", ">8"]
SAMPLING = {"xian": "2-4s", "chengdu": "2-4s", "beijing": "60s", "porto": "15s", "sanfrancisco": "~60s", "bjhoser": "HOSER-pipeline"}
print("city(sampling)         " + "  ".join(f"{l:>16s}" for l in LAB))
for c in e4.ALL:
    A = diag.stage_a(c, 8)
    L = e4.road_meta(c, 16)["length"]
    go, gf, lo, lf = A["gt_off"], A["gt_flat"], A["len_off"], A["len_flat"]
    n = len(go) - 1
    idx = np.random.default_rng(0).choice(n, min(60000, n), replace=False)
    km, exact = [], []
    for i in idx:
        g, s = gf[go[i]:go[i + 1]], lf[lo[i]:lo[i + 1]]
        if len(g) < 2 or len(s) < 1:
            continue
        km.append(L[s].sum() / 1000)
        exact.append(np.array_equal(g, s))
    km, exact = np.array(km), np.array(exact)
    b = np.digitize(km, BINS[1:-1])
    cells = []
    for k in range(len(LAB)):
        m = b == k
        cells.append(f"{exact[m].mean():.2f} (n={m.sum():5d})" if m.sum() >= 200 else f"{'-':>16s}")
    print(f"{c + '(' + SAMPLING[c] + ')':22s} " + "  ".join(f"{x:>16s}" for x in cells) + f"   overall {exact.mean():.2f}")
