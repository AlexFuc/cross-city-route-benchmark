"""Length-weighted top-1 of the MAP6 logit on exactly the RL2 test trips (same 10k draw as rl2.main), so the
recursive-logit table compares both models under the primary metric. Run with CC_F1=length."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e9  # noqa: E402
from rl2 import N_TEST  # noqa: E402

assert e4.F1_MODE == "length"
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
out = {}
for city, src in [("chengdu", "xian"), ("xian", "chengdu"), ("porto", "chengdu")]:
    P = e4.load_pool(city, "test")
    n_seg = e4.road_meta(city, 16)["n_seg"]
    te = np.random.default_rng(0).choice(int(P["n"]), min(N_TEST, int(P["n"])), replace=False)
    gall = diag.ragged_take(P["gt_flat"], P["gt_off"], range(int(P["n"])))
    gts = [np.asarray(gall[i], np.int64) for i in te]
    gt_flow = np.bincount(np.concatenate([g[1:] for g in gts]), minlength=n_seg).astype(float)
    mask = gt_flow >= 5
    orig = np.bincount(np.array([int(g[0]) for g in gts]), minlength=n_seg).astype(float)
    res = {"src": src}
    for name, pool_ in [("map6_incity", e4.load_pool(city, "train")), ("map6_zeroshot", e4.load_pool(src, "train"))]:
        beta = e4.fit_logit(pool_, device)
        p = e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"]))
        top = e9.top1_vec(P, p, P["f1"])
        sel = np.nonzero(np.isin(P["pair_i"], te))[0]
        rep, seg = diag.flatten_rows(P["rflat"], P["roff"], sel)
        fl = np.bincount(seg, weights=p[sel][rep], minlength=n_seg) - orig
        res[name] = {"top1_length": round(float(top[te].mean()), 4),
                     "flow_nmae": round(diag.geh_stats(fl, gt_flow, mask)["nmae"], 4)}
    out[city] = res
    diag.log(f"[{city}] " + json.dumps(res))
(diag.OUT / "RL2_map6_len.json").write_text(json.dumps(out, indent=2))
