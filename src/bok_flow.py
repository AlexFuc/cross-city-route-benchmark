"""Best-of-K vs. distribution quality: flow NMAE of (a) the single length-shortest path and (b) a uniform
distribution over the MAP6 candidates, on the same 100k test trips and segments (flow >= 5) as E4."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402

out = {}
for c in e4.ALL:
    P, meta = e4.load_pool(c, "test"), e4.road_meta(c, 16)
    n = int(P["n"])
    _, gseg = diag.flatten_rows(P["gt_flat"], P["gt_off"], np.arange(n))
    gt_flow = np.bincount(gseg, minlength=meta["n_seg"]).astype(float)
    mask = gt_flow >= 5
    one = (P["pair_k"] == 0).astype(float)
    uni = e4.softmax_by_traj(np.zeros(len(P["pair_i"])), P["pair_i"], n)
    r_len, r_uni = e4.evaluate_probs(P, one, meta, gt_flow, mask), e4.evaluate_probs(P, uni, meta, gt_flow, mask)
    out[c] = {"oracle6": round(float(P["best"].mean()), 4), "len_top1": round(r_len["top1_f1"], 4),
              "len_flow": round(r_len["flow_nmae"], 4), "uniform_flow": round(r_uni["flow_nmae"], 4)}
    print(c, out[c], flush=True)
(diag.OUT / f"BOK_flow{e4.TAG}.json").write_text(json.dumps(out, indent=2))
