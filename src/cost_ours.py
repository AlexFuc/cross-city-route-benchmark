"""Wall-clock cost of the non-deep methods, measured end to end on one target city (same machine as GTG).

Stages (target test = the fixed 100k subset, or all trips if fewer):
  candidates  MAP6 for all test ODs: 3 shortest-path fields + 3 penalty alternatives (10 CPU workers)
  features    candidate dedup, 5 map features
  fit_logit   pooled-source path-size logit (GPU)
  fit_gbdt    in-city GBDT on 100k target training pools (CPU)
  predict     softmax / GBDT scoring for all test candidates
  odhist      build OD history from the full target training set
Candidate-route F1 against the ground truth is evaluation, not inference, and is excluded.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e9  # noqa: E402
import pool  # noqa: E402

WORKERS = 10


def candidates(city, A, rows, meta):
    import routing
    fields = {"LEN": routing.raw_length(city), "FFT": meta["length"] / meta["speed"],
              "CLS": meta["length"] * meta["factor"]}
    csr = {k: routing.csr_from_node_cost(city, np.asarray(v, float)) for k, v in fields.items()}
    o, d = pool.origin_dest(A, rows)
    routes = {k: pool.batched_shortest_paths(csr, [k] * len(rows), o, d, WORKERS) for k in fields}
    pen = pool.penalty_alternatives(city, list(zip(o, d)), WORKERS)
    return [routes["LEN"], routes["FFT"], routes["CLS"], pen[1], pen[2], pen[3]]


def main():
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    t = sys.argv[1]
    out = {"city": t, "workers": WORKERS}
    meta = e4.road_meta(t, 16)
    A_te, A_tr = e4.load_split(t, "test", 8), e4.load_split(t, "train", 8)
    rows = e4.subset(t, "test", A_te, 100000)
    out["n_test"] = int(len(rows))
    s = time.time()
    cands = candidates(t, A_te, rows, meta)
    out["candidates_s"] = round(time.time() - s, 1)
    P = e4.load_pool(t, "test")
    pooled = e4b.concat([e4.load_pool(c, "train") for c in e4.ALL if c != t])
    s = time.time()
    b = e4.fit_logit(pooled, device)
    out["fit_logit_s"] = round(time.time() - s, 1)
    s = time.time()
    p = e4.softmax_by_traj(P["X"] @ b, P["pair_i"], int(P["n"]))
    out["predict_logit_s"] = round(time.time() - s, 2)
    Ptr = e4.load_pool(t, "train")
    s = time.time()
    g = e9.fit_gbdt(e9.design(Ptr), Ptr["label"])
    out["fit_gbdt_s"] = round(time.time() - s, 1)
    s = time.time()
    g.predict_proba(e9.design(P))
    out["predict_gbdt_s"] = round(time.time() - s, 1)
    s = time.time()
    e4.od_history(A_tr, np.arange(len(A_tr["gt_off"]) - 1), meta["n_seg"])
    out["odhist_build_s"] = round(time.time() - s, 1)
    out["n_unique_candidates"] = int(len(P["pair_i"]))
    out["mean_candidates_per_od"] = round(len(P["pair_i"]) / int(P["n"]), 2)
    diag.log(json.dumps(out))
    (diag.OUT / f"COST_{t}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
