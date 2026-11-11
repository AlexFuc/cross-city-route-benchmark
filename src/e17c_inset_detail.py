"""E17c: detail for E17b (with OD-cluster CIs of hit-rate differences) (observed choices on covered trips): selection, harder subset, and OD-cluster CIs (CPU only).

Covered trips = test trips whose observed route is one of their MAP6 candidates (E17b). Many of them are exact
shortest paths, so we also evaluate the trips whose observed route is a candidate but NOT the length-shortest one
(the chosen candidate is not the LEN slot). Models are those of E17b (fitted on covered training trips: in-city,
pooled other four cities, single sources). Per city: covered share and size, share of shortest-path choices among
covered trips, median detour (observed / shortest length) and length of covered vs. uncovered trips, transfer index
(TI) with 95% OD-cluster bootstrap CIs on all covered and on non-shortest covered trips, top-1 hit rates.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e9  # noqa: E402
from e12_sources import take  # noqa: E402
from e17b_inset_ti import covered  # noqa: E402
from e11_stats import cstats  # noqa: E402

DEV, B = "cpu", 2000


def trip_logp(P, beta):
    p = e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"]))
    out = np.zeros(int(P["n"]))
    out[P["pair_i"][P["label"]]] = np.log(np.maximum(p[P["label"]], 1e-12))
    return out


def ti_ci(ls, lt, l0, od, seed=0):
    _, g = np.unique(od, return_inverse=True)
    G = int(g.max()) + 1
    num, den = np.bincount(g, ls - l0, G), np.bincount(g, lt - l0, G)
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, G, G), minlength=G)
        bs[b] = (w @ num) / (w @ den)
    r = lambda v: round(float(v), 4)  # noqa: E731
    return {"TI": r(num.sum() / den.sum()), "ci95": [r(np.percentile(bs, 2.5)), r(np.percentile(bs, 97.5))], "clusters": G}


def hitvec(P, beta):
    return e9.top1_vec(P, e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"])), P["label"].astype(float))


def hit(P, beta, rows):
    return round(float(hitvec(P, beta)[rows].mean()), 4)


def main():
    cities = e4.ALL
    E17b = json.loads((diag.OUT / f"E17b_inset_ti{e4.TAG}.json").read_text())["cities"]
    tr, te, info = {}, {}, {}
    for c in cities:
        Ptr, Pte = e4.load_pool(c, "train"), e4.load_pool(c, "test")
        a, b = covered(Ptr), covered(Pte)
        tr[c], te[c] = take(Ptr, a), take(Pte, b)
        L = e4.road_meta(c, 16)["length"]
        gl = np.array([L[np.asarray(g, np.int64)].sum() for g in diag.ragged_take(Pte["gt_flat"], Pte["gt_off"], range(int(Pte["n"])))])
        det = gl / np.maximum(Pte["min_len"], 1e-9)
        cov = np.zeros(int(Pte["n"]), bool)
        cov[b] = True
        k_lab = Pte["pair_k"][Pte["label"]][np.argsort(Pte["pair_i"][Pte["label"]])]   # slot of each trip's label
        nonsp = cov & (k_lab != 0)
        info[c] = {"covered_share": round(float(cov.mean()), 4), "n_covered": int(cov.sum()),
                   "shortest_share_of_covered": round(float((cov & (k_lab == 0)).sum() / cov.sum()), 4),
                   "n_covered_nonshortest": int(nonsp.sum()),
                   "median_detour": {"covered": round(float(np.median(det[cov])), 3), "uncovered": round(float(np.median(det[~cov])), 3)},
                   "median_km": {"covered": round(float(np.median(gl[cov]) / 1000), 2), "uncovered": round(float(np.median(gl[~cov]) / 1000), 2)},
                   "_nonsp_rows": np.nonzero(nonsp[b])[0], "_od": Pte["od_key"][b]}
        diag.log(f"[E17c {c}] " + json.dumps({k: v for k, v in info[c].items() if not k.startswith("_")}))
    beta = {c: e4.fit_logit(tr[c], DEV) for c in cities}
    res = {"B": B, "cities": {}}
    for t in cities:
        P = te[t]
        n = int(P["n"])
        l0 = trip_logp(P, np.zeros(P["X"].shape[1]))
        lt = trip_logp(P, beta[t])
        bp = e4.fit_logit(e4b.concat([tr[s] for s in cities if s != t]), DEV)
        lp = trip_logp(P, bp)
        assert abs((lp - l0).sum() / (lt - l0).sum() - E17b[t]["pooled"]["TI"]) < 2e-3, t
        rows, od = info[t]["_nonsp_rows"], info[t]["_od"]
        r = {k: v for k, v in info[t].items() if not k.startswith("_")}
        r["pooled_all"] = ti_ci(lp, lt, l0, od)
        r["pooled_nonshortest"] = ti_ci(lp[rows], lt[rows], l0[rows], od[rows], seed=1)
        r["single_nonshortest"] = {}
        for s in cities:
            if s != t:
                ls = trip_logp(P, beta[s])
                r["single_nonshortest"][e4.SHORT[s]] = round(float((ls[rows] - l0[rows]).sum() / (lt[rows] - l0[rows]).sum()), 4)
        r["hit"] = {"incity_all": hit(P, beta[t], np.arange(n)), "pooled_all": hit(P, bp, np.arange(n)),
                    "incity_nonshortest": hit(P, beta[t], rows), "pooled_nonshortest": hit(P, bp, rows)}
        hd = hitvec(P, bp) - hitvec(P, beta[t])
        r["hit_diff_nonshortest"] = cstats(hd[rows], od[rows], seed=2)   # pooled minus in-city, OD-cluster CI
        r["hit_diff_all"] = cstats(hd, od, seed=3)
        res["cities"][t] = r
        diag.log(f"[E17c {t}] " + json.dumps({k: r[k] for k in ("pooled_all", "pooled_nonshortest", "hit")})
                 + " single(non-SP) " + json.dumps(r["single_nonshortest"]))
        (diag.OUT / f"E17c_inset_detail{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E17c done")


if __name__ == "__main__":
    main()
