"""Robustness of the transfer-gap finding to 'shortest-path-like' ground truth (map-matching artefact risk).

Strata on the target test subset:
  detour  = GT route length / LEN-candidate length   (<=1.05 'near-shortest' vs >1.05 'deviating')
  length  = GT route length vs city median           (short vs long)
Per stratum: top-1 F1 of zero-shot logit (pooled other cities), in-city logit, in-city GBDT (no region),
and MAP6 Best-of-K. The finding survives if zero-shot ~= in-city logit also on deviating / long trips.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402


def per_traj_top1(P, p):
    pi = P["pair_i"]
    order = np.lexsort((-p, pi))
    first = np.r_[True, pi[order][1:] != pi[order][:-1]]
    top = order[first]
    out = np.zeros(int(P["n"]))
    out[pi[top]] = P["f1"][top]
    return out


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    cities = e4.ALL
    metas = {c: e4.road_meta(c, 16) for c in cities}
    Pt = {c: e4.load_pool(c, "test") for c in cities}
    Ptr = {c: e4.load_pool(c, "train") for c in cities}
    res = {}
    for t in cities:
        P, meta = Pt[t], metas[t]
        n = int(P["n"])
        L = meta["length"]
        gt_len = np.array([L[P["gt_flat"][P["gt_off"][i]:P["gt_off"][i + 1]]].sum() for i in range(n)])
        len_pair = np.nonzero(P["pair_k"] == 0)[0]
        len_route = np.zeros(n)
        for j in len_pair:
            len_route[P["pair_i"][j]] = L[P["rflat"][P["roff"][j]:P["roff"][j + 1]]].sum()
        detour = gt_len / np.maximum(len_route, 1.0)
        srcs = [s for s in cities if s != t]
        pooled = e4b.concat([Ptr[s] for s in srcs])
        b_zs = e4.fit_logit(pooled, device)
        b_in = e4.fit_logit(Ptr[t], device)
        g_in = e4b.fit(e4b.design_noregion(Ptr[t]), Ptr[t]["label"])
        sc = g_in.predict_proba(e4b.design_noregion(P))[:, 1]
        models = {"zeroshot_logit": e4.softmax_by_traj(P["X"] @ b_zs, P["pair_i"], n),
                  "incity_logit": e4.softmax_by_traj(P["X"] @ b_in, P["pair_i"], n),
                  "incity_gbdt": sc / np.bincount(P["pair_i"], weights=sc, minlength=n)[P["pair_i"]]}
        top = {k: per_traj_top1(P, p) for k, p in models.items()}
        top["MAP6_BoK"] = P["best"]
        strata = {"all": np.ones(n, bool), "near_shortest(<=1.05)": detour <= 1.05, "deviating(>1.05)": detour > 1.05,
                  "deviating(>1.20)": detour > 1.20, "short(<=median)": gt_len <= np.median(gt_len),
                  "long(>median)": gt_len > np.median(gt_len)}
        r = {}
        for s, m in strata.items():
            r[s] = {"share": round(float(m.mean()), 3), **{k: round(float(v[m].mean()), 4) for k, v in top.items()}}
            r[s]["transfer_gap"] = round(r[s]["incity_logit"] - r[s]["zeroshot_logit"], 4)
            r[s]["gbdt_gain"] = round(r[s]["incity_gbdt"] - r[s]["incity_logit"], 4)
        res[t] = r
        diag.log(f"[{t}] " + json.dumps(r))
    (diag.OUT / f"E7_strata{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E7 done")


if __name__ == "__main__":
    main()
