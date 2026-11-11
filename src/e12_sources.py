"""E12: what transfers? Source-set, feature and source-size ablations of the zero-shot map-only logit (CPU only).

Per target city (100k test subset, length-weighted F1, oracle labels as in E9):
  (1) every non-empty subset of the four other cities as pooled source (15 fits): top-1 F1 and transfer index TI
      (TI = (LL_transfer - LL_equal) / (LL_incity - LL_equal), Koppelman-Wilmot, same as E9)
  (2) leave-one-feature-out: zero-shot (pooled four sources) and in-city logit without feature j
  (3) source size: n pooled source trips in {1e2, 1e3, 1e4, 1e5} (3 random draws each) vs all
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e9  # noqa: E402

DEV = "cpu"


def take(P, trips):
    """Sub-pool with the given trips (indices into 0..n-1), pairs re-indexed."""
    keep = np.isin(P["pair_i"], trips)
    remap = np.full(int(P["n"]), -1)
    remap[trips] = np.arange(len(trips))
    return {"X": P["X"][keep], "label": P["label"][keep], "f1": P["f1"][keep],
            "pair_i": remap[P["pair_i"][keep]], "n": np.int64(len(trips))}


def cols(P, c):
    Q = dict(P)
    Q["X"] = P["X"][:, c]
    return Q


def evaluate(P, beta, X=None):
    n, pi = int(P["n"]), P["pair_i"]
    p = e4.softmax_by_traj((P["X"] if X is None else X) @ beta, pi, n)
    return float(e9.top1_vec(P, p, P["f1"]).mean()), e9.loglik(P, p)


def main():
    cities = e4.ALL
    Pt = {c: e4.load_pool(c, "test") for c in cities}
    Ptr = {c: e4.load_pool(c, "train") for c in cities}
    E9 = json.loads((diag.OUT / f"E9_rigor{e4.TAG}.json").read_text())
    nf = Pt[cities[0]]["X"].shape[1]
    res = {"features": e4.FEATS, "cities": {}}
    rng = np.random.default_rng(0)
    for t in cities:
        P, n = Pt[t], int(Pt[t]["n"])
        ll_eq = e9.loglik(P, e4.softmax_by_traj(np.zeros(len(P["pair_i"])), P["pair_i"], n))
        f_loc, ll_loc = evaluate(P, e4.fit_logit(Ptr[t], DEV))
        ti = lambda ll, ref=ll_loc: round((ll - ll_eq) / (ref - ll_eq), 4)  # noqa: E731
        r = {"incity": {"top1": round(f_loc, 4)}, "subsets": {}, "drop": {}, "size": {}}
        assert abs(f_loc - E9[t]["top1"]["incity_logit"]["length"]) < 1.5e-3, (t, f_loc)
        others = [s for s in cities if s != t]
        for k in range(1, len(others) + 1):
            for S in itertools.combinations(others, k):
                f, ll = evaluate(P, e4.fit_logit(e4b.concat([Ptr[s] for s in S]), DEV))
                r["subsets"]["+".join(e4.SHORT[s] for s in S)] = {"top1": round(f, 4), "TI": ti(ll)}
        full = "+".join(e4.SHORT[s] for s in others)
        assert abs(r["subsets"][full]["top1"] - E9[t]["top1"]["zeroshot_pooled"]["length"]) < 1.5e-3, (t, r["subsets"][full])
        pooled = e4b.concat([Ptr[s] for s in others])
        for j in range(nf):
            c = [i for i in range(nf) if i != j]
            fz, llz = evaluate(cols(P, c), e4.fit_logit(cols(pooled, c), DEV))
            fl, lll = evaluate(cols(P, c), e4.fit_logit(cols(Ptr[t], c), DEV))
            r["drop"][e4.FEATS[j]] = {"zeroshot_top1": round(fz, 4), "incity_top1": round(fl, 4),
                                      "gap": round(fl - fz, 4), "TI_vs_full_incity": ti(llz), "TI_same_features": ti(llz, lll)}
        for m in (100, 1000, 10000, 100000):
            vals = []
            for _ in range(3 if m < 100000 else 1):
                sub = take(pooled, np.sort(rng.choice(int(pooled["n"]), m, replace=False)))
                vals.append(evaluate(P, e4.fit_logit(sub, DEV))[0])
            r["size"][str(m)] = {"top1_mean": round(float(np.mean(vals)), 4), "top1_min": round(float(np.min(vals)), 4),
                                 "top1_max": round(float(np.max(vals)), 4)}
        r["size"]["all"] = {"n": int(pooled["n"]), "top1_mean": r["subsets"][full]["top1"]}
        res["cities"][t] = r
        diag.log(f"[E12 {t}] " + json.dumps(r)[:2000])
        (diag.OUT / f"E12_sources{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E12 done")


if __name__ == "__main__":
    main()
