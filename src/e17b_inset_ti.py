"""E17b: transfer index on observed choices, on the trips whose observed route is one of the MAP6 candidates (CPU only).

On these trips the oracle label is the observed route itself (its F1 is 1 and candidates are distinct), so the
logit is an ordinary revealed-preference choice model over the generated set, with no pseudo-label and nothing added
to the set (E17 adds the observed route to the set instead, which lets a model learn which alternative was added).
Per target city: in-city fit on covered target-train trips, pooled zero-shot fit on covered source-train trips,
every single source; TI and TTS on covered target-test trips, as in E9. Also: top-1 hit rate (observed route picked).
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

DEV = "cpu"


def covered(P):
    """Trips whose observed route equals one of their MAP6 candidates (then F1 = 1 for exactly that candidate)."""
    best = np.full(int(P["n"]), -1.0)
    np.maximum.at(best, P["pair_i"], P["f1"])
    routes = diag.ragged_take(P["rflat"], P["roff"], np.nonzero(P["label"])[0])
    gts = diag.ragged_take(P["gt_flat"], P["gt_off"], P["pair_i"][P["label"]])
    same = np.array([len(r) == len(g) and np.array_equal(np.asarray(r, np.int64), np.asarray(g, np.int64)) for r, g in zip(routes, gts)])
    trips = P["pair_i"][P["label"]][same]
    assert np.all(best[trips] > 1 - 1e-9), "covered trips must have an F1 = 1 label"
    return np.sort(trips)


def ll(P, beta):
    return e9.loglik(P, e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"])))


def main():
    cities = e4.ALL
    E9 = json.loads((diag.OUT / f"E9_rigor{e4.TAG}.json").read_text())
    tr, te, share = {}, {}, {}
    for c in cities:
        Ptr, Pte = e4.load_pool(c, "train"), e4.load_pool(c, "test")
        a, b = covered(Ptr), covered(Pte)
        tr[c], te[c], share[c] = take(Ptr, a), take(Pte, b), (len(a) / int(Ptr["n"]), len(b) / int(Pte["n"]))
        diag.log(f"[E17b {c}] covered trips: train {share[c][0]:.3f}, test {share[c][1]:.3f}")
    beta = {c: e4.fit_logit(tr[c], DEV) for c in cities}
    res = {"features": e4.FEATS, "cities": {}}
    for t in cities:
        P = te[t]
        n = int(P["n"])
        l0, lt = ll(P, np.zeros(P["X"].shape[1])), ll(P, beta[t])
        hit = lambda b: float(e9.top1_vec(P, e4.softmax_by_traj(P["X"] @ b, P["pair_i"], n), P["label"].astype(float)).mean())  # noqa: E731
        r = {"covered_share_test": round(share[t][1], 4), "covered_share_train": round(share[t][0], 4), "n_test": n,
             "beta_incity": [round(float(x), 3) for x in beta[t]], "hit_incity": round(hit(beta[t]), 4), "single": {}}
        for s in cities:
            if s != t:
                ls = ll(P, beta[s])
                r["single"][e4.SHORT[s]] = {"TI": round((ls - l0) / (lt - l0), 4), "TTS": round(-2 * (ls - lt), 1)}
        bp = e4.fit_logit(e4b.concat([tr[s] for s in cities if s != t]), DEV)
        lp = ll(P, bp)
        r["pooled"] = {"TI": round((lp - l0) / (lt - l0), 4), "TTS": round(-2 * (lp - lt), 1), "hit": round(hit(bp), 4),
                       "beta": [round(float(x), 3) for x in bp]}
        r["oracle_label_TI_pooled_all_trips"] = E9[t]["transfer"]["zeroshot_pooled"]["TI"]
        res["cities"][t] = r
        diag.log(f"[E17b {t}] " + json.dumps({k: r[k] for k in ("covered_share_test", "pooled", "hit_incity")})
                 + " single TI " + json.dumps({k: v["TI"] for k, v in r["single"].items()}))
        (diag.OUT / f"E17b_inset_ti{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E17b done")


if __name__ == "__main__":
    main()
