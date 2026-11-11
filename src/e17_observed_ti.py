"""E17: transfer index on observed route choices instead of oracle labels (CPU only).

E9's transfer index scores oracle labels (the MAP6 candidate closest to the observed route), so it measures the
transfer of a candidate-selection rule. Here we use the classical revealed-preference setup of route-choice studies:
the observed route is added to the MAP6 choice set (unless it is already one of the candidates) and is the label.
The same five path-size-logit features are computed over the augmented set (ratios to the set minimum, path size
over the set). Per target city (same 100k test / 100k train pools as E9): in-city fit, pooled zero-shot fit (other
four cities) and every single source; TI = (LL(beta_s) - LL(0)) / (LL(beta_t) - LL(0)) on target-test observed
choices, LL(0) = equal shares over the augmented set; TTS = -2 (LL(beta_s) - LL(beta_t)).
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

DEV = "cpu"


def augmented(P, meta):
    """MAP6 candidates of every trip plus its observed route; label = the observed route."""
    n = int(P["n"])
    routes = diag.ragged_take(P["rflat"], P["roff"], range(len(P["pair_i"])))
    gts = diag.ragged_take(P["gt_flat"], P["gt_off"], range(n))
    by_trip = [[] for _ in range(n)]
    for r, i in zip(routes, P["pair_i"]):
        by_trip[int(i)].append(np.asarray(r, np.int64))
    sets, lab, in_set = [], [], np.zeros(n, bool)
    for i in range(n):
        g = np.asarray(gts[i], np.int64)
        c = by_trip[i]
        hit = [k for k, r in enumerate(c) if len(r) == len(g) and np.array_equal(r, g)]
        if hit:
            in_set[i] = True
            lab.append(hit[0])
            sets.append(c)
        else:
            lab.append(len(c))
            sets.append(c + [g])
    pi_ = np.repeat(np.arange(n), [len(c) for c in sets])
    flat = [r for c in sets for r in c]
    rflat, roff = diag.from_list(flat)
    npair = len(flat)
    rep = np.repeat(np.arange(npair), np.diff(roff))
    seg = rflat.astype(np.int64)
    ln = meta["length"][seg]
    L = np.bincount(rep, weights=ln, minlength=npair)
    fft = np.bincount(rep, weights=ln / meta["speed"][seg], minlength=npair)
    major = np.bincount(rep, weights=ln * meta["major"][seg], minlength=npair)
    nseg = np.diff(roff).astype(float)
    _, inv, cnt = np.unique(pi_[rep] * meta["n_seg"] + seg, return_inverse=True, return_counts=True)
    ps = np.bincount(rep, weights=ln / np.maximum(L[rep], 1e-9) / cnt[inv], minlength=npair)

    def log_ratio(v):   # as in e4.pair_table
        mn = np.full(n, np.inf)
        np.minimum.at(mn, pi_, v)
        return np.log(np.maximum(v, 1e-9) / np.maximum(mn[pi_], 1e-9))

    X = np.stack([log_ratio(L), log_ratio(fft), major / np.maximum(L, 1e-9), log_ratio(nseg), np.log(np.maximum(ps, 1e-9))], 1)
    first = np.r_[0, np.cumsum([len(c) for c in sets])[:-1]]
    label = np.zeros(npair, bool)
    label[first + np.array(lab)] = True
    return {"X": X, "pair_i": pi_, "label": label, "f1": np.zeros(npair), "n": np.int64(n)}, float(in_set.mean())


def ll(P, beta):
    return e9.loglik(P, e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"])))


def main():
    cities = e4.ALL
    E9 = json.loads((diag.OUT / f"E9_rigor{e4.TAG}.json").read_text())
    tr, te, share = {}, {}, {}
    for c in cities:
        meta = e4.road_meta(c, 16)
        tr[c], _ = augmented(e4.load_pool(c, "train"), meta)
        te[c], share[c] = augmented(e4.load_pool(c, "test"), meta)
        diag.log(f"[E17 {c}] observed route already in MAP6: {share[c]:.3f} of test trips; "
                 f"{len(te[c]['pair_i']) / int(te[c]['n']):.2f} alternatives per trip")
    beta = {c: e4.fit_logit(tr[c], DEV) for c in cities}
    res = {"features": e4.FEATS, "cities": {}}
    for t in cities:
        P = te[t]
        n = int(P["n"])
        l0 = ll(P, np.zeros(P["X"].shape[1]))
        lt = ll(P, beta[t])
        r = {"observed_in_map6": round(share[t], 4), "beta_incity": [round(float(x), 3) for x in beta[t]],
             "ll_per_trip": {"equal": round(l0 / n, 4), "incity": round(lt / n, 4)}, "single": {}}
        for s in cities:
            if s == t:
                continue
            ls = ll(P, beta[s])
            r["single"][e4.SHORT[s]] = {"TI": round((ls - l0) / (lt - l0), 4), "TTS": round(-2 * (ls - lt), 1)}
        bp = e4.fit_logit(e4b.concat([tr[s] for s in cities if s != t]), DEV)
        lp = ll(P, bp)
        r["pooled"] = {"TI": round((lp - l0) / (lt - l0), 4), "TTS": round(-2 * (lp - lt), 1),
                       "beta": [round(float(x), 3) for x in bp]}
        r["ll_per_trip"]["pooled"] = round(lp / n, 4)
        r["oracle_label_TI_pooled"] = E9[t]["transfer"]["zeroshot_pooled"]["TI"]
        res["cities"][t] = r
        diag.log(f"[E17 {t}] " + json.dumps({k: r[k] for k in ("observed_in_map6", "pooled", "oracle_label_TI_pooled")})
                 + " single TI " + json.dumps({k: v["TI"] for k, v in r["single"].items()}))
        (diag.OUT / f"E17_observed_ti{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E17 done")


if __name__ == "__main__":
    main()
