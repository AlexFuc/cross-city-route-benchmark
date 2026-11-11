"""Where does the in-city GBDT gain come from: capacity (transferable) or local region knowledge?

  G1 in-city GBDT without region ids          (target train -> target test)
  G2 zero-shot GBDT without region ids        (single source train -> target test)
  G3 zero-shot GBDT, both sources pooled
  L3 zero-shot logit, both sources pooled
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag  # noqa: E402
import e4  # noqa: E402

CITIES = e4.ALL


def design_noregion(P):
    X = P["X"]
    pi = P["pair_i"]
    h = P["hour"][pi]
    return np.c_[X, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.log(P["min_len"][pi] + 1.0)]


def fit(Z, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0)
    return clf.fit(Z, y.astype(int))


def concat(Ps):
    out = {k: np.concatenate([P[k] for P in Ps]) for k in ("X", "label", "f1")}
    off, pis = 0, []
    for P in Ps:
        pis.append(P["pair_i"] + off)
        off += int(P["n"])
    out["pair_i"] = np.concatenate(pis)
    out["n"] = np.int64(off)
    return out


def main():
    metas = {c: e4.road_meta(c, 16) for c in CITIES}
    Pt, Ptr = {}, {}
    for c in CITIES:
        Pt[c] = e4.load_pool(c, "test")
        Ptr[c] = e4.load_pool(c, "train")
    Ztr = {c: design_noregion(Ptr[c]) for c in CITIES}
    models = {c: fit(Ztr[c], Ptr[c]["label"]) for c in CITIES}
    res = {}
    for t in CITIES:
        P, meta = Pt[t], metas[t]
        n = int(P["n"])
        _, gt_seg = diag.flatten_rows(P["gt_flat"], P["gt_off"], np.arange(n))
        gt_flow = np.bincount(gt_seg, minlength=meta["n_seg"]).astype(float)
        mask = gt_flow >= 5
        Z = design_noregion(P)

        def ev(score):
            p = score / np.bincount(P["pair_i"], weights=score, minlength=n)[P["pair_i"]]
            r = e4.evaluate_probs(P, p, meta, gt_flow, mask)
            return {k: round(v, 4) for k, v in r.items() if k != "top_idx"}

        r = {"G1_incity_gbdt_noregion": ev(models[t].predict_proba(Z)[:, 1])}
        srcs = [s for s in CITIES if s != t]
        for s in srcs:
            r[f"G2_zeroshot_gbdt_from_{e4.SHORT[s]}"] = ev(models[s].predict_proba(Z)[:, 1])
        pooled = concat([Ptr[s] for s in srcs])
        pooled["hour"] = np.concatenate([Ptr[s]["hour"] for s in srcs])
        pooled["min_len"] = np.concatenate([Ptr[s]["min_len"] for s in srcs])
        g3 = fit(design_noregion(pooled), pooled["label"])
        r["G3_zeroshot_gbdt_pooled"] = ev(g3.predict_proba(Z)[:, 1])
        beta = e4.fit_logit(pooled, "cuda:0")
        r["L3_zeroshot_logit_pooled"] = ev(e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], n))
        res[t] = r
        diag.log(f"[{t}] " + json.dumps(r))
    (diag.OUT / f"E4b_gbdt_transfer{e4.TAG}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
