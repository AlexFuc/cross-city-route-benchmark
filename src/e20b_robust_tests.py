"""E20b: parameter-equality tests with the exact unpenalized MLE and OD-cluster-robust covariances (CPU only).

E20 used the ridge-penalized Adam estimates (lambda 1e-4 on standardized features) and model-based covariances
(inverse Fisher information), which ignore that trips of the same OD are dependent. Here, per city (training pool,
oracle labels):
  beta  = exact maximum-likelihood estimate (Newton-Raphson from the E9/E20 starting point, no penalty);
  V_rob = cluster-robust sandwich covariance I^-1 M I^-1, M = G/(G-1) sum_c S_c S_c', S_c = summed scores of the
          trips of OD cluster c (pooled sources: clusters = city x OD);
  Wald  = (beta_S - beta_T)' (V_S + V_T)^-1 (beta_S - beta_T), compared with chi^2_5 (independent samples).
Reported for comparison: the model-based (naive) Wald and the classical TTS on the target estimation sample; the
TTS has no chi-square reference under clustered data and is not used for inference.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402

DEV = "cpu"


def parts(P, beta):
    n, pi, X = int(P["n"]), P["pair_i"], P["X"]
    p = e4.softmax_by_traj(X @ beta, pi, n)
    xbar = np.zeros((n, X.shape[1]))
    np.add.at(xbar, pi, p[:, None] * X)
    D = X - xbar[pi]
    info = (D * p[:, None]).T @ D
    score = np.zeros((n, X.shape[1]))
    score[pi[P["label"]]] = D[P["label"]]          # x_chosen - xbar, one row per trip
    ll = float(np.log(np.maximum(p[P["label"]], 1e-300)).sum())
    return ll, score, info


def mle(P, beta0, tol=1e-10, it=50):
    beta = beta0.astype(float).copy()
    ll0 = parts(P, beta)[0]
    for _ in range(it):
        ll, score, info = parts(P, beta)
        step = np.linalg.solve(info, score.sum(0))
        beta = beta + step
        if np.abs(step).max() < tol:
            break
    ll, score, info = parts(P, beta)
    assert ll >= ll0 - 1e-6, (ll, ll0)
    return beta, ll, score, info


def robust_cov(score, info, cluster):
    _, g = np.unique(cluster, return_inverse=True)
    G = int(g.max()) + 1
    S = np.zeros((G, score.shape[1]))
    np.add.at(S, g, score)
    Ii = np.linalg.inv(info)
    return Ii @ (G / (G - 1) * S.T @ S) @ Ii, Ii, G


def concat(Ps, keys):
    off, pis = 0, []
    for P in Ps:
        pis.append(P["pair_i"] + off)
        off += int(P["n"])
    return {"X": np.concatenate([P["X"] for P in Ps]), "label": np.concatenate([P["label"] for P in Ps]),
            "pair_i": np.concatenate(pis), "n": np.int64(off)}, np.concatenate(keys)


def log10_sf(x, df=5):
    return float(chi2.logsf(x, df) / np.log(10))


def fit(P, cl):
    b0 = e4.fit_logit(P, DEV)
    b, ll, score, info = mle(P, b0)
    Vr, Vn, G = robust_cov(score, info, cl)
    return {"b0": b0, "beta": b, "Vr": Vr, "Vn": Vn, "G": G}


def main():
    cities = e4.ALL
    tr = {c: e4.load_pool(c, "train") for c in cities}
    key = {c: tr[c]["od_key"].astype(np.int64) + i * 10**11 for i, c in enumerate(cities)}   # city x OD clusters
    F = {c: fit(tr[c], key[c]) for c in cities}
    res = {"features": e4.FEATS, "df": 5, "coef": {}, "targets": {}}
    for c in cities:
        f = F[c]
        res["coef"][c] = {"beta_mle": [round(float(x), 4) for x in f["beta"]],
                          "se_robust": [round(float(x), 4) for x in np.sqrt(np.diag(f["Vr"]))],
                          "se_model": [round(float(x), 4) for x in np.sqrt(np.diag(f["Vn"]))],
                          "design_effect": [round(float(x), 2) for x in np.diag(f["Vr"]) / np.diag(f["Vn"])],
                          "ridge_vs_mle_max_rel": round(float(np.max(np.abs(f["b0"] - f["beta"]) / np.maximum(np.abs(f["beta"]), 1e-9))), 5),
                          "od_clusters": f["G"]}
        diag.log(f"[E20b {c}] " + json.dumps(res["coef"][c]))
    for t in cities:
        lt = parts(tr[t], F[t]["beta"])[0]
        out = {}
        srcs = [(e4.SHORT[s], F[s]) for s in cities if s != t]
        P, k = concat([tr[s] for s in cities if s != t], [key[s] for s in cities if s != t])
        srcs.append(("pooled", fit(P, k)))
        for name, f in srcs:
            d = f["beta"] - F[t]["beta"]
            wr = float(d @ np.linalg.solve(f["Vr"] + F[t]["Vr"], d))
            wn = float(d @ np.linalg.solve(f["Vn"] + F[t]["Vn"], d))
            tts = -2.0 * (parts(tr[t], f["beta"])[0] - lt)
            out[name] = {"Wald_robust": round(wr, 1), "log10_p_robust": round(log10_sf(wr), 1),
                         "Wald_model": round(wn, 1), "TTS_train": round(tts, 1)}
        res["targets"][t] = out
        diag.log(f"[E20b {t}] " + json.dumps({k: (v["Wald_robust"], v["log10_p_robust"]) for k, v in out.items()}))
        (diag.OUT / f"E20b_robust_tests{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E20b done")


if __name__ == "__main__":
    main()
