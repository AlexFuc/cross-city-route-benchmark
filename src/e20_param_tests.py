"""E20: tests of exact parameter equality between cities for the path-size logit (CPU only).

E9's TTS was computed on target TEST trips with beta_T estimated on target TRAINING trips; beta_T is then not the
maximizer of that likelihood, so the statistic has no chi-square reference. Here:
  TTS  = -2 [ LL_T(beta_S) - LL_T(beta_T) ] on the target ESTIMATION sample (training pool), where beta_T is the
         (near-)maximum-likelihood estimate (ridge 1e-4 on standardized features), source estimates treated as
         fixed, compared with chi^2_5 (Koppelman and Wilmot 1982);
  Wald = (beta_S - beta_T)' (V_S + V_T)^{-1} (beta_S - beta_T), V = inverse Fisher information of each estimate
         on its own estimation sample, compared with chi^2_5; this also accounts for the source estimate's
         uncertainty.
For every target: the 4 single sources and the pooled other four cities. Labels: oracle labels, as everywhere.
Also reported: coefficients with standard errors per city.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e9  # noqa: E402

DEV = "cpu"


def fisher(P, beta):
    """Fisher information of the multinomial logit at beta: sum_n sum_j p_nj (x_nj - xbar_n)(x_nj - xbar_n)'."""
    n, pi, X = int(P["n"]), P["pair_i"], P["X"]
    p = e4.softmax_by_traj(X @ beta, pi, n)
    xbar = np.zeros((n, X.shape[1]))
    np.add.at(xbar, pi, p[:, None] * X)
    D = X - xbar[pi]
    return (D * p[:, None]).T @ D


def ll(P, beta):
    return e9.loglik(P, e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"])))


def log10_sf(x, df=5):
    return float(chi2.logsf(x, df) / np.log(10))


def main():
    cities = e4.ALL
    tr = {c: e4.load_pool(c, "train") for c in cities}
    beta = {c: e4.fit_logit(tr[c], DEV) for c in cities}
    V = {c: np.linalg.inv(fisher(tr[c], beta[c])) for c in cities}
    res = {"features": e4.FEATS, "df": 5, "coef": {}, "targets": {}}
    for c in cities:
        se = np.sqrt(np.diag(V[c]))
        res["coef"][c] = {"beta": [round(float(x), 4) for x in beta[c]], "se": [round(float(x), 4) for x in se]}
    for t in cities:
        lt = ll(tr[t], beta[t])
        out = {}
        srcs = [(e4.SHORT[s], beta[s], V[s]) for s in cities if s != t]
        pooled = e4b.concat([tr[s] for s in cities if s != t])
        bp = e4.fit_logit(pooled, DEV)
        srcs.append(("pooled", bp, np.linalg.inv(fisher(pooled, bp))))
        for name, bs, Vs in srcs:
            tts = -2.0 * (ll(tr[t], bs) - lt)
            d = bs - beta[t]
            w = float(d @ np.linalg.solve(Vs + V[t], d))
            out[name] = {"TTS_train": round(float(tts), 1), "log10_p_TTS": round(log10_sf(tts), 1),
                         "Wald": round(w, 1), "log10_p_Wald": round(log10_sf(w), 1)}
        res["targets"][t] = out
        diag.log(f"[E20 {t}] " + json.dumps({k: (v["TTS_train"], v["Wald"]) for k, v in out.items()}))
        (diag.OUT / f"E20_param_tests{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E20 done")


if __name__ == "__main__":
    main()
