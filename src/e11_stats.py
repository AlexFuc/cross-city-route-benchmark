"""E11: statistics for the headline comparisons, with trips of the same OD pair resampled together (CPU only).

Trips that share an OD pair share candidates and mostly outcomes, so trip-level bootstrap CIs (E9) are too narrow
where ODs repeat (Xi'an, Chengdu). Per target city (100k test subset, length-weighted F1, same models as E9):
  transfer gap   = in-city logit - pooled zero-shot logit    (equivalence test against +-DELTA, TOST)
  GBDT gain      = in-city GBDT - in-city logit              (one-sided, > 0)
  gain - gap                                                  (one-sided, > 0)
Per GTG direction and run (official model, same 100k trips): logit - GTG (one-sided, > 0) and GTG - LEN (two-sided).
Means use all trips; SEs are cluster-robust over OD pairs (CR1); 95%/90% CIs from a cluster bootstrap. p-values are
normal approximations from the cluster-robust SE (>= 600 clusters everywhere); Holm over each family.
Robustness: clusters = origin segment (coarser; contains every OD cluster).
"""
import json
import os
import sys
from math import erf, sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e9  # noqa: E402

DELTA = 0.01   # equivalence margin (F1); below the smallest in-city GBDT gain we report (0.011)
DEV = "cpu"
B = 2000
GTG_DIRS = [("chengdu", "xian"), ("beijing", "xian"), ("beijing", "chengdu"), ("xian", "beijing"),
            ("xian", "chengdu"), ("chengdu", "beijing")]


def phi_sf(z):  # P(Z > z)
    return 0.5 * (1 - erf(z / sqrt(2)))


def cstats(x, g, seed=0):
    _, gi = np.unique(g, return_inverse=True)
    G = int(gi.max()) + 1
    S, n = np.bincount(gi, weights=x, minlength=G), np.bincount(gi, minlength=G).astype(float)
    m = float(S.sum() / n.sum())
    se = float(sqrt(G / (G - 1) * ((S - n * m) ** 2).sum()) / n.sum())
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, G, G), minlength=G)
        bs[b] = (w @ S) / (w @ n)
    r4 = lambda v: round(float(v), 5)  # noqa: E731
    return {"mean": r4(m), "se": r4(se), "ci95": [r4(np.percentile(bs, 2.5)), r4(np.percentile(bs, 97.5))],
            "ci90": [r4(np.percentile(bs, 5)), r4(np.percentile(bs, 95))], "clusters": G}


def trip_se(x):
    return float(x.std(ddof=1) / sqrt(len(x)))


def holm(ps):
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = run
    return out


def origin_of(P):
    return P["gt_flat"][P["gt_off"][:-1]]


def main():
    cities = e4.ALL
    Pt = {c: e4.load_pool(c, "test") for c in cities}
    Ptr = {c: e4.load_pool(c, "train") for c in cities}
    betas = {c: e4.fit_logit(Ptr[c], DEV) for c in cities}
    E9 = json.loads((diag.OUT / f"E9_rigor{e4.TAG}.json").read_text())
    res, pfam = {"delta": DELTA, "B": B, "cities": {}, "gtg": {}}, {}
    for t in cities:
        P = Pt[t]
        n, pi, f = int(P["n"]), P["pair_i"], P["f1"]
        pooled = e4b.concat([Ptr[s] for s in cities if s != t])
        pz = e4.softmax_by_traj(P["X"] @ e4.fit_logit(pooled, DEV), pi, n)
        pl = e4.softmax_by_traj(P["X"] @ betas[t], pi, n)
        g = e9.fit_gbdt(e9.design(Ptr[t]), Ptr[t]["label"])
        sc = g.predict_proba(e9.design(P))[:, 1]
        pg = sc / np.bincount(pi, weights=sc, minlength=n)[pi]
        zs, inl, gb = (e9.top1_vec(P, p, f) for p in (pz, pl, pg))
        for k, v in (("zeroshot_pooled", zs), ("incity_logit", inl), ("incity_gbdt", gb)):
            assert abs(v.mean() - E9[t]["top1"][k]["length"]) < 1.5e-3, (t, k, v.mean(), E9[t]["top1"][k]["length"])
        od, orig = P["od_key"], origin_of(P)
        _, cnt = np.unique(od, return_counts=True)
        r = {"n": n, "od_clusters": int(len(cnt)), "share_trips_repeated_od": round(float(cnt[cnt > 1].sum() / n), 4)}
        for name, x in (("gap", inl - zs), ("gain", gb - inl), ("gain_minus_gap", (gb - inl) - (inl - zs))):
            s = cstats(x, od)
            s["origin_clusters"] = cstats(x, orig, seed=1)
            s["se_trip"] = round(trip_se(x), 5)
            s["design_effect"] = round((s["se"] / max(s["se_trip"], 1e-12)) ** 2, 2)
            r[name] = s
        m, se, seo = r["gap"]["mean"], r["gap"]["se"], r["gap"]["origin_clusters"]["se"]
        r["gap"]["p_tost"] = max(phi_sf((DELTA - m) / se), phi_sf((m + DELTA) / se))
        r["gap"]["p_tost_origin"] = max(phi_sf((DELTA - m) / seo), phi_sf((m + DELTA) / seo))
        r["gap"]["equivalent_90ci"] = bool(-DELTA < r["gap"]["ci90"][0] and r["gap"]["ci90"][1] < DELTA)
        r["gain"]["p_gt0"] = phi_sf(r["gain"]["mean"] / r["gain"]["se"])
        r["gain_minus_gap"]["p_gt0"] = phi_sf(r["gain_minus_gap"]["mean"] / r["gain_minus_gap"]["se"])
        for k, key in (("gap", "p_tost"), ("gain", "p_gt0"), ("gain_minus_gap", "p_gt0")):
            pfam[f"{t}/{k}"] = r[k][key]
        res["cities"][t] = r
        diag.log(f"[E11 {t}] " + json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("mean", "ci95", "se", "design_effect")}) for k, v in r.items()}))
        (diag.OUT / f"E11_stats{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    hp = holm(pfam)
    for k, v in hp.items():
        c, name = k.split("/")
        res["cities"][c][name]["p_holm"] = v
    # GTG directions: official model routes on the same 100k trips (gtg_eval.py alignment)
    runs = {"main": diag.OUT / "gtg"}
    for tag in ("exp_102", "exp_103"):
        if (diag.OUT / "gtg/seeds" / tag).exists():
            runs[tag] = diag.OUT / "gtg/seeds" / tag
    gfam = {}
    for src, trg in GTG_DIRS:
        P = Pt[trg]
        n, pi, f = int(P["n"]), P["pair_i"], P["f1"]
        A = diag.stage_a(trg, 8)
        rows = e4.subset(trg, "test", A, 100000)
        assert len(rows) == n and np.array_equal(np.diff(A["gt_off"])[rows], np.diff(P["gt_off"])), trg
        ids = A["traj_id"][rows]
        gts = diag.ragged_take(A["gt_flat"], A["gt_off"], rows)
        L = e4.road_meta(trg, 16)["length"]
        zs = e9.top1_vec(P, e4.softmax_by_traj(P["X"] @ betas[src], pi, n), f)
        ln = np.zeros(n)
        k0 = np.nonzero(P["pair_k"] == 0)[0]
        ln[pi[k0]] = f[k0]
        d = f"{e4.SHORT[src]}->{e4.SHORT[trg]}"
        out = {}
        for rn, rdir in runs.items():
            fcsv = rdir / f"gen_{src}2{trg}.csv"
            if not fcsv.exists():
                continue
            gen = pd.read_csv(fcsv).drop_duplicates("traj_id").set_index("traj_id").reindex(ids)
            routes = [np.fromstring(str(s), dtype=np.int64, sep=",") if isinstance(s, str) else np.zeros(0, np.int64)
                      for s in gen["rid_list"]]
            wl = e9.wlcs_batch_gpu(routes, gts, L, DEV)
            rl = np.array([L[x].sum() for x in routes])
            gl = np.array([L[x].sum() for x in gts])
            p_, r_ = wl / np.maximum(rl, 1e-9), wl / np.maximum(gl, 1e-9)
            gtg = 2 * p_ * r_ / np.maximum(p_ + r_, 1e-9)
            ref = json.loads((rdir / "GTG_eval.json").read_text())[d]
            assert abs(gtg.mean() - ref["GTG_official"]["top1_length"]) < 5e-4, (d, rn, gtg.mean())
            assert abs(zs.mean() - ref["zeroshot_logit_same_src"]["top1_length"]) < 1.5e-3, (d, rn, zs.mean())
            assert abs(ln.mean() - ref["LEN_shortest"]["top1_length"]) < 5e-4, (d, rn, ln.mean())
            o = {"logit_minus_gtg": cstats(zs - gtg, P["od_key"]), "gtg_minus_len": cstats(gtg - ln, P["od_key"])}
            o["logit_minus_gtg"]["p_gt0"] = phi_sf(o["logit_minus_gtg"]["mean"] / o["logit_minus_gtg"]["se"])
            z = o["gtg_minus_len"]["mean"] / o["gtg_minus_len"]["se"]
            o["gtg_minus_len"]["p_two_sided"] = min(1.0, 2 * phi_sf(abs(z)))
            if rn == "main":
                gfam[f"{d}/logit_minus_gtg"] = o["logit_minus_gtg"]["p_gt0"]
                gfam[f"{d}/gtg_minus_len"] = o["gtg_minus_len"]["p_two_sided"]
            out[rn] = o
        res["gtg"][d] = out
        diag.log(f"[E11 GTG {d}] " + json.dumps({rn: {k: (v["mean"], v["ci95"]) for k, v in o.items()} for rn, o in out.items()}))
        (diag.OUT / f"E11_stats{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    for k, v in holm(gfam).items():
        d, name = k.split("/")
        res["gtg"][d]["main"][name]["p_holm"] = v
    (diag.OUT / f"E11_stats{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E11 done")


if __name__ == "__main__":
    main()
