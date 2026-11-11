"""E16: temporal split. In-city references trained on earlier trips and tested on later ones (CPU only).

The published splits draw training and test trips from the same period, which may flatter the in-city references,
OD history most. Per city, the cutoff is the start of the day that contains the 75% quantile of training start
times (days in UTC+8 for the Chinese cities, UTC otherwise). In-city logit and GBDT (MAP6 training pool) and OD
history (all training trips, zero-shot fallback as in E10) are fitted either on the trips before the cutoff
("temporal") or on a random subset of the same size from the whole period ("random"). Both are evaluated on the
test-pool trips after the cutoff. The zero-shot logit (pooled other cities, whole period) is unchanged.
OD-cluster statistics as in E11.
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
from e11_stats import DEV, cstats  # noqa: E402
from e12_sources import take  # noqa: E402

CN = ("xian", "chengdu", "beijing")


def day(st, city):
    return (np.asarray(st, np.int64) + (8 * 3600 if city in CN else 0)) // 86400


def take_d(P, trips):
    Q = take(P, trips)
    for k in ("hour", "min_len", "od_key"):
        Q[k] = P[k][trips]
    return Q


def gbdt_top1(Ptr, P):
    g = e9.fit_gbdt(e9.design(Ptr), Ptr["label"])
    sc = g.predict_proba(e9.design(P))[:, 1]
    return e9.top1_vec(P, sc / np.bincount(P["pair_i"], weights=sc, minlength=int(P["n"]))[P["pair_i"]], P["f1"])


def odh_top1(A_tr, rows, P, fallback, L, n_seg):
    hist = e4.od_history(A_tr, rows, n_seg)
    keys = P["od_key"]
    idx = np.nonzero(np.array([int(k) in hist for k in keys]))[0]
    gts = diag.ragged_take(P["gt_flat"], P["gt_off"], idx)
    pred = [np.frombuffer(hist[int(keys[i])][0][0], dtype=np.int32).astype(np.int64) for i in idx]
    out = fallback.copy()
    out[idx] = diag.route_f1(pred, [np.asarray(g, np.int64) for g in gts], DEV, length=L)
    return out, float(len(idx) / len(keys))


def main():
    cities = e4.ALL
    E9 = json.loads((diag.OUT / f"E9_rigor{e4.TAG}.json").read_text())
    pools = {c: e4.load_pool(c, "train") for c in cities}
    res = {"quantile": 0.75, "cities": {}}
    rng = np.random.default_rng(16)
    for t in cities:
        meta = e4.road_meta(t, 16)
        A_te, A_tr = e4.load_split(t, "test", 8), e4.load_split(t, "train", 8)
        P, Ptr = e4.load_pool(t, "test"), pools[t]
        n = int(P["n"])
        rows_te = e4.subset(t, "test", A_te, 100000)
        rows_tr = e4.subset(t, "train", A_tr, int(Ptr["n"]))
        assert len(rows_te) == n and np.array_equal(np.diff(A_te["gt_off"])[rows_te], np.diff(P["gt_off"])), t
        assert len(rows_tr) == int(Ptr["n"]) and np.array_equal(np.diff(A_tr["gt_off"])[rows_tr], np.diff(Ptr["gt_off"])), t
        d_all = day(A_tr["start_time"], t)
        cut = int(day([np.quantile(A_tr["start_time"], 0.75)], t)[0])
        late = np.nonzero(day(A_te["start_time"][rows_te], t) >= cut)[0]
        early_pool = np.nonzero(day(A_tr["start_time"][rows_tr], t) < cut)[0]
        early_all = np.nonzero(d_all < cut)[0]
        rand_pool = np.sort(rng.choice(int(Ptr["n"]), len(early_pool), replace=False))
        rand_all = np.sort(rng.choice(len(d_all), len(early_all), replace=False))
        b_zs = e4.fit_logit(e4b.concat([pools[s] for s in cities if s != t]), DEV)
        pi = P["pair_i"]
        zs = e9.top1_vec(P, e4.softmax_by_traj(P["X"] @ b_zs, pi, n), P["f1"])
        assert abs(zs.mean() - E9[t]["top1"]["zeroshot_pooled"]["length"]) < 1.5e-3, (t, zs.mean())
        top, cov = {"zeroshot": zs}, {}
        for v, ip, ia in (("temporal", early_pool, early_all), ("random", rand_pool, rand_all)):
            Q = take_d(Ptr, ip)
            top[f"logit_{v}"] = e9.top1_vec(P, e4.softmax_by_traj(P["X"] @ e4.fit_logit(Q, DEV), pi, n), P["f1"])
            top[f"gbdt_{v}"] = gbdt_top1(Q, P)
            top[f"odh_{v}"], cov[v] = odh_top1(A_tr, ia, P, zs, meta["length"], meta["n_seg"])
        od = P["od_key"][late]
        r = {"cutoff_day_utc": cut, "n_late_test": int(len(late)), "n_early_pool": int(len(early_pool)),
             "n_early_all": int(len(early_all)), "share_late_test": round(len(late) / n, 3),
             "odh_coverage": {k: round(x, 4) for k, x in cov.items()},
             "top1_late": {k: round(float(x[late].mean()), 4) for k, x in top.items()}, "diff": {}}
        for v in ("temporal", "random"):
            r["diff"][f"gap_{v}"] = cstats(top[f"logit_{v}"][late] - zs[late], od)
            r["diff"][f"gain_{v}"] = cstats(top[f"gbdt_{v}"][late] - top[f"logit_{v}"][late], od)
            r["diff"][f"odh_minus_gbdt_{v}"] = cstats(top[f"odh_{v}"][late] - top[f"gbdt_{v}"][late], od)
        for m in ("logit", "gbdt", "odh"):
            r["diff"][f"{m}_temporal_minus_random"] = cstats(top[f"{m}_temporal"][late] - top[f"{m}_random"][late], od)
        res["cities"][t] = r
        diag.log(f"[E16 {t}] " + json.dumps({k: r[k] for k in ("n_late_test", "n_early_pool", "odh_coverage", "top1_late")}))
        (diag.OUT / f"E16_temporal{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E16 done")


if __name__ == "__main__":
    main()
