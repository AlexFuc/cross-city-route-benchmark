"""E15: does a choice model profit from more candidates? Path-size logit over the first K link-penalty routes (CPU only).

E14 shows that Oracle@K rises with K, most in Porto and San Francisco. Here the same five map features are computed
over LP105 candidate sets (link penalty, factor 1.05, as in E14) with K = 6 and K = 30, and the logit is estimated
in-city and zero-shot (pooled other four cities), with labels = best candidate by length-weighted F1 (as in E9).
Per city: 5,000 training trips drawn from the MAP6 training pool and the 5,000 E14 test trips. The MAP6 logit is
re-estimated on the same 5,000 training trips per city, so every row uses the same trips.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e9  # noqa: E402
import e14_kcand as e14  # noqa: E402
import routing  # noqa: E402
from e11_stats import cstats  # noqa: E402
from e12_sources import take  # noqa: E402

N_TR = int(os.environ.get("E15_NTR", 5000))
KS = [6, 30]
DEV = "cpu"


def _lp105(ods):
    base = e14._G["length"]
    out = []
    for o, d in ods:
        sp, dsp = e14._sp(base, o, d, np.inf)
        lst, seen = [], set()
        if sp is not None:
            lim = e14.LIMIT * dsp + base[o] + base[d]
            cost = base.copy()
            for _ in range(e14.ITER):
                p, _d = e14._sp(cost, o, d, lim * 2.0)
                if p is None:
                    break
                e14._add(lst, seen, p)
                if len(lst) >= e14.KMAX:
                    break
                cost[np.asarray(p[1:-1], np.int64)] *= 1.05
        out.append(((o, d), lst))
    return out


def generate(ods, workers):
    uniq = sorted(set(ods))
    chunk = max(1, len(uniq) // (workers * 8))
    cand = {}
    with Pool(workers) as pool:
        for part in pool.imap_unordered(_lp105, [uniq[i:i + chunk] for i in range(0, len(uniq), chunk)]):
            cand.update(dict(part))
    return [cand[od] for od in ods]


def pool_k(lists, gts, K, meta, L):
    """MAP6-style pair table over the first K routes, with length-weighted F1 and labels."""
    slots = [[x[k].astype(np.int32) if k < len(x) else np.zeros(0, np.int32) for x in lists] for k in range(K)]
    P = e4.pair_table(slots, gts, meta, DEV)
    routes = diag.ragged_take(P["rflat"], P["roff"], range(len(P["pair_i"])))
    f1 = diag.route_f1(routes, [gts[i] for i in P["pair_i"]], DEV, length=L)
    n = int(P["n"])
    best = np.full(n, -1.0)
    np.maximum.at(best, P["pair_i"], f1)
    lab_idx = np.nonzero(f1 >= best[P["pair_i"]] - 1e-12)[0]
    _, first = np.unique(P["pair_i"][lab_idx], return_index=True)
    label = np.zeros(len(f1), bool)
    label[lab_idx[first]] = True
    P["f1"], P["best"], P["label"] = f1, best, label
    return P


def take_full(P, rows):
    Q = take(P, rows)
    Q["best"], Q["od_key"] = P["best"][rows], P["od_key"][rows]
    return Q


def top1(P, beta):
    return e9.top1_vec(P, e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"])), P["f1"])


def main():
    workers = int(os.environ.get("E15_WORKERS", 8))
    E14 = json.loads((diag.OUT / f"E14_kcand{e4.TAG}.json").read_text())["cities"]
    pools = {}
    for c in e4.ALL:
        n, src, dst, length = routing.city_graph(c)
        e14._G.update(n=n, src=src.astype(np.int64), dst=dst.astype(np.int64), length=length.astype(np.float64))
        meta = e4.road_meta(c, 16)
        L = e4.seg_lengths(c)
        Pt, Ptr = e4.load_pool(c, "test"), e4.load_pool(c, "train")
        te = np.sort(np.random.default_rng(14).choice(int(Pt["n"]), min(e14.N_TRIPS, int(Pt["n"])), replace=False))
        tr = np.sort(np.random.default_rng(15).choice(int(Ptr["n"]), min(N_TR, int(Ptr["n"])), replace=False))
        r = {"MAP6": {"test": take_full(Pt, te), "train": take_full(Ptr, tr)}}
        for split, P, rows in (("test", Pt, te), ("train", Ptr, tr)):
            gts = [np.asarray(x, np.int64) for x in diag.ragged_take(P["gt_flat"], P["gt_off"], rows)]
            lists = generate([(int(g[0]), int(g[-1])) for g in gts], workers)
            for K in KS:
                r.setdefault(f"LP105@{K}", {})[split] = pool_k(lists, gts, K, meta, L)
            diag.log(f"[E15 {c}/{split}] {len(rows)} trips, mean routes {np.mean([len(x) for x in lists]):.2f}")
        best6 = r["MAP6"]["test"]["best"].mean()
        if e14.N_TRIPS != 5000:   # smoke run: E14 reference values are for 5,000 trips
            pools[c] = r
            continue
        assert abs(best6 - E14[c]["MAP6_oracle6"]) < 1e-3, (c, best6)
        for K in KS:
            o = r[f"LP105@{K}"]["test"]["best"].mean()
            assert abs(o - E14[c]["LP105"]["oracle"][str(K)]) < 2e-3, (c, K, o)
        pools[c] = r
    res = {"params": dict(N_TR=N_TR, N_TE=e14.N_TRIPS, KS=KS), "cities": {}}
    for t in e4.ALL:
        out = {}
        for cs in pools[t]:
            Pte = pools[t][cs]["test"]
            b_in = e4.fit_logit(pools[t][cs]["train"], DEV)
            b_zs = e4.fit_logit(e4b.concat([pools[s][cs]["train"] for s in e4.ALL if s != t]), DEV)
            zin, zzs = top1(Pte, b_in), top1(Pte, b_zs)
            out[cs] = {"oracle": round(float(Pte["best"].mean()), 4), "incity": round(float(zin.mean()), 4),
                       "zeroshot": round(float(zzs.mean()), 4), "beta_zeroshot": [round(float(x), 3) for x in b_zs],
                       "routes_per_trip": round(len(Pte["pair_i"]) / int(Pte["n"]), 2),
                       "_zs": zzs, "_in": zin}
        od = pools[t]["MAP6"]["test"]["od_key"]
        for cs in [k for k in out if k != "MAP6"]:
            out[cs]["zeroshot_minus_map6"] = cstats(out[cs]["_zs"] - out["MAP6"]["_zs"], od)
            out[cs]["incity_minus_map6"] = cstats(out[cs]["_in"] - out["MAP6"]["_in"], od)
        for v in out.values():
            v.pop("_zs"); v.pop("_in")
        res["cities"][t] = out
        diag.log(f"[E15 {t}] " + json.dumps({k: (v["oracle"], v["zeroshot"], v["incity"]) for k, v in out.items()}))
        (diag.OUT / f"E15_kcand_logit{e4.TAG}{'' if e14.N_TRIPS == 5000 else '_smoke'}.json").write_text(json.dumps(res, indent=2))
    diag.log("E15 done")


if __name__ == "__main__":
    main()
