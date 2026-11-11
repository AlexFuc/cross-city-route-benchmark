"""E14: is low coverage a property of the city or of six candidates? Oracle@K for larger candidate sets (CPU only).

Per target city, 5,000 random trips of the 100k test subset (fixed seed). Generators on raw segment lengths, each
returning distinct routes in generation order (at most KMAX):
  LP105  link penalty, factor 1.05 per use, up to ITER iterations
  LP2    link penalty, factor 2 (the MAP6 penalty), up to ITER iterations
  LE     link elimination: remove one segment of the shortest path at a time (evenly spaced along it)
  SIM    simulation: lognormal noise on every segment cost (sigma 0.5), ITER draws
  UNION  round-robin merge of the four
Searches are cut at LIMIT x the shortest-path length (2x / 8x that under LP105 / LP2 penalties, 2x for SIM).
Reported: Oracle@K (best length-weighted F1 among the first K routes), coverage at overlap >= 0.8 / 0.9 / 1.0
(overlap = matched length / observed length, as in the choice-set literature), number of distinct routes,
and MAP6 Oracle@6 on the same trips.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e9  # noqa: E402
import routing  # noqa: E402

N_TRIPS, KMAX, ITER, LIMIT, SIGMA = int(os.environ.get("E14_N", 5000)), 30, 40, 4.0, 0.5
KS = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30]
GENS = ["LP105", "LP2", "LE", "SIM"]
_G = {}


def _sp(cost, o, d, lim):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    src, dst, n = _G["src"], _G["dst"], _G["n"]
    w = (cost[src] + cost[dst]) / 2.0
    ok = np.isfinite(w)
    csr = csr_matrix((w[ok], (src[ok], dst[ok])), shape=(n, n))
    dist, pred = dijkstra(csr, directed=True, indices=o, return_predecessors=True, limit=lim)
    if not np.isfinite(dist[d]):
        return None, np.inf
    path = [d]
    while path[-1] != o:
        p = int(pred[path[-1]])
        if p < 0:
            return None, np.inf
        path.append(p)
    return path[::-1], float(dist[d])


def _add(lst, seen, path):
    if path is not None:
        k = tuple(path)
        if k not in seen:
            seen.add(k)
            lst.append(np.asarray(path, np.int64))


def _worker(task):
    ods, seed = task
    base = _G["length"]
    rng = np.random.default_rng(seed)
    out = []
    for o, d in ods:
        sp, dsp = _sp(base, o, d, np.inf)
        res = {}
        if sp is None:
            out.append(((o, d), {g: [] for g in GENS}))
            continue
        lim = LIMIT * dsp + base[o] + base[d]
        for g, factor in (("LP105", 1.05), ("LP2", 2.0)):
            cost, lst, seen = base.copy(), [], set()
            for _ in range(ITER):
                p, _d = _sp(cost, o, d, lim * (8.0 if g == "LP2" else 2.0))
                if p is None:
                    break
                _add(lst, seen, p)
                if len(lst) >= KMAX:
                    break
                cost[np.asarray(p[1:-1], np.int64)] *= factor
            res[g] = lst
        lst, seen = [], set()
        _add(lst, seen, sp)
        inner = sp[1:-1]
        for j in np.unique(np.linspace(0, len(inner) - 1, min(len(inner), 2 * KMAX)).round().astype(int)) if inner else []:
            cost = base.copy()
            cost[inner[j]] = np.inf
            _add(lst, seen, _sp(cost, o, d, lim)[0])
            if len(lst) >= KMAX:
                break
        res["LE"] = lst
        lst, seen = [], set()
        _add(lst, seen, sp)
        for _ in range(ITER):
            cost = base * np.exp(SIGMA * rng.standard_normal(len(base)))
            cost[o], cost[d] = base[o], base[d]
            _add(lst, seen, _sp(cost, o, d, lim * 2)[0])
            if len(lst) >= KMAX:
                break
        res["SIM"] = lst
        out.append(((o, d), res))
    return out


def union(lists):
    out, seen = [], set()
    for k in range(max(len(x) for x in lists)):
        for x in lists:
            if k < len(x) and tuple(x[k]) not in seen:
                seen.add(tuple(x[k]))
                out.append(x[k])
    return out[:KMAX]


def main():
    workers = int(os.environ.get("E14_WORKERS", 8))
    res = {"params": dict(N_TRIPS=N_TRIPS, KMAX=KMAX, ITER=ITER, LIMIT=LIMIT, SIGMA=SIGMA, KS=KS), "cities": {}}
    for t in e4.ALL:
        n, src, dst, length = routing.city_graph(t)
        _G.update(n=n, src=src.astype(np.int64), dst=dst.astype(np.int64), length=length.astype(np.float64))
        P = e4.load_pool(t, "test")
        rng = np.random.default_rng(14)
        pick = np.sort(rng.choice(int(P["n"]), min(N_TRIPS, int(P["n"])), replace=False))
        gts = [np.asarray(x, np.int64) for x in diag.ragged_take(P["gt_flat"], P["gt_off"], pick)]
        ods = [(int(g[0]), int(g[-1])) for g in gts]
        uniq = sorted(set(ods))
        chunk = max(1, len(uniq) // (workers * 8))
        tasks = [(uniq[i:i + chunk], 1000 + i) for i in range(0, len(uniq), chunk)]
        cand = {}
        with Pool(workers) as pool:
            for part in pool.imap_unordered(_worker, tasks):
                cand.update(dict(part))
        sets = {g: [cand[od][g] for od in ods] for g in GENS}
        sets["UNION"] = [union([cand[od][g] for g in GENS]) for od in ods]
        r = {"n": len(pick)}
        best6 = np.full(int(P["n"]), -1.0)
        np.maximum.at(best6, P["pair_i"], P["f1"])
        r["MAP6_oracle6"] = round(float(best6[pick].mean()), 4)
        for g, S in sets.items():
            flat_r, flat_g, owner, rank = [], [], [], []
            for i, lst in enumerate(S):
                for k, x in enumerate(lst):
                    flat_r.append(x); flat_g.append(gts[i]); owner.append(i); rank.append(k)
            owner, rank = np.array(owner), np.array(rank)
            wl = e9.wlcs_batch_gpu(flat_r, flat_g, length, "cpu")
            pl = np.array([length[x].sum() for x in flat_r])
            gl = np.array([length[x].sum() for x in flat_g])
            pr, ov = wl / np.maximum(pl, 1e-9), wl / np.maximum(gl, 1e-9)
            f1 = 2 * pr * ov / np.maximum(pr + ov, 1e-9)
            nd = np.array([len(x) for x in S])
            rr = {"distinct_mean": round(float(nd.mean()), 2), "distinct_median": float(np.median(nd)),
                  "share_full_K": round(float((nd >= KMAX).mean()), 3), "oracle": {}, "coverage": {}}
            for K in KS:
                m = rank < K
                b = np.zeros(len(S)); np.maximum.at(b, owner[m], f1[m])
                rr["oracle"][str(K)] = round(float(b.mean()), 4)
                c = np.zeros(len(S)); np.maximum.at(c, owner[m], ov[m])
                rr["coverage"][str(K)] = {str(th): round(float((c >= th - 1e-9).mean()), 4) for th in (0.8, 0.9, 1.0)}
            r[g] = rr
        res["cities"][t] = r
        diag.log(f"[E14 {t}] " + json.dumps({g: (v["distinct_mean"], v["oracle"]["6"], v["oracle"]["30"]) if isinstance(v, dict) else v
                                            for g, v in r.items()}))
        (diag.OUT / f"E14_kcand{e4.TAG}{os.environ.get('E14_SUFFIX', '')}.json").write_text(json.dumps(res, indent=2))
    diag.log("E14 done")


if __name__ == "__main__":
    main()
