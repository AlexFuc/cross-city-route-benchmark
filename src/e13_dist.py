"""E13: distribution-level metrics (GTG's macro metrics) for every method, and whether they rank methods like route F1.

GTG definitions (gtg_official/evaluate/macro_similarity.py): per route, travel distance = sum of straight-line
distances between consecutive segment GPS points (km), radius = mean distance of those points from their centroid
(km); JSD of histograms (distance: 1000 bins over 0-30 km, radius: 100 bins over 0-10 km, density=True) and of the
segment visit-frequency distribution (LocFreq). GTG projects with EPSG:32649 for every city; we use a local
equirectangular projection (accurate within a city; also valid for Porto/SF). `check` mode compares with the
official function on Xi'an (run in gtg_official/ with the gtgp2 env). All methods see the same test ODs, so OD
statistics are identical by construction. CPU only.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy, kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DIST_BINS = np.arange(0, 30, 30 / 1000).tolist()
RAD_BINS = np.arange(0, 10, 10 / 100).tolist()
R_EARTH = 6371008.8


def jsd(p, q):
    m = (p + q) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def local_xy(gps):
    lon0, lat0 = np.radians(np.nanmean(gps, 0))
    lon, lat = np.radians(gps[:, 0]), np.radians(gps[:, 1])
    return np.c_[R_EARTH * np.cos(lat0) * (lon - lon0), R_EARTH * (lat - lat0)]


def route_stats(routes, xy, n_seg):
    """-> (distance km, radius km) per non-empty route, segment visit counts."""
    rs = [np.asarray(r, np.int64) for r in routes if r is not None and len(r)]
    lens = np.array([len(r) for r in rs])
    flat = np.concatenate(rs)
    p = xy[flat]
    owner = np.repeat(np.arange(len(rs)), lens)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    step[np.cumsum(lens)[:-1] - 1] = 0.0          # no step across route boundaries
    dist = np.bincount(owner[1:], weights=step, minlength=len(rs)) / 1000
    cen = np.c_[np.bincount(owner, weights=p[:, 0]), np.bincount(owner, weights=p[:, 1])] / lens[:, None]
    rad = np.bincount(owner, weights=np.linalg.norm(p - cen[owner], axis=1)) / lens / 1000
    return dist, rad, np.bincount(flat, minlength=n_seg).astype(float)


def macro(gen, gt):
    hg = lambda x, b: np.histogram(x, b, density=True)[0]  # noqa: E731
    return {"Distance": jsd(hg(gen[0], DIST_BINS), hg(gt[0], DIST_BINS)),
            "Radius": jsd(hg(gen[1], RAD_BINS), hg(gt[1], RAD_BINS)),
            "LocFreq": jsd(gen[2] / gen[2].sum(), gt[2] / gt[2].sum())}


def rank_agreement(res):
    """Kendall tau between each JSD (lower = better) and top-1 F1 (higher = better) over the methods of one table."""
    names = [k for k in res if k != "ground_truth"]
    f1 = np.array([res[k]["top1_length"] for k in names])
    return {m: round(float(kendalltau(-f1, [res[k][m] for k in names])[0]), 3) for m in ("Distance", "Radius", "LocFreq")}


def main():
    import diag
    import e4
    import e4b
    import e9
    import gtg_eval
    from hoser_eval import HOSER_OUT, top_routes
    dev = "cpu"
    out = {}
    for t in e4.ALL:
        meta = e4.road_meta(t, 16)
        n_seg, L = meta["n_seg"], meta["length"]
        xy = local_xy(gtg_eval.gps_array(t))
        P, Ptr = e4.load_pool(t, "test"), e4.load_pool(t, "train")
        n = int(P["n"])
        gts = [np.asarray(x, np.int64) for x in diag.ragged_take(P["gt_flat"], P["gt_off"], range(n))]
        pooled = e4b.concat([e4.load_pool(s, "train") for s in e4.ALL if s != t])
        R = {"LEN": [P["rflat"][P["roff"][j]:P["roff"][j + 1]].astype(np.int64) for j in np.nonzero(P["pair_k"] == 0)[0]]}
        R["zeroshot_logit"] = top_routes(P, P["X"] @ e4.fit_logit(pooled, dev), n_seg)
        R["incity_logit"] = top_routes(P, P["X"] @ e4.fit_logit(Ptr, dev), n_seg)
        g = e9.fit_gbdt(e9.design(Ptr), Ptr["label"])
        R["incity_gbdt"] = top_routes(P, g.predict_proba(e9.design(P))[:, 1], n_seg)
        A_tr = e4.load_split(t, "train", 8)
        hist = e4.od_history(A_tr, np.arange(len(A_tr["gt_off"]) - 1), n_seg)
        R["incity_od_history"] = [np.frombuffer(hist[int(k)][0][0], dtype=np.int32).astype(np.int64) if int(k) in hist
                                  else R["zeroshot_logit"][i] for i, k in enumerate(P["od_key"])]
        R["oracle6"] = top_routes(P, P["f1"], n_seg)
        extra = {}
        if t in ("xian", "chengdu", "beijing"):
            A = diag.stage_a(t, 8)
            rows = e4.subset(t, "test", A, 100000)
            ids = A["traj_id"][rows]
            for s, tt in [d for d in [("chengdu", "xian"), ("beijing", "xian"), ("beijing", "chengdu"), ("xian", "beijing"),
                                      ("xian", "chengdu"), ("chengdu", "beijing")] if d[1] == t]:
                gen = pd.read_csv(diag.OUT / f"gtg/gen_{s}2{t}.csv").drop_duplicates("traj_id").set_index("traj_id").reindex(ids)
                R[f"GTG_from_{e4.SHORT[s]}"] = [np.fromstring(str(x), dtype=np.int64, sep=",") if isinstance(x, str)
                                                else np.zeros(0, np.int64) for x in gen["rid_list"]]
        gt_st = route_stats(gts, xy, n_seg)
        res = {}
        for k, routes in R.items():
            routes = [r if r is not None else np.zeros(0, np.int64) for r in routes]
            m = macro(route_stats(routes, xy, n_seg), gt_st)
            m["top1_length"] = float(np.mean(gtg_eval_f1(routes, gts, L, dev)))
            res[k] = {kk: round(float(v), 5) for kk, v in m.items()}
        res["ground_truth"] = {kk: round(float(v), 5) for kk, v in macro(gt_st, gt_st).items()}
        out[t] = {"n": n, "methods": res, "kendall_tau_vs_f1": rank_agreement(res)}
        if t in ("porto", "sanfrancisco"):   # HOSER: same 10k trips as tab_hoser
            pick = np.load(diag.OUT / f"hoser/pick_{t}.npy")
            hrow = np.load(diag.OUT / f"hoser/rows_{t}.npy")
            gen = pd.read_csv(HOSER_OUT / f"gen_{t}.csv").set_index("test_row").reindex(hrow)
            R10 = {k: [v[i] if v[i] is not None else np.zeros(0, np.int64) for i in pick] for k, v in R.items()}
            R10["HOSER"] = [np.array(json.loads(s), np.int64) if isinstance(s, str) else np.zeros(0, np.int64)
                            for s in gen["gene_trace_road_id"]]
            g10 = [gts[i] for i in pick]
            st10 = route_stats(g10, xy, n_seg)
            r10 = {}
            for k, routes in R10.items():
                m = macro(route_stats(routes, xy, n_seg), st10)
                m["top1_length"] = float(np.mean(gtg_eval_f1(routes, g10, L, dev)))
                r10[k] = {kk: round(float(v), 5) for kk, v in m.items()}
            out[t]["hoser_10k"] = {"methods": r10, "kendall_tau_vs_f1": rank_agreement(r10)}
        diag.log(f"[E13 {t}] " + json.dumps(out[t])[:2500])
        (diag.OUT / f"E13_dist{e4.TAG}.json").write_text(json.dumps(out, indent=2))
    diag.log("E13 done")


def gtg_eval_f1(routes, gts, L, dev):
    import e9
    wl = e9.wlcs_batch_gpu(routes, gts, L, dev)
    rl = np.array([L[r].sum() for r in routes])
    gl = np.array([L[g].sum() for g in gts])
    p, r = wl / np.maximum(rl, 1e-9), wl / np.maximum(gl, 1e-9)
    return 2 * p * r / np.maximum(p + r, 1e-9)


def check():
    """Our vectorized statistics (with GTG's own EPSG:32649 projection) vs the official function, Xi'an test routes."""
    import pyproj
    sys.path.insert(1, os.getcwd())
    from evaluate.macro_similarity import calc_macro_similarity_parallel
    df = pd.read_csv("data/xian/traj/test.csv", sep=";").head(20000)
    gps = json.load(open("data/xian/map/rid_to_gps.json"))
    gen = pd.read_csv(sys.argv[2]).head(20000) if len(sys.argv) > 2 else df.sample(frac=1, random_state=0)
    off = calc_macro_similarity_parallel(gen_df=gen, gt_df=df, rid_gps=gps)
    arr = np.zeros((max(int(k) for k in gps) + 1, 2))
    for k, (lon, lat) in gps.items():
        arr[int(k)] = lon, lat
    x, y = pyproj.Transformer.from_crs(4326, 32649).transform(arr[:, 1], arr[:, 0])
    xy = np.c_[x, y]
    rt = lambda d: [np.array([int(v) for v in s.split(",")]) for s in d.rid_list]  # noqa: E731
    k = min(len(df), len(gen))
    ours = macro(route_stats(rt(gen.iloc[:k]), xy, len(arr)), route_stats(rt(df.iloc[:k]), xy, len(arr)))
    ours_local = macro(route_stats(rt(gen.iloc[:k]), local_xy(arr), len(arr)), route_stats(rt(df.iloc[:k]), local_xy(arr), len(arr)))
    print("CHECK official", off, "\nCHECK ours_utm49", ours, "\nCHECK ours_local", ours_local, flush=True)


if __name__ == "__main__":
    check() if sys.argv[1:2] == ["check"] else main()
