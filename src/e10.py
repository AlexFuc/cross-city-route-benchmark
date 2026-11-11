"""E10: length-weighted versions of the OD-history reference and the LOO predictability ceiling, plus the
dataset statistics table (all 5 cities), so every number in the decomposition uses the primary metric.

Per city (target test 100k subset unless noted):
  odhist_full_len     OD-history hybrid top-1 (full target train; fallback pooled zero-shot logit), length-F1
  loo_len_frequent    LOO mode-route ceiling on trips whose OD has >=5 test trips (full test), length-F1,
                      plus zero-shot / in-city logit / in-city GBDT / BoK on the same trips (length-F1)
  stats               segments, trips, median trip km, exact-shortest-path share, OD recurrence
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e9  # noqa: E402


def wf1(pred, gts, L, device):
    wl = e9.wlcs_batch_gpu(pred, gts, L, device)
    pl = np.array([L[p].sum() for p in pred])
    gl = np.array([L[g].sum() for g in gts])
    p, r = wl / np.maximum(pl, 1e-9), wl / np.maximum(gl, 1e-9)
    return 2 * p * r / np.maximum(p + r, 1e-9)


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    cities = e4.ALL
    out = {}
    for t in cities:
        meta = e4.road_meta(t, 16)
        L = meta["length"]
        A_te, A_tr = e4.load_split(t, "test", 8), e4.load_split(t, "train", 8)
        P, Ptr_t = e4.load_pool(t, "test"), e4.load_pool(t, "train")
        n = int(P["n"])
        pi = P["pair_i"]
        rows = e4.subset(t, "test", A_te, 100000)
        gts = [np.asarray(x, np.int64) for x in diag.ragged_take(P["gt_flat"], P["gt_off"], range(n))]
        # per-pair length-F1 (same as E9)
        routes = diag.ragged_take(P["rflat"], P["roff"], range(len(pi)))
        f1w_pair = wf1([np.asarray(r, np.int64) for r in routes], [gts[i] for i in pi], L, device)
        pooled = e4b.concat([e4.load_pool(s, "train") for s in cities if s != t])
        b_zs, b_in = e4.fit_logit(pooled, device), e4.fit_logit(Ptr_t, device)
        p_zs = e4.softmax_by_traj(P["X"] @ b_zs, pi, n)
        p_in = e4.softmax_by_traj(P["X"] @ b_in, pi, n)
        g = e9.fit_gbdt(e9.design(Ptr_t), Ptr_t["label"])
        sc = g.predict_proba(e9.design(P))[:, 1]
        top = {"zeroshot": e9.top1_vec(P, p_zs, f1w_pair), "incity_logit": e9.top1_vec(P, p_in, f1w_pair),
               "incity_gbdt": e9.top1_vec(P, sc, f1w_pair)}
        bok = np.full(n, -1.0)
        np.maximum.at(bok, pi, f1w_pair)
        top["bok"] = bok
        # OD-history hybrid with full target train, length-F1
        hist = e4.od_history(A_tr, np.arange(len(A_tr["gt_off"]) - 1), meta["n_seg"])
        keys = P["od_key"]
        seen = np.array([int(k) in hist for k in keys])
        idx = np.nonzero(seen)[0]
        pred = [np.frombuffer(hist[int(keys[i])][0][0], dtype=np.int32).astype(np.int64) for i in idx]
        od = top["zeroshot"].copy()
        od[idx] = wf1(pred, [gts[i] for i in idx], L, device)
        top["odhist_full"] = od
        # LOO ceiling on frequent ODs (full test set), then restricted to subset trips
        off, flat = A_te["gt_off"], A_te["gt_flat"]
        by_od = defaultdict(lambda: defaultdict(int))
        ok = np.where(np.diff(off) > 1)[0]
        for i in ok:
            r = flat[off[i]:off[i + 1]]
            by_od[(int(r[0]), int(r[-1]))][r.tobytes()] += 1
        size = {k: sum(v.values()) for k, v in by_od.items()}
        tops = {k: sorted(v.items(), key=lambda x: -x[1])[:2] for k, v in by_od.items()}
        pos = {int(r): j for j, r in enumerate(rows)}
        li, lp, lg = [], [], []
        for i in ok:
            r = flat[off[i]:off[i + 1]]
            k = (int(r[0]), int(r[-1]))
            if size[k] < 5 or int(i) not in pos:
                continue
            (r1, c1), *rest = tops[k]
            pbytes = rest[0][0] if (r.tobytes() == r1 and (c1 - 1) < (rest[0][1] if rest else 0)) else r1
            li.append(pos[int(i)])
            lp.append(np.frombuffer(pbytes, dtype=np.int32).astype(np.int64))
            lg.append(np.asarray(r, np.int64))
        res = {k: round(float(v.mean()), 4) for k, v in top.items()}
        if li:
            j = np.array(li)
            loo = wf1(lp, lg, L, device)
            res["frequent_od"] = {"n": int(len(j)), "loo_ceiling": round(float(loo.mean()), 4),
                                  **{k: round(float(v[j].mean()), 4) for k, v in top.items()}}
        # dataset statistics
        gl_km = np.array([L[g].sum() for g in gts]) / 1000
        lens_first = P["pair_k"] == 0
        len_routes = [P["rflat"][P["roff"][q]:P["roff"][q + 1]] for q in np.nonzero(lens_first)[0]]
        exact = np.mean([np.array_equal(a, b) for a, b in zip(len_routes, gts)])
        res["stats"] = {"segments": int(meta["n_seg"]), "train_trips": int(len(A_tr["gt_off"]) - 1),
                        "test_trips": int(len(A_te["gt_off"]) - 1), "median_trip_km": round(float(np.median(gl_km)), 2),
                        "exact_shortest_share": round(float(exact), 3), "od_recurrence": round(float(seen.mean()), 3)}
        out[t] = res
        diag.log(f"[{t}] " + json.dumps(res))
        (diag.OUT / f"E10_len{e4.TAG}.json").write_text(json.dumps(out, indent=2))
    diag.log("E10 done")


if __name__ == "__main__":
    main()
