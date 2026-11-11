"""Evaluate official GTG routes against the same-source zero-shot logit and LEN-SP on identical trips.

Per direction src->trg (target test 100k subset):
  top-1 F1 (count LCS and length-weighted), all-link flow NMAE (GT flow >= 5), mean Hausdorff
  (routing.hausdorff_metric on segment GPS, same formula as the GTG/HOSER code; compared with GTG paper Table 1).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e9  # noqa: E402

DIRS = [("chengdu", "xian"), ("beijing", "xian"), ("beijing", "chengdu"), ("xian", "beijing"), ("xian", "chengdu"),
        ("chengdu", "beijing")]
EXP = {("chengdu", "xian"): "exp_1/chengdu_to_xian", ("beijing", "xian"): "exp_2/beijing_to_xian",
       ("beijing", "chengdu"): "exp_3/beijing_to_chengdu", ("xian", "beijing"): "exp_4/xian_to_beijing",
       ("xian", "chengdu"): "exp_5/xian_to_chengdu", ("chengdu", "beijing"): "exp_6/chengdu_to_beijing"}
if os.environ.get("GTG_DIRS"):  # e.g. "xian:xian,chengdu:chengdu" for the same-city runs
    DIRS = [tuple(d.split(":")) for d in os.environ["GTG_DIRS"].split(",")]


def gps_array(city):
    g = json.load(open(diag.DATA_ROOT / f"data/{city}/map/rid_to_gps.json"))
    arr = np.zeros((max(int(k) for k in g) + 1, 2))
    for k, (lon, lat) in g.items():
        arr[int(k)] = (lat, lon)
    return arr


def route_metrics(routes, gts, meta, gps, device, haus_n=20000):
    L = meta["length"]
    nr = np.array([len(r) for r in routes])
    ng = np.array([len(g) for g in gts])
    f1c = diag.f1_from_lcs(diag.lcs_batch_gpu(routes, gts, device), nr, ng)
    wl = e9.wlcs_batch_gpu(routes, gts, L, device)
    rl = np.array([L[r].sum() for r in routes])
    gl = np.array([L[g].sum() for g in gts])
    p, r = wl / np.maximum(rl, 1e-9), wl / np.maximum(gl, 1e-9)
    f1w = 2 * p * r / np.maximum(p + r, 1e-9)
    gt_flow = np.bincount(np.concatenate(gts), minlength=meta["n_seg"]).astype(float)
    pr_flow = np.bincount(np.concatenate([x for x in routes if len(x)]), minlength=meta["n_seg"]).astype(float)
    from routing import hausdorff_metric
    idx = np.random.default_rng(0).choice(len(gts), min(haus_n, len(gts)), replace=False)
    haus = np.mean([hausdorff_metric(gps[gts[i]], gps[routes[i]]) for i in idx if len(routes[i])])
    return {"top1_count": round(float(f1c.mean()), 4), "top1_length": round(float(f1w.mean()), 4),
            "flow_nmae": round(diag.geh_stats(pr_flow, gt_flow, gt_flow >= 5)["nmae"], 4),
            "hausdorff_mean_20k": round(float(haus), 4)}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    gdir = diag.OUT / "gtg" / (sys.argv[2] if len(sys.argv) > 2 else "")
    res = {}
    for src, trg in DIRS:
        f = gdir / f"gen_{src}2{trg}.csv"
        if not f.exists():
            print("missing", f)
            continue
        meta = e4.road_meta(trg, 16)
        gps = gps_array(trg)
        A = diag.stage_a(trg, 8)
        rows = e4.subset(trg, "test", A, 100000)
        ids = A["traj_id"][rows]
        gts = diag.ragged_take(A["gt_flat"], A["gt_off"], rows)
        gen = pd.read_csv(f).drop_duplicates("traj_id").set_index("traj_id").reindex(ids)
        gtg_routes = [np.fromstring(str(s), dtype=np.int64, sep=",") if isinstance(s, str) else np.zeros(0, np.int64)
                      for s in gen["rid_list"]]
        out = {"n": int(len(rows)), "missing_gtg": int(sum(len(r) == 0 for r in gtg_routes))}
        out["GTG_official"] = route_metrics(gtg_routes, gts, meta, gps, device)
        # same-source zero-shot map-only logit (top-1 route)
        P = e4.load_pool(trg, "test")
        beta = e4.fit_logit(e4.load_pool(src, "train"), device)
        p = e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"]))
        top = e4.evaluate_probs(P, p, meta, np.zeros(meta["n_seg"]), np.zeros(meta["n_seg"], bool))["top_idx"]
        zs = [None] * int(P["n"])
        for j in top:
            zs[P["pair_i"][j]] = P["rflat"][P["roff"][j]:P["roff"][j + 1]].astype(np.int64)
        out["zeroshot_logit_same_src"] = route_metrics(zs, gts, meta, gps, device)
        lenr = [P["rflat"][P["roff"][j]:P["roff"][j + 1]].astype(np.int64) for j in np.nonzero(P["pair_k"] == 0)[0]]
        out["LEN_shortest"] = route_metrics(lenr, gts, meta, gps, device)
        res[f"{e4.SHORT[src]}->{e4.SHORT[trg]}"] = out
        diag.log(f"[{src}->{trg}] " + json.dumps(out))
        (gdir / "GTG_eval.json").write_text(json.dumps(res, indent=2))
    diag.log("GTG eval done")


if __name__ == "__main__":
    main()
