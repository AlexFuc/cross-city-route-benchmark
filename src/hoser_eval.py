"""Official HOSER (in-city, OD-conditioned) vs. our methods on the same 10k target test trips (Porto, SF).

Every method contributes one route per trip; metrics as in gtg_eval.route_metrics (top-1 F1 count/length,
flow NMAE of the top-1 routes on segments with flow >= 5 within the 10k sample, mean Hausdorff).
Run with CC_F1=length so pools, oracle labels and fits match the paper's primary setting.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e9  # noqa: E402
import gtg_eval  # noqa: E402

assert e4.F1_MODE == "length"
HOSER_OUT = Path(os.environ.get("HOSER_OUT", str(Path(os.environ["HOSER_DIR"]) / "gene_eval")))


def route_metrics(routes, gts, meta, gps, device):
    r = gtg_eval.route_metrics(routes, gts, meta, gps, device, haus_n=len(gts))
    r["hausdorff_mean"] = r.pop("hausdorff_mean_20k")
    return r


def top_routes(P, score, n_seg):
    top = e4.evaluate_probs(P, score, {"n_seg": n_seg}, np.zeros(n_seg), np.zeros(n_seg, bool))["top_idx"]
    out = [None] * int(P["n"])
    for j in top:
        out[P["pair_i"][j]] = P["rflat"][P["roff"][j]:P["roff"][j + 1]].astype(np.int64)
    return out


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    res = {}
    for city in ["porto", "sanfrancisco"]:
        f = HOSER_OUT / f"gen_{city}.csv"
        if not f.exists():
            print("missing", f)
            continue
        meta = e4.road_meta(city, 16)
        gps = gtg_eval.gps_array(city)
        P = e4.load_pool(city, "test")
        pick = np.load(diag.OUT / f"hoser/pick_{city}.npy")          # positions in the 100k test subset
        hrow = np.load(diag.OUT / f"hoser/rows_{city}.npy")          # HOSER test.csv rows of the same trips
        gall = diag.ragged_take(P["gt_flat"], P["gt_off"], range(int(P["n"])))
        gts = [np.asarray(gall[i], np.int64) for i in pick]
        gen = pd.read_csv(f).set_index("test_row").reindex(hrow)
        hoser = [np.array(json.loads(s), np.int64) if isinstance(s, str) else np.zeros(0, np.int64)
                 for s in gen["gene_trace_road_id"]]
        out = {"n": int(len(pick)), "hoser_missing": int(sum(len(r) == 0 for r in hoser)),
               "hoser_reached_destination": round(float(np.mean([len(h) > 0 and h[-1] == g[-1] for h, g in zip(hoser, gts)])), 4)}
        out["HOSER_official"] = route_metrics(hoser, gts, meta, gps, device)
        lenr = [P["rflat"][P["roff"][j]:P["roff"][j + 1]].astype(np.int64) for j in np.nonzero(P["pair_k"] == 0)[0]]
        out["LEN_shortest"] = route_metrics([lenr[i] for i in pick], gts, meta, gps, device)
        Ptr = e4.load_pool(city, "train")
        pooled = e4b.concat([e4.load_pool(s, "train") for s in e4.ALL if s != city])
        zs = top_routes(P, P["X"] @ e4.fit_logit(pooled, device), meta["n_seg"])
        out["zeroshot_logit_pooled"] = route_metrics([zs[i] for i in pick], gts, meta, gps, device)
        inl = top_routes(P, P["X"] @ e4.fit_logit(Ptr, device), meta["n_seg"])
        out["incity_logit"] = route_metrics([inl[i] for i in pick], gts, meta, gps, device)
        g = e9.fit_gbdt(e9.design(Ptr), Ptr["label"])
        gb = top_routes(P, g.predict_proba(e9.design(P))[:, 1], meta["n_seg"])
        out["incity_gbdt"] = route_metrics([gb[i] for i in pick], gts, meta, gps, device)
        A_tr = e4.load_split(city, "train", 8)
        hist = e4.od_history(A_tr, np.arange(len(A_tr["gt_off"]) - 1), meta["n_seg"])
        keys = P["od_key"]
        odh = [np.frombuffer(hist[int(keys[i])][0][0], dtype=np.int32).astype(np.int64) if int(keys[i]) in hist else zs[i]
               for i in pick]
        out["incity_od_history"] = route_metrics(odh, gts, meta, gps, device)
        orc = top_routes(P, P["f1"], meta["n_seg"])
        out["oracle6"] = route_metrics([orc[i] for i in pick], gts, meta, gps, device)
        res[city] = out
        diag.log(f"[{city}] " + json.dumps(out))
        (diag.OUT / "hoser/HOSER_eval.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
