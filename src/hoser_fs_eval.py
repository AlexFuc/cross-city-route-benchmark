"""Few-shot comparison on the same 10k test trips (Porto, SF): HOSER trained on n local trips vs. GBDT, in-city
logit and OD history fitted on exactly the same n trips (E5 seed-0 subsets). One route per trip; metrics as in
hoser_eval.route_metrics. Run with CC_F1=length."""
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
import e5  # noqa: E402
import gtg_eval  # noqa: E402
import hoser_eval  # noqa: E402

assert e4.F1_MODE == "length"
FS = Path(os.environ["HOSER_FS"])
IDS = diag.OUT / "hoser" / "fewshot"


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    res = {}
    for city in ["porto", "sanfrancisco"]:
        meta, gps = e4.road_meta(city, 16), gtg_eval.gps_array(city)
        P, Ptr = e4.load_pool(city, "test"), e4.load_pool(city, "train")
        pick, hrow = np.load(diag.OUT / f"hoser/pick_{city}.npy"), np.load(diag.OUT / f"hoser/rows_{city}.npy")
        gall = diag.ragged_take(P["gt_flat"], P["gt_off"], range(int(P["n"])))
        gts = [np.asarray(gall[i], np.int64) for i in pick]
        pooled = e4b.concat([e4.load_pool(s, "train") for s in e4.ALL if s != city])
        zs = hoser_eval.top_routes(P, P["X"] @ e4.fit_logit(pooled, device), meta["n_seg"])
        A_tr = e4.load_split(city, "train", 8)
        tr_rows = e4.subset(city, "train", A_tr, 100000)
        Zt = e4b.design_noregion(P)
        res[city] = {}
        for n in (1000, 10000, 100000):
            idx = np.load(IDS / f"pool_idx_{city}_n{n}.npy")
            S = e5.take(Ptr, idx)
            out = {"n": int(len(idx))}
            f = FS / f"gen_{city}_n{n}.csv"
            if f.exists():
                gen = pd.read_csv(f).set_index("test_row").reindex(hrow)
                h = [np.array(json.loads(s), np.int64) if isinstance(s, str) else np.zeros(0, np.int64) for s in gen["gene_trace_road_id"]]
                out["HOSER"] = hoser_eval.route_metrics(h, gts, meta, gps, device)
            g = e4b.fit(e4b.design_noregion(S), S["label"])
            gb = hoser_eval.top_routes(P, g.predict_proba(Zt)[:, 1], meta["n_seg"])
            out["gbdt"] = hoser_eval.route_metrics([gb[i] for i in pick], gts, meta, gps, device)
            lg = hoser_eval.top_routes(P, P["X"] @ e4.fit_logit(S, device), meta["n_seg"])
            out["logit"] = hoser_eval.route_metrics([lg[i] for i in pick], gts, meta, gps, device)
            hist = e4.od_history(A_tr, tr_rows[np.sort(idx)], meta["n_seg"])
            keys = P["od_key"]
            odh = [np.frombuffer(hist[int(keys[i])][0][0], dtype=np.int32).astype(np.int64) if int(keys[i]) in hist else zs[i]
                   for i in pick]
            out["od_history"] = hoser_eval.route_metrics(odh, gts, meta, gps, device)
            res[city][str(n)] = out
            diag.log(f"[{city} n={n}] " + json.dumps(out))
            (diag.OUT / "hoser/HOSER_fewshot.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
