"""Reviewer check (training budget): HOSER trained on the n=1e3 few-shot subsets for 250 instead of 25 epochs (the
update budget of n=1e4 under the official schedule), evaluated exactly like hoser_fs_eval.py on the same 10k test
trips. Output: results/hoser/HOSER_fewshot_budget.json, with the official-schedule results for comparison."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import gtg_eval  # noqa: E402
import hoser_eval  # noqa: E402

assert e4.F1_MODE == "length"
FS = Path(os.environ["HOSER_FS"])


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    ref = json.loads((diag.OUT / "hoser/HOSER_fewshot.json").read_text())
    res = {"max_epoch": 250, "updates": 25 * 4 * 10, "cities": {}}
    for city in ["porto", "sanfrancisco"]:
        f = FS / f"gen_{city}_n1000_e250.csv"
        if not f.exists():
            continue
        meta, gps = e4.road_meta(city, 16), gtg_eval.gps_array(city)
        P = e4.load_pool(city, "test")
        pick, hrow = np.load(diag.OUT / f"hoser/pick_{city}.npy"), np.load(diag.OUT / f"hoser/rows_{city}.npy")
        gall = diag.ragged_take(P["gt_flat"], P["gt_off"], range(int(P["n"])))
        gts = [np.asarray(gall[i], np.int64) for i in pick]
        gen = pd.read_csv(f).set_index("test_row").reindex(hrow)
        h = [np.array(json.loads(s), np.int64) if isinstance(s, str) else np.zeros(0, np.int64) for s in gen["gene_trace_road_id"]]
        out = {"HOSER_e250": hoser_eval.route_metrics(h, gts, meta, gps, device), "missing": int(sum(len(x) == 0 for x in h)),
               "official_n1000": ref[city]["1000"], "official_n10000_HOSER": ref[city]["10000"]["HOSER"]}
        res["cities"][city] = out
        diag.log(f"[HOSER budget {city}] " + json.dumps({"e250": out["HOSER_e250"], "e25": ref[city]["1000"]["HOSER"]}))
    (diag.OUT / "hoser/HOSER_fewshot_budget.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
