"""Few-shot HOSER: the exact E5 seed-0 training subsets (n in 1e3, 1e4, 1e5) as row indices of HOSER's train.csv
(our train traj_id = row index, see convert_hoser.py). OD of every 500th trip is checked against train.csv."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402

out = diag.OUT / "hoser" / "fewshot"
out.mkdir(parents=True, exist_ok=True)
for city, name in [("porto", "Porto"), ("sanfrancisco", "San_Francisco")]:
    A_tr = e4.load_split(city, "train", 8)
    tr_rows = e4.subset(city, "train", A_tr, 100000)
    Ptr = e4.load_pool(city, "train")
    assert int(Ptr["n"]) == len(tr_rows)
    ht = pd.read_csv(Path(os.environ["HOSER_DATA"]) / f"{name}/train.csv", usecols=["rid_list"])
    for n in (1000, 10000, 100000):
        idx = np.random.default_rng(0).choice(int(Ptr["n"]), min(n, int(Ptr["n"])), replace=False)   # = E5 seed 0
        rows = tr_rows[np.sort(idx)]
        hrow = A_tr["traj_id"][rows].astype(np.int64)
        for k in range(0, len(rows), 500):
            g = A_tr["gt_flat"][A_tr["gt_off"][rows[k]]:A_tr["gt_off"][rows[k] + 1]]
            h = [int(x) for x in ht.rid_list.iloc[hrow[k]].split(",")]
            assert g[0] == h[0] and g[-1] == h[-1], (city, n, k)
        np.save(out / f"train_rows_{city}_n{n}.npy", hrow)
        np.save(out / f"pool_idx_{city}_n{n}.npy", idx)
        print(city, n, len(hrow), "ok", flush=True)
