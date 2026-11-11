"""HOSER evaluation sample: 10k trips drawn (seed 0) from the fixed target test subset used by every other
method; written as row indices of HOSER's own test.csv (our traj_id = 10,000,000 + row, see convert_hoser.py)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
out = diag.OUT / "hoser"
out.mkdir(parents=True, exist_ok=True)
for city, name in [("porto", "Porto"), ("sanfrancisco", "San_Francisco")]:
    A = diag.stage_a(city, 8)
    rows = e4.subset(city, "test", A, 100000)
    pick = np.sort(np.random.default_rng(0).choice(len(rows), min(N, len(rows)), replace=False))
    tid = A["traj_id"][rows[pick]].astype(np.int64)
    hrow = tid - 10_000_000
    # check: OD of our trip equals OD of HOSER's test row
    ht = pd.read_csv(Path(os.environ["HOSER_DATA"]) / f"{name}/test.csv", usecols=["rid_list"])
    for k in range(0, len(pick), 997):
        g = A["gt_flat"][A["gt_off"][rows[pick[k]]]:A["gt_off"][rows[pick[k]] + 1]]
        h = [int(x) for x in ht.rid_list.iloc[hrow[k]].split(",")]
        assert g[0] == h[0] and g[-1] == h[-1], (city, k)
    np.save(out / f"rows_{city}.npy", hrow)
    np.save(out / f"pick_{city}.npy", pick)
    print(city, len(hrow), "ok")
