"""Export the fixed 100k target-test subsets (traj_ids) used by E4-E9, so GTG is evaluated on the same trips."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402

out = diag.OUT / "gtg"
out.mkdir(parents=True, exist_ok=True)
for c in ("xian", "chengdu", "beijing"):
    A = diag.stage_a(c, 8)
    rows = e4.subset(c, "test", A, 100000)
    np.save(out / f"ids_{c}.npy", A["traj_id"][rows])
    print(c, len(rows))
