"""Exact-shortest-path share (raw-length LEN-SP) on the FULL test sets of all cities incl. the HOSER-pipeline
Beijing control; uses the stage-A cache (raw lengths since v2)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402

out = {}
for c in ["xian", "chengdu", "beijing", "porto", "sanfrancisco", "bjhoser"]:
    A = diag.stage_a(c, 8)
    n = len(A["gt_off"]) - 1
    ok = np.diff(A["gt_off"]) > 1
    same = [np.array_equal(A["gt_flat"][A["gt_off"][i]:A["gt_off"][i + 1]], A["len_flat"][A["len_off"][i]:A["len_off"][i + 1]])
            for i in np.nonzero(ok)[0]]
    out[c] = {"n": int(ok.sum()), "exact_shortest_share": round(float(np.mean(same)), 4)}
    print(c, out[c], flush=True)
(diag.OUT / "EXACT_SP_control.json").write_text(json.dumps(out, indent=2))
