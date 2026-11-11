"""Dry run of hoser_eval.py: fake HOSER output = LEN-SP routes; HOSER row must equal LEN_shortest row."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag, e4
d = "/tmp/hoser_dry"; os.makedirs(d, exist_ok=True)
for city in ["porto", "sanfrancisco"]:
    P = e4.load_pool(city, "test")
    pick = np.load(diag.OUT / f"hoser/pick_{city}.npy"); hrow = np.load(diag.OUT / f"hoser/rows_{city}.npy")
    lenr = [P["rflat"][P["roff"][j]:P["roff"][j + 1]].tolist() for j in np.nonzero(P["pair_k"] == 0)[0]]
    pd.DataFrame({"test_row": hrow, "gene_trace_road_id": [json.dumps(lenr[i]) for i in pick]}).to_csv(f"{d}/gen_{city}.csv", index=False)
