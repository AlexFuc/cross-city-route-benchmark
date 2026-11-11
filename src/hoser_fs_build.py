"""Build one working directory per (city, n): a copy of the official HOSER code (unchanged; the dataset name
must stay 'Porto' / 'San_Francisco' because train.py/gene.py branch on it) plus data/<Name>/ with the training
subset. road_network_partition depends only on the map and is copied; zone_trans_mat.npy is recomputed from
the SUBSET with the loop of the official get_zone_trans_mat.py (using the full-data matrix would leak)."""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(os.environ["HOSER_DIR"])
FS = Path(os.environ["HOSER_FS"])
IDS = Path(__file__).resolve().parent / "results/hoser/fewshot"
CODE = ["train.py", "train_accum.py", "_driver_gene.py", "dataset.py", "utils.py", "models", "config"]
for city, name in [("porto", "Porto"), ("sanfrancisco", "San_Francisco")]:
    full = pd.read_csv(H / f"data/{name}/train.csv")
    for n in (1000, 10000, 100000):
        d = FS / f"{city}_n{n}"
        (d / "data" / name).mkdir(parents=True, exist_ok=True)
        for c in CODE:
            src, dst = H / c, d / c
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        dd = d / "data" / name
        for f in ["roadmap.geo", "roadmap.rel", "val.csv", "test.csv"]:
            if not (dd / f).exists():
                os.symlink(H / f"data/{name}/{f}", dd / f)
        shutil.copy2(H / f"data/{name}/road_network_partition", dd / "road_network_partition")
        rows = np.load(IDS / f"train_rows_{city}_n{n}.npy")
        sub = full.iloc[rows]
        sub.to_csv(dd / "train.csv", index=False)
        # official get_zone_trans_mat.py loop, on the subset
        road2zone = [int(l.strip()) for l in open(dd / "road_network_partition")]
        zc = max(road2zone) + 1
        z = np.zeros((zc, zc), dtype=np.int64)
        for _, row in sub.iterrows():
            zl = [road2zone[r] for r in eval(row["rid_list"])]
            for a, b in zip(zl[:-1], zl[1:]):
                if a != b:
                    z[a, b] += 1
        np.save(dd / "zone_trans_mat.npy", z)
        print(city, n, len(sub), "zone transitions", int(z.sum()), flush=True)
