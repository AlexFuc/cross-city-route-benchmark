"""How much does the TARGET train_label.csv shape GTG's target sampling? Official functions only, CPU. Run in gtg_official/ (gtgp2 env): python ../gtg_label_check.py xian chengdu beijing
Counts the subgraph samples gen_cluster_dataset keeps vs the 24 x num_clusters it would draw without labels."""
import graph_tool  # noqa: F401
import os, random, sys
sys.path.insert(0, os.getcwd())
import json
import numpy as np, pandas as pd, torch
from config import cfg
from dataloader import get_network, gen_cluster_dataset
random.seed(0); np.random.seed(0); torch.manual_seed(0)
res = {}
for city in sys.argv[1:]:
    net = get_network(city)
    lab = pd.read_csv(f"data/{city}/traj/train_label.csv")
    ncl = int(net.num_nodes / cfg["local_size"])
    ds = gen_cluster_dataset(net, lab)
    per_slot = lab.groupby("time_index").rid.nunique()
    print(f"LABEL {city}: segments {net.num_nodes}, label rows {len(lab)}, slots {lab.time_index.nunique()}/24, "
          f"labeled segs/slot median {int(per_slot.median())} ({per_slot.median()/net.num_nodes:.0%}), "
          f"samples kept {len(ds.data_list)}/{24*ncl} ({len(ds.data_list)/(24*ncl):.1%})", flush=True)
    res[city] = {"segments": int(net.num_nodes), "label_rows": int(len(lab)), "slots": int(lab.time_index.nunique()),
                 "samples_kept": len(ds.data_list), "samples_label_free": 24 * ncl}
json.dump(res, open("../results/gtg/GTG_target_label_check.json", "w"), indent=2)
