"""Generate GTG routes with the OFFICIAL code (unmodified) for the fixed evaluation subsets.

Run inside gtg_official/ with the gtgp2 env. graph_tool must be imported before torch (libgomp clash).
Usage: python gtg_generate.py <src> <trg> <device> <out_csv> <ids_npy> [model_pth]
"""
import graph_tool  # noqa: F401  (must precede torch)
import sys

import numpy as np
import pandas as pd
import torch

from config import cfg
from dataloader import get_network
from generate import generate_traj
from models import GTGModel


def main():
    src, trg, device, out_csv, ids_npy = sys.argv[1:6]
    model_pth = sys.argv[6] if len(sys.argv) > 6 else f"ckpt/exp_101/{src}_to_{trg}/pref.pth"
    cfg["dataset_source"], cfg["dataset_target"], cfg["device"], cfg["exp_tag"] = src, trg, device, 101
    cfg["model_dir"] = f"ckpt/exp_101/{src}_to_{trg}"
    model = GTGModel(cfg).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    print("loaded", model_pth, flush=True)
    network = get_network(trg)
    ids = set(np.load(ids_npy).tolist())
    gt_df = pd.read_csv(f"data/{trg}/traj/test.csv", sep=";")
    gt_df = gt_df[gt_df.traj_id.isin(ids)].reset_index(drop=True)
    print(f"{src}->{trg}: generating {len(gt_df)} routes", flush=True)
    gen = generate_traj(model=model, network=network, gt_df=gt_df)
    gen.to_csv(out_csv, index=False)
    print(f"{src}->{trg}: wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
