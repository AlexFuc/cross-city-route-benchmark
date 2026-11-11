"""Take over an official GTG run at disentangle epoch 595 and finish it with a bit-identical faster build_graph.

Mirrors main.py: disentangle (remaining epochs 596..599 with official train(), fresh Adam since the official
code does not save optimizer state) -> official preference_train(). The ONLY change is replacing
unsupervise.build_graph with a vectorized version verified bit-identical (edges, float64 weights, 1000/1000
shortest paths; see _fbg_check.py). Run inside gtg_official/ with gtgp2.
Usage: python _bj_takeover.py <src> <trg> <device> [--dry]
  --dry: model_dir -> /tmp scratch, no extra epochs, cfg['debug_mode']=True (1 preference step).
"""
import graph_tool  # noqa: F401  (must precede torch)
import logging
import os
import sys
import time

import numpy as np
import torch
from graph_tool.all import Graph

from config import cfg

src, trg, device = sys.argv[1], sys.argv[2], sys.argv[3]
dry = "--dry" in sys.argv
real_dir = f"ckpt/exp_101/{src}_to_{trg}"
cfg["dataset_source"], cfg["dataset_target"], cfg["device"], cfg["exp_tag"] = src, trg, device, 101
cfg["model_dir"] = f"/tmp/takeover_dry_{src}_to_{trg}" if dry else real_dir
os.makedirs(cfg["model_dir"], exist_ok=True)
if dry:
    cfg["debug_mode"] = True

logging.basicConfig(level=logging.INFO, format="[%(levelname)s](%(asctime)s):%(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("takeover")

import unsupervise  # noqa: E402
from dataloader import get_network, get_traj_dataset  # noqa: E402
from disentangle import get_dataloaders, train  # noqa: E402
from models import GTGModel  # noqa: E402


def build_graph_fast(network, costs):
    ei = network.edge_index.detach().cpu().numpy()
    c = costs.detach().cpu().numpy().astype(np.float64)
    g = Graph(directed=True)
    g.add_edge_list(ei.T)
    wt = g.new_edge_property("double")
    wt.a[:] = (c[ei[0]] + c[ei[1]]) / 2
    g.ep.wt = wt
    return g


unsupervise.build_graph = build_graph_fast          # used by preference_train and validate_pref
log.info(f"takeover {src}->{trg} on {device}; model_dir={cfg['model_dir']}; dry={dry}")

model = GTGModel(cfg).to(device)
state = torch.load(f"{real_dir}/disent.pth", map_location=device)
model.load_state_dict(state)
log.info("loaded disent.pth (official save at epoch 595)")

if not dry:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loaders = get_dataloaders()
    for epoch in range(596, cfg["epoch_disent"]):
        t0 = time.time()
        loss = train(model, optimizer, loaders)
        log.info(f"Disentangle train (takeover), epoch {epoch} " + ",".join(f"{k}: {v:.6f}" for k, v in loss.items())
                 + f" [{time.time() - t0:.0f}s]")
    torch.save(model.state_dict(), f"{cfg['model_dir']}/disent_takeover_e{cfg['epoch_disent'] - 1}.pth")

network = get_network(cfg["dataset_source"])
train_traj, valid_traj = get_traj_dataset(cfg["dataset_source"])
log.info(f"preference training: {len(train_traj)} train / {len(valid_traj)} valid trajectories")
t0 = time.time()
unsupervise.preference_train(model, network, train_traj, valid_traj, save_model=True)
log.info(f"TAKEOVER DONE in {time.time() - t0:.0f}s; pref.pth in {cfg['model_dir']}")
