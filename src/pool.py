"""Candidate generation helpers: batched shortest paths and penalty-based alternative routes (raw lengths)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag  # noqa: E402

DATA_ROOT = diag.DATA_ROOT
CACHE = diag.CACHE
OUT = diag.OUT
log = diag.log
_G = {}


# ---------------------------------------------------------------- generators
def _sp_worker(task):
    key, origins, dests = task
    from scipy.sparse.csgraph import dijkstra
    csr = _G["csr"][key]
    dist, pred = dijkstra(csr, directed=True, indices=origins, return_predecessors=True)
    out = []
    for r, o in enumerate(origins):
        for d, row_ids in dests[r]:
            if not np.isfinite(dist[r, d]):
                path = [o, d]
            else:
                path = [d]
                while path[-1] != o:
                    p = int(pred[r, path[-1]])
                    if p < 0:
                        break
                    path.append(p)
                path = path[::-1]
            out.append((row_ids, path))
    return key, out


def batched_shortest_paths(csr_by_key, row_key, row_o, row_d, workers):
    """row_key[i] selects the graph; returns list of paths aligned with rows."""
    _G["csr"] = csr_by_key
    groups = {}
    for i, (k, o, d) in enumerate(zip(row_key, row_o, row_d)):
        groups.setdefault((k, o), {}).setdefault(d, []).append(i)
    by_key = {}
    for (k, o), dd in groups.items():
        by_key.setdefault(k, []).append((o, list(dd.items())))
    tasks = []
    for k, lst in by_key.items():
        for s in range(0, len(lst), 64):
            chunk = lst[s:s + 64]
            tasks.append((k, np.array([c[0] for c in chunk], np.int64), [c[1] for c in chunk]))
    paths = [None] * len(row_o)
    with Pool(workers) as pool:
        for _, res in pool.imap_unordered(_sp_worker, tasks, chunksize=4):
            for row_ids, path in res:
                for i in row_ids:
                    paths[i] = path
    return paths


def _pen_worker(task):
    ods, factor, n_alt = task
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    src, dst, base = _G["src"], _G["dst"], _G["node_cost"]
    n = len(base)
    out = []
    for o, d in ods:
        cost = base.copy()
        paths = []
        for _ in range(n_alt):
            w = (cost[src] + cost[dst]) / 2.0
            csr = csr_matrix((w, (src, dst)), shape=(n, n))
            dist, pred = dijkstra(csr, directed=True, indices=o, return_predecessors=True)
            if not np.isfinite(dist[d]):
                paths.append([o, d])
                continue
            path = [d]
            while path[-1] != o:
                p = int(pred[path[-1]])
                if p < 0:
                    break
                path.append(p)
            path = path[::-1]
            paths.append(path)
            cost[np.asarray(path[1:-1], dtype=np.int64)] *= factor
        out.append(((o, d), paths))
    return out


def penalty_alternatives(city, ods, workers, factor=2.0, n_alt=4):
    import routing
    _, src, dst, length = routing.city_graph(city)
    _G.update(src=src.astype(np.int64), dst=dst.astype(np.int64), node_cost=length.astype(np.float64))   # raw lengths
    uniq = sorted(set(ods))
    chunk = max(1, len(uniq) // (workers * 8))
    tasks = [(uniq[i:i + chunk], factor, n_alt) for i in range(0, len(uniq), chunk)]
    res = {}
    with Pool(workers) as pool:
        for part in pool.imap_unordered(_pen_worker, tasks):
            res.update(part)
    return [[np.asarray(res[od][k], np.int32) for od in ods] for k in range(n_alt)]


def origin_dest(A, rows):
    off, flat = A["gt_off"], A["gt_flat"]
    return [int(flat[off[i]]) for i in rows], [int(flat[off[i + 1] - 1]) for i in rows]


# ---------------------------------------------------------------- evaluation


