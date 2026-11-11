"""Shared helpers: data loading (stage A: test trips + raw-length shortest paths), exact GPU LCS / route F1,
ragged arrays, road metadata and flow statistics."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data_root")))
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "results"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

DIRECTIONS = [
    ("exp_1/chengdu_to_xian", "chengdu", "xian", "CD", "XA"),
    ("exp_2/beijing_to_xian", "beijing", "xian", "BJ", "XA"),
    ("exp_3/beijing_to_chengdu", "beijing", "chengdu", "BJ", "CD"),
    ("exp_4/xian_to_beijing", "xian", "beijing", "XA", "BJ"),
    ("exp_5/xian_to_chengdu", "xian", "chengdu", "XA", "CD"),
    ("exp_6/chengdu_to_beijing", "chengdu", "beijing", "CD", "BJ"),
]
SLOTS = ["Proto0", "Proto1", "Len-SP", "Proto3"]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
MAJOR_TYPES = {"motorway", "trunk", "primary", "secondary"}
# Local-time periods (UTC+8): night, AM peak, midday, PM peak, evening.
PERIOD_EDGES = [0, 6, 10, 16, 20, 24]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Ragged int arrays: (flat, offsets)
# --------------------------------------------------------------------------
def to_ragged(str_series) -> tuple[np.ndarray, np.ndarray]:
    arrs = [np.fromstring(str(s), dtype=np.int32, sep=",") if isinstance(s, str) and s else np.zeros(0, np.int32)
            for s in str_series]
    return from_list(arrs)


def from_list(arrs) -> tuple[np.ndarray, np.ndarray]:
    lens = np.fromiter((len(a) for a in arrs), dtype=np.int64, count=len(arrs))
    off = np.zeros(len(arrs) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    flat = np.concatenate(arrs).astype(np.int32) if len(arrs) else np.zeros(0, np.int32)
    return flat, off


def ragged_take(flat, off, idx):
    return [flat[off[i]:off[i + 1]] for i in idx]


# --------------------------------------------------------------------------
# Stage A: GT + Len-SP per target city
# --------------------------------------------------------------------------
_G = {}


def _lensp_worker(args):
    origins, od_by_origin = args
    from scipy.sparse.csgraph import dijkstra
    csr = _G["csr"]
    out = {}
    dist, pred = dijkstra(csr, directed=True, indices=origins, return_predecessors=True)
    for r, o in enumerate(origins):
        for d in od_by_origin[r]:
            if not np.isfinite(dist[r, d]):
                out[(o, d)] = [o, d]
                continue
            path = [d]
            p = d
            steps = 0
            while p != o and steps < csr.shape[0] + 5:
                p = int(pred[r, p])
                if p < 0:
                    break
                path.append(p)
                steps += 1
            out[(o, d)] = path[::-1]
    return out


def stage_a(city: str, workers: int) -> dict:
    path = CACHE / f"A_{city}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    sys.path.insert(0, str(DATA_ROOT))
    os.chdir(DATA_ROOT)
    import routing  # noqa: E402

    log(f"[A] {city}: reading GT")
    gt = pd.read_csv(DATA_ROOT / f"data/{city}/traj/test.csv", sep=";").sort_values("traj_id").reset_index(drop=True)
    gflat, goff = to_ragged(gt["rid_list"].tolist())
    n = len(gt)
    lens = np.diff(goff)
    origin = np.where(lens > 0, gflat[np.minimum(goff[:-1], len(gflat) - 1)], -1)
    dest = np.where(lens > 0, gflat[np.maximum(goff[1:] - 1, 0)], -1)

    _G["csr"] = routing.csr_from_node_cost(city, routing.raw_length(city))   # raw lengths
    ods = pd.DataFrame({"o": origin, "d": dest})
    ods = ods[(ods.o >= 0)].drop_duplicates()
    grp = ods.groupby("o")["d"].apply(lambda s: s.to_numpy()).reset_index()
    origins = grp["o"].to_numpy()
    dsts = grp["d"].tolist()
    chunk = max(1, len(origins) // (workers * 4))
    tasks = [(origins[i:i + chunk].astype(np.int64), dsts[i:i + chunk]) for i in range(0, len(origins), chunk)]
    log(f"[A] {city}: {n} trajs, {len(ods)} unique ODs, {len(origins)} origins, {len(tasks)} tasks")
    paths = {}
    with Pool(workers) as pool:
        for res in pool.imap_unordered(_lensp_worker, tasks):
            paths.update(res)
    lsp = [np.asarray(paths[(int(o), int(d))], dtype=np.int32) if o >= 0 else np.zeros(0, np.int32)
           for o, d in zip(origin, dest)]
    lflat, loff = from_list(lsp)
    out = dict(traj_id=gt["traj_id"].to_numpy(np.int64), start_time=gt["start_time"].to_numpy(np.int64),
               gt_flat=gflat, gt_off=goff, len_flat=lflat, len_off=loff)
    np.savez(path, **out)
    log(f"[A] {city}: done")
    return out


# --------------------------------------------------------------------------
# Stage B: exact LCS-F1 on GPU
# --------------------------------------------------------------------------
def lcs_batch_gpu(a_list, b_list, device, max_cells=3e8):
    """Exact LCS lengths for pairs (a_i, b_i).

    Row recurrence: new[j] = cummax_j(max(prev[j], prev[j-1] + [a_i == b_j])),
    which equals the classic LCS DP because a diagonal match always dominates.
    """
    import torch
    n = len(a_list)
    out = np.zeros(n, dtype=np.int64)
    la = np.fromiter((len(x) for x in a_list), np.int64, n)
    lb = np.fromiter((len(x) for x in b_list), np.int64, n)
    order = np.argsort(np.maximum(la, lb), kind="stable")
    i = 0
    while i < n:
        # grow chunk while B * La * Lb stays bounded
        j = i
        La = Lb = 1
        while j < n:
            k = order[j]
            nla, nlb = max(La, int(la[k])), max(Lb, int(lb[k]))
            if j > i and ((j - i + 1) * nla * nlb > max_cells or (j - i + 1) > 200_000):
                break
            La, Lb = nla, nlb
            j += 1
        idx = order[i:j]
        B = len(idx)
        A = np.full((B, max(La, 1)), -1, dtype=np.int32)
        Bm = np.full((B, max(Lb, 1)), -2, dtype=np.int32)
        for r, k in enumerate(idx):
            A[r, :la[k]] = a_list[k]
            Bm[r, :lb[k]] = b_list[k]
        At = torch.from_numpy(A).to(device)
        Bt = torch.from_numpy(Bm).to(device)
        prev = torch.zeros((B, Lb + 1), dtype=torch.int32, device=device)
        for r in range(La):
            m = (At[:, r:r + 1] == Bt).to(torch.int32)
            t = torch.maximum(prev[:, 1:], prev[:, :-1] + m)
            t = torch.cat([torch.zeros((B, 1), dtype=torch.int32, device=device), t], dim=1)
            prev = torch.cummax(t, dim=1).values
        res = prev.gather(1, torch.from_numpy(lb[idx]).to(device).view(-1, 1)).view(-1)
        out[idx] = res.cpu().numpy()
        i = j
    return out


def wlcs_batch_gpu(a_list, b_list, w_of, device, max_cells=3e8):
    """Weighted LCS: a match on element x contributes w_of[x] (segment length). Exact DP via cummax rows."""
    import torch
    n = len(a_list)
    out = np.zeros(n)
    la = np.fromiter((len(x) for x in a_list), np.int64, n)
    lb = np.fromiter((len(x) for x in b_list), np.int64, n)
    order = np.argsort(np.maximum(la, lb), kind="stable")
    w_t = torch.tensor(w_of, dtype=torch.float64, device=device)
    i = 0
    while i < n:
        j, La, Lb = i, 1, 1
        while j < n:
            k = order[j]
            nla, nlb = max(La, int(la[k])), max(Lb, int(lb[k]))
            if j > i and ((j - i + 1) * nla * nlb > max_cells or (j - i + 1) > 200_000):
                break
            La, Lb = nla, nlb
            j += 1
        idx = order[i:j]
        B = len(idx)
        Am = np.full((B, La), -1, np.int64)
        Bm = np.full((B, Lb), -2, np.int64)
        for r, k in enumerate(idx):
            Am[r, :la[k]] = a_list[k]
            Bm[r, :lb[k]] = b_list[k]
        At, Bt = torch.from_numpy(Am).to(device), torch.from_numpy(Bm).to(device)
        Wb = torch.where(Bt >= 0, w_t[Bt.clamp(min=0)], torch.zeros((), dtype=torch.float64, device=device))
        prev = torch.zeros((B, Lb + 1), dtype=torch.float64, device=device)
        for r in range(La):
            m = (At[:, r:r + 1] == Bt).to(torch.float64) * Wb
            t = torch.maximum(prev[:, 1:], prev[:, :-1] + m)
            t = torch.cat([torch.zeros((B, 1), dtype=torch.float64, device=device), t], 1)
            prev = torch.cummax(t, 1).values
        out[idx] = prev.gather(1, torch.from_numpy(lb[idx]).to(device).view(-1, 1)).view(-1).cpu().numpy()
        i = j
    return out


def route_f1(preds, gts, device, length=None):
    """LCS-F1 between routes; count-based if length is None, else length-weighted (matched length / route length)."""
    if length is None:
        l = lcs_batch_gpu(preds, gts, device)
        return f1_from_lcs(l, np.array([len(p) for p in preds]), np.array([len(g) for g in gts]))
    wl = wlcs_batch_gpu(preds, gts, length, device)
    pl = np.array([length[np.asarray(p, np.int64)].sum() for p in preds])
    gl = np.array([length[np.asarray(g, np.int64)].sum() for g in gts])
    pr, rc = wl / np.maximum(pl, 1e-9), wl / np.maximum(gl, 1e-9)
    return np.where((pl > 0) & (gl > 0), 2 * pr * rc / np.maximum(pr + rc, 1e-9), 0.0)


def f1_from_lcs(l, n_gen, n_gt):
    prec = np.where(n_gen > 0, l / np.maximum(n_gen, 1), 0.0)
    rec = np.where(n_gt > 0, l / np.maximum(n_gt, 1), 0.0)
    f1 = 2 * prec * rec / (prec + rec + 1e-5)
    return np.where((n_gen > 0) & (n_gt > 0), f1, 0.0)


# --------------------------------------------------------------------------
# Stage C: diagnostics
# --------------------------------------------------------------------------
def city_meta(city: str, n_regions: int):
    from sklearn.cluster import KMeans
    road = pd.read_csv(DATA_ROOT / f"data/{city}/map/road.csv", usecols=["link_id", "length", "link_type_name"])
    road = road.sort_values("link_id")
    with open(DATA_ROOT / f"data/{city}/map/rid_to_gps.json") as f:
        gps = json.load(f)
    xy = np.zeros((len(road), 2))
    for k, (lon, lat) in gps.items():
        if int(k) < len(xy):
            xy[int(k)] = (float(lon), float(lat))
    xy_m = np.c_[xy[:, 0] * np.cos(np.deg2rad(xy[:, 1].mean())), xy[:, 1]]
    region = KMeans(n_clusters=n_regions, n_init=10, random_state=0).fit_predict(xy_m)
    major = road["link_type_name"].isin(MAJOR_TYPES).to_numpy()
    return dict(length=road["length"].to_numpy(float), region=region, major=major, n_seg=len(road))


def flatten_rows(flat, off, rows):
    """(row index within `rows`, segment id) for every segment of the selected routes."""
    lens = np.diff(off)[rows]
    starts = off[:-1][rows]
    rep = np.repeat(np.arange(len(rows)), lens)
    within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    return rep, flat[np.repeat(starts, lens) + within]


def unique_hits(rep, segs, sensor_pos):
    col = sensor_pos[segs]
    keep = col >= 0
    key = np.unique(rep[keep].astype(np.int64) * 10_000_000 + col[keep])
    return key // 10_000_000, key % 10_000_000


def geh_stats(pred, obs, mask=None):
    if mask is not None:
        pred, obs = pred[mask], obs[mask]
    geh = np.sqrt(2 * (pred - obs) ** 2 / np.maximum(pred + obs, 1e-9))
    return dict(geh_lt5=float((geh < 5).mean()), nmae=float(np.abs(pred - obs).sum() / max(obs.sum(), 1e-9)),
                corr=float(np.corrcoef(pred, obs)[0, 1]) if len(obs) > 2 else float("nan"))


# --------------------------------------------------------------------------


