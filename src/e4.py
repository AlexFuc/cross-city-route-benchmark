"""Week-1 study (clean protocol).

Candidate pool MAP6 is map-only and built identically in every city/split:
  LEN (length SP), FFT (free-flow-time SP by road class), CLS (class-weighted length SP),
  PEN2..PEN4 (length-penalty alternatives).
Route features are map-only: detour length, free-flow time, major-road share, #segments, path size.

Evaluated on the fixed 100k target *test* subset (same rows as diag/pool/logit):
  M1  zero-shot: beta fitted on SOURCE-city train subset only
  M2  in-city logit: beta fitted on TARGET train subset
  M3  in-city GBDT with context features, fitted on TARGET train subset
  M4  in-city OD-history (mode route from TARGET train, full or n-limited), fallback M1
  LOO predictability ceiling on the full target test set (mode of the other trips of the same OD)
Nothing is ever fitted on target test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag  # noqa: E402
import pool  # noqa: E402

DATA_ROOT, CACHE, OUT, log = diag.DATA_ROOT, diag.CACHE, diag.OUT, diag.log
SHORT = {"xian": "XA", "chengdu": "CD", "beijing": "BJ", "porto": "PT", "sanfrancisco": "SF", "bjhoser": "BJH"}
SEED = {"xian": 1, "chengdu": 2, "beijing": 3, "porto": 4, "sanfrancisco": 5, "bjhoser": 6}
ALL = os.environ.get("CC_CITIES", "xian,chengdu,beijing").split(",")
TAG = "" if ALL == ["xian", "chengdu", "beijing"] else "_" + "-".join(SHORT[c] for c in ALL)
F1_MODE = os.environ.get("CC_F1", "count")          # "count" (legacy) or "length" (paper primary metric)
if F1_MODE == "length":
    TAG += "_len"


def seg_lengths(city):
    return pd.read_csv(DATA_ROOT / f"data/{city}/map/road.csv", usecols=["link_id", "length"]).sort_values("link_id")["length"].to_numpy(float)
SPEED_KMH = {"motorway": 80, "trunk": 60, "primary": 45, "secondary": 35, "tertiary": 30, "unclassified": 25}
FACTOR = {"motorway": 0.6, "trunk": 0.7, "primary": 0.8, "secondary": 0.9, "tertiary": 1.0, "unclassified": 1.1}
FEATS = ["log_detour_len", "log_ratio_fft", "major_share", "log_ratio_nseg", "log_path_size"]
SLOTS = ["LEN", "FFT", "CLS", "PEN2", "PEN3", "PEN4"]


# ------------------------------------------------------------------ data
def load_split(city, split, workers):
    if split == "test":
        return diag.stage_a(city, workers)
    path = CACHE / f"T_{city}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    log(f"[data] parsing {city} train")
    df = pd.read_csv(DATA_ROOT / f"data/{city}/traj/train.csv", sep=";", usecols=["traj_id", "start_time", "rid_list"])
    df = df.sort_values("traj_id").reset_index(drop=True)
    gflat, goff = diag.to_ragged(df["rid_list"].tolist())
    out = dict(traj_id=df["traj_id"].to_numpy(np.int64), start_time=df["start_time"].to_numpy(np.int64), gt_flat=gflat, gt_off=goff)
    np.savez(path, **out)
    return out


def subset(city, split, A, n):
    valid = np.where(np.diff(A["gt_off"]) > 1)[0]
    rng = np.random.default_rng(SEED[city] if split == "test" else 100 + SEED[city])
    return np.sort(rng.choice(valid, min(n, len(valid)), replace=False))


def road_meta(city, regions):
    meta = diag.city_meta(city, regions)
    typ = pd.read_csv(DATA_ROOT / f"data/{city}/map/road.csv", usecols=["link_id", "link_type_name"]).sort_values("link_id")
    typ = typ["link_type_name"].fillna("other").to_numpy()
    meta["speed"] = np.array([SPEED_KMH.get(t, 20) for t in typ], float) / 3.6
    meta["factor"] = np.array([FACTOR.get(t, 1.2) for t in typ], float)
    return meta


def load_pool(city, split):
    """Load the largest cached MAP6 pool for (city, split) (subset size may be < 100k for small cities)."""
    files = sorted(CACHE.glob(f"MAP6_{city}_{split}_*.npz"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not files:
        raise FileNotFoundError(f"no MAP6 pool for {city}/{split}")
    return apply_f1_mode(dict(np.load(files[-1])), city, files[-1].stem)


def apply_f1_mode(P, city, stem):
    """In length mode, replace count-based candidate F1, BoK and oracle labels by length-weighted ones."""
    if F1_MODE == "length":
        cf = CACHE / f"F1W_{stem}.npy"
        if cf.exists():
            f1w = np.load(cf)
        else:
            L = seg_lengths(city)
            routes = diag.ragged_take(P["rflat"], P["roff"], range(len(P["pair_i"])))
            gts = diag.ragged_take(P["gt_flat"], P["gt_off"], range(int(P["n"])))
            dev = os.environ.get("CC_DEVICE", "cuda:0")
            f1w = diag.route_f1(routes, [gts[i] for i in P["pair_i"]], dev, length=L)
            np.save(cf, f1w)
        n = int(P["n"])
        best = np.full(n, -1.0)
        np.maximum.at(best, P["pair_i"], f1w)
        lab_idx = np.nonzero(f1w >= best[P["pair_i"]] - 1e-12)[0]
        _, first = np.unique(P["pair_i"][lab_idx], return_index=True)
        label = np.zeros(len(f1w), bool)
        label[lab_idx[first]] = True
        P["f1"], P["best"], P["label"] = f1w, best, label
    return P


def local_hour(st):
    return ((st + 8 * 3600) % 86400) // 3600


# ------------------------------------------------------------------ pool
def build_pool(city, split, A, rows, meta, args):
    path = CACHE / f"MAP6_{city}_{split}_{len(rows)}.npz"
    if path.exists():
        z = np.load(path)
        return apply_f1_mode({k: z[k] for k in z.files}, city, path.stem)
    import routing
    fields = {"LEN": routing.raw_length(city), "FFT": meta["length"] / meta["speed"],
              "CLS": meta["length"] * meta["factor"]}
    csr = {k: routing.csr_from_node_cost(city, np.asarray(v, float)) for k, v in fields.items()}
    o, d = pool.origin_dest(A, rows)
    log(f"[pool] {city}/{split}: {len(rows)} ODs, fields")
    routes = {k: [np.asarray(p, np.int32) for p in pool.batched_shortest_paths(csr, [k] * len(rows), o, d, args.workers)]
              for k in fields}
    pen_path = CACHE / f"PEN4_{city}_{len(rows)}.npz"
    if split == "test" and pen_path.exists():
        z = np.load(pen_path)
        pen = [diag.ragged_take(z[f"f{k}"], z[f"o{k}"], range(len(rows))) for k in range(4)]
    else:
        log(f"[pool] {city}/{split}: penalty alternatives")
        pen = pool.penalty_alternatives(city, list(zip(o, d)), args.workers)
    cands = [routes["LEN"], routes["FFT"], routes["CLS"], pen[1], pen[2], pen[3]]
    gt = diag.ragged_take(A["gt_flat"], A["gt_off"], rows)
    P = pair_table(cands, gt, meta, args.device)
    P["hour"] = local_hour(A["start_time"][rows])
    P["region_o"] = meta["region"][np.array(o)]
    P["region_d"] = meta["region"][np.array(d)]
    P["od_key"] = np.array(o, np.int64) * meta["n_seg"] + np.array(d, np.int64)
    gflat, goff = diag.from_list(gt)
    P["gt_flat"], P["gt_off"] = gflat, goff
    np.savez(path, **P)
    log(f"[pool] {city}/{split}: {len(P['pair_i'])} unique routes ({len(P['pair_i']) / len(rows):.2f}/traj), "
        f"BoK={P['best'].mean():.3f}")
    return apply_f1_mode(P, city, path.stem)


def pair_table(cands, gt, meta, device):
    n, K = len(gt), len(cands)
    uniq = np.ones((n, K), bool)
    for i in range(n):
        seen = set()
        for k in range(K):
            b = cands[k][i].tobytes()
            if b in seen or len(cands[k][i]) == 0:
                uniq[i, k] = False
            seen.add(b)
    pi_, pk_ = np.nonzero(uniq)
    routes = [cands[k][i] for i, k in zip(pi_, pk_)]
    rflat, roff = diag.from_list(routes)
    npair = len(routes)
    rep = np.repeat(np.arange(npair), np.diff(roff))
    seg = rflat.astype(np.int64)
    ln = meta["length"][seg]
    L = np.bincount(rep, weights=ln, minlength=npair)
    fft = np.bincount(rep, weights=ln / meta["speed"][seg], minlength=npair)
    major = np.bincount(rep, weights=ln * meta["major"][seg], minlength=npair)
    nseg = np.diff(roff).astype(float)
    _, inv, cnt = np.unique(pi_[rep].astype(np.int64) * meta["n_seg"] + seg, return_inverse=True, return_counts=True)
    ps = np.bincount(rep, weights=ln / np.maximum(L[rep], 1e-9) / cnt[inv], minlength=npair)

    def log_ratio(v):
        mn = np.full(n, np.inf)
        np.minimum.at(mn, pi_, v)
        return np.log(np.maximum(v, 1e-9) / np.maximum(mn[pi_], 1e-9))

    X = np.stack([log_ratio(L), log_ratio(fft), major / np.maximum(L, 1e-9), log_ratio(nseg), np.log(np.maximum(ps, 1e-9))], 1)
    f1 = diag.f1_from_lcs(diag.lcs_batch_gpu(routes, [gt[i] for i in pi_], device), nseg, np.array([len(gt[i]) for i in pi_]))
    best = np.full(n, -1.0)
    np.maximum.at(best, pi_, f1)
    lab_idx = np.nonzero(f1 >= best[pi_] - 1e-12)[0]
    _, first = np.unique(pi_[lab_idx], return_index=True)
    label = np.zeros(npair, bool)
    label[lab_idx[first]] = True
    mn_len = np.full(n, np.inf)
    np.minimum.at(mn_len, pi_, L)
    return dict(X=X, pair_i=pi_, pair_k=pk_, rflat=rflat, roff=roff, f1=f1, label=label, best=best, n=np.int64(n),
                min_len=mn_len)


# ------------------------------------------------------------------ models
def softmax_by_traj(logit, pair_i, n):
    m = np.full(n, -np.inf)
    np.maximum.at(m, pair_i, logit)
    e = np.exp(logit - m[pair_i])
    s = np.bincount(pair_i, weights=e, minlength=n)
    return e / s[pair_i]


def fit_logit(P, device, iters=800, l2=1e-4):
    import torch
    X = P["X"]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    pi = torch.tensor(P["pair_i"], device=device)
    lab = torch.tensor(P["label"], device=device)
    n = int(P["n"])
    beta = torch.zeros(X.shape[1], device=device, requires_grad=True)
    opt = torch.optim.Adam([beta], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        lg = Xt @ beta
        m = torch.full((n,), -1e30, device=device).scatter_reduce(0, pi, lg, reduce="amax")
        e = torch.exp(lg - m[pi])
        s = torch.zeros(n, device=device).index_add(0, pi, e)
        loss = -(torch.log(e[lab] / s[pi][lab] + 1e-12)).mean() + l2 * (beta ** 2).sum()
        loss.backward()
        opt.step()
    return beta.detach().cpu().numpy() / sd   # raw units


def gbdt_design(P, n_regions):
    X = P["X"]
    pi = P["pair_i"]
    h = P["hour"][pi]
    ctx = np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.log(P["min_len"][pi] + 1.0),
                P["region_o"][pi], P["region_d"][pi]]
    return np.c_[X, ctx]


def fit_gbdt(P, n_regions):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Z = gbdt_design(P, n_regions)
    cat = np.zeros(Z.shape[1], bool)
    cat[-2:] = True
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                         categorical_features=cat, random_state=0)
    clf.fit(Z, P["label"].astype(int))
    return clf


# ------------------------------------------------------------------ evaluation
def evaluate_probs(P, p, meta, gt_flow, mask):
    n = int(P["n"])
    pi = P["pair_i"]
    order = np.lexsort((-p, pi))
    first = np.r_[True, pi[order][1:] != pi[order][:-1]]
    top = order[first]
    rep, seg = diag.flatten_rows(P["rflat"], P["roff"], np.arange(len(pi)))
    fl = np.bincount(seg, weights=p[rep], minlength=meta["n_seg"])
    return {"top1_f1": float(P["f1"][top].mean()), "exp_f1": float(np.bincount(pi, weights=p * P["f1"], minlength=n).mean()),
            "flow_nmae": diag.geh_stats(fl, gt_flow, mask)["nmae"], "top_idx": top}


def route_strings(A, rows=None):
    off, flat = A["gt_off"], A["gt_flat"]
    idx = range(len(off) - 1) if rows is None else rows
    return [flat[off[i]:off[i + 1]].tobytes() for i in idx]


def od_history(A_train, train_rows, n_seg):
    """od_key -> list of (route_bytes, count) sorted by count desc."""
    off, flat = A_train["gt_off"], A_train["gt_flat"]
    hist = defaultdict(lambda: defaultdict(int))
    for i in train_rows:
        r = flat[off[i]:off[i + 1]]
        hist[int(r[0]) * n_seg + int(r[-1])][r.tobytes()] += 1
    return {k: sorted(v.items(), key=lambda x: -x[1]) for k, v in hist.items()}


def eval_od_history(P, hist, gt_flow, mask, meta, fallback_p, device, topn=5):
    n = int(P["n"])
    keys = P["od_key"]
    gt = diag.ragged_take(P["gt_flat"], P["gt_off"], range(n))
    seen = np.array([int(k) in hist for k in keys])
    pred = [np.frombuffer(hist[int(k)][0][0], dtype=np.int32) if s else np.zeros(0, np.int32) for k, s in zip(keys, seen)]
    idx = np.nonzero(seen)[0]
    f1_seen = np.zeros(n)
    if len(idx):
        f1_seen[idx] = diag.route_f1([pred[i] for i in idx], [gt[i] for i in idx], device,
                                     length=meta["length"] if F1_MODE == "length" else None)
    fb = evaluate_probs(P, fallback_p, meta, gt_flow, mask)
    fb_top = np.zeros(n)
    fb_top[P["pair_i"][fb["top_idx"]]] = P["f1"][fb["top_idx"]]
    hybrid = np.where(seen, f1_seen, fb_top)
    # flows: empirical top-n route distribution for seen ODs, fallback probabilities otherwise
    fl = np.zeros(meta["n_seg"])
    for i in idx:
        lst = hist[int(keys[i])][:topn]
        tot = sum(c for _, c in lst)
        for rb, c in lst:
            np.add.at(fl, np.frombuffer(rb, dtype=np.int32), c / tot)
    unseen_pairs = ~seen[P["pair_i"]]
    rep, seg = diag.flatten_rows(P["rflat"], P["roff"], np.nonzero(unseen_pairs)[0])
    fl += np.bincount(seg, weights=fallback_p[unseen_pairs][rep], minlength=meta["n_seg"])
    return {"coverage": float(seen.mean()), "top1_seen_only": float(f1_seen[idx].mean()) if len(idx) else float("nan"),
            "fallback_top1_on_seen": float(fb_top[idx].mean()) if len(idx) else float("nan"),
            "hybrid_top1": float(hybrid.mean()), "hybrid_flow_nmae": diag.geh_stats(fl, gt_flow, mask)["nmae"]}


def loo_ceiling(A_test, device, min_n=5, length=None):
    """Leave-one-out mode-route prediction within each OD on the full test set."""
    off, flat = A_test["gt_off"], A_test["gt_flat"]
    n = len(off) - 1
    lens = np.diff(off)
    ok = np.where(lens > 1)[0]
    rs = {i: flat[off[i]:off[i + 1]] for i in ok}
    by_od = defaultdict(lambda: defaultdict(int))
    for i in ok:
        r = rs[i]
        by_od[(int(r[0]), int(r[-1]))][r.tobytes()] += 1
    tops = {k: sorted(v.items(), key=lambda x: -x[1])[:2] for k, v in by_od.items()}
    sizes = {k: sum(v.values()) for k, v in by_od.items()}
    idx, preds = [], []
    for i in ok:
        r = rs[i]
        k = (int(r[0]), int(r[-1]))
        if sizes[k] < min_n:
            continue
        (r1, c1), *rest = tops[k]
        own = r.tobytes()
        if own == r1 and (c1 - 1) < (rest[0][1] if rest else 0):
            p = rest[0][0]
        else:
            p = r1
        idx.append(i)
        preds.append(np.frombuffer(p, dtype=np.int32))
    gt = [rs[i] for i in idx]
    f1 = diag.route_f1(preds, gt, device, length=length if F1_MODE == "length" else None)
    return np.array(idx), f1


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["pools", "eval", "all"])
    ap.add_argument("--cities", default="xian,chengdu,beijing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--subset", type=int, default=100_000)
    ap.add_argument("--regions", type=int, default=16)
    ap.add_argument("--min-flow", type=float, default=5.0)
    ap.add_argument("--hist-sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000_000])
    args = ap.parse_args()
    cities = args.cities.split(",")
    metas = {c: road_meta(c, args.regions) for c in ALL}
    if args.stage in ("pools", "all"):
        for c in cities:
            for split in ("test", "train"):
                A = load_split(c, split, args.workers)
                build_pool(c, split, A, subset(c, split, A, args.subset), metas[c], args)
    if args.stage == "pools":
        return

    P_train, P_test, A_tr, A_te = {}, {}, {}, {}
    for c in ALL:
        A_te[c], A_tr[c] = load_split(c, "test", args.workers), load_split(c, "train", args.workers)
        P_test[c] = build_pool(c, "test", A_te[c], subset(c, "test", A_te[c], args.subset), metas[c], args)
        P_train[c] = build_pool(c, "train", A_tr[c], subset(c, "train", A_tr[c], args.subset), metas[c], args)
    betas = {c: fit_logit(P_train[c], args.device) for c in P_train}
    for c, b in betas.items():
        log(f"[beta] {c} train: " + ", ".join(f"{f}={v:.2f}" for f, v in zip(FEATS, b)))

    results = {"feats": FEATS, "betas_train": {c: b.tolist() for c, b in betas.items()}, "targets": {}}
    for t in cities:
        P, meta = P_test[t], metas[t]
        n = int(P["n"])
        gt_rep, gt_seg = diag.flatten_rows(P["gt_flat"], P["gt_off"], np.arange(n))
        gt_flow = np.bincount(gt_seg, minlength=meta["n_seg"]).astype(float)
        mask = gt_flow >= args.min_flow
        res = {"n": n, "BoK_MAP6": float(P["best"].mean())}
        # simple rules
        slot0 = P["pair_k"] == 0
        res["LEN_top1"] = float(P["f1"][slot0].sum() / n)
        uni = softmax_by_traj(np.zeros(len(P["pair_i"])), P["pair_i"], n)
        ev = evaluate_probs(P, uni, meta, gt_flow, mask)
        res["uniform"] = {k: v for k, v in ev.items() if k != "top_idx"}
        # M1 zero-shot from each other city
        for s in ALL:
            if s == t:
                continue
            p = softmax_by_traj(P["X"] @ betas[s], P["pair_i"], n)
            ev = evaluate_probs(P, p, meta, gt_flow, mask)
            res[f"M1_zeroshot_from_{SHORT[s]}"] = {k: v for k, v in ev.items() if k != "top_idx"}
        # M2 in-city logit
        p_in = softmax_by_traj(P["X"] @ betas[t], P["pair_i"], n)
        ev = evaluate_probs(P, p_in, meta, gt_flow, mask)
        res["M2_incity_logit"] = {k: v for k, v in ev.items() if k != "top_idx"}
        # M3 in-city GBDT
        clf = fit_gbdt(P_train[t], args.regions)
        sc = clf.predict_proba(gbdt_design(P, args.regions))[:, 1]
        p_g = sc / np.bincount(P["pair_i"], weights=sc, minlength=n)[P["pair_i"]]
        ev = evaluate_probs(P, p_g, meta, gt_flow, mask)
        res["M3_incity_gbdt"] = {k: v for k, v in ev.items() if k != "top_idx"}
        # M4 OD-history with growing target-train size, fallback = best zero-shot source
        srcs = [s for s in ALL if s != t]
        p_zs = softmax_by_traj(P["X"] @ betas[srcs[0]], P["pair_i"], n)
        perm = np.random.default_rng(7).permutation(len(A_tr[t]["gt_off"]) - 1)
        res["M4_od_history"] = {}
        for hs in args.hist_sizes:
            rows_h = perm[: min(hs, len(perm))]
            hist = od_history(A_tr[t], rows_h, meta["n_seg"])
            r = eval_od_history(P, hist, gt_flow, mask, meta, p_zs, args.device)
            r["n_train"] = int(len(rows_h))
            res["M4_od_history"][str(len(rows_h))] = r
            log(f"[M4] {t} n_train={len(rows_h)}: coverage={r['coverage']:.3f} hybrid top1={r['hybrid_top1']:.3f} "
                f"seen-only={r['top1_seen_only']:.3f} vs zero-shot-on-seen={r['fallback_top1_on_seen']:.3f}")
        # LOO predictability ceiling (full test), compared on the overlapping subset trips
        idx, f1_loo = loo_ceiling(A_te[t], args.device, length=meta["length"])
        sub_rows = subset(t, "test", A_te[t], args.subset)
        pos = {int(r): j for j, r in enumerate(sub_rows)}
        both = [(pos[int(i)], f) for i, f in zip(idx, f1_loo) if int(i) in pos]
        res["LOO_ceiling_all"] = {"n_trips": int(len(idx)), "top1": float(f1_loo.mean())}
        if both:
            j = np.array([b[0] for b in both])
            fl = np.array([b[1] for b in both])
            comp = {"n": int(len(j)), "LOO_ceiling": float(fl.mean())}
            for name, p in (("M1_zeroshot", p_zs), ("M2_incity_logit", p_in), ("M3_incity_gbdt", p_g)):
                top = evaluate_probs(P, p, meta, gt_flow, mask)["top_idx"]
                per = np.zeros(n)
                per[P["pair_i"][top]] = P["f1"][top]
                comp[name] = float(per[j].mean())
            per_b = P["best"]
            comp["MAP6_BoK"] = float(per_b[j].mean())
            res["on_frequent_OD_trips"] = comp
        results["targets"][t] = res
        log(f"[{t}] " + json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
                                   for k, v in res.items() if k not in ("M4_od_history",)}, default=str)[:1500])
    (OUT / f"E4_week1{TAG}.json").write_text(json.dumps(results, indent=2))
    log("E4 done")


if __name__ == "__main__":
    main()
