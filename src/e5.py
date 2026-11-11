"""Few-shot curves: how many target trajectories close the transfer gap?

For each target city and n_target in {100, 1k, 10k, 100k} (3 seeds), fit on a random
subset of the target TRAIN pool and evaluate on the fixed 100k target TEST subset:
  logit_target      logit fitted on target subset only
  logit_prior       logit with L2 pull towards the pooled-source beta (classic combined transfer)
  gbdt_target       GBDT (no region ids) on target subset only
  gbdt_stacked      GBDT on target subset with the pooled-source GBDT score as an extra feature
  odhist_hybrid     OD-history from the same n trajectories, fallback zero-shot logit
Zero-shot references (n=0): pooled-source logit and pooled-source GBDT.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402

CITIES = e4.ALL
NS = [100, 1000, 10000, 100000]


def take(P, traj_idx):
    """Restrict a pair table to a subset of trajectories (re-indexed)."""
    keep_traj = np.zeros(int(P["n"]), bool)
    keep_traj[traj_idx] = True
    m = keep_traj[P["pair_i"]]
    remap = -np.ones(int(P["n"]), np.int64)
    remap[np.sort(traj_idx)] = np.arange(len(traj_idx))
    out = {"X": P["X"][m], "label": P["label"][m], "f1": P["f1"][m], "pair_i": remap[P["pair_i"][m]],
           "n": np.int64(len(traj_idx))}
    for k in ("hour", "min_len"):
        out[k] = P[k][np.sort(traj_idx)]
    return out


def fit_logit_prior(P, prior_raw, lam, device, iters=800):
    import torch
    X = P["X"]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    pi = torch.tensor(P["pair_i"], device=device)
    lab = torch.tensor(P["label"], device=device)
    n = int(P["n"])
    prior = torch.tensor(np.asarray(prior_raw) * sd, dtype=torch.float32, device=device)
    beta = prior.clone().requires_grad_(True)
    opt = torch.optim.Adam([beta], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        lg = Xt @ beta
        m = torch.full((n,), -1e30, device=device).scatter_reduce(0, pi, lg, reduce="amax")
        e = torch.exp(lg - m[pi])
        s = torch.zeros(n, device=device).index_add(0, pi, e)
        loss = -(torch.log(e[lab] / s[pi][lab] + 1e-12)).mean() + lam * ((beta - prior) ** 2).sum()
        loss.backward()
        opt.step()
    return beta.detach().cpu().numpy() / sd


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    metas = {c: e4.road_meta(c, 16) for c in CITIES}
    Pt = {c: e4.load_pool(c, "test") for c in CITIES}
    Ptr = {c: e4.load_pool(c, "train") for c in CITIES}
    A_tr = {c: e4.load_split(c, "train", 10) for c in CITIES}
    tr_rows = {c: e4.subset(c, "train", A_tr[c], 100000) for c in CITIES}
    res = {}
    for t in CITIES:
        P, meta = Pt[t], metas[t]
        n = int(P["n"])
        _, gt_seg = diag.flatten_rows(P["gt_flat"], P["gt_off"], np.arange(n))
        gt_flow = np.bincount(gt_seg, minlength=meta["n_seg"]).astype(float)
        mask = gt_flow >= 5
        Zt = e4b.design_noregion(P)

        def ev(p):
            r = e4.evaluate_probs(P, p, meta, gt_flow, mask)
            return {"top1": round(r["top1_f1"], 4), "flow_nmae": round(r["flow_nmae"], 4)}

        srcs = [s for s in CITIES if s != t]
        pooled = e4b.concat([Ptr[s] for s in srcs])
        pooled["hour"] = np.concatenate([Ptr[s]["hour"] for s in srcs])
        pooled["min_len"] = np.concatenate([Ptr[s]["min_len"] for s in srcs])
        beta_src = e4.fit_logit(pooled, device)
        gbdt_src = e4b.fit(e4b.design_noregion(pooled), pooled["label"])
        p_zs = e4.softmax_by_traj(P["X"] @ beta_src, P["pair_i"], n)
        src_score_test = gbdt_src.predict_proba(Zt)[:, 1]
        out = {"zeroshot_logit": ev(p_zs),
               "zeroshot_gbdt": ev(src_score_test / np.bincount(P["pair_i"], weights=src_score_test, minlength=n)[P["pair_i"]]),
               "curves": []}
        for n_t in NS:
            for seed in range(3):
                rng = np.random.default_rng(seed)
                idx = rng.choice(int(Ptr[t]["n"]), min(n_t, int(Ptr[t]["n"])), replace=False)
                S = take(Ptr[t], idx)
                row = {"n_target": int(len(idx)), "seed": seed}
                b = e4.fit_logit(S, device)
                row["logit_target"] = ev(e4.softmax_by_traj(P["X"] @ b, P["pair_i"], n))
                b2 = fit_logit_prior(S, beta_src, lam=1.0 / max(len(idx), 1) * 100, device=device)
                row["logit_prior"] = ev(e4.softmax_by_traj(P["X"] @ b2, P["pair_i"], n))
                Zs = e4b.design_noregion(S)
                if S["label"].sum() >= 20 and len(np.unique(S["label"])) == 2:
                    g = e4b.fit(Zs, S["label"])
                    sc = g.predict_proba(Zt)[:, 1]
                    row["gbdt_target"] = ev(sc / np.bincount(P["pair_i"], weights=sc, minlength=n)[P["pair_i"]])
                    ss = gbdt_src.predict_proba(Zs)[:, 1]
                    g2 = e4b.fit(np.c_[Zs, ss], S["label"])
                    sc2 = g2.predict_proba(np.c_[Zt, src_score_test])[:, 1]
                    row["gbdt_stacked"] = ev(sc2 / np.bincount(P["pair_i"], weights=sc2, minlength=n)[P["pair_i"]])
                hist = e4.od_history(A_tr[t], tr_rows[t][np.sort(idx)], meta["n_seg"])
                h = e4.eval_od_history(P, hist, gt_flow, mask, meta, p_zs, device)
                row["odhist_hybrid"] = {"top1": round(h["hybrid_top1"], 4), "flow_nmae": round(h["hybrid_flow_nmae"], 4),
                                        "coverage": round(h["coverage"], 4)}
                out["curves"].append(row)
                diag.log(f"[{t}] n={len(idx)} s={seed} " + json.dumps({k: v for k, v in row.items() if isinstance(v, dict)}))
        res[t] = out
        (diag.OUT / f"E5_fewshot{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E5 done")


if __name__ == "__main__":
    main()
