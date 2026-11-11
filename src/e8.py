"""E8: counts vs local trajectories on the SAME evaluation links, plus count-realism robustness.

For each target city and sensor draw (rho in {0.05, 0.10}, 3 seeds), every method is scored on the
same link set: links with GT flow >= 5 and WITHOUT any sensor.
  zero-shot logit (pooled sources)                               -- no target data
  C2 link-offset calibration from counts (lam_theta grid)         -- counts only
  C2 under realistic counts: Poisson + multiplicative noise and unknown penetration p (scale fitted)
  in-city GBDT and OD-history hybrid with n target trajectories   -- trajectories only
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
import e5  # noqa: E402

N_PER = len(diag.PERIOD_EDGES) - 1
NS = [1000, 10000, 100000]


def main():
    import torch
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    cities = e4.ALL
    metas = {c: e4.road_meta(c, 16) for c in cities}
    Pt = {c: e4.load_pool(c, "test") for c in cities}
    Ptr = {c: e4.load_pool(c, "train") for c in cities}
    A_tr = {c: e4.load_split(c, "train", 8) for c in cities}
    tr_rows = {c: e4.subset(c, "train", A_tr[c], 100000) for c in cities}
    res = {}
    for t in cities:
        P, meta = Pt[t], metas[t]
        n, n_seg = int(P["n"]), meta["n_seg"]
        pi_np = P["pair_i"]
        npair = len(pi_np)
        rep, seg = diag.flatten_rows(P["rflat"], P["roff"], np.arange(npair))
        grep, gseg = diag.flatten_rows(P["gt_flat"], P["gt_off"], np.arange(n))
        gt_flow = np.bincount(gseg, minlength=n_seg).astype(float)
        per_traj = np.digitize(P["hour"], diag.PERIOD_EDGES[1:-1])
        srcs = [s for s in cities if s != t]
        pooled = e4b.concat([Ptr[s] for s in srcs])
        beta_src = e4.fit_logit(pooled, device)
        p_zs = e4.softmax_by_traj(P["X"] @ beta_src, pi_np, n)
        Zt = e4b.design_noregion(P)
        # trajectory-based methods (fit once per n, seed 0) -> probability vectors / histories
        traj_methods = {}
        for n_t in NS:
            idx = np.random.default_rng(0).choice(int(Ptr[t]["n"]), min(n_t, int(Ptr[t]["n"])), replace=False)
            S = e5.take(Ptr[t], idx)
            g = e4b.fit(e4b.design_noregion(S), S["label"])
            sc = g.predict_proba(Zt)[:, 1]
            traj_methods[f"gbdt@{len(idx)}"] = ("p", sc / np.bincount(pi_np, weights=sc, minlength=n)[pi_np])
            traj_methods[f"odhist@{len(idx)}"] = ("hist", e4.od_history(A_tr[t], tr_rows[t][np.sort(idx)], n_seg))

        X = torch.tensor(P["X"], dtype=torch.float32, device=device)
        pi = torch.tensor(pi_np, device=device)
        b0 = torch.tensor(beta_src, dtype=torch.float32, device=device)

        def probs(u):
            m = torch.full((n,), -1e30, device=device).scatter_reduce(0, pi, u, reduce="amax")
            e = torch.exp(u - m[pi])
            return e / torch.zeros(n, device=device).index_add(0, pi, e)[pi]

        rows = []
        for rho in (0.05, 0.10):
            for seed in range(3):
                r2 = np.random.default_rng(1000 + seed)
                n_sens = max(4, int(round(rho * n_seg)))
                major_ids, other_ids = np.where(meta["major"])[0], np.where(~meta["major"])[0]
                sens = r2.choice(major_ids, n_sens, replace=False) if n_sens <= len(major_ids) else \
                    np.r_[major_ids, r2.choice(other_ids, n_sens - len(major_ids), replace=False)]
                mask = gt_flow >= 5
                mask[sens] = False
                pos = np.full(n_seg, -1, np.int64)
                pos[sens] = np.arange(len(sens))
                pr, col = diag.unique_hits(rep, seg, pos)
                H = e6_sparse(col * N_PER + per_traj[pi_np[pr]], pr, (len(sens) * N_PER, npair), device)
                R = e6_sparse(pr, col, (npair, len(sens)), device)
                ti, sc_ = diag.unique_hits(grep, gseg, pos)
                y_true = np.bincount(sc_ * N_PER + per_traj[ti], minlength=len(sens) * N_PER).astype(np.float64)
                row = {"rho": rho, "seed": seed}
                row["zeroshot"] = round(e4.evaluate_probs(P, p_zs, meta, gt_flow, mask)["flow_nmae"], 4)
                for name, (kind, obj) in traj_methods.items():
                    if kind == "p":
                        row[name] = round(e4.evaluate_probs(P, obj, meta, gt_flow, mask)["flow_nmae"], 4)
                    else:
                        row[name] = round(e4.eval_od_history(P, obj, gt_flow, mask, meta, p_zs, device)["hybrid_flow_nmae"], 4)
                # counts: lambda grid (clean counts) and realism grid (lam=1e-3)
                settings = [("C2_lam1e-4", 1e-4, 1.0, 0.0), ("C2_lam1e-3", 1e-3, 1.0, 0.0), ("C2_lam1e-2", 1e-2, 1.0, 0.0),
                            ("C2_noise0.2", 1e-3, 1.0, 0.2), ("C2_pen0.2", 1e-3, 0.2, 0.0), ("C2_pen0.2_noise0.2", 1e-3, 0.2, 0.2)]
                for name, lam, pen, sig in settings:
                    rng = np.random.default_rng(77 + seed)
                    y = y_true / pen
                    if pen < 1.0 or sig > 0:
                        y = rng.poisson(y).astype(np.float64)
                    if sig > 0:
                        y = y * rng.lognormal(0.0, sig, size=y.shape)
                    yt = torch.tensor(y, dtype=torch.float32, device=device)
                    fit_scale = pen < 1.0
                    beta = b0.clone().requires_grad_(True)
                    theta = torch.zeros(len(sens), device=device, requires_grad=True)
                    log_s = torch.zeros(1, device=device, requires_grad=True)
                    opt = torch.optim.Adam([beta, theta] + ([log_s] if fit_scale else []), lr=0.02)
                    for _ in range(600):
                        opt.zero_grad()
                        p = probs(X @ beta + torch.sparse.mm(R, theta.unsqueeze(1)).squeeze(1))
                        pred = torch.sparse.mm(H, p.unsqueeze(1)).squeeze(1) * torch.exp(log_s)
                        loss = ((pred - yt) ** 2 / (yt + 10.0)).mean() + 1e-2 * ((beta - b0) ** 2).sum() + lam * (theta ** 2).sum()
                        loss.backward()
                        opt.step()
                    with torch.no_grad():
                        pn = probs(X @ beta + torch.sparse.mm(R, theta.unsqueeze(1)).squeeze(1)).cpu().numpy().astype(np.float64)
                    ev = e4.evaluate_probs(P, pn, meta, gt_flow, mask)
                    row[name] = round(ev["flow_nmae"], 4)
                    row[name + "|top1"] = round(ev["top1_f1"], 4)
                    if fit_scale:
                        row[name + "|scale_est"] = round(float(torch.exp(log_s).item()), 3)
                rows.append(row)
                diag.log(f"[{t}] rho={rho} seed={seed} " + json.dumps(row))
        res[t] = rows
        (diag.OUT / f"E8_equiv{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E8 done")


def e6_sparse(rows, cols, shape, device):
    import torch
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.ones(len(rows)), shape).coalesce().to(device)


if __name__ == "__main__":
    main()
