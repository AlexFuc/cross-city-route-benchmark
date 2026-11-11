"""Can aggregate link counts recover the city-specific part that does not transfer?

Base: zero-shot logit with pooled-source beta over the map-only MAP6 pool (target test subset).
Count-calibrated variants (fit only on FIT sensors, evaluated on links WITHOUT any sensor):
  C1 global beta re-fit from counts (L2 pull to source beta)
  C2 C1 + per-sensor-link utility offsets theta_a (L2)
  C3 C1 + origin-region beta deviations delta_r (L2)
References: zero-shot base, in-city GBDT / OD-history (from E5, 100k).
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
N_PER = len(diag.PERIOD_EDGES) - 1


def torch_sparse(rows, cols, shape, device):
    import torch
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.ones(len(rows)), shape).coalesce().to(device)


def main():
    import torch
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    lam_beta, lam_theta, lam_delta = 1e-2, 1e-3, 1e-2
    metas = {c: e4.road_meta(c, 16) for c in CITIES}
    Pt = {c: e4.load_pool(c, "test") for c in CITIES}
    Ptr = {c: e4.load_pool(c, "train") for c in CITIES}
    res = {}
    for t in CITIES:
        P, meta = Pt[t], metas[t]
        n, n_seg = int(P["n"]), meta["n_seg"]
        pi_np = P["pair_i"]
        npair = len(pi_np)
        rep, seg = diag.flatten_rows(P["rflat"], P["roff"], np.arange(npair))
        grep, gseg = diag.flatten_rows(P["gt_flat"], P["gt_off"], np.arange(n))
        gt_flow = np.bincount(gseg, minlength=n_seg).astype(float)
        per_traj = np.digitize(P["hour"], diag.PERIOD_EDGES[1:-1])
        srcs = [s for s in CITIES if s != t]
        pooled = e4b.concat([Ptr[s] for s in srcs])
        beta_src = e4.fit_logit(pooled, device)
        g_in = e4b.fit(e4b.design_noregion(Ptr[t]), Ptr[t]["label"])
        sc_in = g_in.predict_proba(e4b.design_noregion(P))[:, 1]
        p_incity = sc_in / np.bincount(pi_np, weights=sc_in, minlength=n)[pi_np]

        X = torch.tensor(P["X"], dtype=torch.float32, device=device)
        pi = torch.tensor(pi_np, device=device)
        region = torch.tensor(P["region_o"][pi_np], device=device)

        def probs(util):
            m = torch.full((n,), -1e30, device=device).scatter_reduce(0, pi, util, reduce="amax")
            e = torch.exp(util - m[pi])
            s = torch.zeros(n, device=device).index_add(0, pi, e)
            return e / s[pi]

        def flows_of(p):
            return np.bincount(seg, weights=p[rep], minlength=n_seg)

        def top1_of(p):
            order = np.lexsort((-p, pi_np))
            first = np.r_[True, pi_np[order][1:] != pi_np[order][:-1]]
            return float(P["f1"][order[first]].mean())

        out = {"rows": []}
        for rho in (0.02, 0.05, 0.10):
            for seed in range(3):
                r2 = np.random.default_rng(1000 + seed)
                n_sens = max(4, int(round(rho * n_seg)))
                major_ids, other_ids = np.where(meta["major"])[0], np.where(~meta["major"])[0]
                sens = r2.choice(major_ids, n_sens, replace=False) if n_sens <= len(major_ids) else \
                    np.r_[major_ids, r2.choice(other_ids, n_sens - len(major_ids), replace=False)]
                r2.shuffle(sens)
                cut = int(round(0.7 * len(sens)))
                fit_s, hold_s = sens[:cut], sens[cut:]
                eval_mask = (gt_flow >= 5)
                eval_mask[sens] = False          # evaluate only on links without any sensor

                def design(sset):
                    pos = np.full(n_seg, -1, np.int64)
                    pos[sset] = np.arange(len(sset))
                    pr, col = diag.unique_hits(rep, seg, pos)
                    H = torch_sparse(col * N_PER + per_traj[pi_np[pr]], pr, (len(sset) * N_PER, npair), device)
                    ti, sc = diag.unique_hits(grep, gseg, pos)
                    y = np.bincount(sc * N_PER + per_traj[ti], minlength=len(sset) * N_PER).astype(np.float32)
                    R = torch_sparse(pr, col, (npair, len(sset)), device)   # route x sensor incidence
                    return H, torch.tensor(y, device=device), R

                Hf, yf, Rf = design(fit_s)
                Hh, yh, _ = design(hold_s)
                b0 = torch.tensor(beta_src, dtype=torch.float32, device=device)

                def fit(variant, iters=600):
                    beta = b0.clone().requires_grad_(True)
                    theta = torch.zeros(len(fit_s), device=device, requires_grad=True)
                    delta = torch.zeros((16, X.shape[1]), device=device, requires_grad=True)
                    params = [beta] + ([theta] if variant == "C2" else []) + ([delta] if variant == "C3" else [])
                    opt = torch.optim.Adam(params, lr=0.02)

                    def util():
                        u = X @ beta
                        if variant == "C2":
                            u = u + torch.sparse.mm(Rf, theta.unsqueeze(1)).squeeze(1)
                        if variant == "C3":
                            u = u + (X * delta[region]).sum(1)
                        return u

                    for _ in range(iters):
                        opt.zero_grad()
                        r = torch.sparse.mm(Hf, probs(util()).unsqueeze(1)).squeeze(1) - yf
                        loss = (r * r / (yf + 10.0)).mean() + lam_beta * ((beta - b0) ** 2).sum()
                        if variant == "C2":
                            loss = loss + lam_theta * (theta ** 2).sum()
                        if variant == "C3":
                            loss = loss + lam_delta * (delta ** 2).sum()
                        loss.backward()
                        opt.step()
                    with torch.no_grad():
                        return probs(util())

                row = {"rho": rho, "seed": seed, "n_sensors": int(len(sens))}
                with torch.no_grad():
                    variants = {"C0_zeroshot": probs(X @ b0)}
                for v in ("C1", "C2", "C3"):
                    variants[v] = fit(v)
                variants["REF_incity_gbdt"] = torch.tensor(p_incity, dtype=torch.float32, device=device)
                for name, p in variants.items():
                    pn = p.detach().cpu().numpy().astype(np.float64)
                    hold_pred = torch.sparse.mm(Hh, p.unsqueeze(1)).squeeze(1).cpu().numpy()
                    yhn = yh.cpu().numpy()
                    row[name] = {"nonsensor_nmae": round(diag.geh_stats(flows_of(pn), gt_flow, eval_mask)["nmae"], 4),
                                 "hold_sensor_nmae": round(diag.geh_stats(hold_pred, yhn, yhn >= 5)["nmae"], 4),
                                 "top1": round(top1_of(pn), 4)}
                out["rows"].append(row)
                diag.log(f"[{t}] rho={rho} seed={seed} " + json.dumps({k: v for k, v in row.items() if isinstance(v, dict)}))
        res[t] = out
        (diag.OUT / f"E6_countcal{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E6 done")


if __name__ == "__main__":
    main()
