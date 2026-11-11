"""Recursive logit v2: log-space value iteration, transition-level map features, existence handling.

u(k->a) = b1*len_km(a) + b2*len_km(a)*major(a) + b3 + b4*|turn|/pi + b5*uturn(|turn|>150deg)
V_d(k) = logsumexp_{a in succ(k)} [u(k->a) + V_d(a)],  V_d(d) = 0      (batched over destinations, GPU)
log P(path) = sum_t u(a_{t-1}->a_t) - V_d(o)
Gradient: observed minus expected transition-feature sums (forward mass propagation).
Transfer: if V does not exist on the target network under source beta, lower b3 in 0.25 steps until it does.
Usage: python rl2.py <target_city> <source_city|-> [device]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e9  # noqa: E402

N_TRAIN, N_TEST, ITERS = 5000, 10000, 500
K = 5


class RL2:
    def __init__(self, city, device):
        import torch
        self.t, self.dev = torch, device
        self.meta = e4.road_meta(city, 16)
        e = pd.read_csv(diag.DATA_ROOT / f"data/{city}/map/edge.csv", usecols=["src", "trg", "angle"])
        self.N, self.E = self.meta["n_seg"], len(e)
        self.src, self.dst = e.src.to_numpy(np.int64), e.trg.to_numpy(np.int64)
        turn = np.abs((e.angle.to_numpy() + np.pi) % (2 * np.pi) - np.pi)
        L = self.meta["length"][self.dst] / 1000.0
        maj = self.meta["major"][self.dst].astype(float)
        self.X = np.stack([L, L * maj, np.ones(self.E), turn / np.pi, (turn > np.deg2rad(150)).astype(float)], 1)
        self.Xt = torch.tensor(self.X, dtype=torch.float64, device=device)
        self.s_t = torch.tensor(self.src, device=device)
        self.d_t = torch.tensor(self.dst, device=device)
        self.eid = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.src, self.dst))}
        self.batch = 512 if self.E < 40000 else 192

    def values(self, u, dests):
        t = self.t
        D = len(dests)
        dt = t.tensor(dests, device=self.dev)
        cols = t.arange(D, device=self.dev)
        V = t.full((self.N, D), -float("inf"), dtype=t.float64, device=self.dev)
        V[dt, cols] = 0.0
        idx = self.s_t.view(-1, 1).expand(-1, D)
        ok = True
        for it in range(ITERS):
            val = u.view(-1, 1) + V[self.d_t]                                   # E x D
            m = t.full((self.N, D), -float("inf"), dtype=t.float64, device=self.dev).scatter_reduce(0, idx, val, "amax")
            mf = t.where(t.isfinite(m), m, t.zeros_like(m))
            s = t.zeros((self.N, D), dtype=t.float64, device=self.dev).scatter_add(0, idx, t.exp(val - mf[self.s_t]))
            Vn = t.where(s > 0, mf + t.log(s.clamp_min(1e-300)), t.full_like(s, -float("inf")))
            Vn[dt, cols] = 0.0
            fin = t.isfinite(Vn) & t.isfinite(V)
            delta = (Vn - V)[fin].abs().max().item() if fin.any() else float("inf")
            V = Vn
            if float(V[t.isfinite(V)].max().item()) > 700:
                ok = False
                break
            if it > 10 and delta < 1e-10:
                break
        else:
            ok = ok and delta < 1e-6
        self.exists = getattr(self, "exists", True) and ok
        return V

    def pass_(self, beta, paths):
        """mean-free sums: loglik, observed and expected transition-feature totals, expected link visits."""
        t = self.t
        u = self.Xt @ beta
        obs, exp_, ll = np.zeros(K), np.zeros(K), 0.0
        visits = np.zeros(self.N)
        by_d = {}
        for p in paths:
            by_d.setdefault(int(p[-1]), []).append(p)
        ds = list(by_d)
        self.exists = True
        u_np = u.cpu().numpy()
        for s0 in range(0, len(ds), self.batch):
            dd = ds[s0:s0 + self.batch]
            V = self.values(u, dd)
            ps = [p for d in dd for p in by_d[d]]
            col = {d: j for j, d in enumerate(dd)}
            jj = t.tensor([col[int(p[-1])] for p in ps], device=self.dev)
            oo = t.tensor([int(p[0]) for p in ps], device=self.dev)
            de = t.tensor([int(p[-1]) for p in ps], device=self.dev)
            ll -= float(V[oo, jj].sum().item())
            for p in ps:
                es = [self.eid.get((int(a), int(b))) for a, b in zip(p[:-1], p[1:])]
                es = [e for e in es if e is not None]
                obs += self.X[es].sum(0)
                ll += float(u_np[es].sum())
            Vp = V[:, jj]                                                   # N x P
            P = len(ps)
            pc = t.arange(P, device=self.dev)
            logp = u.view(-1, 1) + Vp[self.d_t] - Vp[self.s_t]              # E x P transition log-probs
            prob = t.where(t.isfinite(logp), t.exp(logp), t.zeros_like(logp))
            f = t.zeros((self.N, P), dtype=t.float64, device=self.dev)
            f[oo, pc] = 1.0
            ecnt = t.zeros(self.E, dtype=t.float64, device=self.dev)
            vis = t.zeros(self.N, dtype=t.float64, device=self.dev)
            for _ in range(ITERS):
                f[de, pc] = 0.0
                flow = f[self.s_t] * prob                                    # E x P
                if float(flow.sum().item()) < 1e-9:
                    break
                ecnt += flow.sum(1)
                f = t.zeros_like(f).index_add(0, self.d_t, flow)
                vis += f.sum(1)
            exp_ += (ecnt @ self.Xt).cpu().numpy()
            visits += vis.cpu().numpy()
        return ll, obs, exp_, visits

    def fit(self, paths, beta0, maxiter=80):
        from scipy.optimize import minimize
        t = self.t
        n = len(paths)
        trace = []

        def f(b):
            ll, obs, exp_, _ = self.pass_(t.tensor(b, dtype=t.float64, device=self.dev), paths)
            if not self.exists or not np.isfinite(ll):
                return 1e6, np.zeros(K)
            trace.append(ll / n)
            return -ll / n, -(obs - exp_) / n

        bounds = [(-50, 0), (-20, 20), (-20, -0.05), (-20, 5), (-20, 5)]
        r = minimize(f, np.asarray(beta0, float), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        self.fit_info = {"success": bool(r.success), "nit": int(r.nit), "message": str(r.message)}
        return np.asarray(r.x), trace

    def fit_robust(self, paths, beta0, maxiter=60):
        """Projected gradient ascent with Armijo backtracking that never leaves the existence region."""
        t = self.t
        n = len(paths)
        lo = np.array([-50, -20, -20, -20, -20.0])
        hi = np.array([0, 20, -0.05, 5, 5.0])

        def ev(b):
            ll, obs, exp_, _ = self.pass_(t.tensor(b, dtype=t.float64, device=self.dev), paths)
            return (ll / n if self.exists and np.isfinite(ll) else -np.inf), (obs - exp_) / n

        b = np.clip(np.asarray(beta0, float), lo, hi)
        f, g = ev(b)
        step, trace, stall = 0.1, [f], 0
        for it in range(maxiter):
            accepted = False
            for _ in range(25):
                bn = np.clip(b + step * g, lo, hi)
                fn, gn = ev(bn)
                if np.isfinite(fn) and fn >= f + 1e-4 * np.dot(g, bn - b):
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            stall = stall + 1 if fn - f < 1e-6 else 0
            b, f, g = bn, fn, gn
            trace.append(f)
            step *= 1.5
            if stall >= 3 or np.linalg.norm(g) < 1e-4:
                break
        self.fit_info = {"success": True, "nit": it + 1, "message": "robust-armijo", "final_grad_norm": float(np.linalg.norm(g))}
        return b, trace

    def ensure_exists(self, beta, probe_paths):
        """Lower the link constant until the value function exists on this network (transfer rule)."""
        t = self.t
        b = np.array(beta, float)
        shift = 0.0
        for _ in range(40):
            self.pass_(t.tensor(b, dtype=t.float64, device=self.dev), probe_paths)
            if self.exists:
                return b, shift
            b[2] -= 0.25
            shift -= 0.25
        return b, shift

    def most_likely(self, beta, ods):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        w = np.maximum(-(self.X @ beta), 1e-6)
        csr = csr_matrix((w, (self.src, self.dst)), shape=(self.N, self.N))
        by_o = {}
        for i, (o, d) in enumerate(ods):
            by_o.setdefault(o, []).append((d, i))
        res = [None] * len(ods)
        og = list(by_o)
        for s0 in range(0, len(og), 256):
            chunk = og[s0:s0 + 256]
            _, pred = dijkstra(csr, directed=True, indices=chunk, return_predecessors=True)
            for r_, o in enumerate(chunk):
                for d, i in by_o[o]:
                    path = [d]
                    while path[-1] != o and pred[r_, path[-1]] >= 0:
                        path.append(int(pred[r_, path[-1]]))
                    res[i] = np.array(path[::-1], np.int64)
        return res


def main():
    city, src = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None)
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:1"
    P = e4.load_pool(city, "test")
    te = np.random.default_rng(0).choice(int(P["n"]), min(N_TEST, int(P["n"])), replace=False)
    gall = diag.ragged_take(P["gt_flat"], P["gt_off"], range(int(P["n"])))
    gts = [np.asarray(gall[i], np.int64) for i in te]

    def train_paths(c):
        Ptr = e4.load_pool(c, "train")
        g = diag.ragged_take(Ptr["gt_flat"], Ptr["gt_off"], range(int(Ptr["n"])))
        sel = np.random.default_rng(1).choice(len(g), min(N_TRAIN, len(g)), replace=False)
        return [np.asarray(g[i], np.int64) for i in sel]

    rl = RL2(city, device)
    out = {"city": city, "N_links": int(rl.N), "N_edges": int(rl.E)}
    b0 = np.array([-1.0, 0.0, -1.0, -1.0, -2.0])
    # sanity: log-space value iteration vs exact linear solve z = (I - A)^-1 e_d (row d of A zeroed)
    from scipy.sparse import csr_matrix, identity
    from scipy.sparse.linalg import spsolve
    u0 = rl.Xt @ rl.t.tensor(b0, dtype=rl.t.float64, device=device)
    errs = []
    for d in (int(gts[0][-1]), int(gts[1][-1])):
        V = rl.values(u0, [d])[:, 0].cpu().numpy()
        A = csr_matrix((np.exp(u0.cpu().numpy()), (rl.src, rl.dst)), shape=(rl.N, rl.N)).tolil()
        A[d, :] = 0
        b = np.zeros(rl.N)
        b[d] = 1.0
        z = spsolve((identity(rl.N) - A.tocsr()).tocsc(), b)
        m = (z > 1e-250) & np.isfinite(V)
        errs.append(float(np.max(np.abs(np.log(z[m]) - V[m]))))
    out["logspace_vs_exact_max_abs_err"] = errs
    fits = [("incity", city)] + ([("zeroshot", src)] if src else [])
    tag = os.environ.get("RL2_TAG", "")
    L = rl.meta["length"]
    gt_flow = np.bincount(np.concatenate([g[1:] for g in gts]), minlength=rl.N).astype(float)
    mask = gt_flow >= 5
    for name, c in fits:
        t0 = time.time()
        fitter = rl if c == city else RL2(c, device)
        warm = os.environ.get("RL2_WARM")
        if os.environ.get("RL2_ROBUST") == "1":
            beta, trace = fitter.fit_robust(train_paths(c), np.array(json.loads(warm)) if warm else b0)
        else:
            beta, trace = fitter.fit(train_paths(c), b0)
        out[f"{name}_fit"] = {"sec": round(time.time() - t0, 1), **fitter.fit_info, "beta": np.round(beta, 4).tolist(),
                              "ll_trace": [round(x, 4) for x in trace[::5]] + ([round(trace[-1], 4)] if trace else [])}
        shift = 0.0
        if c != city:
            beta, shift = rl.ensure_exists(beta, gts[:2000])
        out[f"{name}_transfer_link_const_shift"] = shift
        pred = rl.most_likely(beta, [(int(g[0]), int(g[-1])) for g in gts])
        nr, ng = np.array([len(p) for p in pred]), np.array([len(g) for g in gts])
        f1c = diag.f1_from_lcs(diag.lcs_batch_gpu(pred, gts, device), nr, ng)
        wl = e9.wlcs_batch_gpu(pred, gts, L, device)
        rlen, glen = np.array([L[p].sum() for p in pred]), np.array([L[g].sum() for g in gts])
        pr_, rc_ = wl / np.maximum(rlen, 1e-9), wl / np.maximum(glen, 1e-9)
        f1w = 2 * pr_ * rc_ / np.maximum(pr_ + rc_, 1e-9)
        ll, _, _, visits = rl.pass_(rl.t.tensor(beta, dtype=rl.t.float64, device=device), gts)
        out[f"{name}_eval"] = {"exists": rl.exists, "test_ll_per_trip": round(ll / len(gts), 4),
                               "top1_count": round(float(f1c.mean()), 4), "top1_length": round(float(f1w.mean()), 4),
                               "flow_nmae": round(diag.geh_stats(visits, gt_flow, mask)["nmae"], 4)}
        diag.log(f"[{city}] {name}: " + json.dumps(out[f"{name}_eval"]) + f" beta={np.round(beta, 3).tolist()}")
    Ptr = e4.load_pool(city, "train")
    orig = np.bincount(np.array([int(g[0]) for g in gts]), minlength=rl.N).astype(float)
    pools = [("map6_incity", Ptr)] + ([("map6_zeroshot", e4.load_pool(src, "train"))] if src else [])
    for name, pool_ in pools:
        beta = e4.fit_logit(pool_, device)
        p = e4.softmax_by_traj(P["X"] @ beta, P["pair_i"], int(P["n"]))
        top_c = e9.top1_vec(P, p, P["f1"])
        sel = np.nonzero(np.isin(P["pair_i"], te))[0]
        rep, seg = diag.flatten_rows(P["rflat"], P["roff"], sel)
        fl = np.bincount(seg, weights=p[sel][rep], minlength=rl.N) - orig
        out[f"{name}_same10k"] = {"top1_count": round(float(top_c[te].mean()), 4),
                                  "flow_nmae": round(diag.geh_stats(fl, gt_flow, mask)["nmae"], 4)}
    print(json.dumps(out, indent=1))
    (diag.OUT / f"RL2_{city}{tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
