"""E9: rigor add-ons on the 100k target test subsets.

  (1) length-weighted LCS-F1 (matched length / route length), removing segment-granularity bias
  (2) Koppelman-Wilmot transfer index TI and transferability test statistic TTS for the logit
      (target-test log-likelihood of oracle labels; base model = equal probabilities over candidates)
  (3) bootstrap 95% CIs (trajectory resampling) for transfer gap and GBDT gain, count- and length-weighted F1
  (4) in-city GBDT hyper-parameter sensitivity; with/without time-of-day features
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402


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


def top1_vec(P, p, f):
    pi = P["pair_i"]
    order = np.lexsort((-p, pi))
    first = np.r_[True, pi[order][1:] != pi[order][:-1]]
    top = order[first]
    v = np.zeros(int(P["n"]))
    v[pi[top]] = f[top]
    return v


def loglik(P, p):
    return float(np.log(np.maximum(p[P["label"]], 1e-12)).sum())


def boot_ci(x, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return [round(float(x.mean()), 4), round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def design(P, hour=True):
    Z = e4b.design_noregion(P)
    return Z if hour else np.delete(Z, [Z.shape[1] - 3, Z.shape[1] - 2], axis=1)


def fit_gbdt(Z, y, leaf=63, lr=0.1):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=lr, max_leaf_nodes=leaf, random_state=0).fit(Z, y.astype(int))


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    cities = e4.ALL
    metas = {c: e4.road_meta(c, 16) for c in cities}
    Pt = {c: e4.load_pool(c, "test") for c in cities}
    Ptr = {c: e4.load_pool(c, "train") for c in cities}
    betas = {c: e4.fit_logit(Ptr[c], device) for c in cities}
    res = {}
    for t in cities:
        P, meta = Pt[t], metas[t]
        n = int(P["n"])
        L = meta["length"]
        # (1) length-weighted F1 for every unique candidate
        routes = diag.ragged_take(P["rflat"], P["roff"], range(len(P["pair_i"])))
        gts = diag.ragged_take(P["gt_flat"], P["gt_off"], range(n))
        gt_of_pair = [gts[i] for i in P["pair_i"]]
        wl = wlcs_batch_gpu(routes, gt_of_pair, L, device)
        rl = np.array([L[r].sum() for r in routes])
        gl = np.array([L[g].sum() for g in gt_of_pair])
        prec, rec = wl / np.maximum(rl, 1e-9), wl / np.maximum(gl, 1e-9)
        f1w = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        f1c = P["f1"]
        srcs = [s for s in cities if s != t]
        pooled = e4b.concat([Ptr[s] for s in srcs])
        b_zs = e4.fit_logit(pooled, device)
        pi = P["pair_i"]
        probs = {"zeroshot_pooled": e4.softmax_by_traj(P["X"] @ b_zs, pi, n),
                 "incity_logit": e4.softmax_by_traj(P["X"] @ betas[t], pi, n)}
        for s in srcs:
            probs[f"zeroshot_from_{e4.SHORT[s]}"] = e4.softmax_by_traj(P["X"] @ betas[s], pi, n)
        g = fit_gbdt(design(Ptr[t]), Ptr[t]["label"])
        sc = g.predict_proba(design(P))[:, 1]
        probs["incity_gbdt"] = sc / np.bincount(pi, weights=sc, minlength=n)[pi]
        uni = e4.softmax_by_traj(np.zeros(len(pi)), pi, n)
        r = {"n": n, "BoK_count": round(float(P["best"].mean()), 4)}
        bestw = np.full(n, -1.0)
        np.maximum.at(bestw, pi, f1w)
        r["BoK_length"] = round(float(bestw.mean()), 4)
        r["top1"] = {k: {"count": round(float(top1_vec(P, p, f1c).mean()), 4),
                         "length": round(float(top1_vec(P, p, f1w).mean()), 4)} for k, p in probs.items()}
        # (2) transfer index / TTS (oracle labels under count F1)
        ll_c, ll_loc = loglik(P, uni), loglik(P, probs["incity_logit"])
        r["transfer"] = {}
        for k in [k for k in probs if k.startswith("zeroshot")]:
            ll_tr = loglik(P, probs[k])
            r["transfer"][k] = {"TI": round((ll_tr - ll_c) / (ll_loc - ll_c), 4), "TTS": round(-2 * (ll_tr - ll_loc), 1),
                                "LL_per_trip_gap": round((ll_loc - ll_tr) / n, 4)}
        # (3) bootstrap CIs
        r["ci"] = {}
        for fname, f in (("count", f1c), ("length", f1w)):
            zs, inl, gb = (top1_vec(P, probs[k], f) for k in ("zeroshot_pooled", "incity_logit", "incity_gbdt"))
            r["ci"][fname] = {"transfer_gap": boot_ci(inl - zs), "gbdt_gain": boot_ci(gb - inl)}
        # (4) GBDT sensitivity and time features
        sens = {}
        for leaf, lr in ((31, 0.1), (127, 0.1), (63, 0.05)):
            gg = fit_gbdt(design(Ptr[t]), Ptr[t]["label"], leaf, lr)
            s2 = gg.predict_proba(design(P))[:, 1]
            sens[f"leaf{leaf}_lr{lr}"] = round(float(top1_vec(P, s2, f1c).mean()), 4)
        gg = fit_gbdt(design(Ptr[t], hour=False), Ptr[t]["label"])
        s3 = gg.predict_proba(design(P, hour=False))[:, 1]
        sens["no_hour_leaf63"] = round(float(top1_vec(P, s3, f1c).mean()), 4)
        sens["default_leaf63_lr0.1"] = r["top1"]["incity_gbdt"]["count"]
        r["gbdt_sensitivity_top1"] = sens
        res[t] = r
        diag.log(f"[{t}] " + json.dumps(r)[:1500])
        (diag.OUT / f"E9_rigor{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E9 done")


if __name__ == "__main__":
    main()
