"""E21: mechanism analysis (CPU only; every model is fitted on training trips, target-test trips are only scored).

Usage: python e21_why.py A|B|C|D ...     (each part writes results/E21_<part><TAG>.json)

A  Why transfer is cheap. Per target and source set (pooled, single):
   - decision identity: gap = P(rules differ) * E[F1_T - F1_S | differ] (exact), win rates among disagreements;
   - Taylor expansion of the target log-loss around the target MLE:
       actual  = [LL(b_T) - LL(b_S)] / n                   (target train; the score vanishes at the MLE)
       quad    = 0.5 d' H d, H = mean within-OD feature covariance under b_T, d = b_S - b_T
       ortho   = 0.5 sum_j d_j^2 H_jj                      (loss if coefficient differences acted independently)
     quad / ortho is invariant to rescaling any feature; < 1 means the differences compensate each other;
   - scale vs direction: 1-D refit of s in s*b_S on target train (top-1 is invariant to s);
   - restricted models on the composite cost c = log length + log free-flow time + log #segments ratios.
B  Anatomy of GTG routes against LEN and the same-source logit (all runs, plus in-city GTG).
C  What local data buy: gains by whether the OD occurred in training; HOSER by candidate membership.
E  Parameter-free composite rule: accuracy, significance against GTG, flows, Hausdorff, and top-1 over a simplex of weights.
F  Checks of Sec. VII: first-order term on test trips, rules designed without the target city, OD-cluster sampling.
D  Sample complexity of the logit: effective dimension d = tr(H^-1 J), predicted excess log-loss d/(2n), crossover with
   the transferred model, against empirical curves.
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
import e4b  # noqa: E402
import e5  # noqa: E402
import e9  # noqa: E402
import e20b_robust_tests as e20b  # noqa: E402
from e11_stats import GTG_DIRS, cstats  # noqa: E402

assert e4.F1_MODE == "length"
DEV = "cpu"
CITIES = e4.ALL
S = e4.SHORT
I_LEN, I_FFT, I_MAJOR, I_NSEG, I_PS = 0, 1, 2, 3, 4


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (float, np.floating)):
        return round(float(x), 6)
    if isinstance(x, (np.integer, np.bool_)):
        return int(x)
    return x


def save(part, res):
    (diag.OUT / f"E21_{part}{e4.TAG}.json").write_text(json.dumps(clean(res), indent=1))


def tops(P, score):
    pi, n = P["pair_i"], int(P["n"])
    order = np.lexsort((-score, pi))
    first = np.r_[True, pi[order][1:] != pi[order][:-1]]
    top = order[first]
    out = np.full(n, -1, np.int64)
    out[pi[top]] = top
    assert (out >= 0).all()
    return out


def len_index(P):
    out = np.full(int(P["n"]), -1, np.int64)
    k0 = np.nonzero(P["pair_k"] == 0)[0]
    out[P["pair_i"][k0]] = k0
    assert (out >= 0).all()
    return out


def decision(fa, fb, same):
    """Rule a against rule b per trip: mean(fa - fb) = P(differ) * E[fa - fb | differ], exactly."""
    d, dis = fa - fb, ~same
    cond = float(d[dis].mean()) if dis.any() else 0.0
    return {"gap": d.mean(), "differ": dis.mean(), "cond_gain": cond, "abs_cond": np.abs(d[dis]).mean() if dis.any() else 0.0,
            "a_better": (d[dis] > 1e-9).mean() if dis.any() else 0.0, "b_better": (d[dis] < -1e-9).mean() if dis.any() else 0.0,
            "identity_err": abs(d.mean() - dis.mean() * cond)}


def od_of(P, n_seg):
    n = int(P["n"])
    o = P["gt_flat"][P["gt_off"][:-1]].astype(np.int64)
    d = P["gt_flat"][P["gt_off"][1:] - 1].astype(np.int64)
    assert len(o) == n
    return o * n_seg + d


def load(cities=CITIES):
    return {c: e4.load_pool(c, "test") for c in cities}, {c: e4.load_pool(c, "train") for c in cities}


def pooled_of(Ptr, t):
    srcs = [s for s in CITIES if s != t]
    Q = e4b.concat([Ptr[s] for s in srcs])
    Q["hour"] = np.concatenate([Ptr[s]["hour"] for s in srcs])
    Q["min_len"] = np.concatenate([Ptr[s]["min_len"] for s in srcs])
    return Q


def composite(P, mode):
    X = P["X"]
    c = X[:, I_LEN] + X[:, I_FFT] + X[:, I_NSEG]
    Q = dict(P)
    Q["X"] = {"1p": c[:, None], "3p": np.c_[c, X[:, I_MAJOR], X[:, I_PS]]}[mode]
    return Q


# ----------------------------------------------------------------------------- A
def part_a():
    Pt, Ptr = load()
    single = {c: e4.fit_logit(Ptr[c], DEV) for c in CITIES}
    res = {"features": e4.FEATS, "targets": {}}
    for t in CITIES:
        t0 = time.time()
        P, Q = Pt[t], Ptr[t]
        n_tr, n_te = int(Q["n"]), int(P["n"])
        b_T = single[t]
        b_mle, _, _, info = e20b.mle(Q, b_T)
        H = info / n_tr
        D = np.sqrt(np.diag(H))
        C = H / np.outer(D, D)
        ev, V = np.linalg.eigh(C)
        pooled = pooled_of(Ptr, t)
        srcs = {"pooled": e4.fit_logit(pooled, DEV)}
        srcs.update({S[s]: single[s] for s in CITIES if s != t})
        ll_u_te = e20b.parts(P, np.zeros(5))[0]
        ll_T_te = e20b.parts(P, b_T)[0]
        ll_T_tr = e20b.parts(Q, b_mle)[0]
        top_T = tops(P, P["X"] @ b_T)
        f = P["f1"]
        lenx = len_index(P)
        tr = {"beta_T": b_T, "H_corr": C, "H_corr_eig": ev,
              "cost_sum_T": b_T[[I_LEN, I_FFT, I_NSEG]].sum(),
              "n_cand_mean": np.bincount(P["pair_i"], minlength=n_te).mean(),
              "share_single_candidate": (np.bincount(P["pair_i"], minlength=n_te) == 1).mean(),
              "sources": {}}
        for name, b_S in srcs.items():
            top_S = tops(P, P["X"] @ b_S)
            dec = decision(f[top_T], f[top_S], top_T == top_S)
            d = b_S - b_mle
            actual = (ll_T_tr - e20b.parts(Q, b_S)[0]) / n_tr
            quad = 0.5 * d @ H @ d
            ortho = 0.5 * float((d ** 2 * np.diag(H)).sum())
            dt = D * d
            coords = V.T @ dt
            # 1-D scale refit on target train
            s = 1.0
            for _ in range(40):
                _, sc, inf = e20b.parts(Q, s * b_S)
                g, h = b_S @ sc.sum(0), b_S @ inf @ b_S
                step = g / max(h, 1e-12)
                s += step
                if abs(step) < 1e-8:
                    break
            actual_s = (ll_T_tr - e20b.parts(Q, s * b_S)[0]) / n_tr
            ll_S_te = e20b.parts(P, b_S)[0]
            r = {"beta_S": b_S, "cost_sum_S": b_S[[I_LEN, I_FFT, I_NSEG]].sum(), "decision": dec,
                 "ti_test": (ll_S_te - ll_u_te) / (ll_T_te - ll_u_te),
                 "dnll_test": (ll_T_te - ll_S_te) / n_te,
                 "dnll_train_actual": actual, "dnll_train_quad": quad, "dnll_train_ortho": ortho,
                 "quad_over_ortho": quad / max(ortho, 1e-12),
                 "delta_std_coords_on_eig": coords, "cos_fisher": (b_S @ H @ b_mle) / np.sqrt((b_S @ H @ b_S) * (b_mle @ H @ b_mle)),
                 "scale_star": s, "dnll_train_after_scale": actual_s,
                 "scale_share_of_loss": 1.0 - actual_s / max(actual, 1e-12)}
            tr["sources"][name] = r
        # selection potential and restricted models (pooled source)
        best = P["best"]
        tr["selection"] = {"len": f[lenx].mean(), "oracle": best.mean(), "in_logit": f[top_T].mean(),
                           "zs_logit": f[tops(P, P["X"] @ srcs["pooled"])].mean()}
        zs = tops(P, P["X"] @ srcs["pooled"])
        tr["zs_vs_len"] = decision(f[zs], f[lenx], zs == lenx)
        tr["in_vs_len"] = decision(f[top_T], f[lenx], top_T == lenx)
        Xc = P["X"][:, I_LEN] + P["X"][:, I_FFT] + P["X"][:, I_NSEG]
        rs = {"0p_composite_rule": {"top1": f[tops(P, -Xc)].mean()}}
        for mode in ("1p", "3p"):
            Rs, Rt, Rq = composite(pooled, mode), composite(P, mode), composite(Q, mode)
            b = e4.fit_logit(Rs, DEV)
            b_in = e4.fit_logit(Rq, DEV)
            ll = e20b.parts(Rt, b)[0]
            ll_in = e20b.parts(Rt, b_in)[0]
            rs[mode] = {"beta": b, "top1": f[tops(Rt, Rt["X"] @ b)].mean(), "top1_in": f[tops(Rt, Rt["X"] @ b_in)].mean(),
                        "ti_vs_in5": (ll - ll_u_te) / (ll_T_te - ll_u_te), "ll_trip": ll / n_te, "ll_trip_in": ll_in / n_te}
        rs["5p"] = {"top1": f[zs].mean(), "top1_in": f[top_T].mean(),
                    "ll_trip": e20b.parts(P, srcs["pooled"])[0] / n_te, "ll_trip_in": ll_T_te / n_te, "ll_trip_uniform": ll_u_te / n_te}
        tr["restricted"] = rs
        res["targets"][t] = tr
        save("A", res)
        diag.log(f"[E21A {t}] done in {time.time() - t0:.0f}s gap={tr['sources']['pooled']['decision']}")
    diag.log("E21A done")


# ----------------------------------------------------------------------------- D
def part_d():
    Pt, Ptr = load()
    res = {}
    NS = [30, 100, 300, 1000, 3000, 10000]
    for t in CITIES:
        P, Q = Pt[t], Ptr[t]
        n_te, n_tr = int(P["n"]), int(Q["n"])
        b_in = e4.fit_logit(Q, DEV)
        b_mle, _, score, info = e20b.mle(Q, b_in)
        od = Q["od_key"].astype(np.int64)
        H = info / n_tr
        Hi = np.linalg.inv(H)
        d_iid = np.trace(Hi @ (score.T @ score / n_tr))
        _, g = np.unique(od, return_inverse=True)
        G = int(g.max()) + 1
        Sg = np.zeros((G, score.shape[1]))
        np.add.at(Sg, g, score)
        d_cl = np.trace(Hi @ (Sg.T @ Sg / n_tr))
        pooled = pooled_of(Ptr, t)
        b_zs = e4.fit_logit(pooled, DEV)
        ll_in, ll_zs = e20b.parts(P, b_in)[0] / n_te, e20b.parts(P, b_zs)[0] / n_te
        f = P["f1"]
        top_in, top_zs = f[tops(P, P["X"] @ b_in)].mean(), f[tops(P, P["X"] @ b_zs)].mean()
        dT = ll_in - ll_zs
        r = {"d_iid": d_iid, "d_cluster": d_cl, "ods_train": G, "dnll_transfer_test": dT, "ll_in": ll_in, "ll_zs": ll_zs,
             "top1_in": top_in, "top1_zs": top_zs,
             "n_star_pred_iid": d_iid / (2 * dT) if dT > 0 else None, "n_star_pred_cluster": d_cl / (2 * dT) if dT > 0 else None,
             "curve": []}
        for n in NS:
            lls, t1 = [], []
            for seed in range(10):
                idx = np.random.default_rng(seed).choice(n_tr, n, replace=False)
                b = e4.fit_logit(e5.take(Q, idx), DEV, iters=400)
                lls.append(e20b.parts(P, b)[0] / n_te)
                t1.append(f[tops(P, P["X"] @ b)].mean())
            r["curve"].append({"n": n, "ll": float(np.mean(lls)), "ll_range": [min(lls), max(lls)], "excess": ll_in - float(np.mean(lls)),
                               "excess_pred_iid": d_iid / (2 * n), "top1": float(np.mean(t1)), "top1_range": [min(t1), max(t1)]})
            diag.log(f"[E21D {t}] n={n} " + json.dumps(clean(r["curve"][-1])))
        res[t] = r
        save("D", res)
    diag.log("E21D done")


# ----------------------------------------------------------------------------- B
def csv_routes(fcsv, ids):
    gen = pd.read_csv(fcsv).drop_duplicates("traj_id").set_index("traj_id").reindex(ids)
    return [np.fromstring(str(s), dtype=np.int64, sep=",") if isinstance(s, str) else np.zeros(0, np.int64) for s in gen["rid_list"]]


def wf1(routes, gts, L):
    wl = e9.wlcs_batch_gpu(routes, gts, L, DEV)
    rl = np.array([L[x].sum() for x in routes])
    gl = np.array([L[x].sum() for x in gts])
    p_, r_ = wl / np.maximum(rl, 1e-9), wl / np.maximum(gl, 1e-9)
    return 2 * p_ * r_ / np.maximum(p_ + r_, 1e-9)


def pair_route(P, j):
    return P["rflat"][P["roff"][j]:P["roff"][j + 1]].astype(np.int64)


def candidate_keys(P):
    keys = [set() for _ in range(int(P["n"]))]
    for j, i in enumerate(P["pair_i"]):
        keys[i].add(pair_route(P, j).tobytes())
    return keys


def attr(routes, L, major):
    ln = np.array([L[r].sum() for r in routes])
    mj = np.array([(L[r] * major[r]).sum() for r in routes])
    return ln, mj


def route_anatomy(tag, routes, P, gts, meta, top_logit, keys, lenx):
    L, major = meta["length"], meta["major"].astype(float)
    f_g = wf1(routes, gts, L)
    f = P["f1"]
    f_len, f_log = f[lenx], f[top_logit]
    n = int(P["n"])
    key_len = [pair_route(P, j).tobytes() for j in lenx]
    g_eqlen = np.array([routes[i].tobytes() == key_len[i] for i in range(n)])
    g_inset = np.array([routes[i].tobytes() in keys[i] for i in range(n)])
    l_eqlen = top_logit == lenx
    same = np.array([routes[i].tobytes() == pair_route(P, top_logit[i]).tobytes() for i in range(n)])
    out = {"tag": tag, "n": n, "missing": int(sum(len(r) == 0 for r in routes)),
           "f1": {"gen": f_g.mean(), "len": f_len.mean(), "logit": f_log.mean()},
           "gen_eq_len": g_eqlen.mean(), "gen_in_map6": g_inset.mean(), "logit_eq_len": l_eqlen.mean(), "gen_eq_logit": same.mean(),
           "gen_vs_len": decision(f_g, f_len, g_eqlen), "logit_vs_len": decision(f_log, f_len, l_eqlen),
           "logit_vs_gen": decision(f_log, f_g, same),
           "tail_f1_lt_0.5": {"gen": (f_g < 0.5).mean(), "len": (f_len < 0.5).mean(), "logit": (f_log < 0.5).mean()},
           "tail_f1_lt_0.3": {"gen": (f_g < 0.3).mean(), "len": (f_len < 0.3).mean(), "logit": (f_log < 0.3).mean()}}
    # four cells of (generator deviates from LEN, logit deviates from LEN); contributions sum to logit - generator
    cells = {}
    for a, an in ((True, "gen_len"), (False, "gen_dev")):
        for b, bn in ((True, "log_len"), (False, "log_dev")):
            m = (g_eqlen == a) & (l_eqlen == b)
            cells[f"{an}&{bn}"] = {"share": m.mean(), "contrib": ((f_log - f_g) * m).mean(), "mean_diff": (f_log - f_g)[m].mean() if m.any() else 0.0}
    out["cells"] = cells
    # deviations inside vs outside the candidate set
    for nm, m in (("dev_in_map6", (~g_eqlen) & g_inset), ("dev_out_map6", (~g_eqlen) & ~g_inset)):
        out[nm] = {"share": m.mean(), "gain_over_len": (f_g - f_len)[m].mean() if m.any() else 0.0, "contrib_over_len": ((f_g - f_len) * m).mean()}
    # do the routes move in the observed direction relative to the shortest path?
    ln_len = np.array([L[pair_route(P, j)].sum() for j in lenx])
    mj_len = np.array([(L[pair_route(P, j)] * major[pair_route(P, j)]).sum() for j in lenx])
    ln_gt, mj_gt = attr(gts, L, major)
    ln_g, mj_g = attr(routes, L, major)
    rt_l = [pair_route(P, j) for j in top_logit]
    ln_l, mj_l = attr(rt_l, L, major)
    out["attrs"] = {"detour_mean": {"gt": (ln_gt / ln_len).mean(), "gen": (ln_g / ln_len).mean(), "logit": (ln_l / ln_len).mean()},
                    "major_share": {"gt": mj_gt.sum() / ln_gt.sum(), "gen": mj_g.sum() / ln_g.sum(), "logit": mj_l.sum() / ln_l.sum(), "len": mj_len.sum() / ln_len.sum()}}
    # among trips where generator deviates: does it move toward the observed major-road use / detour?
    dev = ~g_eqlen
    dgt, dgen = (ln_gt / ln_len) - 1, (ln_g / ln_len) - 1
    out["detour_corr_on_dev"] = np.corrcoef(dgt[dev], dgen[dev])[0, 1] if dev.sum() > 10 else None
    ldev = ~l_eqlen
    dlog = (ln_l / ln_len) - 1
    out["detour_corr_on_logit_dev"] = np.corrcoef(dgt[ldev], dlog[ldev])[0, 1] if ldev.sum() > 10 else None
    return out


def part_b():
    res = {"runs": {}}
    cache = {}
    betas = {}
    def ctx(trg):
        if trg in cache:
            return cache[trg]
        P = e4.load_pool(trg, "test")
        A = diag.stage_a(trg, 8)
        rows = e4.subset(trg, "test", A, 100000)
        assert len(rows) == int(P["n"]) and np.array_equal(np.diff(A["gt_off"])[rows], np.diff(P["gt_off"]))
        ids = A["traj_id"][rows]
        gts = [np.asarray(g, np.int64) for g in diag.ragged_take(A["gt_flat"], A["gt_off"], rows)]
        meta = e4.road_meta(trg, 16)
        cache[trg] = (P, ids, gts, meta, candidate_keys(P), len_index(P))
        return cache[trg]

    def beta(c):
        if c not in betas:
            betas[c] = e4.fit_logit(e4.load_pool(c, "train"), DEV)
        return betas[c]

    runs = [(s, t, "main", diag.OUT / f"gtg/gen_{s}2{t}.csv") for s, t in GTG_DIRS]
    for tag in ("exp_102", "exp_103"):
        for s, t in GTG_DIRS:
            f = diag.OUT / f"gtg/seeds/{tag}/gen_{s}2{t}.csv"
            if f.exists():
                runs.append((s, t, tag, f))
    for c in ("xian", "chengdu"):
        runs.append((c, c, "incity", diag.OUT / f"gtg/samecity/gen_{c}2{c}.csv"))
    for s, t, tag, f in runs:
        t0 = time.time()
        P, ids, gts, meta, keys, lenx = ctx(t)
        top = tops(P, P["X"] @ beta(s))
        routes = csv_routes(f, ids)
        o = route_anatomy(f"{S[s]}->{S[t]}/{tag}", routes, P, gts, meta, top, keys, lenx)
        res["runs"][o["tag"]] = o
        save("B", res)
        diag.log(f"[E21B] {o['tag']} {time.time() - t0:.0f}s f1={clean(o['f1'])} eqlen={o['gen_eq_len']:.3f} inset={o['gen_in_map6']:.3f} "
                 f"gen_dev={clean(o['gen_vs_len'])} logit_dev={clean(o['logit_vs_len'])}")
    diag.log("E21B done")


# ----------------------------------------------------------------------------- C
def part_c():
    from pathlib import Path
    Pt, Ptr = load()
    HOSER_OUT = Path(os.environ["HOSER_DIR"]) / "gene_eval"
    res = {}
    for c in CITIES:
        t0 = time.time()
        P, Q = Pt[c], Ptr[c]
        meta = e4.road_meta(c, 16)
        n_seg = meta["n_seg"]
        n = int(P["n"])
        pi, f = P["pair_i"], P["f1"]
        odt = P["od_key"].astype(np.int64)
        assert np.array_equal(od_of(P, n_seg), odt)
        seen100 = np.isin(odt, Q["od_key"].astype(np.int64))
        A_tr = e4.load_split(c, "train", 8)
        o = A_tr["gt_flat"][A_tr["gt_off"][:-1]].astype(np.int64)
        d = A_tr["gt_flat"][A_tr["gt_off"][1:] - 1].astype(np.int64)
        seen_all = np.isin(odt, o * n_seg + d)
        b_in = e4.fit_logit(Q, DEV)
        top_l = tops(P, P["X"] @ b_in)
        g_in = e9.fit_gbdt(e9.design(Q), Q["label"])
        top_g = tops(P, g_in.predict_proba(e9.design(P))[:, 1])
        g5 = e9.fit_gbdt(Q["X"], Q["label"])
        top_g5 = tops(P, g5.predict_proba(P["X"])[:, 1])
        pooled = pooled_of(Ptr, c)
        b_zs = e4.fit_logit(pooled, DEV)
        top_zl = tops(P, P["X"] @ b_zs)
        g_zs = e9.fit_gbdt(e4b.design_noregion(pooled), pooled["label"])
        top_zg = tops(P, g_zs.predict_proba(e9.design(P))[:, 1])
        fl, fg, fg5, fzl, fzg = f[top_l], f[top_g], f[top_g5], f[top_zl], f[top_zg]
        r = {"share_seen_100k": seen100.mean(), "share_seen_all": seen_all.mean(),
             "gbdt_in_vs_logit_in": decision(fg, fl, top_g == top_l), "gbdt_zs_vs_logit_zs": decision(fzg, fzl, top_zg == top_zl),
             "gbdt5_in_vs_logit_in": decision(fg5, fl, top_g5 == top_l)}

        def strat(name, x, seen):
            out = {}
            for nm, m in (("seen", seen), ("unseen", ~seen)):
                out[nm] = {"share": m.mean(), "gain": x[m].mean() if m.any() else None, "contrib": (x * m).mean()}
                if m.sum() > 200:
                    st = cstats(x[m], odt[m])
                    out[nm]["ci95"] = st["ci95"]
                    out[nm]["clusters"] = st["clusters"]
            return out

        r["gbdt_gain_by_od"] = strat("g", fg - fl, seen100)
        r["gbdt5_gain_by_od"] = strat("g5", fg5 - fl, seen100)
        r["transfer_gap_by_od"] = strat("t", fl - fzl, seen100)
        if c in ("porto", "sanfrancisco"):
            pick = np.load(diag.OUT / f"hoser/pick_{c}.npy")
            hrow = np.load(diag.OUT / f"hoser/rows_{c}.npy")
            gall = diag.ragged_take(P["gt_flat"], P["gt_off"], range(n))
            gts = [np.asarray(gall[i], np.int64) for i in pick]
            gen = pd.read_csv(HOSER_OUT / f"gen_{c}.csv").set_index("test_row").reindex(hrow)
            hoser = [np.array(json.loads(s), np.int64) if isinstance(s, str) else np.zeros(0, np.int64) for s in gen["gene_trace_road_id"]]
            fh = wf1(hoser, gts, meta["length"])
            keys = candidate_keys(P)
            inset = np.array([hoser[k].tobytes() in keys[i] for k, i in enumerate(pick)])
            lenx = len_index(P)
            key_len = [pair_route(P, lenx[i]).tobytes() for i in pick]
            eqlen = np.array([hoser[k].tobytes() == key_len[k] for k in range(len(pick))])
            sub_seen = seen_all[pick]
            fl_s, fg_s = fl[pick], fg[pick]
            x = fh - fl_s
            best = P["best"][pick]
            h = {"n": len(pick), "f1": fh.mean(), "logit": fl_s.mean(), "gbdt": fg_s.mean(), "share_in_map6": inset.mean(), "share_eq_len": eqlen.mean(),
                 "share_seen_all": sub_seen.mean(),
                 "hoser_vs_oracle6": (fh - best).mean()}
            for nm, m in (("seen", sub_seen), ("unseen", ~sub_seen)):
                h[nm] = {"share": m.mean(), "gain_logit": x[m].mean(), "gain_gbdt": (fh - fg_s)[m].mean(), "f1": fh[m].mean(), "logit_f1": fl_s[m].mean()}
            for nm, m in (("in_map6", inset), ("out_map6", ~inset)):
                h[nm] = {"share": m.mean(), "gain_logit": x[m].mean(), "contrib": (x * m).mean(), "f1": fh[m].mean(), "oracle6": best[m].mean(), "logit_f1": fl_s[m].mean()}
            r["hoser"] = h
        res[c] = r
        save("C", res)
        diag.log(f"[E21C {c}] {time.time() - t0:.0f}s " + json.dumps(clean({k: r[k] for k in ("share_seen_100k", "share_seen_all", "gbdt_gain_by_od")})))
    diag.log("E21C done")



# ----------------------------------------------------------------------------- E
def part_e():
    """Parameter-free composite rule (no coefficient estimated from any trajectories), effective dimension, weight plateau."""
    from math import sqrt
    import gtg_eval
    from e11_stats import holm, phi_sf
    Pt, Ptr = load()
    res = {"cities": {}, "gtg": {}}
    fam = {}
    betas_src = {c: e4.fit_logit(Ptr[c], DEV) for c in CITIES}
    grid = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]
    rule_routes = {}
    for c in CITIES:
        t0 = time.time()
        P, Q = Pt[c], Ptr[c]
        meta = e4.road_meta(c, 16)
        n, pi, f, od, X = int(P["n"]), P["pair_i"], P["f1"], P["od_key"].astype(np.int64), P["X"]
        _, gt_seg = diag.flatten_rows(P["gt_flat"], P["gt_off"], np.arange(n))
        gt_flow = np.bincount(gt_seg, minlength=meta["n_seg"]).astype(float)
        mask = gt_flow >= 5
        def ev(p):
            r = e4.evaluate_probs(P, p, meta, gt_flow, mask)
            return {"top1": r["top1_f1"], "flow_nmae": r["flow_nmae"]}
        Xc = X[:, I_LEN] + X[:, I_FFT] + X[:, I_NSEG]
        pooled = pooled_of(Ptr, c)
        b5z, b5i = e4.fit_logit(pooled, DEV), betas_src[c]
        top0 = tops(P, -Xc)
        lenx = len_index(P)
        f0, f5z, f5i, fl = f[top0], f[tops(P, X @ b5z)], f[tops(P, X @ b5i)], f[lenx]
        r = {"top1": {"len": fl.mean(), "rule0": f0.mean(), "logit5_zs": f5z.mean(), "logit5_in": f5i.mean(), "oracle6": P["best"].mean()},
             "rule0_vs_len": decision(f0, fl, top0 == lenx), "rule0_minus_zs5": cstats(f0 - f5z, od), "rule0_minus_in5": cstats(f0 - f5i, od),
             "rule0_minus_len": cstats(f0 - fl, od), "same_pick_rule0_zs5": (top0 == tops(P, X @ b5z)).mean()}
        models = {"5p": (b5z, b5i, P, pooled, Q)}
        for mode in ("1p", "3p"):
            Rs, Rt, Rq = composite(pooled, mode), composite(P, mode), composite(Q, mode)
            models[mode] = (e4.fit_logit(Rs, DEV), e4.fit_logit(Rq, DEV), Rt, Rs, Rq)
        ll_u = e20b.parts(P, np.zeros(5))[0]
        for mode, (bz, bi, Rt, _, _) in models.items():
            lz, li = e20b.parts(Rt, bz)[0], e20b.parts(Rt, bi)[0]
            pz, pi_ = e4.softmax_by_traj(Rt["X"] @ bz, pi, n), e4.softmax_by_traj(Rt["X"] @ bi, pi, n)
            r[mode] = {"beta_zs": bz, "beta_in": bi, "ll_zs": lz / n, "ll_in": li / n, "ti_own": (lz - ll_u) / (li - ll_u), "zs": ev(pz), "in": ev(pi_)}
        r["uniform"] = ev(e4.softmax_by_traj(np.zeros(len(pi)), pi, n))
        r["ll_uniform"] = ll_u / n
        vals = [float(f[tops(P, -(a * X[:, I_LEN] + b * X[:, I_FFT] + w * X[:, I_NSEG]))].mean()) for a, b, w in grid]
        eq = float(f0.mean())
        v = np.array(vals)
        def simplex(b):
            s = b[[I_LEN, I_FFT, I_NSEG]]
            return (s / s.sum()).tolist()
        r["simplex"] = {"grid": grid, "top1": vals, "equal": eq, "best": v.max(), "worst": v.min(), "best_weights": grid[int(v.argmax())],
                        "share_within_0.005_of_best": (v >= v.max() - 0.005).mean(), "share_within_0.01_of_best": (v >= v.max() - 0.01).mean(),
                        "corners": {"len": vals[grid.index((1.0, 0.0, 0.0))], "fft": vals[grid.index((0.0, 1.0, 0.0))], "nseg": vals[grid.index((0.0, 0.0, 1.0))]},
                        "w_in": simplex(b5i), "w_zs": simplex(b5z)}
        res["cities"][c] = r
        rule_routes[c] = [pair_route(P, j) for j in top0]
        save("E", res)
        diag.log(f"[E21E {c}] {time.time() - t0:.0f}s " + json.dumps(clean({k: r[k] for k in ("top1", "rule0_minus_zs5")})))
    # GTG
    runs = [(s, t, "main", diag.OUT / f"gtg/gen_{s}2{t}.csv") for s, t in GTG_DIRS]
    for tag in ("exp_102", "exp_103"):
        for s, t in GTG_DIRS:
            f_ = diag.OUT / f"gtg/seeds/{tag}/gen_{s}2{t}.csv"
            if f_.exists():
                runs.append((s, t, tag, f_))
    for c in ("xian", "chengdu"):
        runs.append((c, c, "incity", diag.OUT / f"gtg/samecity/gen_{c}2{c}.csv"))
    ctx = {}
    for s, t, tag, fcsv in runs:
        if t not in ctx:
            P = Pt[t]
            A = diag.stage_a(t, 8)
            rows = e4.subset(t, "test", A, 100000)
            assert len(rows) == int(P["n"]) and np.array_equal(np.diff(A["gt_off"])[rows], np.diff(P["gt_off"]))
            gts = [np.asarray(g, np.int64) for g in diag.ragged_take(A["gt_flat"], A["gt_off"], rows)]
            meta = e4.road_meta(t, 16)
            m = gtg_eval.route_metrics(rule_routes[t], gts, meta, gtg_eval.gps_array(t), DEV)
            res.setdefault("rule0_route_metrics", {})[t] = m
            ctx[t] = (P, A["traj_id"][rows], gts, meta)
        P, ids, gts, meta = ctx[t]
        od, f = P["od_key"].astype(np.int64), P["f1"]
        fg = wf1(csv_routes(fcsv, ids), gts, meta["length"])
        Xc = P["X"][:, I_LEN] + P["X"][:, I_FFT] + P["X"][:, I_NSEG]
        f0 = f[tops(P, -Xc)]
        flog = f[tops(P, P["X"] @ betas_src[s])]
        name = f"{S[s]}->{S[t]}/{tag}"
        st = cstats(f0 - fg, od)
        st["p_one_sided"] = phi_sf(st["mean"] / st["se"])
        if tag != "incity":
            fam[name] = st["p_one_sided"]
        lenx = len_index(P)
        fl = f[lenx]
        key_len = [pair_route(P, j).tobytes() for j in lenx]
        rts = csv_routes(fcsv, ids)
        dev = np.array([rts[i].tobytes() != key_len[i] for i in range(int(P["n"]))])
        dv = cstats((fg - fl)[dev], od[dev])
        res["gtg"][name] = {"rule0_minus_gtg": st, "f1_gtg": fg.mean(), "f1_rule0": f0.mean(), "f1_logit": flog.mean(),
                            "gtg_dev_value": dv, "gtg_dev_share": dev.mean()}
        save("E", res)
        diag.log(f"[E21E] {name} rule0-gtg {st['mean']:+.4f} {st['ci95']} p={st['p_one_sided']:.2e} dev_value {dv['mean']:+.4f} {dv['ci95']}")
    res["holm_rule0_gt_gtg"] = holm(fam)
    save("E", res)
    diag.log("E21E done")



# ----------------------------------------------------------------------------- F
def fast_top1(pi, score, f, starts, gid):
    """Top-1 F1 per trip; pair_i must be sorted. Ties resolve to the first pair of the trip, as in tops()."""
    m = np.maximum.reduceat(score, starts)
    idx = np.nonzero(score >= m[gid])[0]
    _, first = np.unique(gid[idx], return_index=True)
    return f[idx[first]]


def part_f():
    """Checks of Sec. VII: (1) first-order term of the expansion on test trips; (2) rules designed without the target city;
    (3) in-city logit learning curves when whole OD pairs are sampled instead of single trips."""
    from itertools import product
    import gtg_eval
    from e11_stats import holm, phi_sf
    Pt, Ptr = load()
    single = {c: e4.fit_logit(Ptr[c], DEV) for c in CITIES}
    res = {"expansion": {}, "loco": {}, "same_source": {}, "gtg": {}, "cluster_sampling": {}}
    # (1) expansion on target-test trips around the target-train MLE
    for t in CITIES:
        P, Q = Pt[t], Ptr[t]
        n_te = int(P["n"])
        b_mle = e20b.mle(Q, single[t])[0]
        ll_T, sc, info = e20b.parts(P, b_mle)
        g, H_te = sc.sum(0) / n_te, info / n_te
        srcs = {"pooled": e4.fit_logit(pooled_of(Ptr, t), DEV)}
        srcs.update({S[s]: single[s] for s in CITIES if s != t})
        out = {}
        for name, b_S in srcs.items():
            d = b_S - b_mle
            actual = (ll_T - e20b.parts(P, b_S)[0]) / n_te
            first, quad = -float(g @ d), 0.5 * float(d @ H_te @ d)
            out[name] = {"actual": actual, "first": first, "quad": quad, "pred": first + quad}
        res["expansion"][t] = out
        save("F", res)
    diag.log("[E21F] expansion done")
    # (2) rules designed on source cities only
    pools = {}
    for c in CITIES:
        for split, PP in (("train", Ptr[c]), ("test", Pt[c])):
            pi = PP["pair_i"]
            assert (np.diff(pi) >= 0).all()
            starts = np.r_[0, np.nonzero(np.diff(pi))[0] + 1]
            gid = np.cumsum(np.r_[0, (np.diff(pi) > 0).astype(np.int64)])
            pools[(c, split)] = (PP["X"], pi, PP["f1"], starts, gid)
    cache = {}
    def ev(c, split, w):
        k = (c, split, tuple(w))
        if k not in cache:
            X, pi, f, st, gid = pools[(c, split)]
            cache[k] = float(fast_top1(pi, X @ np.asarray(w, float), f, st, gid).mean())
        return cache[k]
    INT = [w for w in product((0, 1, 2), repeat=5) if any(w)]
    assert len(INT) == 242
    TRI = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]
    EQ = np.array([-1.0, -1.0, 0.0, -1.0, 0.0])
    def design(srcs):
        sgn = np.sign(np.mean([single[s] for s in srcs], axis=0))
        fam = {w: sgn * np.array(w, float) for w in INT}
        vals = {w: float(np.mean([ev(s, "train", v) for s in srcs])) for w, v in fam.items()}
        best = max(sorted(vals, key=lambda w: (sum(1 for x in w if x), sum(w))), key=vals.get)
        tri = {w: np.array([-w[0], -w[1], 0.0, -w[2], 0.0]) for w in TRI}
        tvals = {w: float(np.mean([ev(s, "train", v) for s in srcs])) for w, v in tri.items()}
        tbest = max(tvals, key=tvals.get)
        eq_src = float(np.mean([ev(s, "train", EQ) for s in srcs]))
        return {"sign": sgn, "int_best": best, "int_weights": fam[best], "int_source_top1": vals[best], "tri_best": tbest, "tri_weights": tri[tbest],
                "tri_source_top1": tvals[tbest], "equal_source_top1": eq_src, "equal_rank_of_242": 1 + sum(v > vals.get((1, 1, 0, 1, 0), -1) for v in vals.values())}
    for t in CITIES:
        srcs = [s for s in CITIES if s != t]
        d = design(srcs)
        P = Pt[t]
        b5z = e4.fit_logit(pooled_of(Ptr, t), DEV)
        X, f, od = P["X"], P["f1"], P["od_key"].astype(np.int64)
        f_eq, f_i, f_t, f_5z, f_5i = f[tops(P, X @ EQ)], f[tops(P, X @ d["int_weights"])], f[tops(P, X @ d["tri_weights"])], f[tops(P, X @ b5z)], f[tops(P, X @ single[t])]
        res["loco"][t] = {**d, "heldout": {"frozen_equal": f_eq.mean(), "int_design": f_i.mean(), "tri_design": f_t.mean(), "logit5_zs": f_5z.mean(), "logit5_in": f_5i.mean()},
                          "int_minus_frozen": cstats(f_i - f_eq, od), "tri_minus_frozen": cstats(f_t - f_eq, od), "int_minus_zs5": cstats(f_i - f_5z, od), "int_minus_in5": cstats(f_i - f_5i, od)}
        save("F", res)
        diag.log(f"[E21F LOCO {t}] int {d['int_best']} tri {d['tri_best']} heldout " + json.dumps(clean(res["loco"][t]["heldout"])))
    # same information as GTG: rule designed on GTG's single source
    ctx, routes_done, fam = {}, {}, {}
    runs = [(s, t, "main", diag.OUT / f"gtg/gen_{s}2{t}.csv") for s, t in GTG_DIRS]
    for tag in ("exp_102", "exp_103"):
        for s, t in GTG_DIRS:
            f_ = diag.OUT / f"gtg/seeds/{tag}/gen_{s}2{t}.csv"
            if f_.exists():
                runs.append((s, t, tag, f_))
    for s, t, tag, fcsv in runs:
        key = f"{S[s]}->{S[t]}"
        if key not in res["same_source"]:
            res["same_source"][key] = design([s])
        P = Pt[t]
        if t not in ctx:
            A = diag.stage_a(t, 8)
            rows = e4.subset(t, "test", A, 100000)
            assert len(rows) == int(P["n"]) and np.array_equal(np.diff(A["gt_off"])[rows], np.diff(P["gt_off"]))
            ctx[t] = (A["traj_id"][rows], [np.asarray(g, np.int64) for g in diag.ragged_take(A["gt_flat"], A["gt_off"], rows)], e4.road_meta(t, 16), gtg_eval.gps_array(t))
        ids, gts, meta, gps = ctx[t]
        w = res["same_source"][key]["int_weights"]
        top = tops(P, P["X"] @ w)
        f_r = P["f1"][top]
        od = P["od_key"].astype(np.int64)
        fg = wf1(csv_routes(fcsv, ids), gts, meta["length"])
        st = cstats(f_r - fg, od)
        st["p_one_sided"] = phi_sf(st["mean"] / st["se"])
        name = f"{key}/{tag}"
        fam[name] = st["p_one_sided"]
        res["gtg"][name] = {"rule_minus_gtg": st, "f1_rule": f_r.mean(), "f1_gtg": fg.mean()}
        if tag == "main":
            res["same_source"][key]["route_metrics"] = gtg_eval.route_metrics([pair_route(P, j) for j in top], gts, meta, gps, DEV)
        save("F", res)
        diag.log(f"[E21F GTG] {name} rule {f_r.mean():.4f} gtg {fg.mean():.4f} diff {st['mean']:+.4f} {st['ci95']}")
    res["holm_rule_gt_gtg"] = holm(fam)
    save("F", res)
    # (3) learning curves when whole ODs are sampled
    NS = [100, 300, 1000, 3000, 10000]
    for t in CITIES:
        P, Q = Pt[t], Ptr[t]
        n_te = int(P["n"])
        b_in, b_zs = single[t], e4.fit_logit(pooled_of(Ptr, t), DEV)
        ll_in, ll_zs = e20b.parts(P, b_in)[0] / n_te, e20b.parts(P, b_zs)[0] / n_te
        uq, inv, cnt = np.unique(Q["od_key"].astype(np.int64), return_inverse=True, return_counts=True)
        order = np.argsort(inv, kind="stable")
        st_ = np.r_[0, np.cumsum(cnt)[:-1]]
        r = {"ll_in": ll_in, "ll_zs": ll_zs, "ods": len(uq), "max_od_trips": int(cnt.max()), "share_trips_top1pct_ods": float(np.sort(cnt)[::-1][:max(1, len(cnt) // 100)].sum() / cnt.sum()), "curve": []}
        for n in NS:
            lls, real = [], []
            for seed in range(10):
                perm = np.random.default_rng(seed).permutation(len(uq))
                k = int(np.searchsorted(np.cumsum(cnt[perm]), n)) + 1
                idx = np.concatenate([order[st_[c]:st_[c] + cnt[c]] for c in perm[:k]])
                b = e4.fit_logit(e5.take(Q, idx), DEV, iters=400)
                lls.append(e20b.parts(P, b)[0] / n_te)
                real.append(len(idx))
            r["curve"].append({"n": n, "ll_mean": float(np.mean(lls)), "ll_median": float(np.median(lls)), "ll_range": [min(lls), max(lls)], "realized_mean": float(np.mean(real)), "excess": ll_in - float(np.mean(lls))})
            diag.log(f"[E21F cluster {t}] n={n} " + json.dumps(clean(r["curve"][-1])))
        res["cluster_sampling"][t] = r
        save("F", res)
    diag.log("E21F done")


if __name__ == "__main__":
    funcs = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e, "F": part_f}
    for a in sys.argv[1:]:
        funcs[a]()
