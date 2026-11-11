"""E19: prospective count calibration. Counts from earlier days calibrate the model; later days' flows are evaluated.

E8 calibrates on counts from the same test trips whose flows it then scores (on disjoint links): a transductive
reconstruction. Here the test pool is split at the E16 cutoff (start of the day that contains the 75% quantile of
training start times). The C2 calibration of E8 (pooled zero-shot beta + per-sensor-link offsets, lam 1e-3, sensors
on 10% of the segments, major roads first, same three placements) is fitted on counts of the EARLY test trips and
applied to the LATE test trips' candidate sets; flows of the late trips are scored on links without a sensor
(flow >= 5). References on the same late trips and links: zero-shot, the same calibration fitted on the late counts
themselves (transductive, as in E8), and in-city GBDT / OD history fitted on training trips before the cutoff
(prospective) or on the whole 100k training pool (as in Tab. counts).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e4b  # noqa: E402
from e5 import take as take5  # noqa: E402
from e8 import N_PER, e6_sparse  # noqa: E402
from e16_temporal import day  # noqa: E402

RHO, LAM, SEEDS = 0.10, 1e-3, 3


def sub(P, trips):
    """Pair table restricted to the given trips, with their candidate routes and observed routes."""
    trips = np.sort(trips)
    keep = np.zeros(int(P["n"]), bool)
    keep[trips] = True
    m = keep[P["pair_i"]]
    remap = -np.ones(int(P["n"]), np.int64)
    remap[trips] = np.arange(len(trips))
    pairs = np.nonzero(m)[0]
    rf, ro = diag.from_list(diag.ragged_take(P["rflat"], P["roff"], pairs))
    gf, go = diag.from_list(diag.ragged_take(P["gt_flat"], P["gt_off"], trips))
    out = {"X": P["X"][m], "label": P["label"][m], "f1": P["f1"][m], "pair_i": remap[P["pair_i"][m]],
           "n": np.int64(len(trips)), "rflat": rf, "roff": ro, "gt_flat": gf, "gt_off": go}
    for k in ("hour", "min_len", "od_key"):
        out[k] = P[k][trips]
    return out


def main():
    import torch
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    cities = e4.ALL
    Ptr = {c: e4.load_pool(c, "train") for c in cities}
    res = {"rho": RHO, "lam": LAM, "cities": {}}
    for t in cities:
        meta = e4.road_meta(t, 16)
        n_seg = meta["n_seg"]
        P = e4.load_pool(t, "test")
        A_te, A_tr = e4.load_split(t, "test", 8), e4.load_split(t, "train", 8)
        rows_te = e4.subset(t, "test", A_te, 100000)
        rows_tr = e4.subset(t, "train", A_tr, int(Ptr[t]["n"]))
        assert np.array_equal(np.diff(A_te["gt_off"])[rows_te], np.diff(P["gt_off"])), t
        assert np.array_equal(np.diff(A_tr["gt_off"])[rows_tr], np.diff(Ptr[t]["gt_off"])), t
        cut = int(day([np.quantile(A_tr["start_time"], 0.75)], t)[0])
        d_te = day(A_te["start_time"][rows_te], t)
        E, L = sub(P, np.nonzero(d_te < cut)[0]), sub(P, np.nonzero(d_te >= cut)[0])
        early_pool = np.nonzero(day(A_tr["start_time"][rows_tr], t) < cut)[0]

        beta_src = e4.fit_logit(e4b.concat([Ptr[s] for s in cities if s != t]), "cpu")
        nL = int(L["n"])
        p_zs = e4.softmax_by_traj(L["X"] @ beta_src, L["pair_i"], nL)
        refs = {}
        for tag, idx in (("prospective", early_pool), ("full_pool", np.arange(int(Ptr[t]["n"])))):
            S = take5(Ptr[t], idx)
            g = e4b.fit(e4b.design_noregion(S), S["label"])
            sc = g.predict_proba(e4b.design_noregion(L))[:, 1]
            refs[f"gbdt_{tag}"] = ("p", sc / np.bincount(L["pair_i"], weights=sc, minlength=nL)[L["pair_i"]])
            refs[f"odhist_{tag}"] = ("hist", e4.od_history(A_tr, rows_tr[np.sort(idx)], n_seg))

        def design(Q, sens, device):
            nq, pi_np = int(Q["n"]), Q["pair_i"]
            rep, seg = diag.flatten_rows(Q["rflat"], Q["roff"], np.arange(len(pi_np)))
            grep, gseg = diag.flatten_rows(Q["gt_flat"], Q["gt_off"], np.arange(nq))
            per = np.digitize(Q["hour"], diag.PERIOD_EDGES[1:-1])
            pos = np.full(n_seg, -1, np.int64)
            pos[sens] = np.arange(len(sens))
            pr, col = diag.unique_hits(rep, seg, pos)
            H = e6_sparse(col * N_PER + per[pi_np[pr]], pr, (len(sens) * N_PER, len(pi_np)), device)
            R = e6_sparse(pr, col, (len(pi_np), len(sens)), device)
            ti, sc_ = diag.unique_hits(grep, gseg, pos)
            y = np.bincount(sc_ * N_PER + per[ti], minlength=len(sens) * N_PER).astype(np.float32)
            gt_flow = np.bincount(gseg, minlength=n_seg).astype(float)
            return H, R, torch.tensor(y, device=device), gt_flow

        def probs(Q, u):
            nq = int(Q["n"])
            pi = torch.tensor(Q["pair_i"], device=device)
            m = torch.full((nq,), -1e30, device=device).scatter_reduce(0, pi, u, reduce="amax")
            e = torch.exp(u - m[pi])
            return e / torch.zeros(nq, device=device).index_add(0, pi, e)[pi]

        def calibrate(Q, H, R, y, nsens):
            """C2 of E8 (clean counts, lam 1e-3): beta pulled to the source beta + per-sensor-link offsets."""
            X = torch.tensor(Q["X"], dtype=torch.float32, device=device)
            b0 = torch.tensor(beta_src, dtype=torch.float32, device=device)
            beta = b0.clone().requires_grad_(True)
            theta = torch.zeros(nsens, device=device, requires_grad=True)
            opt = torch.optim.Adam([beta, theta], lr=0.02)
            for _ in range(600):
                opt.zero_grad()
                p = probs(Q, X @ beta + torch.sparse.mm(R, theta.unsqueeze(1)).squeeze(1))
                pred = torch.sparse.mm(H, p.unsqueeze(1)).squeeze(1)
                loss = ((pred - y) ** 2 / (y + 10.0)).mean() + 1e-2 * ((beta - b0) ** 2).sum() + LAM * (theta ** 2).sum()
                loss.backward()
                opt.step()
            return beta.detach(), theta.detach()

        def apply(Q, R, beta, theta):
            X = torch.tensor(Q["X"], dtype=torch.float32, device=device)
            with torch.no_grad():
                return probs(Q, X @ beta + torch.sparse.mm(R, theta.unsqueeze(1)).squeeze(1)).cpu().numpy().astype(np.float64)

        rows = []
        for seed in range(SEEDS):
            r2 = np.random.default_rng(1000 + seed)           # same placements as E8
            n_sens = max(4, int(round(RHO * n_seg)))
            major_ids, other_ids = np.where(meta["major"])[0], np.where(~meta["major"])[0]
            sens = r2.choice(major_ids, n_sens, replace=False) if n_sens <= len(major_ids) else \
                np.r_[major_ids, r2.choice(other_ids, n_sens - len(major_ids), replace=False)]
            HE, RE, yE, _ = design(E, sens, device)
            HL, RL, yL, gt_late = design(L, sens, device)
            mask = gt_late >= 5
            mask[sens] = False
            row = {"seed": seed, "n_sensors": int(len(sens)), "eval_links": int(mask.sum())}
            row["zeroshot"] = round(e4.evaluate_probs(L, p_zs, meta, gt_late, mask)["flow_nmae"], 4)
            b, th = calibrate(E, HE, RE, yE, len(sens))
            ev = e4.evaluate_probs(L, apply(L, RL, b, th), meta, gt_late, mask)
            row["counts_prospective"], row["counts_prospective|top1"] = round(ev["flow_nmae"], 4), round(ev["top1_f1"], 4)
            b, th = calibrate(L, HL, RL, yL, len(sens))
            ev = e4.evaluate_probs(L, apply(L, RL, b, th), meta, gt_late, mask)
            row["counts_transductive"], row["counts_transductive|top1"] = round(ev["flow_nmae"], 4), round(ev["top1_f1"], 4)
            for name, (kind, obj) in refs.items():
                if kind == "p":
                    row[name] = round(e4.evaluate_probs(L, obj, meta, gt_late, mask)["flow_nmae"], 4)
                else:
                    row[name] = round(e4.eval_od_history(L, obj, gt_late, mask, meta, p_zs, "cpu")["hybrid_flow_nmae"], 4)
            rows.append(row)
            diag.log(f"[E19 {t}] " + json.dumps(row))
        res["cities"][t] = {"cutoff_day_utc": cut, "n_early": int(E["n"]), "n_late": nL, "rows": rows}
        (diag.OUT / f"E19_temporal_counts{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E19 done")


if __name__ == "__main__":
    main()
