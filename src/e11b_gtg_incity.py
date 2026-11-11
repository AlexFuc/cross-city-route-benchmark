"""E11b: GTG trained in the target city itself (run_gtg_samecity.sh) vs. its zero-shot runs (CPU only).

Separates GTG's model limits from its transfer loss. Per target (Xi'an, Chengdu; same 100k test trips as E11):
  GTG transfer gap    = in-city GTG - zero-shot GTG from source s       (one-sided, > 0; every zero-shot run)
  logit transfer gap  = in-city logit - zero-shot logit from the same s
  gap difference      = GTG gap - logit gap                             (one-sided, > 0)
  in-city GTG - in-city logit, - in-city GBDT, - LEN                    (two-sided)
OD-cluster statistics and Holm correction as in E11.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag  # noqa: E402
import e4  # noqa: E402
import e9  # noqa: E402
from e11_stats import DEV, cstats, holm, phi_sf  # noqa: E402

TARGETS = {"xian": ["chengdu", "beijing"], "chengdu": ["beijing", "xian"]}


def gtg_f1(fcsv, ids, gts, L):
    gen = pd.read_csv(fcsv).drop_duplicates("traj_id").set_index("traj_id").reindex(ids)
    routes = [np.fromstring(str(s), dtype=np.int64, sep=",") if isinstance(s, str) else np.zeros(0, np.int64)
              for s in gen["rid_list"]]
    wl = e9.wlcs_batch_gpu(routes, gts, L, DEV)
    rl = np.array([L[x].sum() for x in routes])
    gl = np.array([L[x].sum() for x in gts])
    p_, r_ = wl / np.maximum(rl, 1e-9), wl / np.maximum(gl, 1e-9)
    return 2 * p_ * r_ / np.maximum(p_ + r_, 1e-9), int(gen["rid_list"].isna().sum())


def main():
    runs = {"main": diag.OUT / "gtg"}
    for tag in ("exp_102", "exp_103"):
        if (diag.OUT / "gtg/seeds" / tag).exists():
            runs[tag] = diag.OUT / "gtg/seeds" / tag
    E9 = json.loads((diag.OUT / f"E9_rigor{e4.TAG}.json").read_text())
    ref_in = json.loads((diag.OUT / "gtg/samecity/GTG_eval.json").read_text())
    betas = {c: e4.fit_logit(e4.load_pool(c, "train"), DEV) for c in e4.ALL}
    res, fam = {"targets": {}}, {}
    for trg, sources in TARGETS.items():
        P = e4.load_pool(trg, "test")
        n, pi, f, od = int(P["n"]), P["pair_i"], P["f1"], P["od_key"]
        A = diag.stage_a(trg, 8)
        rows = e4.subset(trg, "test", A, 100000)
        assert len(rows) == n and np.array_equal(np.diff(A["gt_off"])[rows], np.diff(P["gt_off"])), trg
        ids = A["traj_id"][rows]
        gts = diag.ragged_take(A["gt_flat"], A["gt_off"], rows)
        L = e4.road_meta(trg, 16)["length"]
        T = e4.SHORT[trg]
        gin, miss = gtg_f1(diag.OUT / f"gtg/samecity/gen_{trg}2{trg}.csv", ids, gts, L)
        ref = ref_in[f"{T}->{T}"]
        assert miss == 0 and abs(gin.mean() - ref["GTG_official"]["top1_length"]) < 5e-4, (trg, gin.mean())
        lin = e9.top1_vec(P, e4.softmax_by_traj(P["X"] @ betas[trg], pi, n), f)
        assert abs(lin.mean() - E9[trg]["top1"]["incity_logit"]["length"]) < 1.5e-3, (trg, lin.mean())
        Ptr = e4.load_pool(trg, "train")
        g = e9.fit_gbdt(e9.design(Ptr), Ptr["label"])
        sc = g.predict_proba(e9.design(P))[:, 1]
        gb = e9.top1_vec(P, sc / np.bincount(pi, weights=sc, minlength=n)[pi], f)
        assert abs(gb.mean() - E9[trg]["top1"]["incity_gbdt"]["length"]) < 1.5e-3, (trg, gb.mean())
        ln = np.zeros(n)
        k0 = np.nonzero(P["pair_k"] == 0)[0]
        ln[pi[k0]] = f[k0]
        r = {"gtg_incity": round(float(gin.mean()), 4), "logit_incity": round(float(lin.mean()), 4),
             "gbdt_incity": round(float(gb.mean()), 4), "len": round(float(ln.mean()), 4), "sources": {}}
        for name, x in (("gtg_minus_logit", gin - lin), ("gtg_minus_gbdt", gin - gb), ("gtg_minus_len", gin - ln)):
            s = cstats(x, od)
            s["p_two_sided"] = min(1.0, 2 * phi_sf(abs(s["mean"]) / s["se"]))
            fam[f"{T}/{name}"] = s["p_two_sided"]
            r[name] = s
        for src in sources:
            S = e4.SHORT[src]
            lzs = e9.top1_vec(P, e4.softmax_by_traj(P["X"] @ betas[src], pi, n), f)
            o = {"logit_zeroshot": round(float(lzs.mean()), 4), "runs": {}}
            for rn, rdir in runs.items():
                fcsv = rdir / f"gen_{src}2{trg}.csv"
                if not fcsv.exists():
                    continue
                gzs, _ = gtg_f1(fcsv, ids, gts, L)
                gref = json.loads((rdir / "GTG_eval.json").read_text())[f"{S}->{T}"]["GTG_official"]["top1_length"]
                assert abs(gzs.mean() - gref) < 5e-4, (src, trg, rn, gzs.mean())
                gg, lg = gin - gzs, lin - lzs
                q = {"gtg_zeroshot": round(float(gzs.mean()), 4), "gtg_gap": cstats(gg, od), "logit_gap": cstats(lg, od),
                     "gap_difference": cstats(gg - lg, od)}
                for k in ("gtg_gap", "gap_difference"):
                    q[k]["p_gt0"] = phi_sf(q[k]["mean"] / q[k]["se"])
                    if rn == "main":
                        fam[f"{S}->{T}/{k}"] = q[k]["p_gt0"]
                o["runs"][rn] = q
            r["sources"][S] = o
        res["targets"][T] = r
        diag.log(f"[E11b {T}] " + json.dumps({k: (v if not isinstance(v, dict) else v.get("mean", "")) for k, v in r.items()}))
    for k, v in holm(fam).items():
        key, name = k.split("/")
        if "->" in key:
            s_, t_ = key.split("->")
            res["targets"][t_]["sources"][s_]["runs"]["main"][name]["p_holm"] = v
        else:
            res["targets"][key][name]["p_holm"] = v
    (diag.OUT / f"E11b_gtg_incity{e4.TAG}.json").write_text(json.dumps(res, indent=2))
    diag.log("E11b done")


if __name__ == "__main__":
    main()
