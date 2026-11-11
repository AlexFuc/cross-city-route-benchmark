"""Build every figure, table and in-text number of the ICDE paper from the result JSONs.

Primary metric = length-weighted LCS-F1 (files with the _len tag); nothing here is typed by hand.
Outputs (paper_out/): figs/*.pdf, tables/*.tex, numbers.json, numbers.tex (\\newcommand per number).
"""
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

R = Path(os.environ.get("PA_RESULTS", str(Path(__file__).resolve().parent / "results")))
OUT = Path(os.environ.get("PA_OUT", str(Path(__file__).resolve().parent / "paper_out")))
(OUT / "figs").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
TAG = os.environ.get("PA_TAG", "_XA-CD-BJ-PT-SF_len")   # count-based files only for smoke tests
CITIES = ["xian", "chengdu", "beijing", "porto", "sanfrancisco"]
AB = {"xian": "XA", "chengdu": "CD", "beijing": "BJ", "porto": "PT", "sanfrancisco": "SF"}
NAME = {"xian": "Xi'an", "chengdu": "Chengdu", "beijing": "Beijing", "porto": "Porto", "sanfrancisco": "San Francisco"}
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8f98", "#1f2328", "#5b6068"
COL1, COL2 = 3.5, 7.16

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "lines.linewidth": 1.2,
})


def load(name):
    return json.loads((R / name).read_text())


E4, E4b, E5 = load(f"E4_week1{TAG}.json")["targets"], load(f"E4b_gbdt_transfer{TAG}.json"), load(f"E5_fewshot{TAG}.json")
E7, E8, E9, E10 = load(f"E7_strata{TAG}.json"), load(f"E8_equiv{TAG}.json"), load(f"E9_rigor{TAG}.json"), load(f"E10_len{TAG}.json")
E11 = load(f"E11_stats{TAG}.json")
N = {}  # macro name -> formatted string


def num(key, value, fmt="{:.3f}"):
    N[key] = fmt.format(value) if not isinstance(value, str) else value
    return N[key]


def f3(x):
    return f"{x:.3f}"


def words(k):
    return ["no", "one", "two", "three", "four", "five", "six"][k]


def sg(x, nd=3, plus=True):
    """Signed number for LaTeX: math minus, no '-0.000'."""
    if abs(x) < 0.5 * 10 ** -nd:
        return f"{0:.{nd}f}"
    t = f"{abs(x):.{nd}f}"
    return ("$-$" if x < 0 else ("$+$" if plus else "")) + t


# ---------------------------------------------------------------- consistency checks between files
for c in CITIES if TAG.endswith("_len") else []:
    assert abs(E9[c]["top1"]["zeroshot_pooled"]["length"] - E10[c]["zeroshot"]) < 2e-3, c
    assert abs(E9[c]["BoK_length"] - E10[c]["bok"]) < 2e-3, c
    assert abs(E4[c]["BoK_MAP6"] - E10[c]["bok"]) < 2e-3, (c, E4[c]["BoK_MAP6"], E10[c]["bok"])
    for k9, k10 in (("incity_logit", "incity_logit"), ("incity_gbdt", "incity_gbdt")):
        assert abs(E9[c]["top1"][k9]["length"] - E10[c][k10]) < 2e-3, (c, k9, E9[c]["top1"][k9]["length"], E10[c][k10])

# ---------------------------------------------------------------- Table: datasets
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"City & Segs. & Train & Test & Km & Exact SP & OD rec. \\", r"\midrule"]
for c in CITIES:
    s = E10[c]["stats"]
    lines.append(f"{NAME[c]} ({AB[c]}) & {s['segments']:,} & {s['train_trips'] / 1e3:,.0f}k & {s['test_trips'] / 1e3:,.0f}k & "
                 f"{s['median_trip_km']:.1f} & {s['exact_shortest_share']:.2f} & {s['od_recurrence']:.2f} \\\\")
    num(f"exactSP{AB[c]}", s["exact_shortest_share"], "{:.2f}")
    num(f"odRecur{AB[c]}", s["od_recurrence"], "{:.2f}")
    num(f"odRecurPct{AB[c]}", f"{100 * s['od_recurrence']:.0f}\\%", "{}")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_data.tex").write_text("\n".join(lines) + "\n")
num("totalTripsM", sum(E10[c]["stats"]["train_trips"] + E10[c]["stats"]["test_trips"] for c in CITIES) / 1e6, "{:.1f}")

# ---------------------------------------------------------------- decomposition numbers + table (frequent ODs)
for c in CITIES:
    a, e = AB[c], E10[c]
    num(f"bok{a}", e["bok"]); num(f"zs{a}", e["zeroshot"]); num(f"inLogit{a}", e["incity_logit"])
    num(f"inGbdt{a}", e["incity_gbdt"]); num(f"odh{a}", e["odhist_full"]); num(f"len{a}", E4[c]["LEN_top1"])
    t9 = {k: E9[c]["top1"][k]["length"] for k in ("zeroshot_pooled", "incity_logit", "incity_gbdt")}
    # gaps from E9: same fitted models as the bootstrap CIs below
    num(f"covLoss{a}", 1 - e["bok"]); num(f"gap{a}", t9["incity_logit"] - t9["zeroshot_pooled"])
    num(f"gbdtGain{a}", t9["incity_gbdt"] - t9["incity_logit"])
    # 95% CIs: bootstrap over OD pairs (E11; trips of one OD resampled together), same models as E9
    s11 = E11["cities"][c]
    assert abs(s11["gap"]["mean"] - (t9["incity_logit"] - t9["zeroshot_pooled"])) < 1.5e-3, c
    num(f"gapCI{a}", f"[{sg(s11['gap']['ci95'][0], 4, plus=False)}, {sg(s11['gap']['ci95'][1], 4, plus=False)}]")
    num(f"gainCI{a}", f"[{sg(s11['gain']['ci95'][0], plus=False)}, {sg(s11['gain']['ci95'][1], plus=False)}]")
    fo = e["frequent_od"]
    num(f"loo{a}", fo["loo_ceiling"]); num(f"freqN{a}", f"{fo['n']:,}")
    num(f"freqBok{a}", fo["bok"]); num(f"freqZs{a}", fo["zeroshot"]); num(f"freqGbdt{a}", fo["incity_gbdt"])
    num(f"freqOdh{a}", fo["odhist_full"])
cityhdr = " & ".join(AB[c] for c in CITIES)
ladder_row = lambda lab, f: lab + " & " + " & ".join(f3(f(c)) for c in CITIES) + r" \\"  # noqa: E731
fod = lambda c: E10[c]["frequent_od"]  # noqa: E731
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", f"Top-1 F1 & {cityhdr} \\\\", r"\midrule",
         r"\multicolumn{6}{l}{\emph{All test trips}} \\",
         ladder_row(r"\LEN{}", lambda c: E4[c]["LEN_top1"]),
         ladder_row("Zero-shot logit", lambda c: E10[c]["zeroshot"]),
         ladder_row("In-city logit", lambda c: E10[c]["incity_logit"]),
         ladder_row("In-city GBDT", lambda c: E10[c]["incity_gbdt"]),
         ladder_row("OD history", lambda c: E10[c]["odhist_full"]),
         ladder_row("Oracle@6 (coverage bound)", lambda c: E10[c]["bok"]),
         r"\addlinespace",
         r"\multicolumn{6}{l}{\emph{Frequent ODs ($\geq$5 test trips)}} \\",
         ladder_row("Zero-shot logit", lambda c: fod(c)["zeroshot"]),
         ladder_row("OD history", lambda c: fod(c)["odhist_full"]),
         ladder_row("LOO reference", lambda c: fod(c)["loo_ceiling"]),
         ladder_row("Oracle@6", lambda c: fod(c)["bok"]),
         r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_decomp.tex").write_text("\n".join(lines) + "\n")
# LOO route-repeatability reference: not an upper bound -- OD history from training trips reaches it on frequent ODs
num("looCnMin", min(E10[c]["frequent_od"]["loo_ceiling"] for c in ("xian", "chengdu", "beijing")))
num("looCnMax", max(E10[c]["frequent_od"]["loo_ceiling"] for c in ("xian", "chengdu", "beijing")))
assert all(E10[c]["frequent_od"]["odhist_full"] >= E10[c]["frequent_od"]["loo_ceiling"] - 0.003 for c in CITIES), "OD history reaches the LOO reference"

# ---------------------------------------------------------------- Figure: ladder (top-1 and flow)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL1, 1.95), sharey=True, gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.08})
ys = np.arange(len(CITIES))[::-1]
top_series = [("LEN-SP", lambda c: E4[c]["LEN_top1"], dict(marker="D", mfc="white", mec=GRAY, ms=4.2)),
              ("Zero-shot logit", lambda c: E10[c]["zeroshot"], dict(marker="o", mfc=BLUE, mec="white", mew=0.6, ms=5.5)),
              ("In-city logit", lambda c: E10[c]["incity_logit"], dict(marker="o", mfc="white", mec=BLUE, mew=1.0, ms=5.5)),
              ("In-city GBDT", lambda c: E10[c]["incity_gbdt"], dict(marker="^", mfc=ORANGE, mec="white", mew=0.6, ms=6)),
              ("OD history", lambda c: E10[c]["odhist_full"], dict(marker="s", mfc="white", mec=ORANGE, mew=1.0, ms=4.5)),
              ("Oracle@6", lambda c: E10[c]["bok"], dict(marker="|", mec=INK, mew=1.4, ms=8))]
for y, c in zip(ys, CITIES):
    vals = [f(c) for _, f, _ in top_series]
    ax1.plot([min(vals), max(vals)], [y, y], color="#d7dadf", lw=1.0, zorder=1)
for lab, f, st in top_series:
    ax1.plot([f(c) for c in CITIES], ys, ls="none", label=lab, zorder=3, **st)
ax1.set_xlabel("Top-1 route F1 (length-weighted) $\\rightarrow$")
ax1.set_yticks(ys, [AB[c] for c in CITIES])
ax1.set_xlim(0.48, 0.96)
ax1.grid(axis="x", color="#eceef1", lw=0.5)
flow_series = [("Zero-shot logit", lambda c: E4b[c]["L3_zeroshot_logit_pooled"]["flow_nmae"], top_series[1][2]),
               ("In-city logit", lambda c: E4[c]["M2_incity_logit"]["flow_nmae"], top_series[2][2]),
               ("In-city GBDT", lambda c: E4b[c]["G1_incity_gbdt_noregion"]["flow_nmae"], top_series[3][2]),
               ("OD history", lambda c: E4[c]["M4_od_history"][max(E4[c]["M4_od_history"], key=int)]["hybrid_flow_nmae"],
                top_series[4][2])]
for y, c in zip(ys, CITIES):
    vals = [f(c) for _, f, _ in flow_series]
    ax2.plot([min(vals), max(vals)], [y, y], color="#d7dadf", lw=1.0, zorder=1)
for lab, f, st in flow_series:
    ax2.plot([f(c) for c in CITIES], ys, ls="none", zorder=3, **st)
ax2.set_xlabel("$\\leftarrow$ Link-flow NMAE")
ax2.grid(axis="x", color="#eceef1", lw=0.5)
ax2.tick_params(axis="y", length=0)
h, l_ = ax1.get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, handletextpad=0.2,
           columnspacing=0.9)
fig.savefig(OUT / "figs/fig_ladder.pdf")
plt.close(fig)
for c in CITIES:
    num(f"flowZs{AB[c]}", E4b[c]["L3_zeroshot_logit_pooled"]["flow_nmae"])
    num(f"flowGbdt{AB[c]}", E4b[c]["G1_incity_gbdt_noregion"]["flow_nmae"])
    num(f"flowOdh{AB[c]}", flow_series[3][1](c))

# ---------------------------------------------------------------- Figure: transfer-index matrix
cols = CITIES + ["pooled"]
M = np.full((len(CITIES), len(cols)), np.nan)
for i, t in enumerate(CITIES):
    tr = E9[t]["transfer"]
    for j, s in enumerate(cols):
        k = "zeroshot_pooled" if s == "pooled" else f"zeroshot_from_{AB[s]}"
        if k in tr:
            M[i, j] = tr[k]["TI"]
cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc0ee", BLUE, "#174a8c"])
vmin = 0.7   # drawn as panel (a) of fig_transfer below
for i, t in enumerate(CITIES):
    num(f"tiPooled{AB[t]}", M[i, -1], "{:.2f}")
single = M[:, :-1][~np.isnan(M[:, :-1])]
num("tiSingleMin", single.min(), "{:.2f}"); num("tiSingleMax", single.max(), "{:.2f}")

# ---------------------------------------------------------------- Figure: few-shot small multiples
fig, axes = plt.subplots(2, 5, figsize=(COL2, 2.55), sharex=True)
styles = [("logit_target", "In-city logit", dict(color=BLUE, marker="o", mfc="white", ms=4)),
          ("gbdt_target", "In-city GBDT", dict(color=ORANGE, marker="^", ms=4.5)),
          ("odhist_hybrid", "OD history (+zero-shot fallback)", dict(color=GRAY, marker="s", mfc="white", ms=3.8))]
crossings = {}
for j, c in enumerate(CITIES):
    curves = E5[c]["curves"]
    ns = sorted({r["n_target"] for r in curves})
    for i, metric in enumerate(["top1", "flow_nmae"]):
        ax = axes[i, j]
        zs = E5[c]["zeroshot_logit"][metric]
        ax.axhline(zs, color=BLUE, lw=1.0, ls=(0, (4, 2)), zorder=1)
        for key, lab, st in styles:
            m = np.array([np.mean([r[key][metric] for r in curves if r["n_target"] == n]) for n in ns])
            lo = np.array([np.min([r[key][metric] for r in curves if r["n_target"] == n]) for n in ns])
            hi = np.array([np.max([r[key][metric] for r in curves if r["n_target"] == n]) for n in ns])
            ax.fill_between(ns, lo, hi, color=st["color"], alpha=0.15, lw=0)
            ax.plot(ns, m, label=lab, lw=1.2, **st)
            if metric == "top1" and key == "gbdt_target":
                better = [n for n, v in zip(ns, m) if v > zs]
                crossings[c] = better[0] if better else None
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"$10^{int(np.log10(n))}$" for n in ns])
        ax.grid(axis="y", color="#eceef1", lw=0.5)
        if i == 0:
            ax.set_title(NAME[c])
    axes[0, 0].set_ylabel("Top-1 F1 $\\rightarrow$")
    axes[1, 0].set_ylabel("$\\leftarrow$ Flow NMAE")
for ax in axes[1]:
    ax.set_xlabel("Target trips")
h, l_ = axes[0, 0].get_legend_handles_labels()
h.append(plt.Line2D([], [], color=BLUE, lw=1.0, ls=(0, (4, 2))))
l_.append("Zero-shot logit (0 target trips)")
fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.4)
fig.savefig(OUT / "figs/fig_fewshot.pdf")
plt.close(fig)
for c, n in crossings.items():
    num(f"gbdtCross{AB[c]}", "none" if n is None else f"{n:,}", "{}")

# ---------------------------------------------------------------- Table: strata (transfer gap on hard trips)
STR = [("all", "All trips"), ("deviating(>1.05)", "Detour $>$5\\%"), ("deviating(>1.20)", "Detour $>$20\\%"),
       ("long(>median)", "Long trips")]
share_txt = lambda k: (lambda lo, hi: f"{lo:.2f}" if lo == hi else f"{lo:.2f}--{hi:.2f}")(  # noqa: E731
    min(E7[c][k]["share"] for c in CITIES), max(E7[c][k]["share"] for c in CITIES))
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", f"Top-1 F1 & {cityhdr} & Trips \\\\", r"\midrule",
         r"\multicolumn{7}{l}{\emph{Transfer gap}} \\"]
for k, lab in STR:
    lines.append(lab + " & " + " & ".join(sg(E7[c][k]["transfer_gap"]) for c in CITIES) + f" & {share_txt(k)} \\\\")
lines += [r"\addlinespace", r"\multicolumn{7}{l}{\emph{Local-flexibility gain}} \\",
          "All trips & " + " & ".join(sg(E7[c]["all"]["gbdt_gain"]) for c in CITIES) + r" & 1.00 \\"]
for k, lab in STR[1:3]:  # same hard strata as the transfer gap, so the two are compared like for like
    lines.append(lab + " & " + " & ".join(sg(E7[c][k]["gbdt_gain"]) for c in CITIES) + f" & {share_txt(k)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_strata.tex").write_text("\n".join(lines) + "\n")
for c in CITIES:
    num(f"gapDev{AB[c]}", sg(E7[c]["deviating(>1.20)"]["transfer_gap"]), "{}")
    num(f"shareDev{AB[c]}", E7[c]["deviating(>1.05)"]["share"], "{:.2f}")
# hard strata, same-stratum contrasts (RQ2 "Harder trips", intro F2)
_DEV, _CN3 = ("deviating(>1.05)", "deviating(>1.20)"), ("xian", "chengdu", "beijing")
num("gapDevCnMax", max(abs(E7[c][k]["transfer_gap"]) for c in _CN3 for k in _DEV))
num("gainDevPT", sg(E7["porto"][_DEV[1]]["gbdt_gain"]), "{}")
num("gapStrataAbsMax", max(abs(v["transfer_gap"]) for c in CITIES for v in E7[c].values()))
assert all(E7[c][k]["gbdt_gain"] >= 3 * abs(E7[c][k]["transfer_gap"]) for c in _CN3 for k in _DEV)  # "several times smaller"
assert E7["porto"][_DEV[1]]["transfer_gap"] > E7["porto"][_DEV[1]]["gbdt_gain"]                     # Porto exception
assert all(E7["sanfrancisco"][k]["transfer_gap"] < 0 for k in _DEV)                                 # SF zero-shot ahead
_NS = "near_shortest(<=1.05)"  # gaps of opposite sign offset in the all-trip average
assert E7["porto"][_NS]["transfer_gap"] < 0 < E7["porto"][_DEV[0]]["transfer_gap"]
assert E7["sanfrancisco"][_DEV[0]]["transfer_gap"] < 0 < E7["sanfrancisco"][_NS]["transfer_gap"]

# ---------------------------------------------------------------- Table: counts vs trajectories (rho = 0.10)
def e8mean(c, key, rho=0.1):
    v = [r[key] for r in E8[c] if abs(r["rho"] - rho) < 1e-9 and key in r]
    return float(np.mean(v)) if v else float("nan")


# lower block: flows of the later days only (E19); counts / trajectories of the earlier days, same placements
E19t = load(f"E19_temporal_counts{TAG}.json")["cities"]
e19m = lambda c, k: float(np.mean([x[k] for x in E19t[c]["rows"]]))  # noqa: E731


def bold_if(v, best):
    return rf"\textbf{{{f3(v)}}}" if abs(v - best) < 1e-12 else f3(v)


BASE = [("Zero-shot logit", "zeroshot"), (r"In-city GBDT, $10^5$ trips", "gbdt@100000"), (r"OD history, $10^5$ trips", "odhist@100000")]
best_full = {c: min(e8mean(c, k) for k in [b for _, b in BASE] + ["C2_lam1e-3"]) for c in CITIES}
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", f"Flow NMAE $\\downarrow$ & {cityhdr} \\\\", r"\midrule",
         r"\multicolumn{6}{l}{\emph{Without counts}} \\"]
for lab, key in BASE:
    lines.append(lab + " & " + " & ".join(bold_if(e8mean(c, key), best_full[c]) for c in CITIES) + r" \\")
lines += [r"\addlinespace", r"\multicolumn{6}{l}{\emph{Counts on 10\% of the segments}} \\",
          "Exact counts & " + " & ".join(bold_if(e8mean(c, "C2_lam1e-3"), best_full[c]) for c in CITIES) + r" \\",
          "20\\% noise & " + " & ".join(f3(e8mean(c, "C2_noise0.2")) for c in CITIES) + r" \\",
          "20\\% penetration, rate fitted & " + " & ".join(f3(e8mean(c, "C2_pen0.2")) for c in CITIES) + r" \\",
          r"\addlinespace", r"\multicolumn{6}{l}{\emph{Later days only (last quarter of the test trips)}} \\"]
early = {c: (e19m(c, "counts_prospective"), min(e19m(c, "gbdt_prospective"), e19m(c, "odhist_prospective"))) for c in CITIES}
lines += ["Counts, same later days & " + " & ".join(f3(e19m(c, "counts_transductive")) for c in CITIES) + r" \\",
          "Counts, earlier days & " + " & ".join(bold_if(early[c][0], min(early[c])) for c in CITIES) + r" \\",
          "Best trajectory model, earlier days & " + " & ".join(bold_if(early[c][1], min(early[c])) for c in CITIES) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_counts.tex").write_text("\n".join(lines) + "\n")
# sensors on only 5% of the segments (same placements logic; every method re-scored on that placement's non-sensor segments)
win5 = [c for c in CITIES if e8mean(c, "C2_lam1e-3", 0.05) < min(e8mean(c, "gbdt@100000", 0.05), e8mean(c, "odhist@100000", 0.05))]
assert sorted(win5) == sorted(["beijing", "porto", "sanfrancisco"]), win5
for c in win5:
    num(f"cntFive{AB[c]}", e8mean(c, "C2_lam1e-3", 0.05))
    num(f"cntFiveBest{AB[c]}", min(e8mean(c, "gbdt@100000", 0.05), e8mean(c, "odhist@100000", 0.05)))
for c in CITIES:
    a = AB[c]
    num(f"cntFlow{a}", e8mean(c, "C2_lam1e-3")); num(f"cntZs{a}", e8mean(c, "zeroshot"))
    num(f"cntGfive{a}", e8mean(c, "gbdt@100000")); num(f"cntScale{a}", e8mean(c, "C2_pen0.2|scale_est"), "{:.1f}")

# ---------------------------------------------------------------- Table: recursive logit
RLF = {"chengdu": "RL2_chengdu.json", "xian": "RL2_xian.json", "porto": "RL2_porto_robust.json"}
RLZ = {"porto": "RL2_porto.json"}  # zero-shot RL for Porto comes from the (non-robust) run with source CD
M6 = load("RL2_map6_len.json")
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"& \multicolumn{4}{c}{Top-1 F1 $\uparrow$} & \multicolumn{4}{c}{Flow NMAE $\downarrow$} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"& \multicolumn{2}{c}{RL} & \multicolumn{2}{c}{\MAPsix{}} & \multicolumn{2}{c}{RL} & \multicolumn{2}{c}{\MAPsix{}} \\",
         r"Target (source) & In & ZS & In & ZS & In & ZS & In & ZS \\", r"\midrule"]
for c in ["chengdu", "xian", "porto"]:
    r = load(RLF[c])
    rz = load(RLZ.get(c, RLF[c]))
    evs = [r["incity_eval"], rz["zeroshot_eval"], M6[c]["map6_incity"], M6[c]["map6_zeroshot"]]
    tops, flows = [e["top1_length"] for e in evs], [e["flow_nmae"] for e in evs]
    lines.append(f"{NAME[c]} ({AB[M6[c]['src']]}) & " + " & ".join([bold_if(v, max(tops)) for v in tops] + [bold_if(v, min(flows)) for v in flows]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_rl.tex").write_text("\n".join(lines) + "\n")
for c in ["chengdu", "xian", "porto"]:
    e_, m_ = load(RLF[c])["incity_eval"], M6[c]["map6_incity"]
    num(f"rlTop{AB[c]}", e_["top1_length"]); num(f"mapTop{AB[c]}", m_["top1_length"])
    num(f"rlFlow{AB[c]}", e_["flow_nmae"]); num(f"mapFlow{AB[c]}", m_["flow_nmae"])
# observed-route log-likelihood lost by transferring the recursive logit (Table VII; in-city minus zero-shot, per trip)
rl_gap = {c: load(RLF[c])["incity_eval"]["test_ll_per_trip"] - load(RLZ.get(c, RLF[c]))["zeroshot_eval"]["test_ll_per_trip"]
          for c in ("chengdu", "xian", "porto")}
assert all(v > 0 for v in rl_gap.values()) and rl_gap["porto"] > max(rl_gap["chengdu"], rl_gap["xian"])
num("rlLLGapCnMin", min(rl_gap["chengdu"], rl_gap["xian"]), "{:.2f}"); num("rlLLGapCnMax", max(rl_gap["chengdu"], rl_gap["xian"]), "{:.2f}")
num("rlLLGapPT", rl_gap["porto"], "{:.2f}")

# ---------------------------------------------------------------- Table: RQ1 deep generators
G = load("gtg/GTG_eval.json")
PAPER_HD = {"CD->XA": 0.188, "BJ->XA": 0.187, "XA->BJ": 0.292, "CD->BJ": 0.315, "XA->CD": 0.125, "BJ->CD": 0.117}
ORDER = ["BJ->XA", "CD->XA", "XA->CD", "BJ->CD", "XA->BJ", "CD->BJ"]
lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"& \multicolumn{4}{c}{Top-1 F1 (length-weighted) $\uparrow$} & \multicolumn{3}{c}{Flow NMAE $\downarrow$} & "
         r"\multicolumn{3}{c}{Hausdorff, km $\downarrow$} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Direction & GTG & \LEN & Logit & Logit$-$GTG & GTG & \LEN & Logit & GTG & GTG paper & Logit \\", r"\midrule"]
for i_d, d in enumerate(ORDER):
    lab = d.replace("->", r"$\rightarrow$")
    if i_d and d[-2:] != ORDER[i_d - 1][-2:]:
        lines.append(r"\addlinespace")
    g = G[d]
    tops = [g[m]["top1_length"] for m in ("GTG_official", "LEN_shortest", "zeroshot_logit_same_src")]
    flows = [g[m]["flow_nmae"] for m in ("GTG_official", "LEN_shortest", "zeroshot_logit_same_src")]
    hg, hl = g["GTG_official"]["hausdorff_mean_20k"], g["zeroshot_logit_same_src"]["hausdorff_mean_20k"]
    assert hl < min(hg, PAPER_HD[d]), ("logit Hausdorff below GTG run and GTG paper", d)
    cells = [bold_if(v, max(tops)) for v in tops] + [sg(tops[2] - tops[0])] + [bold_if(v, min(flows)) for v in flows]
    lines.append(f"{lab} & " + " & ".join(cells) + rf" & {f3(hg)} & {PAPER_HD[d]:.3f} & \textbf{{{f3(hl)}}} \\")
    k = d.replace("->", "to").replace("XA", "Xa").replace("CD", "Cd").replace("BJ", "Bj")
    num(f"gtg{k}", g["GTG_official"]["top1_length"]); num(f"lensp{k}", g["LEN_shortest"]["top1_length"])
    num(f"logit{k}", g["zeroshot_logit_same_src"]["top1_length"])
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_rq1.tex").write_text("\n".join(lines) + "\n")
GP = load("gtg/prelim/GTG_eval.json")
num("gtgPrelimXatoBj", GP["XA->BJ"]["GTG_official"]["top1_length"])
num("gtgLenWins", str(sum(G[d]["GTG_official"]["top1_length"] < G[d]["LEN_shortest"]["top1_length"] for d in G)), "{}")
num("nDirs", words(len(G)), "{}")

# ---------------------------------------------------------------- Table: cost (wall clock, one shared machine)
def unit(v):
    return (v, "s") if v < 100 else ((v / 60, "min") if v < 3600 else (v / 3600, "h"))


def rng_txt(vals, fmt=None):
    """Range of wall-clock seconds as 'a--b unit' (one unit if both ends share it)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return r"\emph{pending}"
    lo, hi = min(vals), max(vals)
    if hi < 1:
        return "$<$1\\,s"
    (a, ua), (b, ub) = unit(max(lo, 1)), unit(hi)
    if lo < 1:
        return f"$\\leq${b:.0f}\\,{ub}"
    if f"{a:.0f}{ua}" == f"{b:.0f}{ub}":
        return f"{a:.0f}\\,{ua}"
    return f"{a:.0f}--{b:.0f}\\,{ub}" if ua == ub else f"{a:.0f}\\,{ua}--{b:.0f}\\,{ub}"


sec = None
GC = load("GTG_costs.json")
CO = {c: load(f"COST_{c}.json") for c in CITIES if (R / f"COST_{c}.json").exists()}
HC = load("hoser/HOSER_costs.json") if (R / "hoser/HOSER_costs.json").exists() else {}
cn = [d for d in GC if not d.startswith("BJ")]
rl_fit = [load(f)["incity_fit"]["sec"] for f in ("RL2_chengdu.json", "RL2_xian.json", "RL2_porto_robust.json")]
rows = [
    ("GTG, CN source (per direction)", rng_txt([(GC[d]["disentangle_h"] + GC[d]["preference_h"]) * 3600 for d in cn], sec),
     rng_txt([GC[d]["generate_s"] for d in cn], sec)),
    ("GTG, BJ source (per direction)$^\\dagger$", rng_txt([(GC[d]["disentangle_h"] + GC[d]["preference_h"]) * 3600 for d in GC if d.startswith("BJ")], sec),
     rng_txt([GC[d]["generate_s"] for d in GC if d.startswith("BJ")], sec)),
    ("HOSER, in-city (per city)", rng_txt([HC[c]["train_s"] for c in HC], sec), rng_txt([HC[c]["generate_s_per_1e5"] for c in HC], sec)),
    ("\\MAPsix{} candidates (per city)", "--", rng_txt([CO[c]["candidates_s"] for c in CO], sec)),
    ("Path-size logit", rng_txt([CO[c]["fit_logit_s"] for c in CO], sec), rng_txt([CO[c]["predict_logit_s"] for c in CO])),
    ("GBDT, in-city", rng_txt([CO[c]["fit_gbdt_s"] for c in CO], sec), rng_txt([CO[c]["predict_gbdt_s"] for c in CO], sec)),
    ("OD history, in-city", rng_txt([CO[c]["odhist_build_s"] for c in CO], sec), "$<$1\\,s"),
    ("Recursive logit (5k trips)", rng_txt(rl_fit, sec), "--"),
]
lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Method & Training & Inference ($10^5$ ODs) \\", r"\midrule"]
for i_r, (a, b, c_) in enumerate(rows):
    if i_r == 3:
        lines.append(r"\addlinespace")
    if a == "Path-size logit":
        a, b, c_ = rf"\textbf{{{a}}}", rf"\textbf{{{b}}}", rf"\textbf{{{c_}}}"
    lines.append(f"{a} & {b} & {c_} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_cost.tex").write_text("\n".join(lines) + "\n")
num("hoserTrainRange", rows[2][1], "{}"); num("candTimeRange", rows[3][2], "{}"); num("logitFitRange", rows[4][1], "{}")
gtg_h = [GC[d]["disentangle_h"] + GC[d]["preference_h"] for d in GC]
num("gtgTrainHMin", min(gtg_h), "{:.0f}"); num("gtgTrainHMax", max(gtg_h), "{:.0f}")
num("logitFitSMax", max(CO[c]["fit_logit_s"] for c in CO), "{:.0f}")
num("mapsixCandMin", min(CO[c]["mean_candidates_per_od"] for c in CO), "{:.1f}")
num("mapsixCandMax", max(CO[c]["mean_candidates_per_od"] for c in CO), "{:.1f}")

# ---------------------------------------------------------------- Best-of-K vs distribution (uniform over MAP6)
BF = load(f"BOK_flow{TAG}.json")
for c in CITIES:
    num(f"uniFlow{AB[c]}", BF[c]["uniform_flow"]); num(f"lenFlow{AB[c]}", BF[c]["len_flow"])
num("uniWorseCount", words(sum(BF[c]["uniform_flow"] > BF[c]["len_flow"] for c in CITIES)), "{}")

# ---------------------------------------------------------------- Table: deep in-city generator (HOSER), same 10k trips
HE = load("hoser/HOSER_eval.json") if (R / "hoser/HOSER_eval.json").exists() else {}
HROWS = [("LEN_shortest", "\\LEN{}"), ("zeroshot_logit_pooled", "Zero-shot logit"),
         ("incity_logit", "In-city logit"), ("incity_gbdt", "In-city GBDT"), ("incity_od_history", "OD history"),
         ("HOSER_official", "HOSER (in-city)"), ("oracle6", "Oracle@6 (bound)")]
hc = [c for c in ["porto", "sanfrancisco"] if c in HE]
lines = [r"\begin{tabular}{l" + "rrr" * max(len(hc), 1) + "}", r"\toprule"]
if hc:
    lines.append("& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{NAME[c]}}}" for c in hc) + r" \\")
    lines.append("".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(hc))))
    lines.append("Method & " + " & ".join(["F1", "Flow", "Haus."] * len(hc)) + r" \\")
    lines.append(r"\midrule")
    methods = [k for k, _ in HROWS if k != "oracle6"]   # the bound is not a method, so it is never marked best
    for key, lab in HROWS:
        cells = []
        for c in hc:
            m = HE[c][key]
            for met, pick in (("top1_length", max), ("flow_nmae", min), ("hausdorff_mean", min)):
                best = pick(HE[c][k][met] for k in methods)
                cells.append(bold_if(m[met], best) if key != "oracle6" else f3(m[met]))
        if key == "oracle6":
            lines.append(r"\addlinespace")
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    for c in hc:
        num(f"hoserFone{AB[c]}", HE[c]["HOSER_official"]["top1_length"])
        num(f"hoserFlow{AB[c]}", HE[c]["HOSER_official"]["flow_nmae"])
        num(f"hoserHaus{AB[c]}", HE[c]["HOSER_official"]["hausdorff_mean"])
        num(f"hoserReach{AB[c]}", HE[c]["hoser_reached_destination"], "{:.3f}")
        h, e = HE[c]["HOSER_official"], HE[c]
        num(f"hoserGainLogit{AB[c]}", h["top1_length"] - e["incity_logit"]["top1_length"])
        num(f"hoserGainGbdt{AB[c]}", h["top1_length"] - e["incity_gbdt"]["top1_length"])
        num(f"hoserBelowOracle{AB[c]}", e["oracle6"]["top1_length"] - h["top1_length"])
        others = [e[k]["flow_nmae"] for k in ("LEN_shortest", "zeroshot_logit_pooled", "incity_logit", "incity_gbdt", "incity_od_history", "oracle6")]
        num(f"hoserFlowBestOther{AB[c]}", min(others))
        gap9 = e["incity_logit"]["top1_length"] - e["zeroshot_logit_pooled"]["top1_length"]   # same 10k trips
        num(f"hoserGainOverGap{AB[c]}", (h["top1_length"] - e["incity_logit"]["top1_length"]) / gap9, "{:.0f}")
        num(f"incityLogitTen{AB[c]}", e["incity_logit"]["top1_length"]); num(f"gbdtTen{AB[c]}", e["incity_gbdt"]["top1_length"])
        num(f"odhTen{AB[c]}", e["incity_od_history"]["top1_length"]); num(f"oracleTen{AB[c]}", e["oracle6"]["top1_length"])
else:
    lines.append(r"\multicolumn{4}{c}{\emph{HOSER runs pending}} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_hoser.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: few-shot HOSER vs simple models (same 10k trips)
HF = load("hoser/HOSER_fewshot.json") if (R / "hoser/HOSER_fewshot.json").exists() else {}
FSM = [("logit", "In-city logit"), ("gbdt", "GBDT"), ("od_history", "OD history"), ("HOSER", "HOSER")]
fc = [c for c in ["porto", "sanfrancisco"] if c in HF]
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule"]
if fc:
    NS = ("1000", "10000", "100000")
    lines.append("& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{NAME[c]}}}" for c in fc) + r" \\")
    lines.append("".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(fc))))
    lines.append(r"Local trips & " + " & ".join([r"$10^3$", r"$10^4$", r"$10^5$"] * len(fc)) + r" \\")
    lines.append(r"\midrule")
    for i_m, (met, title, pick) in enumerate((("top1_length", r"Top-1 F1 $\uparrow$", max), ("flow_nmae", r"Flow NMAE $\downarrow$", min))):
        if i_m:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
        for key, lab in FSM:
            cells = [bold_if(HF[c][n][key][met], pick(HF[c][n][k][met] for k, _ in FSM)) for c in fc for n in NS]
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
    for c in fc:
        for n in NS:
            word = {"1000": "Three", "10000": "Four", "100000": "Five"}[n]
            num(f"hoserFs{AB[c]}{word}", HF[c][n]["HOSER"]["top1_length"])
            num(f"gbdtFs{AB[c]}{word}", HF[c][n]["gbdt"]["top1_length"])
else:
    lines.append(r"\multicolumn{8}{c}{\emph{few-shot HOSER runs pending}} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_hoser_fs.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- derived claims (so no number in the prose is typed by hand)


# RQ1 (GTG table)
gd = {d: G[d] for d in ORDER if d in G}
diff_len = [g["LEN_shortest"]["top1_length"] - g["GTG_official"]["top1_length"] for g in gd.values()]
below = [x for x in diff_len if x > 0]
num("gtgBelowLenCount", words(len(below)), "{}")
num("gtgBelowLenMin", min(below) if below else 0.0); num("gtgBelowLenMax", max(below) if below else 0.0)
num("gtgAboveLenCount", words(len(diff_len) - len(below)), "{}")
best = [max(("GTG_official", "LEN_shortest", "zeroshot_logit_same_src"), key=lambda m: g[m]["top1_length"]) for g in gd.values()]
num("logitBestCount", words(sum(b == "zeroshot_logit_same_src" for b in best)), "{}")
num("logitOverGtgCount", words(sum(g["zeroshot_logit_same_src"]["top1_length"] > g["GTG_official"]["top1_length"] for g in gd.values())), "{}")
num("logitOverGtgMax", max(g["zeroshot_logit_same_src"]["top1_length"] - g["GTG_official"]["top1_length"] for g in gd.values()))
num("logitFlowBeatsGtgCount", words(sum(g["zeroshot_logit_same_src"]["flow_nmae"] < g["GTG_official"]["flow_nmae"] for g in gd.values())), "{}")
num("gtgFlowBeatsLenCount", words(sum(g["GTG_official"]["flow_nmae"] < g["LEN_shortest"]["flow_nmae"] for g in gd.values())), "{}")
num("gtgPrelimMinusLenXatoBj", sg(GP["XA->BJ"]["GTG_official"]["top1_length"] - G["XA->BJ"]["LEN_shortest"]["top1_length"]), "{}")
num("gtgFinalMinusPrelimXatoBj", sg(G["XA->BJ"]["GTG_official"]["top1_length"] - GP["XA->BJ"]["GTG_official"]["top1_length"]), "{}")
# Best-of-K
gapsb = [BF[c]["oracle6"] - BF[c]["len_top1"] for c in CITIES]
num("bokOverLenMin", min(gapsb), "{:.2f}"); num("bokOverLenMax", max(gapsb), "{:.2f}")
# exact shortest-path share incl. HOSER-pipeline Beijing control (full test sets)
if (R / "EXACT_SP_control.json").exists():
    EX = load("EXACT_SP_control.json")
    for c, a in list(AB.items()) + [("bjhoser", "BJH")]:
        num(f"exactFull{a.replace('BJH', 'Bjh')}", f"{100 * EX[c]['exact_shortest_share']:.0f}\\%", "{}")
# counts: top-1 change and preference-only refit (C1) at rho = 0.10
E6 = load(f"E6_countcal{TAG}.json")
for c in CITIES:
    num(f"cntTopDelta{AB[c]}", sg(e8mean(c, "C2_lam1e-3|top1") - E10[c]["zeroshot"]), "{}")
    rows6 = [r for r in E6[c]["rows"] if abs(r["rho"] - 0.1) < 1e-9]
    num(f"cOne{AB[c]}", float(np.mean([r["C1"]["nonsensor_nmae"] for r in rows6])))
    num(f"cZero{AB[c]}", float(np.mean([r["C0_zeroshot"]["nonsensor_nmae"] for r in rows6])))
# robustness sentences
sens = [abs(v - E9[c]["gbdt_sensitivity_top1"]["default_leaf63_lr0.1"]) for c in CITIES
        for k, v in E9[c]["gbdt_sensitivity_top1"].items() if not isinstance(v, dict) and not isinstance(E9[c]["gbdt_sensitivity_top1"]["default_leaf63_lr0.1"], dict)]
if sens:
    num("gbdtSensMax", max(sens))
st = [abs(np.mean([r["gbdt_stacked"]["top1"] for r in E5[c]["curves"] if r["n_target"] == n and "gbdt_stacked" in r]) -
          np.mean([r["gbdt_target"]["top1"] for r in E5[c]["curves"] if r["n_target"] == n and "gbdt_target" in r]))
      for c in CITIES for n in (1000, 10000, 100000)]
num("stackedMaxDiff", max(st))
num("freqZsBelowLooMin", min(E10[c]["frequent_od"]["loo_ceiling"] - E10[c]["frequent_od"]["zeroshot"] for c in ["xian", "chengdu", "beijing"]), "{:.2f}")
num("freqZsBelowLooMax", max(E10[c]["frequent_od"]["loo_ceiling"] - E10[c]["frequent_od"]["zeroshot"] for c in ["xian", "chengdu", "beijing"]), "{:.2f}")
num("bokCnMin", min(E10[c]["bok"] for c in ["xian", "chengdu", "beijing"]), "{:.2f}")
num("bokCnMax", max(E10[c]["bok"] for c in ["xian", "chengdu", "beijing"]), "{:.2f}")

ratios = [g9 / t9 for g9, t9 in zip(
    [E9[c]["top1"]["incity_gbdt"]["length"] - E9[c]["top1"]["incity_logit"]["length"] for c in CITIES],
    [E9[c]["top1"]["incity_logit"]["length"] - E9[c]["top1"]["zeroshot_pooled"]["length"] for c in CITIES]) if t9 > 1e-9]
num("gainGapRatioMin", np.floor(min(ratios) * 2) / 2, "{:.1f}")
if HE:
    num("hoserTenGapMax", max(HE[c]["incity_logit"]["top1_length"] - HE[c]["zeroshot_logit_pooled"]["top1_length"] for c in hc))
    num("hoserGainLogitMin", min(HE[c]["HOSER_official"]["top1_length"] - HE[c]["incity_logit"]["top1_length"] for c in hc))
    num("hoserGainLogitMax", max(HE[c]["HOSER_official"]["top1_length"] - HE[c]["incity_logit"]["top1_length"] for c in hc))

for c in CITIES:   # smallest n at which OD history (mean of 3 samples) has lower flow NMAE than GBDT
    cv = E5[c]["curves"]
    ns = sorted({r["n_target"] for r in cv})
    m = lambda k, n: np.mean([r[k]["flow_nmae"] for r in cv if r["n_target"] == n and k in r])
    ok = [m("odhist_hybrid", n) < m("gbdt_target", n) - 0.005 for n in ns]   # material margin
    stay = [n for i, n in enumerate(ns) if all(ok[i:])]          # better from n on, for every larger n
    num(f"odhCross{AB[c]}", "never" if not stay else f"$10^{int(np.log10(stay[0]))}$", "{}")
    t = lambda k, n: np.mean([r[k]["top1"] for r in cv if r["n_target"] == n and k in r])
    okg = [t("gbdt_target", n) > E5[c]["zeroshot_logit"]["top1"] + 0.005 for n in ns]   # material margin
    stayg = [n for i, n in enumerate(ns) if all(okg[i:])]
    N[f"gbdtCrossExp{AB[c]}"] = "never" if not stayg else f"$10^{int(np.log10(stayg[0]))}$"

BT = np.array(load(f"E4_week1{TAG}.json")["betas_train"][c] for c in CITIES) if False else \
    np.array([load(f"E4_week1{TAG}.json")["betas_train"][c] for c in CITIES])
# largest between-city magnitude ratio among the three features whose sign agrees everywhere (length, time, #segments)
num("betaRatioMax", words(int(round(max(np.abs(BT[:, j]).max() / np.abs(BT[:, j]).min() for j in (0, 1, 3))))), "{}")
assert (BT[:, [0, 1, 3]] < 0).all(), "sign claim in RQ2 no longer holds"
assert (BT[[i for i, c in enumerate(CITIES) if c != "beijing"], 2] > 0).all() and BT[CITIES.index("beijing"), 2] < 0, "major-road sign claim"
num("tiMinTTS", min(x["TTS"] for c in CITIES for x in E9[c]["transfer"].values()), "{:,.0f}")

# GTG repeat runs (official code sets no seed): main run + exp_102 + exp_103
SEEDS = [load(f"gtg/seeds/{t}/GTG_eval.json") for t in ("exp_102", "exp_103") if (R / f"gtg/seeds/{t}/GTG_eval.json").exists()]
if SEEDS:
    for d in ("CD->XA", "XA->BJ"):
        runs = [G[d]["GTG_official"]["top1_length"]] + [sd[d]["GTG_official"]["top1_length"] for sd in SEEDS]
        hs = [G[d]["GTG_official"]["hausdorff_mean_20k"]] + [sd[d]["GTG_official"]["hausdorff_mean_20k"] for sd in SEEDS]
        k = d.replace("->", "to").replace("XA", "Xa").replace("CD", "Cd").replace("BJ", "Bj")
        num(f"gtgRunMin{k}", min(runs)); num(f"gtgRunMax{k}", max(runs)); num(f"gtgRunSpread{k}", max(runs) - min(runs))
        num(f"gtgHausMin{k}", min(hs)); num(f"gtgHausMax{k}", max(hs))
        assert G[d]["zeroshot_logit_same_src"]["top1_length"] > max(runs), "logit-beats-every-GTG-run claim"
    num("gtgNRuns", words(1 + len(SEEDS)), "{}")
    # directions with a single GTG run: the logit's margin there is below the largest run-to-run spread we observed
    rep_dirs = [d for d in gd if all(d in sd for sd in SEEDS)]
    spread = max(max(r) - min(r) for r in ([G[d]["GTG_official"]["top1_length"]] + [sd[d]["GTG_official"]["top1_length"] for sd in SEEDS] for d in rep_dirs))
    single = [g["zeroshot_logit_same_src"]["top1_length"] - g["GTG_official"]["top1_length"] for d, g in gd.items() if d not in rep_dirs]
    num("logitMarginSingleMin", min(single)); num("logitMarginSingleMax", max(single))
    assert len(single) == 4 and max(single) < spread, "single-run margins below the run spread"
# Hausdorff (GTG's headline metric; no model here is trained on it): logit lowest in every direction, below GTG's published values
assert all(g["zeroshot_logit_same_src"]["hausdorff_mean_20k"] < min(g["GTG_official"]["hausdorff_mean_20k"], g["LEN_shortest"]["hausdorff_mean_20k"], PAPER_HD[d])
           for d, g in gd.items()), "logit Hausdorff claim"
# HOSER official schedule at every few-shot size (config/<city>.yaml: max_epoch 25, batch_size 256; DataLoader keeps the
# last partial batch): optimizer updates = 25 * ceil(n / 256)
upd = lambda n: 25 * int(np.ceil(n / 256))  # noqa: E731
num("hoserUpdThree", f"{upd(1000):,}", "{}"); num("hoserUpdFive", f"{int(round(upd(100000), -2)):,}".replace(",", "{,}"), "{}")
# HOSER on 1e3 trips trained 10x longer (250 epochs = update budget of 1e4 trips; HOSER_fewshot_budget.json)
HB = load("hoser/HOSER_fewshot_budget.json")["cities"]
for c in ("porto", "sanfrancisco"):
    hb, ref = HB[c], HF[c]["1000"]
    assert abs(hb["official_n1000"]["HOSER"]["top1_length"] - ref["HOSER"]["top1_length"]) < 1e-9 and hb["missing"] == 0
    simple_f1 = max(ref[k]["top1_length"] for k in ("logit", "gbdt", "od_history"))
    simple_fl = min(ref[k]["flow_nmae"] for k in ("logit", "gbdt", "od_history"))
    e = hb["HOSER_e250"]
    assert ref["HOSER"]["top1_length"] < e["top1_length"] < simple_f1 < HF[c]["10000"]["HOSER"]["top1_length"], c
    assert e["flow_nmae"] < simple_fl, c                     # best flows among methods trained on the same 1e3 trips
    num(f"hoserBudFone{AB[c]}", e["top1_length"]); num(f"hoserBudFlow{AB[c]}", e["flow_nmae"])
    num(f"simpleFlowBestThree{AB[c]}", simple_fl)
# few-shot HOSER: smallest n from which HOSER beats every simple model trained on the same n (top-1)
if HF:
    for c in fc:
        ns = sorted(HF[c], key=int)
        win = [n for n in ns if HF[c][n]["HOSER"]["top1_length"] > max(HF[c][n][k]["top1_length"] for k in ("logit", "gbdt", "od_history"))]
        num(f"hoserFsWin{AB[c]}", f"$10^{int(np.log10(int(win[0])))}$" if win else "never", "{}")
        lose = HF[c]["1000"]
        num(f"hoserFsLoseThree{AB[c]}", lose["HOSER"]["top1_length"])
        num(f"hoserFsBestSimpleThree{AB[c]}", max(lose[k]["top1_length"] for k in ("logit", "gbdt", "od_history")))
        num(f"hoserFsFour{AB[c]}", HF[c]["10000"]["HOSER"]["top1_length"])
        num(f"gbdtFsFiveSame{AB[c]}", HF[c]["100000"]["gbdt"]["top1_length"])
        num(f"hoserFsFlowFour{AB[c]}", HF[c]["10000"]["HOSER"]["flow_nmae"])

# statistics (E11): OD-cluster bootstrap, equivalence of the transfer gap, Holm-corrected one-sided tests
DELTA = E11["delta"]
num("eqDelta", DELTA, "{:.2f}")
gci = [S for c in CITIES for S in (E11["cities"][c]["gap"]["ci95"],)]
num("gapCILo", sg(np.floor(min(x[0] for x in gci) * 1000) / 1000, plus=False), "{}")
num("gapCIHi", sg(np.ceil(max(x[1] for x in gci) * 1000) / 1000, plus=False), "{}")
eq_min = np.ceil(max(max(abs(v["gap"]["ci90"][0]), abs(v["gap"]["ci90"][1])) for v in E11["cities"].values()) * 1000) / 1000
num("eqMinMargin", eq_min)   # smallest margin for which every city's 90% CI lies inside (margin chosen after the results)
assert eq_min < DELTA
num("bootB", f"{E11['B']:,}", "{}")
S11 = E11["cities"]
assert all(-DELTA < S11[c]["gap"]["ci90"][0] and S11[c]["gap"]["ci90"][1] < DELTA for c in CITIES), "equivalence claim"
assert all(S11[c]["gap"]["p_holm"] < 1e-6 and S11[c]["gap"]["p_tost_origin"] < 1e-3 for c in CITIES), "TOST p claim"
assert {c for c in CITIES if S11[c]["gap"]["ci95"][0] <= 0 <= S11[c]["gap"]["ci95"][1]} == {"xian", "chengdu"}, "gap CI includes 0 only in XA/CD"
assert all(S11[c][k]["p_holm"] < 1e-6 and S11[c][k]["origin_clusters"]["ci95"][0] > 0 for c in CITIES for k in ("gain", "gain_minus_gap")), "gain claims"
num("ciWidenMax", np.sqrt(max(S11[c][k]["design_effect"] for c in CITIES for k in ("gap", "gain"))), "{:.1f}")
num("repOdPctXA", f"{100 * S11['xian']['share_trips_repeated_od']:.0f}\\%", "{}")
num("repOdPctCD", f"{100 * S11['chengdu']['share_trips_repeated_od']:.0f}\\%", "{}")
G11 = E11["gtg"]
lg = [G11[d]["main"]["logit_minus_gtg"]["p_holm"] for d in G11]
assert all(o["logit_minus_gtg"]["ci95"][0] > 0 for d in G11 for o in G11[d].values()), "logit beats every GTG run (CI)"
num("logitGtgHolmMax", max(lg), "{:.3f}")
neg = [d for d in G11 if G11[d]["main"]["gtg_minus_len"]["mean"] < 0]
pos = [d for d in G11 if G11[d]["main"]["gtg_minus_len"]["mean"] > 0]
assert sorted(neg) == sorted(d for d in G11 if not d.endswith("BJ")) and all(d.endswith("BJ") for d in pos), "GTG vs LEN split"
assert all(G11[d]["main"]["gtg_minus_len"]["p_holm"] < 0.05 for d in pos), "BJ-target GTG advantage significant"
num("gtgLenHolmMin", min(G11[d]["main"]["gtg_minus_len"]["p_holm"] for d in neg), "{:.2f}")
assert min(G11[d]["main"]["gtg_minus_len"]["p_holm"] for d in neg) > 0.05, "GTG deficits not significant"

# distribution-level JSDs (E13: GTG's macro metrics for every method; own implementation, verified identical)
E13 = load(f"E13_dist{TAG}.json")
taus = [v for c in CITIES for v in E13[c]["kendall_tau_vs_f1"].values()]
num("jsdTauMin", min(taus), "{:.2f}"); num("jsdTauMax", max(taus), "{:.2f}")
mc = E13["chengdu"]["methods"]
gk = [k for k in mc if k.startswith("GTG_from")]
assert gk and all(mc[k]["Distance"] < mc["zeroshot_logit"]["Distance"] and mc[k]["Radius"] < mc["zeroshot_logit"]["Radius"]
                  and mc[k]["top1_length"] < mc["zeroshot_logit"]["top1_length"] for k in gk), "CD JSD reversal claim"
num("jsdGtgFoneMaxCD", max(mc[k]["top1_length"] for k in gk)); num("jsdZsFoneCD", mc["zeroshot_logit"]["top1_length"])
ow = [c for c in CITIES if E13[c]["methods"]["oracle6"]["Distance"] > E13[c]["methods"]["incity_od_history"]["Distance"] + 1e-9]
num("oracleWorseDistCount", words(len(ow)), "{}")

# GTG's official training also loads the TARGET traj/train_label.csv (trajectory-derived segment travel times); it only
# drops target subgraph samples without any labeled segment (gtg_label_check.py). Claim in Sec. IV: none is dropped.
if (R / "gtg/GTG_target_label_check.json").exists():
    TL = load("gtg/GTG_target_label_check.json")
    assert all(v["samples_kept"] == v["samples_label_free"] and v["slots"] == 24 for v in TL.values()), "GTG target-label claim"

# in-city GTG (official code with source = target, E11b): GTG's own transfer gap vs. the logit's on the same directions
GI = load(f"E11b_gtg_incity{TAG}.json")["targets"]
for t in ("XA", "CD"):
    num(f"gtgIn{t}", GI[t]["gtg_incity"])
    num(f"gtgInMinusLogit{t}", sg(GI[t]["gtg_minus_logit"]["mean"]), "{}")
    assert all(GI[t]["gtg_incity"] > q["gtg_zeroshot"] for o in GI[t]["sources"].values() for q in o["runs"].values()), "in-city GTG above every zero-shot run"
    assert GI[t]["gtg_minus_gbdt"]["ci95"][1] < 0, "in-city GTG below in-city GBDT"
assert GI["XA"]["gtg_minus_logit"]["ci95"][0] > 0 and GI["CD"]["gtg_minus_logit"]["ci95"][0] < 0 < GI["CD"]["gtg_minus_logit"]["ci95"][1], "GTG vs in-city logit"
gm = [o["runs"]["main"] for t in GI for o in GI[t]["sources"].values()]
ga = [q for t in GI for o in GI[t]["sources"].values() for q in o["runs"].values()]
num("gtgOwnGapMin", min(q["gtg_gap"]["mean"] for q in gm)); num("gtgOwnGapMax", max(q["gtg_gap"]["mean"] for q in gm))
num("gtgOwnGapRunMax", max(q["gtg_gap"]["mean"] for q in ga))
num("logitOwnGapMax", max(q["logit_gap"]["mean"] for q in gm))
num("gtgGapDiffHolmMax", max(q["gap_difference"]["p_holm"] for q in gm), "{:.3f}")
assert all(q["gap_difference"]["ci95"][0] > 0 for q in ga) and max(q["gap_difference"]["p_holm"] for q in gm) < 0.01, "GTG gap > logit gap"

# what transfers (E12): source subsets, source size, leave-one-feature-out of the pooled zero-shot logit
E12 = load(f"E12_sources{TAG}.json")["cities"]
CN = ["xian", "chengdu", "beijing"]
sgap = {c: {k: E12[c]["incity"]["top1"] - v["top1"] for k, v in E12[c]["subsets"].items()} for c in CITIES}
for c in CITIES:  # E12 re-fits E9's in-city and pooled models
    assert abs(sgap[c]["+".join(AB[s] for s in CITIES if s != c)] - (E9[c]["top1"]["incity_logit"]["length"] - E9[c]["top1"]["zeroshot_pooled"]["length"])) < 1.5e-3, c
assert all(len(E12[c]["subsets"]) == 15 for c in CITIES)
num("nSubsetsCity", "15", "{}")
num("subGapMax", max(max(v.values()) for v in sgap.values()))
assert max(sgap["porto"], key=sgap["porto"].get) == "SF" and max(sgap["sanfrancisco"], key=sgap["sanfrancisco"].get) == "PT", "PT<->SF worst"
num("ptsfGapPT", sgap["porto"]["SF"]); num("ptsfGapSF", sgap["sanfrancisco"]["PT"])
assert sgap["porto"]["XA+CD+BJ"] < sgap["porto"]["SF"] and sgap["sanfrancisco"]["XA+CD+BJ"] < sgap["sanfrancisco"]["PT"], "CN sources better for PT/SF"
num("cnOnlyGapPT", sg(sgap["porto"]["XA+CD+BJ"], plus=False), "{}"); num("cnOnlyGapSF", sg(sgap["sanfrancisco"]["XA+CD+BJ"], plus=False), "{}")
cont = [E12[c]["subsets"]["PT+SF"]["top1"] - E12[c]["subsets"]["+".join(AB[s] for s in CN if s != c)]["top1"] for c in CN]
num("contDiffMax", max(abs(x) for x in cont))
assert max(abs(x) for x in cont) < 0.005, "other continent as good as same country"
full = {c: E12[c]["size"]["all"]["top1_mean"] for c in CITIES}
assert all(E12[c]["size"]["all"]["n"] == 400000 for c in CITIES)
num("sizeTwoWorst", max(full[c] - E12[c]["size"]["100"]["top1_min"] for c in CITIES))
num("sizeThreeWorst", max(full[c] - E12[c]["size"]["1000"]["top1_min"] for c in CITIES))
num("sizeFourWorst", max(full[c] - E12[c]["size"]["10000"]["top1_min"] for c in CITIES))
num("anySrcGapMax", max(max(max(v.values()) for v in sgap.values()), max(E12[c]["incity"]["top1"] - E12[c]["size"]["1000"]["top1_min"] for c in CITIES)))
dz = {(c, f): v["zeroshot_top1"] - full[c] for c in CITIES for f, v in E12[c]["drop"].items()}
num("dropMax", max(abs(x) for x in dz.values()))
assert min(dz, key=dz.get) == ("porto", "log_ratio_nseg"), "segment-count ratio matters most"
num("dropNsegPT", -dz[("porto", "log_ratio_nseg")])
num("dropGapMax", max(v["gap"] for c in CITIES for v in E12[c]["drop"].values()))
num("dropNsegSF", -dz[("sanfrancisco", "log_ratio_nseg")]); num("dropLenSF", -dz[("sanfrancisco", "log_detour_len")])

# K-sensitivity of coverage (E14): Oracle@K of four alternative-route generators (+ union), 5,000 test trips per city
E14 = load(f"E14_kcand{TAG}.json")
num("kN", f"{E14['params']['N_TRIPS']:,}".replace(",", "{,}"), "{}")
E14 = E14["cities"]
KG = ["LP105", "LP2", "LE", "SIM", "UNION"]
o6 = {c: max(E14[c][g]["oracle"]["6"] for g in KG) - E14[c]["MAP6_oracle6"] for c in CITIES}
num("kSixDiffMax", max(abs(x) for x in o6.values()))
b30 = {c: max(KG, key=lambda g: E14[c][g]["oracle"]["30"]) for c in CITIES}
k30 = {c: E14[c][b30[c]]["oracle"]["30"] for c in CITIES}
kg = {c: k30[c] - E14[c]["MAP6_oracle6"] for c in CITIES}
num("kGainCnMin", min(kg[c] for c in CN)); num("kGainCnMax", max(kg[c] for c in CN))
for c in ("porto", "sanfrancisco"):
    num(f"kGain{AB[c]}", kg[c]); num(f"kThirty{AB[c]}", k30[c])
    num(f"covNine{AB[c]}", f"{100 * E14[c][b30[c]]['coverage']['30']['0.9']:.0f}\\%", "{}")
num("covNineCnMin", f"{100 * min(E14[c][b30[c]]['coverage']['30']['0.9'] for c in CN):.0f}\\%", "{}")
assert max(k30["porto"], k30["sanfrancisco"]) < min(E14[c]["MAP6_oracle6"] for c in CN), "PT/SF at K=30 below CN at K=6"
assert min(kg["porto"], kg["sanfrancisco"]) > max(kg[c] for c in CN), "K matters more in PT/SF"

# choice model over 30 candidates (E15): LP105 K=6/30 vs MAP6, logit re-estimated (5,000 train / 5,000 test trips per city)
E15 = load(f"E15_kcand_logit{TAG}.json")
assert E15["params"]["N_TR"] == E15["params"]["N_TE"] == int(N["kN"].replace("{,}", "")), "E15 sizes = kN"
E15 = E15["cities"]
for c in CITIES:
    assert abs(E15[c]["MAP6"]["oracle"] - E14[c]["MAP6_oracle6"]) < 1e-3 and abs(E15[c]["LP105@30"]["oracle"] - E14[c]["LP105"]["oracle"]["30"]) < 2e-3, c
d30 = {(c, k): E15[c]["LP105@30"][f"{k}_minus_map6"] for c in CITIES for k in ("zeroshot", "incity")}
assert all(v["ci95"][0] < 0 for v in d30.values()), "no significant gain from 30 candidates"
num("kLogitGainMax", max(v["mean"] for v in d30.values()))
num("kLogitLossSF", -d30[("sanfrancisco", "zeroshot")]["mean"])
assert d30[("sanfrancisco", "zeroshot")]["ci95"][1] < 0 and min(v["mean"] for v in d30.values()) == d30[("sanfrancisco", "zeroshot")]["mean"], "SF loss"
num("kGapThirtyMax", max(abs(E15[c]["LP105@30"]["incity"] - E15[c]["LP105@30"]["zeroshot"]) for c in CITIES))

# observed route choices (E17b): trips whose observed route is a MAP6 candidate, where the oracle label is the observed
# choice; E17 adds every observed route to its set instead (biased without a sampling correction, reported for completeness)
E17b, E17 = load(f"E17b_inset_ti{TAG}.json")["cities"], load(f"E17_observed_ti{TAG}.json")["cities"]
cov = [E17b[c]["covered_share_test"] for c in CITIES]
num("covMin", f"{100 * min(cov):.0f}\\%", "{}"); num("covMax", f"{100 * max(cov):.0f}\\%", "{}")
num("tiObsMin", min(E17b[c]["pooled"]["TI"] for c in CITIES), "{:.2f}"); num("tiObsMax", max(E17b[c]["pooled"]["TI"] for c in CITIES), "{:.2f}")
num("tiObsSingleMin", min(v["TI"] for c in CITIES for v in E17b[c]["single"].values()), "{:.2f}")
assert all(b[j] < 0 for c in CITIES for b in (E17b[c]["beta_incity"], E17b[c]["pooled"]["beta"]) for j in (0, 1, 3)), "observed-choice signs"
num("tiAugMin", min(E17[c]["pooled"]["TI"] for c in CITIES), "{:.2f}"); num("tiAugMax", max(E17[c]["pooled"]["TI"] for c in CITIES), "{:.2f}")
assert max(E17[c]["pooled"]["TI"] for c in CITIES) < min(E17b[c]["pooled"]["TI"] for c in CITIES), "augmented design lower"
# selection in the covered subset (E17c): mostly shortest-path choices; on covered trips that are NOT shortest paths,
# compare how often the pooled vs. in-city observed-choice logit picks the observed route (top-1 hit rate)
E17c = load(f"E17c_inset_detail{TAG}.json")["cities"]
for c in CITIES:
    assert abs(E17c[c]["pooled_all"]["TI"] - E17b[c]["pooled"]["TI"]) < 2e-3, c
sp = [E17c[c]["shortest_share_of_covered"] for c in CITIES]
num("spShareMin", f"{100 * min(sp):.0f}\\%", "{}"); num("spShareMax", f"{100 * max(sp):.0f}\\%", "{}")
assert all(E17c[c]["median_detour"]["covered"] <= E17c[c]["median_detour"]["uncovered"] and
           E17c[c]["median_km"]["covered"] < E17c[c]["median_km"]["uncovered"] for c in CITIES), "covered trips shorter"
hd = {c: E17c[c]["hit"]["pooled_nonshortest"] - E17c[c]["hit"]["incity_nonshortest"] for c in CITIES}
num("nonSpHitDiffMax", np.ceil(max(abs(hd[c]) for c in CITIES if c != "porto") * 100) / 100, "{:.2f}")
hci = E17c["porto"]["hit_diff_nonshortest"]["ci95"]
assert hci[1] < -0.15, hci
num("nonSpHitDiffCIPT", f"[{sg(hci[0], 2, plus=False)}, {sg(hci[1], 2, plus=False)}]", "{}")
assert max(abs(hd[c]) for c in CITIES if c != "porto") < 0.07 and hd["porto"] < -0.15, "non-shortest hit-rate claim"
num("nonSpHitPoolPT", E17c["porto"]["hit"]["pooled_nonshortest"], "{:.2f}"); num("nonSpHitInPT", E17c["porto"]["hit"]["incity_nonshortest"], "{:.2f}")

# prospective counts (E19): calibrate on counts of the early test trips, score the flows of the late test trips
E19 = load(f"E19_temporal_counts{TAG}.json")["cities"]
k19 = ("counts_prospective", "counts_transductive", "gbdt_prospective", "odhist_prospective", "gbdt_full_pool", "odhist_full_pool")
m19 = {c: {k: float(np.mean([x[k] for x in E19[c]["rows"]])) for k in k19} for c in CITIES}
num("cntProDiffMax", max(abs(m19[c]["counts_prospective"] - m19[c]["counts_transductive"]) for c in CITIES))
win19 = [c for c in CITIES if m19[c]["counts_prospective"] < min(m19[c][k] for k in k19[2:])]
assert sorted(win19) == sorted(["beijing", "porto", "sanfrancisco"]), win19   # same three cities as Tab. counts
num("cntProWinCount", words(len(win19)), "{}")
num("cntProDiffPlacementMax", max(abs(x["counts_prospective"] - x["counts_transductive"]) for c in CITIES for x in E19[c]["rows"]))
import datetime as _dt  # noqa: E402
_cut = _dt.date(1970, 1, 1) + _dt.timedelta(days=E19["porto"]["cutoff_day_utc"])
num("cutoffPT", f"{_cut.day} {_cut.strftime('%B')} {_cut.year}", "{}")

# exact parameter equality (E20b): unpenalized MLE, OD-cluster-robust covariances, Wald chi^2_5, Holm over the 15
# distinct tests (10 city pairs, symmetric, + 5 pooled); E20 (model-based covariances) overstates the evidence
E20b = load(f"E20b_robust_tests{TAG}.json")
uniq = {}
for t, o in E20b["targets"].items():
    for s, v in o.items():
        uniq[("pooled", t) if s == "pooled" else tuple(sorted((AB[t], s)))] = v["log10_p_robust"]
lp = sorted(uniq.values())
assert len(lp) == 15
holm_max = max(min(0.0, np.log10(len(lp) - i) + x) for i, x in enumerate(lp))   # log10 of the largest Holm-adjusted p
assert holm_max < -6, holm_max
num("waldRobMin", f"{min(v['Wald_robust'] for o in E20b['targets'].values() for v in o.values()):.0f}", "{}")
de = [x for c in ("xian", "chengdu") for x in E20b["coef"][c]["design_effect"]]
num("deffCnMin", f"{min(de):.0f}", "{}"); num("deffCnMax", f"{max(de):.0f}", "{}")

# temporal split (E16): in-city references fitted before a cutoff (75% quantile day) vs a random set of equal size
E16 = load(f"E16_temporal{TAG}.json")["cities"]
assert all(E16[c]["cutoff_day_utc"] == E19[c]["cutoff_day_utc"] for c in CITIES), "E16/E19 share the cutoff"
td = {(c, m): E16[c]["diff"][f"{m}_temporal_minus_random"]["mean"] for c in CITIES for m in ("logit", "gbdt", "odh")}
num("tempDiffMax", max(abs(x) for x in td.values()))
assert max(td, key=lambda k: abs(td[k])) == ("porto", "odh"), "OD history in Porto moves most"
num("tempShiftMax", max(abs(E16[c]["diff"][f"{k}_temporal"]["mean"] - E16[c]["diff"][f"{k}_random"]["mean"]) for c in CITIES for k in ("gap", "gain")))
assert all(-DELTA < E16[c]["diff"]["gap_temporal"]["ci90"][0] and E16[c]["diff"]["gap_temporal"]["ci90"][1] < DELTA for c in CITIES), "temporal gap equivalent"
assert all(E16[c]["diff"]["gain_temporal"]["mean"] > E16[c]["diff"]["gap_temporal"]["mean"] and E16[c]["diff"]["gain_temporal"]["ci95"][0] > 0 for c in CITIES), "temporal gain > gap"
num("tempLateShareMin", f"{100 * min(E16[c]['share_late_test'] for c in CITIES):.0f}\\%", "{}")
num("tempLateShareMax", f"{100 * max(E16[c]['share_late_test'] for c in CITIES):.0f}\\%", "{}")

# robustness to the count-based F1 (v2 count-mode E9 run + GTG eval computes both variants)
if (R / "E9_rigor_XA-CD-BJ-PT-SF.json").exists():
    EC = load("E9_rigor_XA-CD-BJ-PT-SF.json")
    cg = {c: EC[c]["top1"]["incity_logit"]["count"] - EC[c]["top1"]["zeroshot_pooled"]["count"] for c in CITIES}
    ci_ = {c: EC[c]["top1"]["incity_gbdt"]["count"] - EC[c]["top1"]["incity_logit"]["count"] for c in CITIES}
    num("countGapMax", max(cg.values()))
    num("countRatioMin", np.floor(min(ci_[c] / cg[c] for c in CITIES if cg[c] > 1e-9) * 2) / 2, "{:.1f}")
    num("countTiMin", min(EC[c]["transfer"]["zeroshot_pooled"]["TI"] for c in CITIES), "{:.2f}")
num("gtgBelowLenCountCount", words(sum(g["GTG_official"]["top1_count"] < g["LEN_shortest"]["top1_count"] for g in gd.values())), "{}")
num("logitBestCountCount", words(sum(g["zeroshot_logit_same_src"]["top1_count"] > max(g["GTG_official"]["top1_count"], g["LEN_shortest"]["top1_count"]) for g in gd.values())), "{}")

# ---------------------------------------------------------------- Table: estimated preferences per city (E20b: unpenalized MLE, OD-cluster-robust SE)
coefs = E20b["coef"]
BM = np.array([coefs[c]["beta_mle"] for c in CITIES])
assert (np.sign(BM) == np.sign(BT))[:, :4].all(), "MLE and ridge coefficients agree in sign on the four features discussed"
assert words(int(round(max(np.abs(BM[:, j]).max() / np.abs(BM[:, j]).min() for j in (0, 1, 3))))) == N["betaRatioMax"], "ratio claim also holds for the MLE"
num("tStatMin", min(abs(coefs[c]["beta_mle"][j]) / coefs[c]["se_robust"][j] for c in CITIES for j in (0, 1, 3)), "{:.1f}")
# trip-level intervals would turn the two OD-level intervals that include zero into "significant" gaps
assert all(abs(S11[c]["gap"]["mean"]) > 1.96 * S11[c]["gap"]["se_trip"] for c in ("xian", "chengdu")), "trip-level CI excludes zero in XA/CD"
FN = ["Log length ratio", "Log free-flow-time ratio", "Share on major roads", "Log segment-count ratio", "Log path size"]
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", f"Coefficient & {cityhdr} \\\\", r"\midrule"]
for j, lab in enumerate(FN):
    lines.append(lab + " & " + " & ".join(sg(coefs[c]["beta_mle"][j], 2, plus=False) for c in CITIES) + r" \\")
    lines.append(" & " + " & ".join(f"({coefs[c]['se_robust'][j]:.2f})" for c in CITIES) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_beta.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: robustness of the transfer gap (E12)
other = {c: ("PT+SF" if c in CN else "XA+CD+BJ") for c in CITIES}
pooled_key = {c: "+".join(AB[s] for s in CITIES if s != c) for c in CITIES}
ROWS12 = [("Pooled sources (default)", lambda c: sgap[c][pooled_key[c]]),
          ("Worst of 15 subsets", lambda c: max(sgap[c].values())),
          ("Other continents only", lambda c: sgap[c][other[c]]),
          (r"$10^3$ source trips (worst)", lambda c: E12[c]["incity"]["top1"] - E12[c]["size"]["1000"]["top1_min"]),
          ("Worst feature removed", lambda c: max(v["gap"] for v in E12[c]["drop"].values()))]
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", f"Transfer gap & {cityhdr} \\\\", r"\midrule"]
lines += [lab + " & " + " & ".join(sg(f(c)) for c in CITIES) + r" \\" for lab, f in ROWS12]
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_sources.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: how much evaluation choices matter
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", f"& {cityhdr} \\\\", r"\midrule",
         r"\multicolumn{6}{l}{\emph{Best-of-$K$ against link flows}} \\",
         ladder_row(r"\LEN{}, top-1 F1", lambda c: BF[c]["len_top1"]),
         ladder_row(r"\MAPsix{}, Best-of-6 F1", lambda c: BF[c]["oracle6"]),
         ladder_row(r"\LEN{}, flow NMAE", lambda c: BF[c]["len_flow"]),
         ladder_row(r"Uniform over \MAPsix{}, flow NMAE", lambda c: BF[c]["uniform_flow"]),
         r"\addlinespace", r"\multicolumn{6}{l}{\emph{Trip-level 95\% CIs are too narrow by a factor of}} \\"]
for key, lab in (("gap", "Transfer gap"), ("gain", "Local-flexibility gain")):
    lines.append(lab + " & " + " & ".join(f"{np.sqrt(S11[c][key]['design_effect']):.1f}" for c in CITIES) + r" \\")
lines += [r"\addlinespace", r"\multicolumn{6}{l}{\emph{Kendall's $\tau$ between divergence and route F1}} \\"]
for key, lab in (("Distance", "Travel distance"), ("Radius", "Radius of gyration"), ("LocFreq", "Segment visit frequency")):
    lines.append(lab + " & " + " & ".join(f"{E13[c]['kendall_tau_vs_f1'][key]:.2f}" for c in CITIES) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_pitfalls.tex").write_text("\n".join(lines) + "\n")
num("ciWidenOtherMax", np.sqrt(max(S11[c][k]["design_effect"] for c in ("beijing", "porto", "sanfrancisco") for k in ("gap", "gain"))), "{:.1f}")

# ---------------------------------------------------------------- Figure: transfer loss vs. what local data buys
from matplotlib.lines import Line2D  # noqa: E402
AQUA = "#1baf7a"
GRID = "#e6e8eb"
fig, ax = plt.subplots(figsize=(COL1, 1.55))
yy = np.arange(len(CITIES))[::-1].astype(float)
for key, lab, col, mk, off in (("gap", "Transfer gap", BLUE, "o", 0.22), ("gain", "Local-flexibility gain (GBDT)", ORANGE, "^", -0.02)):
    m = np.array([S11[c][key]["mean"] for c in CITIES])
    lo = np.array([S11[c][key]["ci95"][0] for c in CITIES])
    hi = np.array([S11[c][key]["ci95"][1] for c in CITIES])
    ax.errorbar(m, yy + off, xerr=[m - lo, hi - m], fmt=mk, color=col, ms=4.8, mec="white", mew=0.5, elinewidth=1.1, capsize=0, label=lab, zorder=3)
for i_c, c in enumerate(CITIES):
    if c in HE:
        v = HE[c]["HOSER_official"]["top1_length"] - HE[c]["incity_logit"]["top1_length"]
        ax.plot([v], [yy[i_c] - 0.26], ls="", marker="D", ms=4.6, color=AQUA, mec="white", mew=0.5, zorder=3, label="Deep in-city gain (HOSER)" if c == "porto" else None)
        ax.annotate(sg(v).replace("$+$", "+"), (v, yy[i_c] - 0.26), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK)
ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
ax.set_yticks(yy); ax.set_yticklabels([AB[c] for c in CITIES], fontsize=8)
ax.set_ylim(-0.6, len(CITIES) - 0.4)
ax.set_xlim(-0.006, 0.13)
ax.set_xlabel("Difference in top-1 F1", fontsize=8.5); ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
h, l_ = ax.get_legend_handles_labels()
order = [l_.index(t) for t in ("Transfer gap", "Local-flexibility gain (GBDT)", "Deep in-city gain (HOSER)")]
h, l_ = [h[i] for i in order], [l_[i] for i in order]
fig.legend(h, ["Transfer gap", "Flexibility gain (GBDT)", "Deep in-city gain (HOSER)"], loc="lower center", bbox_to_anchor=(0.53, 0.97), ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.2, fontsize=7.6)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "figs/fig_gap.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure: coverage as a function of K (E14)
KS = load(f"E14_kcand{TAG}.json")["params"]["KS"]
fig, ax = plt.subplots(figsize=(COL1, 1.85))
ends = []
for c in CITIES:
    col = BLUE if c in CN else ORANGE
    y = [E14[c][b30[c]]["oracle"][str(k)] for k in KS]
    ax.plot(KS, y, color=col, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.4, zorder=2)
    ax.plot([6], [E14[c]["MAP6_oracle6"]], ls="", marker="D", ms=4.4, mfc="white", mec=INK, mew=0.7, zorder=4)
    ends.append([y[-1], AB[c]])
ends.sort()
pos = []
for v, _ in ends:   # keep the direct labels apart
    pos.append(max(v, pos[-1] + 0.038) if pos else v)
for (v, a), yl in zip(ends, pos):
    ax.annotate(a, (KS[-1], yl), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color=INK, annotation_clip=False)
ax.set_xscale("log")
ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()
ax.set_xlim(0.9, 34)
ax.set_xlabel("Candidates per OD pair ($K$, log scale)")
ax.set_ylabel("Oracle@$K$ route F1")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3.2, lw=1.3, label="Chinese cities"),
                    Line2D([], [], color=ORANGE, marker="o", ms=3.2, lw=1.3, label="Porto, San Francisco"),
                    Line2D([], [], ls="", marker="D", ms=4.4, mfc="white", mec=INK, mew=0.7, label="MAP6 ($K$=6)")],
           loc="lower center", bbox_to_anchor=(0.53, 0.97), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "figs/fig_kcand.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Mechanism analysis (E21): why so little suffices
WA, WB, WC, WD, WE = (load(f"E21_{p}{TAG}.json") for p in "ABCDE")
TA, RB, R0 = WA["targets"], WB["runs"], WE["cities"]
pct = lambda x: f"{100 * x:.0f}" + r"\%"  # noqa: E731
for c in CITIES:  # E21 re-fits the models of E9
    assert abs(R0[c]["top1"]["logit5_zs"] - E9[c]["top1"]["zeroshot_pooled"]["length"]) < 1.5e-3, c
    assert abs(R0[c]["top1"]["logit5_in"] - E9[c]["top1"]["incity_logit"]["length"]) < 1.5e-3, c
    assert abs(R0[c]["top1"]["oracle6"] - E10[c]["bok"]) < 2e-3, c
    assert abs(R0[c]["1p"]["zs"]["top1"] - R0[c]["top1"]["rule0"]) < 1e-9, "one-parameter top-1 = parameter-free rule (scale invariance)"
    d_ = TA[c]["sources"]["pooled"]["decision"]
    assert d_["identity_err"] < 1e-9, "decision identity is exact"
    assert abs(d_["gap"] - (R0[c]["top1"]["logit5_in"] - R0[c]["top1"]["logit5_zs"])) < 1e-6, c
for d_, g in gd.items():  # the anatomy uses the routes of the main GTG table
    s_, t_ = d_.split("->") if "->" in d_ else (None, None)
    o = RB[f"{s_}->{t_}/main"]
    assert abs(o["f1"]["gen"] - g["GTG_official"]["top1_length"]) < 1e-3 and abs(o["f1"]["len"] - g["LEN_shortest"]["top1_length"]) < 1e-3 and abs(o["f1"]["logit"] - g["zeroshot_logit_same_src"]["top1_length"]) < 1e-3, d_

# one dimension: the parameter-free rule and the one-parameter logit
FULL = {v: k for k, v in AB.items()}
others = [c for c in CITIES if c != "chengdu"]
d_in = {c: R0[c]["rule0_minus_in5"]["mean"] for c in CITIES}
assert d_in["chengdu"] < 0 and R0["chengdu"]["rule0_minus_in5"]["ci95"][1] < 0, "rule below the in-city logit in Chengdu"
assert all(d_in[c] > -5e-4 for c in others), "rule at least level with the in-city logit in the other four cities"
assert all(R0[c]["rule0_minus_zs5"]["ci95"][0] > 0 for c in others), "rule significantly above the zero-shot logit in four cities"
assert R0["chengdu"]["rule0_minus_zs5"]["ci95"][1] < 0
num("ruleInMin", sg(min(d_in[c] for c in others))); num("ruleInMax", sg(max(d_in[c] for c in others))); num("ruleInCD", f3(abs(d_in["chengdu"])))
num("ruleLenMin", f3(min(R0[c]["rule0_minus_len"]["mean"] for c in CITIES))); num("ruleLenMax", f3(max(R0[c]["rule0_minus_len"]["mean"] for c in CITIES)))
zr = {k: v for k, v in WE["gtg"].items() if not k.endswith("/incity")}
assert len(zr) == 10 and all(v["rule0_minus_gtg"]["mean"] > 0 and v["rule0_minus_gtg"]["ci95"][0] > 0 for v in zr.values()), "rule above GTG in every zero-shot run"
num("ruleGtgMin", f3(min(v["rule0_minus_gtg"]["mean"] for v in zr.values()))); num("ruleGtgMax", f3(max(v["rule0_minus_gtg"]["mean"] for v in zr.values())))
num("ruleGtgHolmMax", max(WE["holm_rule0_gt_gtg"].values()), "{:.3f}")
assert max(WE["holm_rule0_gt_gtg"].values()) < 0.05
assert WE["gtg"]["XA->XA/incity"]["rule0_minus_gtg"]["ci95"][1] < 0 and WE["gtg"]["CD->CD/incity"]["rule0_minus_gtg"]["ci95"][0] < 0 < WE["gtg"]["CD->CD/incity"]["rule0_minus_gtg"]["ci95"][1]
num("ruleBelowGtgInXA", f3(abs(WE["gtg"]["XA->XA/incity"]["rule0_minus_gtg"]["mean"])))
for d_, g in gd.items():
    s_, t_ = d_.split("->")
    m_ = WE["rule0_route_metrics"][FULL[t_]]
    assert m_["flow_nmae"] < g["GTG_official"]["flow_nmae"] and m_["hausdorff_mean_20k"] < g["GTG_official"]["hausdorff_mean_20k"], d_
cv = lambda v: float(np.std(v) / abs(np.mean(v)))  # noqa: E731
COST = [0, 1, 3]
num("betaCvCostMin", min(cv(BM[:, j]) for j in COST), "{:.2f}"); num("betaCvCostMax", max(cv(BM[:, j]) for j in COST), "{:.2f}"); num("betaCvSum", cv(BM[:, COST].sum(1)), "{:.2f}")
assert cv(BM[:, COST].sum(1)) < 0.5 * min(cv(BM[:, j]) for j in COST)
b1 = [R0[c]["1p"]["beta_in"][0] for c in CITIES]
assert max(b1) < 0
num("betaOneMin", sg(min(b1), 2, plus=False)); num("betaOneMax", sg(max(b1), 2, plus=False)); num("betaOneRatio", min(b1) / max(b1), "{:.1f}")
ti1, ti5 = [R0[c]["1p"]["ti_own"] for c in CITIES], [R0[c]["5p"]["ti_own"] for c in CITIES]
num("tiOneMin", min(ti1), "{:.2f}"); num("tiOneMax", max(ti1), "{:.2f}")
assert all(a > b for a, b in zip(ti1, ti5)), "one-parameter model transfers better than the five-parameter one in every city"
assert all(abs(R0[c]["5p"]["ti_own"] - TA[c]["sources"]["pooled"]["ti_test"]) < 2e-3 for c in CITIES), "independent Adam refits agree"
num("llOneBetterCount", words(sum(R0[c]["1p"]["ll_zs"] > R0[c]["5p"]["ll_zs"] for c in CITIES)), "{}")
num("flowOneDiffMax", max(abs(R0[c]["1p"]["zs"]["flow_nmae"] - R0[c]["5p"]["zs"]["flow_nmae"]) for c in CITIES))
num("simplexN", len(R0["xian"]["simplex"]["grid"]), "{}")
num("simplexRegretMax", max(R0[c]["simplex"]["best"] - R0[c]["simplex"]["equal"] for c in CITIES))
assert R0["porto"]["simplex"]["best_weights"][1] >= 0.5 and R0["sanfrancisco"]["simplex"]["best_weights"][0] >= 0.5, "Porto best on time, San Francisco on length"

# why different coefficients cost little
PR = [v for t in CITIES for v in TA[t]["sources"].values()]
ra = np.array([v["dnll_train_quad"] / v["dnll_train_actual"] for v in PR])
qo = np.array([v["quad_over_ortho"] for v in PR])
num("whyPairs", len(PR), "{}"); assert len(PR) == 25
num("quadRatioMin", ra.min(), "{:.2f}"); num("quadRatioMax", ra.max(), "{:.2f}"); num("quadRatioMed", float(np.median(ra)), "{:.2f}")
num("quadCorr", float(np.corrcoef([v["dnll_train_actual"] for v in PR], [v["dnll_train_quad"] for v in PR])[0, 1]), "{:.3f}")
num("quadOrthoMin", qo.min(), "{:.2f}"); num("quadOrthoMax", qo.max(), "{:.2f}"); num("quadOrthoMed", float(np.median(qo)), "{:.2f}")
PO = {t: TA[t]["sources"]["pooled"] for t in CITIES}
num("scaleShareMax", pct(max(v["scale_share_of_loss"] for v in PO.values())), "{}")
agree = [1 - v["decision"]["differ"] for v in PO.values()]
num("agreeMin", pct(min(agree)), "{}"); num("agreeMax", pct(max(agree)), "{}")
wins = [x for v in PO.values() for x in (v["decision"]["a_better"], v["decision"]["b_better"])]
num("winMin", pct(min(wins)), "{}"); num("winMax", pct(max(wins)), "{}")
num("absCondMin", min(v["decision"]["abs_cond"] for v in PO.values()), "{:.2f}"); num("absCondMax", max(v["decision"]["abs_cond"] for v in PO.values()), "{:.2f}")
num("condAbsMax", max(abs(v["decision"]["cond_gain"]) for v in PO.values()), "{:.2f}")

# why deep transfer loses: value of deviations from the shortest path
ZK = [k for k in RB if not k.endswith("/incity")]
assert len(ZK) == 10
for k in ZK:
    o = RB[k]
    assert o["logit_vs_len"]["cond_gain"] > o["gen_vs_len"]["cond_gain"], k
    assert abs(sum(v["contrib"] for v in o["cells"].values()) - o["logit_vs_gen"]["gap"]) < 1e-5, k
CNK = ["CD->XA/main", "BJ->XA/main", "BJ->CD/main", "XA->CD/main"]
BJK = ["XA->BJ/main", "CD->BJ/main"]
IN = {"XA": "XA->XA/incity", "CD": "CD->CD/incity"}
for k in CNK:
    assert WE["gtg"][k]["gtg_dev_value"]["ci95"][0] <= 0, ("not significantly positive", k)
for k in ZK:
    t_ = k.split("->")[1].split("/")[0]
    if t_ in IN:
        assert RB[IN[t_]]["gen_vs_len"]["cond_gain"] > RB[k]["gen_vs_len"]["cond_gain"], k
gv = lambda ks, f: [f(RB[k]) for k in ks]  # noqa: E731
share_ = lambda o: o["gen_vs_len"]["differ"]  # noqa: E731
val_ = lambda o: o["gen_vs_len"]["cond_gain"]  # noqa: E731
lval_ = lambda o: o["logit_vs_len"]["cond_gain"]  # noqa: E731
num("gtgDevShareCnMin", pct(min(gv(CNK, share_))), "{}"); num("gtgDevShareCnMax", pct(max(gv(CNK, share_))), "{}")
num("gtgInDevShareMin", pct(min(gv(IN.values(), share_))), "{}"); num("gtgInDevShareMax", pct(max(gv(IN.values(), share_))), "{}")
assert max(gv(CNK, val_)) < 0
num("gtgDevValCnMin", sg(min(gv(CNK, val_)), 3, plus=False)); num("gtgDevValCnMax", sg(max(gv(CNK, val_)), 3, plus=False))
num("gtgDevValXA", sg(val_(RB[IN["XA"]]))); num("gtgDevValCD", sg(val_(RB[IN["CD"]])))
num("logDevValCnMin", sg(min(gv(CNK, lval_)))); num("logDevValCnMax", sg(max(gv(CNK, lval_))))
num("gtgDevValBjMin", sg(min(gv(BJK, val_)))); num("gtgDevValBjMax", sg(max(gv(BJK, val_))))
num("logDevValBjMin", sg(min(gv(BJK, lval_)))); num("logDevValBjMax", sg(max(gv(BJK, lval_))))
dvshare = [(RB[k]["cells"]["gen_dev&log_len"]["contrib"] + RB[k]["cells"]["gen_dev&log_dev"]["contrib"]) / RB[k]["logit_vs_gen"]["gap"] for k in ZK]
num("devDiffShareMin", pct(min(dvshare)), "{}")

# what local data buy
sn = [WC[c]["gbdt_gain_by_od"]["seen"]["gain"] for c in CITIES]; un = [WC[c]["gbdt_gain_by_od"]["unseen"]["gain"] for c in CITIES]
num("gbSeenMin", f3(min(sn))); num("gbSeenMax", f3(max(sn))); num("gbUnseenMin", f3(min(un))); num("gbUnseenMax", f3(max(un)))
num("gbUnseenPT", f3(WC["porto"]["gbdt_gain_by_od"]["unseen"]["gain"])); num("gbSeenPT", f3(WC["porto"]["gbdt_gain_by_od"]["seen"]["gain"]))
assert all(WC[c]["gbdt_gain_by_od"]["unseen"]["ci95"][0] > 0 for c in CITIES), "gain on unseen ODs is significant in every city"
f5 = [WC[c]["gbdt5_in_vs_logit_in"]["gap"] / WC[c]["gbdt_in_vs_logit_in"]["gap"] for c in CITIES]
num("gbFiveShareMin", pct(min(f5)), "{}"); num("gbFiveShareMax", pct(max(f5)), "{}")
gi, gz = [WC[c]["gbdt_in_vs_logit_in"] for c in CITIES], [WC[c]["gbdt_zs_vs_logit_zs"] for c in CITIES]
num("gbDiffMin", pct(min(x["differ"] for x in gi)), "{}"); num("gbDiffMax", pct(max(x["differ"] for x in gi)), "{}")
num("gbWinMin", pct(min(x["a_better"] for x in gi)), "{}"); num("gbWinMax", pct(max(x["a_better"] for x in gi)), "{}")
num("zsgbDiffMin", pct(min(x["differ"] for x in gz)), "{}"); num("zsgbDiffMax", pct(max(x["differ"] for x in gz)), "{}")
num("zsgbWinMin", pct(min(x["a_better"] for x in gz)), "{}"); num("zsgbWinMax", pct(max(x["a_better"] for x in gz)), "{}")
HO = {c: WC[c]["hoser"] for c in ("porto", "sanfrancisco")}
for c, h in HO.items():
    assert abs(h["f1"] - HE[c]["HOSER_official"]["top1_length"]) < 1e-3, c
num("hoSeenPT", f3(HO["porto"]["seen"]["gain_logit"])); num("hoSeenSF", f3(HO["sanfrancisco"]["seen"]["gain_logit"]))
num("hoUnseenPT", f3(HO["porto"]["unseen"]["gain_logit"])); num("hoUnseenSF", f3(HO["sanfrancisco"]["unseen"]["gain_logit"]))
oc = [h["out_map6"]["contrib"] / (h["f1"] - h["logit"]) for h in HO.values()]
num("hoOutContribMin", pct(min(oc)), "{}"); num("hoOutContribMax", pct(max(oc)), "{}")
num("hoOutShareMin", pct(min(h["out_map6"]["share"] for h in HO.values())), "{}"); num("hoOutShareMax", pct(max(h["out_map6"]["share"] for h in HO.values())), "{}")

# how many trips: effective dimension and predicted crossover
num("dMin", min(WD[c]["d_iid"] for c in CITIES), "{:.1f}"); num("dMax", max(WD[c]["d_iid"] for c in CITIES), "{:.1f}")
num("nStarMin", min(WD[c]["n_star_pred_iid"] for c in CITIES), "{:.0f}"); num("nStarMax", max(WD[c]["n_star_pred_iid"] for c in CITIES), "{:.0f}")
num("excessErrMax", max(abs(x["excess"] - x["excess_pred_iid"]) for c in CITIES for x in WD[c]["curve"] if x["n"] >= 1000))
num("excessThree", min(x["excess"] for c in CITIES for x in WD[c]["curve"] if x["n"] == 1000), "{:.3f}")
def bracket(c):
    xs = WD[c]["curve"]
    i_ = next(i for i, x in enumerate(xs) if x["ll"] >= WD[c]["ll_zs"])
    assert i_ > 0 and all(x["ll"] >= WD[c]["ll_zs"] for x in xs[i_:]), c
    return xs[i_ - 1]["n"], xs[i_]["n"]
BR = {c: bracket(c) for c in CITIES}
num("nStarObsLow", min(b[0] for b in BR.values()), "{}"); num("nStarObsHigh", max(b[1] for b in BR.values()), "{}")

# checks of Sec. VII (E21 part F): first-order term on test trips, rules designed without the target, OD-cluster sampling
WF = load(f"E21_F{TAG}.json")
EXP = [v for t in CITIES for v in WF["expansion"][t].values()]
assert len(EXP) == 25 and all(v["actual"] > 0 for v in EXP)
num("firstMax", max(abs(v["first"]) for v in EXP))
num("firstMedPct", pct(float(np.median([abs(v["first"]) / v["actual"] for v in EXP]))), "{}")
num("quadTestRatioMin", min(v["quad"] / v["actual"] for v in EXP), "{:.2f}"); num("quadTestRatioMax", max(v["quad"] / v["actual"] for v in EXP), "{:.2f}")
gs1 = [(R0[c]["1p"]["ll_in"] - R0[c]["ll_uniform"]) / (R0[c]["5p"]["ll_in"] - R0[c]["ll_uniform"]) for c in CITIES]
num("oneGainMin", pct(min(gs1)), "{}"); num("oneGainMax", pct(max(gs1)), "{}")
eig_ = [np.array(TA[c]["H_corr_eig"]) for c in CITIES]
num("eigTopMin", pct(min(float(e.max() / e.sum()) for e in eig_)), "{}"); num("eigTopMax", pct(max(float(e.max() / e.sum()) for e in eig_)), "{}")
num("effRankMin", min(float(e.sum() ** 2 / (e ** 2).sum()) for e in eig_), "{:.1f}"); num("effRankMax", max(float(e.sum() ** 2 / (e ** 2).sum()) for e in eig_), "{:.1f}")
tc1 = {c: TA[c]["restricted"]["1p"]["ti_vs_in5"] for c in CITIES}
num("tiCommonOneMin", min(tc1.values()), "{:.2f}"); num("tiCommonOneMax", max(tc1.values()), "{:.2f}")
hi_ = sum(tc1[c] > TA[c]["sources"]["pooled"]["ti_test"] for c in CITIES)
num("tiCommonHigher", words(hi_), "{}"); num("tiCommonLower", words(5 - hi_), "{}")
LO = WF["loco"]
assert all(LO[c]["int_best"][2] == 0 and LO[c]["int_best"][4] == 0 and min(LO[c]["int_best"][0], LO[c]["int_best"][1], LO[c]["int_best"][3]) >= 1 for c in CITIES), "held-out design keeps exactly the three costs"
num("ruleFamilyN", 242, "{}")
lf_ = [LO[c]["int_minus_frozen"]["mean"] for c in CITIES]
num("locoFrozenMin", sg(min(lf_))); num("locoFrozenMax", sg(max(lf_)))
lz_ = [LO[c]["int_minus_zs5"]["mean"] for c in others]
assert min(lz_) > -5e-4 and LO["chengdu"]["int_minus_zs5"]["ci95"][1] < 0
num("locoZsMin", sg(min(lz_))); num("locoZsMax", sg(max(lz_))); num("locoZsCD", f3(abs(LO["chengdu"]["int_minus_zs5"]["mean"])))
HB = WF["holm_rule_gt_gtg"]
sig_ = [k for k, p in HB.items() if p < 0.05]
assert len(HB) == 10 and set(HB) - set(sig_) == {"XA->CD/main", "BJ->CD/main"} and max(HB[k] for k in sig_) < 0.001, "same-source design: eight of ten runs significant"
assert all(WF["gtg"][k]["rule_minus_gtg"]["ci95"][0] < 0 < WF["gtg"][k]["rule_minus_gtg"]["ci95"][1] for k in set(HB) - set(sig_)), "level in the two runs into Chengdu"
num("sameRuleSigCount", {8: "eight"}[len(sig_)], "{}")
num("sameRuleMin", f3(min(WF["gtg"][k]["rule_minus_gtg"]["mean"] for k in sig_))); num("sameRuleMax", f3(max(WF["gtg"][k]["rule_minus_gtg"]["mean"] for k in sig_)))
SS = WF["same_source"]
num("sameFlowBetter", words(sum(SS[d_]["route_metrics"]["flow_nmae"] < g_["GTG_official"]["flow_nmae"] for d_, g_ in gd.items())), "{}")
num("sameHausBetter", words(sum(SS[d_]["route_metrics"]["hausdorff_mean_20k"] < g_["GTG_official"]["hausdorff_mean_20k"] for d_, g_ in gd.items())), "{}")
CS = WF["cluster_sampling"]
def bracket_cl(c):
    xs = CS[c]["curve"]
    i_ = next(i for i, x in enumerate(xs) if x["ll_mean"] >= CS[c]["ll_zs"])
    assert i_ > 0 and all(x["ll_mean"] >= CS[c]["ll_zs"] for x in xs[i_:]), c
    return xs[i_ - 1]["n"], xs[i_]["n"]
BC = {c: bracket_cl(c) for c in CITIES}
assert BC["xian"] == (3000, 10000) and BC["chengdu"] == (3000, 10000) and all(BC[c] == (100, 300) for c in ("beijing", "porto", "sanfrancisco")), BC
num("dClCnMin", min(WD[c]["d_cluster"] for c in ("xian", "chengdu")), "{:.0f}"); num("dClCnMax", max(WD[c]["d_cluster"] for c in ("xian", "chengdu")), "{:.0f}")
num("dClOtherMin", min(WD[c]["d_cluster"] for c in ("beijing", "porto", "sanfrancisco")), "{:.1f}"); num("dClOtherMax", max(WD[c]["d_cluster"] for c in ("beijing", "porto", "sanfrancisco")), "{:.1f}")
num("nStarClMin", min(WD[c]["n_star_pred_cluster"] for c in CITIES), "{:.0f}"); num("nStarClMax", max(WD[c]["n_star_pred_cluster"] for c in CITIES), "{:,.0f}")
kn = lambda v: f"{v // 1000}k" if v >= 1000 else str(v)  # noqa: E731

lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", f"& {cityhdr} \\\\", r"\midrule",
         r"\multicolumn{6}{l}{\emph{Top-1 F1}} \\",
         ladder_row(r"\LEN{}", lambda c: E4[c]["LEN_top1"]),
         ladder_row("Post-hoc rule", lambda c: R0[c]["top1"]["rule0"]),
         ladder_row("Source-selected rule", lambda c: LO[c]["heldout"]["int_design"]),
         ladder_row("Zero-shot logit, 5 par.", lambda c: E10[c]["zeroshot"]),
         ladder_row("In-city logit, 5 par.", lambda c: E10[c]["incity_logit"]),
         r"\addlinespace", r"\multicolumn{6}{l}{\emph{Test log-likelihood per trip}} \\",
         "Zero-shot logit, 1 par. & " + " & ".join(sg(R0[c]["1p"]["ll_zs"], 3, plus=False) for c in CITIES) + r" \\",
         "Zero-shot logit, 5 par. & " + " & ".join(sg(R0[c]["5p"]["ll_zs"], 3, plus=False) for c in CITIES) + r" \\",
         "In-city logit, 5 par. & " + " & ".join(sg(R0[c]["5p"]["ll_in"], 3, plus=False) for c in CITIES) + r" \\",
         r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_rules.tex").write_text("\n".join(lines) + "\n")
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", f"Sampling & {cityhdr} \\\\", r"\midrule",
         r"Random trips: predicted & " + " & ".join(f"{WD[c]['n_star_pred_iid']:.0f}" for c in CITIES) + r" \\",
         r"\quad observed & " + " & ".join(f"{BR[c][0]}--{BR[c][1]}" for c in CITIES) + r" \\",
         r"Whole ODs: predicted & " + " & ".join(f"{WD[c]['n_star_pred_cluster']:.0f}" for c in CITIES) + r" \\",
         r"\quad observed & " + " & ".join(f"{kn(BC[c][0])}--{kn(BC[c][1])}" for c in CITIES) + r" \\",
         r"\bottomrule", r"\end{tabular}"]
(OUT / "tables/tab_nstar.tex").write_text("\n".join(lines) + "\n")

# Figure (RQ2): (a) transfer-index matrix, (b) cost coefficients by city, (c) transfer loss against the quadratic prediction
fig, (axa, axb, ax1) = plt.subplots(1, 3, figsize=(COL2, 1.72), gridspec_kw={"width_ratios": [1.5, 1.0, 0.86], "wspace": 0.42})
axa.imshow(np.where(np.isnan(M), np.nan, M), cmap=cmap, vmin=vmin, vmax=1.0, aspect="auto")
for i_ in range(M.shape[0]):
    for j_ in range(M.shape[1]):
        if np.isnan(M[i_, j_]):
            axa.text(j_, i_, "in-city", ha="center", va="center", color=MUTED, fontsize=6.5)
        else:
            dark = (M[i_, j_] - vmin) / (1 - vmin) > 0.55
            axa.text(j_, i_, f"{M[i_, j_]:.2f}", ha="center", va="center", color="white" if dark else INK, fontsize=7.5)
axa.axvline(len(CITIES) - 0.5, color="white", lw=2.5)
axa.set_xticks(range(len(cols)), [AB[c] for c in CITIES] + ["Pooled"], fontsize=8)
axa.set_yticks(range(len(CITIES)), [AB[c] for c in CITIES], fontsize=8)
axa.set_xlabel("Source city (parameters estimated here)", fontsize=8.5); axa.set_ylabel("Target city", fontsize=8.5)
axa.tick_params(length=0)
for s_ in axa.spines.values():
    s_.set_visible(False)
axa.set_title("(a) Transfer index", loc="left", fontsize=8.5)
yy_ = np.arange(len(CITIES))[::-1].astype(float)
left_ = np.zeros(len(CITIES))
for j_, lab_, col_ in ((0, "Length", "#174a8c"), (1, "Free-flow time", BLUE), (3, "Segments", "#9cc0ee")):
    w_ = -BM[:, j_]
    assert (w_ > 0).all()
    axb.barh(yy_, w_, left=left_, height=0.62, color=col_, edgecolor="white", linewidth=0.8, label=lab_, zorder=3)
    left_ = left_ + w_
for y_, t_ in zip(yy_, left_):
    axb.text(t_ + 0.25, y_, f"{t_:.1f}", ha="left", va="center", fontsize=7.5, color=INK)
axb.set_yticks(yy_); axb.set_yticklabels([AB[c] for c in CITIES], fontsize=8)
axb.set_xlim(0, left_.max() * 1.22); axb.set_ylim(-0.55, len(CITIES) + 0.45)
axb.set_xlabel(r"Weight on route cost, $-\beta$", fontsize=8.5)
axb.tick_params(axis="y", length=0); axb.tick_params(axis="x", labelsize=8)
axb.grid(axis="x", color=GRID, lw=0.5); axb.set_axisbelow(True)
axb.set_title("(b) Cost weights", loc="left", fontsize=8.5)
axb.legend(loc="upper center", bbox_to_anchor=(0.54, 1.04), ncol=3, frameon=False, handlelength=0.8, handletextpad=0.25, columnspacing=0.5, borderpad=0.1, fontsize=6.8)
single = [v for t in CITIES for k, v in TA[t]["sources"].items() if k != "pooled"]
for vs, lab, col, mk in ((single, "Single source", ORANGE, "o"), (list(PO.values()), "Pooled", BLUE, "D")):
    ax1.plot([v["dnll_train_actual"] for v in vs], [v["dnll_train_quad"] for v in vs], ls="", marker=mk, ms=4.4, color=col, mec="white", mew=0.4, label=lab, zorder=3)
xr = [min(v["dnll_train_actual"] for v in PR) * 0.8, max(v["dnll_train_actual"] for v in PR) * 1.25]
ax1.plot(xr, xr, color=MUTED, lw=0.7, zorder=1)
ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_xlim(xr); ax1.set_ylim(xr)
ax1.set_xlabel("Actual loss per trip", fontsize=8.5); ax1.set_ylabel(r"Predicted, $\frac{1}{2}\delta^\top H\delta$", fontsize=8.5)
ax1.set_title("(c) Transfer loss", loc="left", fontsize=8.5)
ax1.legend(loc="lower right", frameon=False, handletextpad=0.1, borderpad=0.1, labelspacing=0.2, fontsize=7.5)
ax1.grid(color=GRID, lw=0.5); ax1.set_axisbelow(True); ax1.minorticks_off()
ticks_ = [0.01, 0.03, 0.1]
ax1.set_xticks(ticks_); ax1.set_xticklabels(["0.01", "0.03", "0.1"]); ax1.set_yticks(ticks_); ax1.set_yticklabels(["0.01", "0.03", "0.1"])
ax1.tick_params(labelsize=8)
fig.savefig(OUT / "figs/fig_transfer.pdf")
plt.close(fig)

# Figure (RQ1): how often a method leaves the length-shortest path, and what leaving it is worth
fig, (axs, ax2) = plt.subplots(1, 2, figsize=(COL1, 1.5), gridspec_kw={"wspace": 0.5})
groups = ["XA", "CD", "BJ"]
for ax_, key_ in ((axs, "differ"), (ax2, "cond_gain")):
    sc_ = 100.0 if key_ == "differ" else 1.0
    for gi_, tg in enumerate(groups):
        zk = [k for k in ZK if k.split("->")[1].split("/")[0] == tg]
        xs = gi_ - 0.24 + np.linspace(-0.07, 0.07, len(zk))
        ax_.plot(xs, [sc_ * RB[k]["gen_vs_len"][key_] for k in zk], ls="", marker="o", ms=4.6, color=ORANGE, mec="white", mew=0.4, zorder=3, label="GTG, zero-shot" if gi_ == 0 else None)
        if tg in IN:
            ax_.plot([gi_], [sc_ * RB[IN[tg]]["gen_vs_len"][key_]], ls="", marker="s", ms=4.8, color=AQUA, mec="white", mew=0.4, zorder=3, label="GTG, in-city" if gi_ == 0 else None)
        lk = sorted({k.split("/")[0] for k in zk})
        ys = [sc_ * RB[f"{k}/main"]["logit_vs_len"][key_] for k in lk]
        ax_.plot(gi_ + 0.24 + np.linspace(-0.05, 0.05, len(ys)), ys, ls="", marker="D", ms=4.4, color=BLUE, mec="white", mew=0.4, zorder=3, label="Logit, zero-shot" if gi_ == 0 else None)
    ax_.set_xticks(range(3)); ax_.set_xticklabels(groups, fontsize=8); ax_.tick_params(axis="y", labelsize=8); ax_.set_xlim(-0.55, 2.55)
    ax_.grid(axis="y", color=GRID, lw=0.5); ax_.set_axisbelow(True); ax_.tick_params(axis="x", length=0)
axs.set_ylim(0, None)
axs.set_ylabel("Trips off the shortest path (%)", fontsize=8.5); axs.set_title("(a) How often", loc="left", fontsize=8.5)
ax2.axhline(0, color=MUTED, lw=0.7, zorder=1)
ax2.set_ylabel("F1 change on those trips", fontsize=8.5); ax2.set_title("(b) What it is worth", loc="left", fontsize=8.5)
h_, l_ = axs.get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.53, 0.98), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.2, fontsize=8)
fig.savefig(OUT / "figs/fig_dev.pdf")
plt.close(fig)

# ---------------------------------------------------------------- aggregate numbers
gaps = [E9[c]["top1"]["incity_logit"]["length"] - E9[c]["top1"]["zeroshot_pooled"]["length"] for c in CITIES]
gains = [E9[c]["top1"]["incity_gbdt"]["length"] - E9[c]["top1"]["incity_logit"]["length"] for c in CITIES]
num("gapMax", max(gaps)); num("gainMin", min(gains)); num("gainMax", max(gains))
num("zsGbdtMaxDiff", max(abs(E4b[c]["G3_zeroshot_gbdt_pooled"]["top1_f1"] - E4b[c]["L3_zeroshot_logit_pooled"]["top1_f1"])
                         for c in CITIES))
(OUT / "numbers.json").write_text(json.dumps(N, indent=1))
tex = ["% Generated by paper_artifacts.py -- do not edit by hand."]
for k, v in sorted(N.items()):
    assert k.isalpha(), k
    tex.append(rf"\newcommand{{\n{k}}}{{{v}\xspace}}")
(OUT / "numbers.tex").write_text("\n".join(tex) + "\n")
print(f"wrote {len(N)} numbers, figs and tables to {OUT}")
