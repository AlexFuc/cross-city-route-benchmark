#!/bin/bash
# Reproduce every result file used by the paper (after data download/conversion and setup_third_party.sh).
# Order: candidate pools -> experiments E4-E10 -> recursive-logit comparison -> deep baselines -> extras -> paper.
source "$(dirname "$0")/../config.sh"
cd "$SRC"
L=reproduce.log
echo "[$(date '+%F %T')] start" >> $L
CC_DEVICE=cuda:0 $PY e4.py --stage pools --cities xian,chengdu,beijing --device cuda:0 --workers 8 > pools_cn.log 2>&1 &
CC_DEVICE=cuda:1 $PY e4.py --stage pools --cities porto,sanfrancisco --device cuda:1 --workers 8 > pools_eu.log 2>&1 &
wait
./run_len.sh                                   # E4, E9, E5, E8, E7, E4b, E6 (length-weighted F1)
CC_DEVICE=cuda:1 $PY e10.py cuda:1 > e10.log 2>&1
CC_F1=count CC_DEVICE=cuda:0 $PY e9.py cuda:0 > e9_count.log 2>&1   # count-based F1 robustness
for c in "chengdu xian" "xian chengdu" "porto chengdu"; do set -- $c; $PY rl2.py $1 $2 cuda:0 > rl2_$1.log 2>&1; done
RL2_ROBUST=1 RL2_TAG=_robust RL2_WARM="[-1.2782,0.0873,-0.3399,-1.912,-1.0506]" $PY rl2.py porto - cuda:0 > rl2_porto_robust.log 2>&1
CC_DEVICE=cuda:0 $PY rl_map6_len.py cuda:0 > rl_map6.log 2>&1
# GTG: official training (6 directions, ~1-3 days each) + generation + evaluation; repeat runs
./run_gtg2.sh && ./run_gtg_seeds.sh
$PY gtg_eval.py cuda:0 > gtg_eval.log 2>&1
./run_gtg_samecity.sh cuda:0 cuda:1          # in-city GTG reference (official main.py, --src = --trg)
( cd "$GTG_DIR" && CUDA_VISIBLE_DEVICES= $PY_GTGP2 ../gtg_label_check.py xian chengdu beijing )  # target-label check
python3 gtg_costs.py
# HOSER: official training per city (with gradient-equivalent fallback), generation, evaluation; few-shot
$PY hoser_ids.py 10000
( ./hoser_fallback.sh San_Francisco SF 0 & ./hoser_fallback.sh Porto PT 1 & wait )
./run_hoser_pipeline.sh
$PY hoser_fs_ids.py && $PY_HOSER hoser_fs_build.py && ./run_hoser_fs.sh
( ./run_hoser_budget.sh sanfrancisco San_Francisco 0 & ./run_hoser_budget.sh porto Porto 1 & wait ) && $PY hoser_fs_budget_eval.py cuda:0 > hoser_fs_budget_eval.log 2>&1  # 1e3 trips, 1e4 update budget
# extras and timing
$PY bok_flow.py > bok_flow.log 2>&1
$PY exact_sp_control.py > exact_sp.log 2>&1
./run_costs.sh
# statistics and analyses added for the EAB (CPU only)
CUDA_VISIBLE_DEVICES= CC_DEVICE=cpu $PY e11_stats.py > e11.log 2>&1      # OD-cluster bootstrap, TOST, Holm
CUDA_VISIBLE_DEVICES= CC_DEVICE=cpu $PY e11b_gtg_incity.py > e11b.log 2>&1  # in-city GTG vs its zero-shot runs
CUDA_VISIBLE_DEVICES= CC_DEVICE=cpu $PY e12_sources.py > e12.log 2>&1    # source subsets, features, source size
CUDA_VISIBLE_DEVICES= $PY e13_dist.py > e13.log 2>&1                      # distribution-level JSDs
CUDA_VISIBLE_DEVICES= E14_WORKERS=8 $PY e14_kcand.py > e14.log 2>&1       # candidate-set size (Oracle@K)
CUDA_VISIBLE_DEVICES= E15_WORKERS=8 $PY e15_kcand_logit.py > e15.log 2>&1  # logit over 6 vs 30 candidates
CUDA_VISIBLE_DEVICES= $PY e16_temporal.py > e16.log 2>&1                  # temporal split of in-city references
CUDA_VISIBLE_DEVICES= $PY e17_observed_ti.py > e17.log 2>&1               # observed route added to the set (not relied on)
CUDA_VISIBLE_DEVICES= $PY e17b_inset_ti.py > e17b.log 2>&1               # observed choices on covered trips
CUDA_VISIBLE_DEVICES= $PY e17c_inset_detail.py > e17c.log 2>&1           # selection and non-shortest covered trips
CUDA_VISIBLE_DEVICES= $PY e19_temporal_counts.py cpu > e19.log 2>&1      # counts from earlier days -> later flows
CUDA_VISIBLE_DEVICES= $PY e20_param_tests.py > e20.log 2>&1               # TTS on estimation data + Wald tests
CUDA_VISIBLE_DEVICES= $PY e20b_robust_tests.py > e20b.log 2>&1           # unpenalized MLE + OD-cluster-robust Wald (reported)
CC_DEVICE=cpu CUDA_VISIBLE_DEVICES= $PY e21_why.py A B C D E F > e21.log 2>&1   # mechanism analysis (Sec. VII)
$PY paper_artifacts.py
echo "[$(date '+%F %T')] done" >> $L
