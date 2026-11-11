#!/bin/bash
# Unpack the released routes of GTG and HOSER to where the evaluation scripts read them, so that both deep
# generators can be re-scored without retraining (gtg_eval.py, e11_stats.py, e11b_gtg_incity.py, hoser_eval.py,
# hoser_fs_eval.py, hoser_fs_budget_eval.py; run gtg_ids.py, hoser_ids.py and hoser_fs_ids.py first).
set -e
source "$(dirname "$0")/../config.sh"
find "$SRC/results/gtg" -name "gen_*.csv.gz" -exec gunzip -kf {} \;
mkdir -p "$HOSER_DIR/gene_eval" "$HOSER_FS"
for f in "$REPO"/routes/hoser/gene_eval/*.csv.gz; do gunzip -c "$f" > "$HOSER_DIR/gene_eval/$(basename "${f%.gz}")"; done
for f in "$REPO"/routes/hoser/fewshot/*.csv.gz; do gunzip -c "$f" > "$HOSER_FS/$(basename "${f%.gz}")"; done
echo "routes unpacked"
