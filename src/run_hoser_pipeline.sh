#!/bin/bash
source "$(dirname "$0")/../config.sh"
# After official HOSER training per city: OD-conditioned generation (official model + _driver_gene.py, 8 processes,
# the official default) on the 10k evaluation trips, then hoser_eval.py once both cities are generated.
H=$HOSER_DIR
C=$SRC
PY=$PY_HOSER
LOG=$H/pipeline.log
mkdir -p $H/gene_eval
gen_city () {  # dataset city cuda statusprefix
  ds=$1; city=$2; dev=$3; tag=$4
  until [ -f $H/save/$ds/seed0/best.pth ] && ! pgrep -f "train.py --dataset $ds " > /dev/null; do sleep 60; done
  echo "[$(date '+%F %T')] $ds trained; generating" >> $LOG
  s=$(date +%s)
  ( cd $H && $PY _driver_gene.py --dataset $ds --seed 0 --cuda $dev --processes 8 --test_rows $C/results/hoser/rows_$city.npy \
      --out_csv $H/gene_eval/gen_$city.csv > $H/gene_$city.log 2>&1 )
  rc=$?; e=$(date +%s)
  echo "[$(date '+%F %T')] $ds gen exit=$rc wall=$((e-s)) s" >> $LOG
  echo "$city $((e-s))" >> $H/gene_wall.txt
}
gen_city San_Francisco sanfrancisco 0 SF &
gen_city Porto porto 1 PT &
wait
cd $C && DATA_ROOT=$C/root CC_CITIES=xian,chengdu,beijing,porto,sanfrancisco CC_F1=length CC_DEVICE=cuda:0 \
  $PY hoser_eval.py cuda:0 > hoser_eval.log 2>&1
echo "[$(date '+%F %T')] eval exit=$?" >> $LOG
python3 - <<PY >> $LOG 2>&1
import json, re
H = "$H"
out = {}
for city, ds, tag in [("porto", "Porto", "PT"), ("sanfrancisco", "San_Francisco", "SF")]:
    tr = open(f"{H}/train_{tag}.log").read()
    m = re.findall(r"train wall ([0-9.]+) s", tr)
    g = dict(l.split() for l in open(f"{H}/gene_wall.txt"))
    out[city] = {"train_s": float(m[-1]) if m else None, "generate_s_10k": int(g[city]),
                 "generate_s_per_1e5": int(g[city]) * 10, "gen_processes": 8}
json.dump(out, open("$C/results/hoser/HOSER_costs.json", "w"), indent=2)
print("costs", out)
PY
echo "[$(date '+%F %T')] PIPELINE DONE" >> $LOG
