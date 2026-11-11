#!/bin/bash
source "$(dirname "$0")/../config.sh"
# In-city GTG: the unmodified official main.py with --src = --trg (cost prediction and preference learning
# on the city's own training trajectories; test trips come from traj/test.csv and are never seen in
# training). Separates GTG's model limits from its transfer loss. Usage: ./run_gtg_samecity.sh <devXA> <devCD>
cd $SRC
G=conda
LOG=gtg_samecity.log
TAG=110
out=results/gtg/samecity; mkdir -p $out
run_one () {  # city device
  c=$1; dev=$2
  echo "[$(date "+%F %T")] train exp_$TAG $c->$c on $dev" >> $LOG
  ( cd gtg_official && $G run --no-capture-output -n gtgp2 python main.py --exp_tag $TAG --src $c --trg $c --device $dev > ../train_samecity_${c}.log 2>&1 )
  echo "[$(date "+%F %T")] train exp_$TAG $c->$c exit=$?" >> $LOG
  ( cd gtg_official && timeout 3h $G run --no-capture-output -n gtgp2 python _driver_generate.py $c $c $dev ../$out/gen_${c}2${c}.csv ../results/gtg/ids_${c}.npy ckpt/exp_$TAG/${c}_to_${c}/pref.pth > ../gen_samecity_${c}.log 2>&1 )
  echo "[$(date "+%F %T")] gen exp_$TAG $c->$c exit=$?" >> $LOG
}
run_one xian ${1:-cuda:0} &
run_one chengdu ${2:-cuda:1} &
wait
GTG_DIRS=xian:xian,chengdu:chengdu $G run --no-capture-output -n gtg python gtg_eval.py cuda:0 samecity > gtg_eval_samecity.log 2>&1
echo "[$(date "+%F %T")] eval samecity exit=$?" >> $LOG
echo "[$(date "+%F %T")] SAMECITYDONE" >> $LOG
