#!/bin/bash
source "$(dirname "$0")/../config.sh"
# Repeat runs of the official GTG code (no seed is set in the official code, so each run differs in random
# init and data order). exp_tag 102/103 x {CD->XA, XA->BJ}; then official-model generation + our evaluation
# into results/gtg/seeds/exp_<tag>/.
cd $SRC
G=conda
LOG=gtg_seeds.log
run_one () {  # tag src trg device
  tag=$1; s=$2; t=$3; dev=$4
  out=results/gtg/seeds/exp_$tag; mkdir -p $out
  echo "[$(date '+%F %T')] train exp_$tag $s->$t on $dev" >> $LOG
  ( cd gtg_official && $G run --no-capture-output -n gtgp2 python main.py --exp_tag $tag --src $s --trg $t --device $dev > ../train_seed${tag}_${s}2${t}.log 2>&1 )
  echo "[$(date '+%F %T')] train exp_$tag $s->$t exit=$?" >> $LOG
  ( cd gtg_official && timeout 3h $G run --no-capture-output -n gtgp2 python _driver_generate.py $s $t $dev ../$out/gen_${s}2${t}.csv ../results/gtg/ids_${t}.npy ckpt/exp_$tag/${s}_to_${t}/pref.pth > ../gen_seed${tag}_${s}2${t}.log 2>&1 )
  echo "[$(date '+%F %T')] gen exp_$tag $s->$t exit=$?" >> $LOG
}
run_one 102 chengdu xian cuda:0 &
run_one 103 chengdu xian cuda:1 &
run_one 102 xian beijing cuda:1 &
run_one 103 xian beijing cuda:0 &
wait
for tag in 102 103; do
  $G run --no-capture-output -n gtg python gtg_eval.py cuda:0 seeds/exp_$tag > gtg_eval_seed$tag.log 2>&1
  echo "[$(date '+%F %T')] eval exp_$tag exit=$?" >> $LOG
done
echo "[$(date '+%F %T')] SEEDSDONE" >> $LOG
