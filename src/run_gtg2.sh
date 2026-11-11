#!/bin/bash
source "$(dirname "$0")/../config.sh"
# Per-direction pipeline v2 (takeover-aware): as soon as a direction's official training exits, generate (official code) and re-run eval.
cd $SRC
G=conda
LOG=gtg_pipeline.log
[ -f results/gtg/ids_beijing.npy ] || $G run --no-capture-output -n gtg python gtg_ids.py >> $LOG 2>&1
cp gtg_generate.py gtg_official/_driver_generate.py
PAIRS="chengdu:xian beijing:xian beijing:chengdu xian:beijing xian:chengdu chengdu:beijing"
while true; do
  left=0
  i=0
  for pair in $PAIRS; do
    s=${pair%%:*}; t=${pair##*:}; dev=cuda:$((i%2)); i=$((i+1))
    out=results/gtg/gen_${s}2${t}.csv
    [ -f $out ] && continue
    [ -f results/gtg/.running_${s}2${t} ] && { left=$((left+1)); continue; }
    if ps aux | grep "[m]ain.py --exp_tag 101 --src $s --trg $t" | grep -q python || ps aux | grep "[_]bj_takeover.py $s $t" | grep -q python || { [ "$s" = beijing ] && [ ! -f results/gtg/.takeover_done_${s}2${t} ] && ! grep -q "TAKEOVER DONE" takeover_${s}2${t}.log 2>/dev/null; }; then left=$((left+1)); continue; fi
    if [ ! -f gtg_official/ckpt/exp_101/${s}_to_${t}/pref.pth ]; then echo "[$(date +%T)] $s->$t training gone but no pref.pth" >> $LOG; continue; fi
    touch results/gtg/.running_${s}2${t}; left=$((left+1))
    echo "[$(date +%T)] generating $s->$t on $dev" >> $LOG
    ( cd gtg_official && timeout 3h $G run --no-capture-output -n gtgp2 python _driver_generate.py $s $t $dev ../$out ../results/gtg/ids_${t}.npy > ../gen_${s}2${t}.log 2>&1
      echo "[$(date +%T)] gen $s->$t exit=$?" >> ../$LOG; rm -f ../results/gtg/.running_${s}2${t}
      cd .. && $G run --no-capture-output -n gtg python gtg_eval.py cuda:$((i%2)) > gtg_eval.log 2>&1
      echo "[$(date +%T)] eval refreshed after $s->$t" >> $LOG ) &
  done
  [ $left -eq 0 ] && break
  sleep 300
done
wait
echo "[$(date +%T)] ALLDONE" >> $LOG
