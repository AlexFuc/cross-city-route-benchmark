#!/bin/bash
source "$(dirname "$0")/../config.sh"
cd $SRC
export DATA_ROOT=$DATA_ROOT CC_CITIES=xian,chengdu,beijing,porto,sanfrancisco CC_F1=length CC_DEVICE=cuda:0
for c in xian chengdu beijing porto sanfrancisco; do
  $PY cost_ours.py $c cuda:0 >> cost_ours.log 2>&1
  echo "[$(date +%T)] $c exit=$?" >> cost_ours.log
done
echo COSTDONE >> cost_ours.log
