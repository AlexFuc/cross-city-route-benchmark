#!/bin/bash
source "$(dirname "$0")/../config.sh"
# Re-run the paper experiments with length-weighted F1 for candidate scores, oracle labels, OD-history and LOO.
cd $SRC
export DATA_ROOT=$DATA_ROOT CC_CITIES=xian,chengdu,beijing,porto,sanfrancisco CC_F1=length CC_DEVICE=cuda:1
P="$PY"
L=len_runs.log
echo "[$(date +%T)] start" >> $L
$P e4.py --stage eval --cities xian,chengdu,beijing,porto,sanfrancisco --device cuda:1 > len_e4.log 2>&1; echo "[$(date +%T)] e4 exit=$?" >> $L
$P e9.py cuda:1 > len_e9.log 2>&1; echo "[$(date +%T)] e9 exit=$?" >> $L
$P e5.py cuda:1 > len_e5.log 2>&1; echo "[$(date +%T)] e5 exit=$?" >> $L
$P e8.py cuda:1 > len_e8.log 2>&1; echo "[$(date +%T)] e8 exit=$?" >> $L
$P e7.py cuda:1 > len_e7.log 2>&1; echo "[$(date +%T)] e7 exit=$?" >> $L
$P e4b.py > len_e4b.log 2>&1; echo "[$(date +%T)] e4b exit=$?" >> $L
$P e6.py cuda:1 > len_e6.log 2>&1; echo "[$(date +%T)] e6 exit=$?" >> $L
echo "[$(date +%T)] ALLDONE" >> $L
