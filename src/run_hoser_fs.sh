#!/bin/bash
source "$(dirname "$0")/../config.sh"
# Few-shot HOSER: train (train_accum.py, gradient-equivalent to the official batch) + OD-conditioned generation on
# the 10k evaluation trips, for n in 1e3/1e4/1e5 (E5 seed-0 subsets). One queue per GPU.
FS=$HOSER_FS
C=$SRC
PY=$PY_HOSER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HOSER_ACCUM=2
queue () {  # city Name cuda
  city=$1; name=$2; dev=$3
  for n in 1000 10000 100000; do
    d=$FS/${city}_n$n; cd $d
    s=$(date +%s)
    $PY train_accum.py --dataset $name --seed 0 --cuda $dev > train.log 2>&1; rc=$?
    t=$(date +%s); echo "[$(date '+%F %T')] $city n=$n train exit=$rc wall=$((t-s))" >> $FS/fs.log
    [ -f save/$name/seed0/best.pth ] || continue
    $PY _driver_gene.py --dataset $name --seed 0 --cuda $dev --processes 8 --test_rows $C/results/hoser/rows_$city.npy \
        --out_csv $FS/gen_${city}_n$n.csv > gene.log 2>&1; rc=$?
    g=$(date +%s); echo "[$(date '+%F %T')] $city n=$n gen exit=$rc wall=$((g-t))" >> $FS/fs.log
  done
}
queue sanfrancisco San_Francisco 0 &
queue porto Porto 1 &
wait
cd $C && DATA_ROOT=$C/root CC_CITIES=xian,chengdu,beijing,porto,sanfrancisco CC_F1=length CC_DEVICE=cuda:0 \
  $PY hoser_fs_eval.py cuda:0 > hoser_fs_eval.log 2>&1
echo "[$(date '+%F %T')] eval exit=$?" >> $FS/fs.log
echo "[$(date '+%F %T')] FSDONE" >> $FS/fs.log
