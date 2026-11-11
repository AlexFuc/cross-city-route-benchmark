#!/bin/bash
source "$(dirname "$0")/../config.sh"
# Reviewer check (training budget): HOSER on the n=1e3 few-shot subsets with the update budget of n=1e4 under the
# official schedule. Only max_epoch differs from the official config (250 instead of 25: 4 batches x 250 = 1,000
# updates = 40 x 25); train_accum.py, data subsets, validation per epoch and best-checkpoint selection as in
# run_hoser_fs.sh. Usage: ./run_hoser_budget.sh <city> <Name> <cuda>
FS=$HOSER_FS
C=$SRC
PY=$PY_HOSER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HOSER_ACCUM=2
city=$1; name=$2; dev=$3
src=$FS/${city}_n1000; d=$FS/${city}_n1000_e250
mkdir -p $d && cp -a $src/{config,data,dataset.py,_driver_gene.py,models,train.py,train_accum.py,utils.py} $d/
sed -i 's/^  max_epoch: 25$/  max_epoch: 250/' $d/config/$name.yaml
grep -q '^  max_epoch: 250$' $d/config/$name.yaml || { echo "config edit failed" >> $FS/fs_budget.log; exit 1; }
diff <(sed 's/max_epoch: 250/max_epoch: 25/' $d/config/$name.yaml) $src/config/$name.yaml > /dev/null || { echo "config differs beyond max_epoch" >> $FS/fs_budget.log; exit 1; }
cmp -s $d/data/$name/train.csv $src/data/$name/train.csv || { echo "train subset differs" >> $FS/fs_budget.log; exit 1; }
cd $d
s=$(date +%s)
$PY train_accum.py --dataset $name --seed 0 --cuda $dev > train.log 2>&1; rc=$?
t=$(date +%s); echo "[$(date '+%F %T')] $city n=1000 e250 train exit=$rc wall=$((t-s))" >> $FS/fs_budget.log
[ -f save/$name/seed0/best.pth ] || exit 1
$PY _driver_gene.py --dataset $name --seed 0 --cuda $dev --processes 8 --test_rows $C/results/hoser/rows_$city.npy \
    --out_csv $FS/gen_${city}_n1000_e250.csv > gene.log 2>&1; rc=$?
g=$(date +%s); echo "[$(date '+%F %T')] $city n=1000 e250 gen exit=$rc wall=$((g-t))" >> $FS/fs_budget.log
echo "[$(date '+%F %T')] $city BUDGETDONE" >> $FS/fs_budget.log
