#!/bin/bash
source "$(dirname "$0")/../config.sh"
# Usage: hoser_fallback.sh <Dataset> <TAG> <cuda> [wait_for_script]
# Runs official train.py (unless an official run of this city is already in progress: then waits for it); if the
# official run ends with a CUDA OOM, reruns with train_accum.py (memory-only change, gradient-equivalent).
ds=$1; tag=$2; dev=$3; waitfor=$4
cd $HOSER_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$PY_HOSER
if [ -n "$waitfor" ]; then
  while pgrep -f "^/bin/bash ./$waitfor$" > /dev/null; do sleep 60; done
else
  echo "[$(date "+%F %T")] $tag official start (GTG repeat runs paused; allocator: expandable_segments)" >> train_status.log
  /usr/bin/time -f "$tag train wall %e s, maxrss %M KB" $PY train.py --dataset $ds --seed 0 --cuda $dev > train_$tag.log 2>&1
  echo "$tag exit=$?" >> train_status.log
fi
if [ -f save/$ds/seed0/best.pth ] && ! grep -aq "OutOfMemoryError" train_$tag.log; then exit 0; fi
if grep -aq "OutOfMemoryError" train_$tag.log; then
  mv train_$tag.log train_${tag}_oom_official.log
  echo "[$(date "+%F %T")] $tag official run OOM on shared GPU -> train_accum.py (HOSER_ACCUM=2, gradient-equivalent)" >> train_status.log
  HOSER_ACCUM=2 /usr/bin/time -f "$tag train wall %e s, maxrss %M KB" $PY train_accum.py --dataset $ds --seed 0 --cuda $dev > train_$tag.log 2>&1
  echo "$tag accum exit=$?" >> train_status.log
fi
