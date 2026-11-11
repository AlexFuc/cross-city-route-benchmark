#!/bin/bash
source "$(dirname "$0")/../config.sh"
# Wait until a Beijing-source official run logs disentangle epoch 595 and saves disent.pth, then hand over.
cd $GTG_DIR
G=conda
LOG=../bj_takeover.log
watch_one() {
  s=$1; t=$2; dev=$3; tl=train_${s}2${t}.log; ck=ckpt/exp_101/${s}_to_${t}
  echo "[$(date '+%F %T')] watching $s->$t ($tl)" >> $LOG
  while ! grep -aq "Disentangle train, epoch 595 " $tl; do sleep 60; done
  now=$(date +%s)
  echo "[$(date '+%F %T')] $s->$t logged epoch 595" >> $LOG
  for i in $(seq 1 30); do m=$(stat -c %Y $ck/disent.pth); [ $m -ge $((now - 180)) ] && break; sleep 10; done
  echo "[$(date '+%F %T')] $s->$t disent.pth mtime $(date -d @$(stat -c %Y $ck/disent.pth) '+%F %T')" >> $LOG
  sleep 20
  cp $ck/disent.pth $ck/disent_e595_backup.pth
  pid=$(ps aux | grep "[m]ain.py --exp_tag 101 --src $s --trg $t" | grep -v "conda run" | awk '{print $2}')
  echo "[$(date '+%F %T')] $s->$t killing official pid $pid" >> $LOG
  [ -n "$pid" ] && kill $pid
  sleep 15
  nohup $G run --no-capture-output -n gtgp2 python _bj_takeover.py $s $t $dev > ../takeover_${s}2${t}.log 2>&1 < /dev/null &
  echo "[$(date '+%F %T')] $s->$t takeover launched on $dev" >> $LOG
}
watch_one beijing xian cuda:0 &
watch_one beijing chengdu cuda:1 &
wait
