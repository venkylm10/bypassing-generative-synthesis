#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
LOG=/workspace/output/train.log
: > "$LOG"
for i in 0 1 2; do
  if [ "$i" = "0" ]; then TAG=seed0; else TAG=seed$i; fi
  echo "=========== STARTING RUN seed=$i tag=$TAG ===========" | tee -a "$LOG"
  python3 -u train.py --seed "$i" --tag "$TAG" 2>&1 | tee -a "$LOG"
  echo "=========== FINISHED RUN seed=$i tag=$TAG ===========" | tee -a "$LOG"
done
echo "ALL_SEEDS_COMPLETE" | tee -a "$LOG"
