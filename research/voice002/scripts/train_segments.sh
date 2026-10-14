#!/bin/zsh
# Train in separate processes of SEGMENT steps each; every process resumes weights, AdamW
# state and step from results/train/<run>/last.pt (or from <init> for a new run).
# On MPS a training process could stay broken after a corrupt step (non-finite gradients,
# 2026-10-13/14), so a failed segment is retried from the last saved point, at most
# MAX_FAILS times in a row.
#
#   scripts/train_segments.sh <run> <total-steps> <init.pt> [train.py args...]
#   scripts/train_segments.sh combo-v2 20000 results/train/film-v3b/step-2000.pt \
#       --variant both-sim --other-weight 3 --lr 5e-5 --lr-cond 5e-4 --val-set ls-dev2 --workers 7
#
# Env: SEGMENT (1000), MAX_FAILS (3).
set -u
cd "${0:A:h}/.."
if (( $# < 3 )); then
  echo "usage: $0 <run> <total-steps> <init.pt> [train.py args...]" >&2
  exit 2
fi
run=$1 total=$2 init=$3
shift 3
segment=${SEGMENT:-1000}
max_fails=${MAX_FAILS:-3}
last=results/train/$run/last.pt
variant=
# Same value as train.py's argparse: "--variant X" or "--variant=X", the last one wins.
args=("$@")
for (( i = 1; i <= $#args; i++ )); do
  case ${args[i]} in
    --variant) (( i < $#args )) && variant=${args[i+1]} ;;
    --variant=*) variant=${args[i]#--variant=} ;;
  esac
done

# Step a checkpoint resumes at; like train.load_init, another variant (e.g. film -> both-sim)
# starts at 0.
ckpt_step() {
  uv run python -c "
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(c.get('step', 0) if c['variant'] == sys.argv[2] else 0)" "$1" "$variant" 2>/dev/null
}

start_step() {
  if [[ -f $last ]]; then ckpt_step "$last"; else ckpt_step "$init"; fi
}

fails=0
while true; do
  step=$(start_step)
  if [[ ! $step == <-> ]]; then
    echo "cannot read the step from $last or $init" >&2
    exit 1
  fi
  (( step >= total )) && { echo "done: step $step"; exit 0 }
  from=$init
  [[ -f $last ]] && from=$last
  n=$(( total - step < segment ? total - step : segment ))
  echo "segment of $n steps from step $step ($from)"
  if uv run python -m voice002.train --run "$run" --init "$from" \
      --max-steps "$n" --val-every "$n" "$@"; then
    if (( $(start_step) <= step )); then
      echo "segment finished without a newer checkpoint" >&2
      exit 1
    fi
    fails=0
  else
    fails=$((fails + 1))
    echo "segment from step $step failed ($fails/$max_fails)" >&2
    (( fails >= max_fails )) && exit 1
  fi
done
