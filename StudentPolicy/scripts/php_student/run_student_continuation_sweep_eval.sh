#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${STUDENT_CONTINUATION_SWEEP_ROOT:-$repo_dir/reports/php_student/continuation_collapse_v1}"
seed="${STUDENT_CONTINUATION_SWEEP_SEED:-43}"
[[ "$seed" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid seed: $seed" >&2; exit 2; }

for step in 001 005 010 025 050; do
    checkpoint="$root/sweep/baseline/model_step_$step.pt"
    report_dir="$root/eval/step${step}_seed$seed"
    report="$report_dir/full_clip_seed$seed.json"
    [[ -f "$checkpoint" ]] || { echo "[ERROR] Missing checkpoint: $checkpoint" >&2; exit 2; }
    if [[ -s "$report" ]]; then
        echo "[SKIP] $report"
        continue
    fi
    echo "[EVAL] step=$step checkpoint=$checkpoint"
    STUDENT_CHECKPOINT="$checkpoint" \
    STUDENT_EVAL_EPISODES_PER_MOTION=1 \
    STUDENT_EVAL_NUM_ENVS=40 \
    STUDENT_EVAL_MAX_STEPS=20000 \
    STUDENT_EVAL_SEED="$seed" \
    STUDENT_EVAL_OBSTACLE_SEED="$seed" \
    STUDENT_EVAL_REPORT_DIR="$report_dir" \
    STUDENT_EVAL_TAKEOVER_MODE=none \
    PHP_EVAL_PROFILE=clean \
    bash "$repo_dir/scripts/php_student/eval_student_full.sh"
done

"${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}" \
    "$repo_dir/scripts/php_student/summarize_student_continuation_sweep.py" \
    --root "$root" --seed "$seed"
