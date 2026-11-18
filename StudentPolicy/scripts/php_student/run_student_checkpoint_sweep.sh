#!/usr/bin/env bash
# Evaluate intermediate checkpoints from one training run under one fixed contract.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
train_dir="${STUDENT_SWEEP_TRAIN_DIR:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion}"
root="${STUDENT_SWEEP_ROOT:-$repo_dir/reports/php_student/checkpoint_sweep_v1}"
seed="${STUDENT_SWEEP_SEED:-43}"
episodes="${STUDENT_SWEEP_EPISODES:-1}"
checkpoint_ids="${STUDENT_SWEEP_CHECKPOINTS:-00000 00500 01000 02500 05000 07500 10000 15000 20000 25000 29999}"
mode="${1:-run}"

check() {
    [[ -x "$python" && -d "$train_dir" ]] || {
        echo "[ERROR] Missing Python or training directory" >&2; exit 2;
    }
    for id in $checkpoint_ids; do
        [[ -f "$train_dir/model_${id}.pt" ]] || {
            echo "[ERROR] Missing $train_dir/model_${id}.pt" >&2; exit 2;
        }
    done
}

evaluate() {
    mkdir -p "$root"
    for id in $checkpoint_ids; do
        out="$root/model_${id}"
        report="$out/full_clip_seed${seed}.json"
        if [[ -s "$report" ]]; then
            echo "[SKIP] $report"
            continue
        fi
        STUDENT_CHECKPOINT="$train_dir/model_${id}.pt" \
        STUDENT_EVAL_REPORT_DIR="$out" \
        STUDENT_EVAL_EPISODES_PER_MOTION="$episodes" \
        STUDENT_EVAL_NUM_ENVS=40 STUDENT_EVAL_MAX_STEPS=6000 \
        STUDENT_EVAL_SEED="$seed" STUDENT_EVAL_OBSTACLE_SEED="$seed" \
        STUDENT_EVAL_DIAGNOSTICS=1 STUDENT_EVAL_TAKEOVER_MODE=none \
        PHP_EVAL_PROFILE=clean \
          bash "$repo_dir/scripts/php_student/eval_student_full.sh"
    done
}

summarize() {
    "$python" "$repo_dir/scripts/php_student/audit_student_checkpoint_sweep.py" \
      --train-dir "$train_dir" --eval-root "$root" --output "$root/summary.json"
}

check
case "$mode" in
    run) evaluate; summarize ;;
    summarize) summarize ;;
    *) echo "Usage: bash $0 run|summarize" >&2; exit 2 ;;
esac
