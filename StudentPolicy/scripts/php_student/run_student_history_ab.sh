#!/usr/bin/env bash
# Controlled A/B: one-frame versus three-frame proprioceptive Student history.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
checkpoint="${STUDENT_HISTORY_SOURCE:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
root="${STUDENT_HISTORY_ROOT:-$repo_dir/reports/php_student/history_ab_v1}"
mode="${1:-help}"
envs="${STUDENT_HISTORY_NUM_ENVS:-256}"
iterations="${STUDENT_HISTORY_ITERATIONS:-200}"
lr="${STUDENT_HISTORY_LR:-0.0001}"
eval_seed="${STUDENT_HISTORY_EVAL_SEED:-43}"

check() {
    [[ -f "$checkpoint" && -x "$python" && -f "$official_root/wbt_training/training_runs/setup_env.sh" ]] || {
        echo "[ERROR] Missing checkpoint, Python, or official setup" >&2; exit 2;
    }
}
activate() {
    source "$official_root/wbt_training/training_runs/setup_env.sh"
    export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
    php_launcher_init
    php_activate_sim
}
train_arm() {
    local arm="$1" target="$2" count="$3" nenv="$4"
    [[ ! -e "$target" ]] || { echo "[ERROR] Refusing to overwrite $target" >&2; exit 2; }
    "$HSSIM_PYTHON" "$repo_dir/scripts/php_student/train_student_history_ab.py" \
        --checkpoint "$checkpoint" --output-dir "$target" --arm "$arm" \
        --num-envs "$nenv" --iterations "$count" --learning-rate "$lr"
}
eval_arm() {
    local arm="$1" ckpt
    [[ -s "$root/train/$arm/checkpoint.txt" ]] || { echo "[ERROR] Train $arm first" >&2; exit 2; }
    ckpt="$(<"$root/train/$arm/checkpoint.txt")"
    [[ -f "$ckpt" ]] || { echo "[ERROR] Missing checkpoint $ckpt" >&2; exit 2; }
    STUDENT_CHECKPOINT="$ckpt" STUDENT_EVAL_REPORT_DIR="$root/eval/$arm" \
    STUDENT_EVAL_EPISODES_PER_MOTION="${STUDENT_HISTORY_EVAL_EPISODES:-3}" \
    STUDENT_EVAL_NUM_ENVS="${STUDENT_HISTORY_EVAL_ENVS:-40}" \
    STUDENT_EVAL_MAX_STEPS="${STUDENT_HISTORY_EVAL_MAX_STEPS:-6000}" \
    STUDENT_EVAL_SEED="$eval_seed" STUDENT_EVAL_OBSTACLE_SEED="$eval_seed" \
    STUDENT_EVAL_TAKEOVER_MODE=none STUDENT_EVAL_DIAGNOSTICS=1 PHP_EVAL_PROFILE=clean \
    bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}
summary() {
    "$python" "$repo_dir/scripts/php_student/summarize_student_history_ab.py" \
        --root "$root" --seed "$eval_seed"
}
case "$mode" in
    verify)
        check
        for arm in single history3; do
            "$python" "$repo_dir/scripts/php_student/train_student_history_ab.py" \
                --checkpoint "$checkpoint" --arm "$arm" --verify-only
        done ;;
    preflight)
        check; activate
        train_arm single "$root/preflight/single" 2 32
        train_arm history3 "$root/preflight/history3" 2 32 ;;
    train)
        check; activate
        train_arm single "$root/train/single" "$iterations" "$envs"
        train_arm history3 "$root/train/history3" "$iterations" "$envs" ;;
    eval)
        check; eval_arm single; eval_arm history3; summary ;;
    summary) summary ;;
    all)
        check; activate
        train_arm single "$root/train/single" "$iterations" "$envs"
        train_arm history3 "$root/train/history3" "$iterations" "$envs"
        eval_arm single; eval_arm history3; summary ;;
    *) echo "Usage: bash $0 verify|preflight|train|eval|summary|all"; [[ "$mode" == help ]] || exit 2 ;;
esac
