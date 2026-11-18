#!/usr/bin/env bash
# Causal probe: only desired_kl changes (0.01 -> 0.03), limiting early LR collapse.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
teacher="${TEACHER_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261015_072805-php-teacher-seed42-locomotion/model_19999.pt}"
bundle="${PHP_BUNDLE:-/home/sjtu/.cache/php-parkour/motions-low-climb-76}"
root="${STUDENT_LR_PROBE_ROOT:-$repo_dir/reports/php_student/lr_probe_kl003_v1}"
run_name="${STUDENT_LR_PROBE_NAME:-student-lr-probe-kl003-seed42}"
mode="${1:-run}"

check() {
    [[ -x "$python" && -f "$teacher" && -d "$bundle" ]] || {
        echo "[ERROR] Missing Python, Teacher checkpoint, or bundle" >&2; exit 2;
    }
}

train() {
    check; mkdir -p "$root"
    if [[ -s "$root/train_dir.txt" ]]; then echo "[SKIP] Training already recorded"; return; fi
    local before newest
    before="$(date +%s)"
    cd "$official_root"
    TEACHER_CHECKPOINT="$teacher" REGISTRY="file://$bundle" FINETUNE=1 LOGGER=disabled NGPUS=1 \
      bash wbt_training/training_runs/run_terrain_warp_distill.sh \
        --training.headless True --training.num-envs 2048 --training.seed 42 \
        --training.name "$run_name" --training.export-onnx False \
        --algo.config.num-learning-iterations 2501 --algo.config.save-interval 500 \
        --algo.config.desired-kl 0.03
    newest="$(find "$official_root/logs/WBT-Holosoma" -mindepth 1 -maxdepth 1 -type d \
      -name "*-${run_name}-locomotion" -newermt "@$before" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    [[ -n "$newest" && -f "$newest/model_02500.pt" ]] || {
        echo "[ERROR] Could not resolve completed probe directory" >&2; exit 1;
    }
    printf '%s\n' "$newest" > "$root/train_dir.txt"
}

evaluate() {
    local train_dir id out report
    [[ -s "$root/train_dir.txt" ]] || { echo "[ERROR] Run train first" >&2; exit 2; }
    train_dir="$(<"$root/train_dir.txt")"
    for id in 00500 01000 02500; do
        out="$root/eval/model_${id}"; report="$out/full_clip_seed43.json"
        [[ ! -s "$report" ]] || { echo "[SKIP] $report"; continue; }
        STUDENT_CHECKPOINT="$train_dir/model_${id}.pt" STUDENT_EVAL_REPORT_DIR="$out" \
        STUDENT_EVAL_EPISODES_PER_MOTION=1 STUDENT_EVAL_NUM_ENVS=40 \
        STUDENT_EVAL_MAX_STEPS=6000 STUDENT_EVAL_SEED=43 STUDENT_EVAL_OBSTACLE_SEED=43 \
        STUDENT_EVAL_DIAGNOSTICS=1 STUDENT_EVAL_TAKEOVER_MODE=none PHP_EVAL_PROFILE=clean \
          bash "$repo_dir/scripts/php_student/eval_student_full.sh"
    done
    "$python" "$repo_dir/scripts/php_student/gate_student_lr_probe.py" \
      --baseline-root "$repo_dir/reports/php_student/checkpoint_sweep_v1" \
      --candidate-root "$root/eval" --output "$root/gate.json"
}

case "$mode" in
    train) train ;;
    eval) evaluate ;;
    run) train; evaluate ;;
    *) echo "Usage: bash $0 train|eval|run" >&2; exit 2 ;;
esac
