#!/usr/bin/env bash
# Three independent probes: one-clip specialization, observation ceiling, frozen Teacher rescue.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
default_checkpoint="$repo_dir/reports/php_student/teacher_rollin_early15_ab_v1/train/teacher/model_30098.pt"
checkpoint="${STUDENT_STRUCTURAL_CHECKPOINT:-$default_checkpoint}"
teacher_checkpoint="${STUDENT_STRUCTURAL_TEACHER_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261015_072805-php-teacher-seed42-locomotion/model_19999.pt}"
bundle="${PHP_BUNDLE:-/home/sjtu/.cache/php-parkour/motions-low-climb-76}"
root="${STUDENT_STRUCTURAL_ROOT:-$repo_dir/reports/php_student/structural_tests_v1}"
source_traces="${STUDENT_STRUCTURAL_TRACES:-$repo_dir/reports/php_student/continuation_collapse_v1/paired_rollout_v1/source/traces}"
clip="${STUDENT_STRUCTURAL_CLIP:-forward_walk0.6s_step0.7s}"
iterations="${STUDENT_STRUCTURAL_ITERATIONS:-200}"
mode="${1:-help}"

check() {
    [[ -x "$python" && -f "$checkpoint" && -f "$teacher_checkpoint" && -d "$bundle" ]] || {
        echo "[ERROR] Missing PHP Python, checkpoint, or bundle" >&2; exit 2;
    }
    [[ "$iterations" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid iterations" >&2; exit 2; }
    [[ "$clip" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "[ERROR] Invalid clip name" >&2; exit 2; }
    [[ -f "$bundle/${clip}_motion.npz" && -f "$bundle/${clip}_terrain.npy" ]] || {
        echo "[ERROR] Motion and terrain must both exist: $clip" >&2; exit 2;
    }
    [[ -d "$source_traces" && -f "$source_traces/../motion_names.txt" ]] || {
        echo "[ERROR] Missing paired autonomous Student traces" >&2; exit 2;
    }
    if [[ "$(realpath "$checkpoint")" != "$(realpath "$default_checkpoint")" && -z "${STUDENT_STRUCTURAL_TRACES:-}" ]]; then
        echo "[ERROR] A different checkpoint requires matching STUDENT_STRUCTURAL_TRACES" >&2; exit 2;
    fi
    local count
    count="$(find "$source_traces" -maxdepth 1 -type f -name 'motion_*.npz' | wc -l)"
    [[ "$count" -eq 40 ]] || { echo "[ERROR] Expected 40 source traces, got $count" >&2; exit 2; }
    local contract expected
    contract="$root/input_contract.txt"
    expected="$(realpath "$checkpoint")|$(realpath "$teacher_checkpoint")|$(realpath "$bundle")|$(realpath "$source_traces")|$clip|$iterations"
    mkdir -p "$root"
    if [[ -f "$contract" ]]; then
        [[ "$(<"$contract")" == "$expected" ]] || {
            echo "[ERROR] Root belongs to another input contract: $contract" >&2; exit 2;
        }
    else
        printf '%s\n' "$expected" > "$contract"
    fi
    echo "[PASS] checkpoint=$checkpoint"
    echo "[PASS] original paired bundle=$bundle; clip=$clip; traces=$count"
}

prepare_single() {
    local subset="$root/single_clip/bundle"
    mkdir -p "$subset"
    if [[ ! -f "$subset/${clip}_motion.npz" ]]; then cp -- "$bundle/${clip}_motion.npz" "$subset/"; fi
    if [[ ! -f "$subset/${clip}_terrain.npy" ]]; then cp -- "$bundle/${clip}_terrain.npy" "$subset/"; fi
    [[ "$(find "$subset" -maxdepth 1 -type f -name '*_motion.npz' | wc -l)" -eq 1 ]] || {
        echo "[ERROR] Refusing a mixed or reused single-clip bundle: $subset" >&2; exit 2;
    }
    cmp -s "$bundle/${clip}_motion.npz" "$subset/${clip}_motion.npz"
    cmp -s "$bundle/${clip}_terrain.npy" "$subset/${clip}_terrain.npy"
    echo "[PASS] Paired motion/terrain subset is byte-identical to source"
}

single_baseline() {
    local report="$root/single_clip/baseline/full_clip_seed43.json"
    [[ ! -s "$report" ]] || { echo "[SKIP] $report"; return; }
    STUDENT_CHECKPOINT="$checkpoint" PHP_BUNDLE="$root/single_clip/bundle" \
      STUDENT_EVAL_REPORT_DIR="$root/single_clip/baseline" \
      STUDENT_EVAL_EPISODES_PER_MOTION=20 STUDENT_EVAL_NUM_ENVS=40 \
      STUDENT_EVAL_MAX_STEPS=12000 STUDENT_EVAL_SEED=43 STUDENT_EVAL_OBSTACLE_SEED=43 \
      PHP_EVAL_PROFILE=clean bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}

single_teacher_gate() {
    local folder="$root/single_clip/teacher"
    [[ ! -s "$folder/full_clip_seed43.json" ]] || { echo "[SKIP] Teacher gate already evaluated"; return; }
    TEACHER_CHECKPOINT="$teacher_checkpoint" PHP_BUNDLE="$root/single_clip/bundle" \
      PHP_EVAL_REPORT_DIR="$folder" PHP_EVAL_EPISODES_PER_MOTION=20 \
      PHP_EVAL_NUM_ENVS=40 PHP_TEACHER_MAX_EVAL_STEPS=12000 PHP_EVAL_SEED=43 \
      PHP_EVAL_PROFILE=clean bash "$repo_dir/../new_teacherpolicy/scripts/php_official/eval_teacher_full.sh"
}

single_train() {
    local output="$root/single_clip/train"
    [[ ! -s "$output/model_final.pt" ]] || { echo "[SKIP] $output/model_final.pt"; return; }
    [[ ! -e "$output" ]] || { echo "[ERROR] Training output already exists: $output" >&2; exit 2; }
    export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
    source /home/sjtu/php_parkour_official/wbt_training/training_runs/setup_env.sh
    php_launcher_init
    php_activate_sim
    "$python" "$repo_dir/scripts/php_student/train_student_single_clip_probe.py" \
      --checkpoint "$checkpoint" --bundle "$root/single_clip/bundle" \
      --output-dir "$output" --num-envs 256 --iterations "$iterations"
}

single_eval() {
    local trained="$root/single_clip/train/model_final.pt"
    [[ -f "$trained" ]] || { echo "[ERROR] Run single-train first" >&2; exit 2; }
    [[ ! -s "$root/single_clip/specialized/full_clip_seed43.json" ]] || {
        echo "[SKIP] Single-clip specialized policy already evaluated"; return;
    }
    STUDENT_CHECKPOINT="$trained" PHP_BUNDLE="$root/single_clip/bundle" \
      STUDENT_EVAL_REPORT_DIR="$root/single_clip/specialized" \
      STUDENT_EVAL_EPISODES_PER_MOTION=20 STUDENT_EVAL_NUM_ENVS=40 \
      STUDENT_EVAL_MAX_STEPS=12000 STUDENT_EVAL_SEED=43 STUDENT_EVAL_OBSTACLE_SEED=43 \
      PHP_EVAL_PROFILE=clean bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}

observability() {
    local output="$root/observability/observation_vs_state.json"
    if [[ ! -s "$output" ]]; then
        PHP_BUNDLE="$bundle" "$python" "$repo_dir/scripts/php_student/probe_observation_vs_state.py" \
          --checkpoint "$checkpoint" --teacher-traces "$source_traces" \
          --output "$output" --steps 600 --phase-start 0.25 --phase-end 0.65 --seed 42
    fi
    local trainable="$root/observability/trainable_depth.json"
    if [[ ! -s "$trainable" ]]; then
        PHP_BUNDLE="$bundle" "$python" "$repo_dir/scripts/php_student/probe_trainable_depth.py" \
          --checkpoint "$checkpoint" --teacher-traces "$source_traces" \
          --output "$trainable" --steps 600 --folds 5 --seed 42 --device cpu
    fi
}

recovery() {
    STUDENT_RECOVERY_CHECKPOINT="$checkpoint" STUDENT_RECOVERY_ROOT="$root/recovery" \
      STUDENT_RECOVERY_SEED=43 bash "$repo_dir/scripts/php_student/run_student_recovery_map.sh" pilot
}

summary() {
    "$python" "$repo_dir/scripts/php_student/summarize_student_structural_tests.py" \
      --root "$root" --source-traces "$source_traces"
}

case "$mode" in
    preflight) check; prepare_single ;;
    single-teacher-gate) check; prepare_single; single_teacher_gate; summary ;;
    single-baseline) check; prepare_single; single_baseline; summary ;;
    single-train) check; prepare_single; single_train ;;
    single-eval) check; prepare_single; single_eval; summary ;;
    observability) check; observability; summary ;;
    recovery) check; recovery; summary ;;
    summary) check; summary ;;
    all) check; prepare_single; single_teacher_gate; single_baseline; single_train; single_eval; observability; recovery; summary ;;
    *) echo "Usage: bash $0 preflight|single-teacher-gate|single-baseline|single-train|single-eval|observability|recovery|summary|all"; [[ "$mode" == help ]] || exit 2 ;;
esac
