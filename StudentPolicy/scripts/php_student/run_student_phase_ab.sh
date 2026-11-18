#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
checkpoint="${STUDENT_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
arm="${1:?Usage: $0 baseline|focus}"
case "$arm" in baseline|focus) ;; *) echo "[ERROR] Expected baseline or focus" >&2; exit 2;; esac
num_envs="${STUDENT_PHASE_AB_NUM_ENVS:-256}"
iterations="${STUDENT_PHASE_AB_ITERATIONS:-100}"
multiplier="${STUDENT_PHASE_AB_MULTIPLIER:-2.0}"
event_traces="${STUDENT_PHASE_AB_EVENT_TRACES:-$repo_dir/reports/php_student/contact_geometry_teacher_trace_seed43_20261030/traces}"
output_dir="${STUDENT_PHASE_AB_OUTPUT_ROOT:-$repo_dir/reports/php_student/phase_ab_2026_10_27}/$arm"
extra_args=()
if [[ "${STUDENT_PHASE_AB_SAVE_LOADED_ONLY:-0}" == 1 ]]; then
    extra_args+=(--save-loaded-only)
fi
if [[ -n "${STUDENT_PHASE_AB_SAVE_MILESTONES:-}" ]]; then
    extra_args+=(--save-milestones "$STUDENT_PHASE_AB_SAVE_MILESTONES")
fi
if [[ -n "${STUDENT_PHASE_AB_COLD_GATE_FALLBACK:-}" ]]; then
    extra_args+=(--cold-gate-fallback "$STUDENT_PHASE_AB_COLD_GATE_FALLBACK")
fi
if [[ -n "${STUDENT_PHASE_AB_ANCHOR_PREFIX_FRAMES:-}" ]]; then
    extra_args+=(--anchor-prefix-frames "$STUDENT_PHASE_AB_ANCHOR_PREFIX_FRAMES")
fi
if [[ -n "${STUDENT_PHASE_AB_ANCHOR_COEF:-}" ]]; then
    extra_args+=(--anchor-coef "$STUDENT_PHASE_AB_ANCHOR_COEF")
fi
if [[ -n "${STUDENT_PHASE_AB_GATE_REBUILD_VALID_SAMPLES:-}" ]]; then
    extra_args+=(--gate-rebuild-valid-samples "$STUDENT_PHASE_AB_GATE_REBUILD_VALID_SAMPLES")
fi
if [[ -n "${STUDENT_PHASE_AB_GATE_REBUILD_MODE:-}" ]]; then
    extra_args+=(--gate-rebuild-mode "$STUDENT_PHASE_AB_GATE_REBUILD_MODE")
fi
if [[ -n "${STUDENT_PHASE_AB_GATE_REBUILD_STATE:-}" ]]; then
    extra_args+=(--gate-rebuild-state "$STUDENT_PHASE_AB_GATE_REBUILD_STATE")
fi
if [[ -n "${STUDENT_PHASE_AB_EARLY_PPO_FRAMES:-}" ]]; then
    extra_args+=(--early-ppo-frames "$STUDENT_PHASE_AB_EARLY_PPO_FRAMES")
fi
if [[ -n "${STUDENT_PHASE_AB_EARLY_PPO_SCALE:-}" ]]; then
    extra_args+=(--early-ppo-scale "$STUDENT_PHASE_AB_EARLY_PPO_SCALE")
fi
if [[ -n "${STUDENT_PHASE_AB_REPLAY_TRACES:-}" ]]; then
    extra_args+=(--replay-traces "$STUDENT_PHASE_AB_REPLAY_TRACES")
fi
if [[ -n "${STUDENT_PHASE_AB_REPLAY_MODE:-}" ]]; then
    extra_args+=(--replay-mode "$STUDENT_PHASE_AB_REPLAY_MODE")
fi
if [[ -n "${STUDENT_PHASE_AB_REPLAY_BATCH_SIZE:-}" ]]; then
    extra_args+=(--replay-batch-size "$STUDENT_PHASE_AB_REPLAY_BATCH_SIZE")
fi
if [[ -n "${STUDENT_PHASE_AB_REPLAY_COEF:-}" ]]; then
    extra_args+=(--replay-coef "$STUDENT_PHASE_AB_REPLAY_COEF")
fi
[[ -f "$checkpoint" && -x "$python" ]] || { echo "[ERROR] Missing checkpoint or Python" >&2; exit 2; }
[[ -d "$event_traces" && -f "$event_traces/../motion_names.txt" ]] || { echo "[ERROR] Missing event traces or motion names: $event_traces" >&2; exit 2; }
[[ ! -e "$output_dir" ]] || { echo "[ERROR] Refusing to overwrite $output_dir" >&2; exit 2; }
source "$official_root/wbt_training/training_runs/setup_env.sh"
export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
php_launcher_init
php_activate_sim
"$HSSIM_PYTHON" "$repo_dir/scripts/php_student/train_student_phase_ab.py" \
    --checkpoint "$checkpoint" --output-dir "$output_dir" --arm "$arm" \
    --num-envs "$num_envs" --iterations "$iterations" --focus-multiplier "$multiplier" \
    --event-traces "$event_traces" "${extra_args[@]}"
[[ -f "$output_dir/ab_contract.json" && ! -e "$output_dir/run_failed.marker" ]] || { echo "[ERROR] Training failed or produced no contract" >&2; exit 1; }
if [[ "$iterations" -gt 0 && "${STUDENT_PHASE_AB_SAVE_LOADED_ONLY:-0}" != 1 ]]; then
    compgen -G "$output_dir/model_*.pt" >/dev/null || {
        echo "[ERROR] Training did not produce an expected checkpoint" >&2; exit 1;
    }
fi
