#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
checkpoint="${STUDENT_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
bundle="${PHP_BUNDLE:-/home/sjtu/.cache/php-parkour/motions-low-climb-76}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
episodes="${STUDENT_EVAL_EPISODES_PER_MOTION:-100}"
num_envs="${STUDENT_EVAL_NUM_ENVS:-40}"
max_steps="${STUDENT_EVAL_MAX_STEPS:-120000}"
seed="${STUDENT_EVAL_SEED:-42}"
obstacle_seed="${STUDENT_EVAL_OBSTACLE_SEED:-42}"
diagnostics="${STUDENT_EVAL_DIAGNOSTICS:-0}"
trace_ids="${STUDENT_EVAL_TRACE_MOTION_IDS:-}"
trace_stride="${STUDENT_EVAL_TRACE_DEPTH_STRIDE:-5}"
takeover_mode="${STUDENT_EVAL_TAKEOVER_MODE:-none}"
takeover_start="${STUDENT_EVAL_TAKEOVER_START_FRACTION:-0.25}"
takeover_steps="${STUDENT_EVAL_TAKEOVER_STEPS:-0}"
takeover_joint_ids="${STUDENT_EVAL_TAKEOVER_JOINT_IDS:-}"
rescue_max_events="${STUDENT_EVAL_RESCUE_MAX_EVENTS:-3}"
rescue_sustain="${STUDENT_EVAL_RESCUE_SUSTAIN:-3}"
rescue_threshold="${STUDENT_EVAL_RESCUE_ACTION_THRESHOLD:-1.0}"
rescue_cooldown="${STUDENT_EVAL_RESCUE_COOLDOWN_STEPS:-40}"
takeover_start_map="${STUDENT_EVAL_TAKEOVER_START_MAP:-}"
profile="${PHP_EVAL_PROFILE:-clean}"
report_dir="${STUDENT_EVAL_REPORT_DIR:-$repo_dir/reports/php_student/full_clip_${profile}}"
case "$report_dir" in
    /*) ;;
    *) report_dir="$repo_dir/$report_dir" ;;
esac

[[ -f "$checkpoint" ]] || { echo "[ERROR] Missing checkpoint: $checkpoint" >&2; exit 2; }
[[ -d "$bundle" ]] || { echo "[ERROR] Missing bundle: $bundle" >&2; exit 2; }
[[ -x "$python" ]] || { echo "[ERROR] Missing PHP Python: $python" >&2; exit 2; }
[[ -f "$official_root/wbt_training/training_runs/setup_env.sh" ]] || { echo "[ERROR] Missing official setup" >&2; exit 2; }
for value in "$episodes" "$num_envs" "$max_steps" "$seed" "$obstacle_seed"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Expected positive integer: $value" >&2; exit 2; }
done
case "$profile" in clean|official_randomized) ;; *) echo "[ERROR] Invalid PHP_EVAL_PROFILE: $profile" >&2; exit 2;; esac
case "$diagnostics" in 0|1) ;; *) echo "[ERROR] STUDENT_EVAL_DIAGNOSTICS must be 0 or 1" >&2; exit 2;; esac
case "$takeover_mode" in none|window|all|mapped_tail|mapped_window|rescue_events) ;; *) echo "[ERROR] Invalid STUDENT_EVAL_TAKEOVER_MODE" >&2; exit 2;; esac
if [[ "$takeover_mode" != none && "$diagnostics" != 1 ]]; then
    echo "[ERROR] Teacher takeover requires STUDENT_EVAL_DIAGNOSTICS=1" >&2; exit 2
fi
[[ "$takeover_steps" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid STUDENT_EVAL_TAKEOVER_STEPS" >&2; exit 2; }
if [[ "$takeover_mode" == mapped_tail || "$takeover_mode" == mapped_window ]]; then
    [[ -f "$takeover_start_map" ]] || { echo "[ERROR] $takeover_mode requires STUDENT_EVAL_TAKEOVER_START_MAP" >&2; exit 2; }
    takeover_start_map="$(cd "$(dirname "$takeover_start_map")" && pwd)/$(basename "$takeover_start_map")"
fi
[[ "$trace_stride" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid STUDENT_EVAL_TRACE_DEPTH_STRIDE" >&2; exit 2; }
if [[ "$takeover_mode" == mapped_window && "$takeover_steps" -eq 0 ]]; then
    echo "[ERROR] mapped_window requires STUDENT_EVAL_TAKEOVER_STEPS > 0" >&2; exit 2
fi
if [[ "$takeover_mode" == rescue_events ]]; then
    [[ "$takeover_steps" -gt 0 && "$rescue_max_events" =~ ^[1-9][0-9]*$ && "$rescue_sustain" =~ ^[1-9][0-9]*$ && "$rescue_cooldown" =~ ^[0-9]+$ ]] || {
        echo "[ERROR] rescue_events requires positive steps/max events/sustain and nonnegative cooldown" >&2; exit 2;
    }
fi
if [[ -n "$trace_ids" && "$diagnostics" != 1 ]]; then
    echo "[ERROR] Trace capture requires STUDENT_EVAL_DIAGNOSTICS=1" >&2; exit 2
fi
[[ -z "$trace_ids" || "$trace_ids" == all || "$trace_ids" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "[ERROR] Invalid STUDENT_EVAL_TRACE_MOTION_IDS" >&2; exit 2; }
mkdir -p "$report_dir"
report_dir="$(cd "$report_dir" && pwd)"
report="$report_dir/full_clip_seed${seed}.json"
names="$report_dir/motion_names.txt"
[[ ! -s "$report" ]] || { echo "[ERROR] Report exists; set a fresh STUDENT_EVAL_REPORT_DIR: $report" >&2; exit 2; }

find "$bundle" -maxdepth 1 -type f -name '*_motion.npz' -printf '%f\n' \
    | sort | sed 's/_motion\.npz$//' > "$names"
motion_count="$(wc -l < "$names")"
[[ "$motion_count" -gt 0 ]] || { echo "[ERROR] No motions in $bundle" >&2; exit 2; }
echo "[STUDENT FULL EVAL] profile=$profile motions=$motion_count episodes/motion=$episodes envs=$num_envs"
echo "[STUDENT FULL EVAL] report=$report"
: > "$report"

export HSSIM_PYTHON="$python" LOGGER=disabled
export PYTHONUNBUFFERED=1
export REGISTRY="file://$bundle" CHECKPOINT="$checkpoint"
export PHP_EVAL_PROFILE="$profile" PHP_EVAL_REPORT_OUTPUT="$report"
export PHP_EVAL_EPISODES_PER_MOTION="$episodes" PHP_EVAL_MOTION_NAMES_FILE="$names"
export PHP_EVAL_OBSTACLE_SEED="$obstacle_seed" PHP_EVAL_DIAGNOSTICS="$diagnostics"
export PHP_EVAL_TAKEOVER_MODE="$takeover_mode" PHP_EVAL_TAKEOVER_START_FRACTION="$takeover_start"
export PHP_EVAL_TAKEOVER_STEPS="$takeover_steps"
export PHP_EVAL_TAKEOVER_JOINT_IDS="$takeover_joint_ids"
export PHP_EVAL_RESCUE_MAX_EVENTS="$rescue_max_events"
export PHP_EVAL_RESCUE_SUSTAIN="$rescue_sustain"
export PHP_EVAL_RESCUE_ACTION_THRESHOLD="$rescue_threshold"
export PHP_EVAL_RESCUE_COOLDOWN_STEPS="$rescue_cooldown"
export PHP_EVAL_TAKEOVER_START_MAP="$takeover_start_map"
export PHP_EVAL_TRACE_MOTION_IDS="$trace_ids" PHP_EVAL_TRACE_DEPTH_STRIDE="$trace_stride"
export PHP_EVAL_TRACE_DIR="$report_dir/traces"
source "$official_root/wbt_training/training_runs/setup_env.sh"
php_launcher_init
php_prepare_inputs eval_student "$CHECKPOINT"
CHECKPOINT="$PHP_CHECKPOINTS"
php_activate_sim
"$HSSIM_PYTHON" "$repo_dir/scripts/php_student/eval_student_full.py" \
    --checkpoint "$CHECKPOINT" \
    --training.registry-name "$REGISTRY" \
    --training.headless True \
    --training.num-envs "$num_envs" \
    --training.max-eval-steps "$max_steps" \
    --training.seed "$seed" \
    --training.export-onnx False \
    --terrain.terrain-term.spawn.randomize-tiles True \
    --simulator.config.sim.max-episode-length-s 20.0 \
    logger:disabled --logger.video.enabled False \
    --training.preprocess-hook-kwargs "{\"add_onpath_obstacle\": true, \"num_variants\": 1, \"obstacle_seed\": $obstacle_seed}" \
    "$@"

[[ -s "$report" ]] || {
    echo "[ERROR] Student evaluator exited without producing a report: $report" >&2
    exit 1
}
"$python" - "$report" "$episodes" "$motion_count" <<'PY'
import json
import sys
path, target, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as stream:
    report = json.load(stream)
rows = report.get("motions", [])
if not report.get("complete") or len(rows) != count or any(row["episodes"] < target for row in rows):
    raise SystemExit(f"[ERROR] Incomplete student full-clip report: {path}")
total = sum(row["episodes"] for row in rows)
success = sum(row["successes"] for row in rows)
print(f"[PASS] Student full-clip report: {path}")
print(f"[RESULT] {success}/{total} successes ({success / total:.1%}) across {len(rows)} motions")
PY
