#!/usr/bin/env bash
# Frozen-policy feasibility test: paired Student baseline, finite Teacher rescue, and release.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
checkpoint="${STUDENT_RECOVERY_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
root="${STUDENT_RECOVERY_ROOT:-$repo_dir/reports/php_student/recovery_map_v1}"
seed="${STUDENT_RECOVERY_SEED:-43}"
mode="${1:-help}"

check() {
    [[ -f "$checkpoint" && -x "$python" ]] || { echo "[ERROR] Missing checkpoint/Python" >&2; exit 2; }
    [[ "$seed" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid seed" >&2; exit 2; }
    mkdir -p "$root"
    local resolved
    resolved="$(realpath "$checkpoint")"
    if [[ -f "$root/checkpoint.txt" ]]; then
        [[ "$(<"$root/checkpoint.txt")" == "$resolved" ]] || {
            echo "[ERROR] Root belongs to another checkpoint: $root" >&2; exit 2;
        }
    else
        printf '%s\n' "$resolved" > "$root/checkpoint.txt"
    fi
}

eval_case() {
    local name="$1" takeover="$2" start_map="${3:-}" steps="${4:-0}" folder
    if [[ "$name" == baseline ]]; then folder="$root/baseline"; else folder="$root/eval/$name"; fi
    if [[ -s "$folder/full_clip_seed${seed}.json" ]]; then
        echo "[SKIP] $name already evaluated"; return
    fi
    STUDENT_CHECKPOINT="$checkpoint" STUDENT_EVAL_REPORT_DIR="$folder" \
    STUDENT_EVAL_EPISODES_PER_MOTION=1 STUDENT_EVAL_NUM_ENVS=40 \
    STUDENT_EVAL_MAX_STEPS=6000 STUDENT_EVAL_SEED="$seed" \
    STUDENT_EVAL_OBSTACLE_SEED="$seed" STUDENT_EVAL_DIAGNOSTICS=1 \
    STUDENT_EVAL_TRACE_MOTION_IDS="$(if [[ "$name" == baseline ]]; then printf all; fi)" \
    STUDENT_EVAL_TAKEOVER_MODE="$takeover" STUDENT_EVAL_TAKEOVER_START_MAP="$start_map" \
    STUDENT_EVAL_TAKEOVER_STEPS="$steps" PHP_EVAL_PROFILE=clean \
      bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}

prepare() {
    eval_case baseline none
    for lead in 20 0 -10; do
        "$python" "$repo_dir/scripts/php_student/audit_student_failure_boundary.py" \
          --trace-dir "$root/baseline/traces" \
          --output "$root/boundary_lead${lead}.json" \
          --start-map "$root/start_lead${lead}.json" \
          --lead-frames "$lead"
    done
    "$python" - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for lead in (20, 0, -10):
    report = json.loads((root / f"boundary_lead{lead}.json").read_text())
    if report["motion_count"] != 40 or any(
        row["recovery_start_source"] != "action_divergence" for row in report["rows"]
    ):
        raise SystemExit(f"[ERROR] Missing sustained divergence on some motions, lead={lead}")
print("[PASS] 40 paired baseline traces and divergence start maps")
PY
}

branch() {
    local lead="$1" steps="$2" name
    name="lead${lead}_steps${steps}"
    [[ -s "$root/start_lead${lead}.json" ]] || { echo "[ERROR] Run prepare first" >&2; exit 2; }
    eval_case "$name" mapped_window "$root/start_lead${lead}.json" "$steps"
}

case "$mode" in
    prepare) check; prepare ;;
    pilot) check; prepare; branch 20 80; "$python" "$repo_dir/scripts/php_student/summarize_student_recovery_map.py" --root "$root" --seed "$seed" ;;
    focused) check; prepare; branch 0 20; branch 0 80; branch -10 80; \
             "$python" "$repo_dir/scripts/php_student/summarize_student_recovery_map.py" --root "$root" --seed "$seed" ;;
    run) check; prepare; for lead in 20 0 -10; do for steps in 20 40 80; do branch "$lead" "$steps"; done; done; \
         "$python" "$repo_dir/scripts/php_student/summarize_student_recovery_map.py" --root "$root" --seed "$seed" ;;
    summary) check; "$python" "$repo_dir/scripts/php_student/summarize_student_recovery_map.py" --root "$root" --seed "$seed" ;;
    *) echo "Usage: bash $0 prepare|pilot|focused|run|summary"; [[ "$mode" == help ]] || exit 2 ;;
esac
