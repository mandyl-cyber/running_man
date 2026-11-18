#!/usr/bin/env bash
# Matched short A/B from the early roll-in Student: only Teacher rescue action differs.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
checkpoint="${STUDENT_RECOVERY_AB_SOURCE:-$repo_dir/reports/php_student/teacher_rollin_early15_ab_v1/train/teacher/model_30098.pt}"
root="${STUDENT_RECOVERY_AB_ROOT:-$repo_dir/reports/php_student/recovery_ab_v1}"
mode="${1:-help}"
envs="${STUDENT_RECOVERY_AB_ENVS:-256}"
iterations="${STUDENT_RECOVERY_AB_ITERATIONS:-100}"
rollouts="${STUDENT_RECOVERY_AB_PROBE_ROLLOUTS:-4}"
steps="${STUDENT_RECOVERY_AB_STEPS:-80}"
max_events="${STUDENT_RECOVERY_AB_MAX_EVENTS:-3}"
cooldown="${STUDENT_RECOVERY_AB_COOLDOWN:-40}"
sustain="${STUDENT_RECOVERY_AB_SUSTAIN:-3}"
threshold="${STUDENT_RECOVERY_AB_THRESHOLD:-1.0}"

check() {
    [[ -f "$checkpoint" && -x "$python" && -f "$official_root/wbt_training/training_runs/setup_env.sh" ]] || {
        echo "[ERROR] Missing source checkpoint/Python/official setup" >&2; exit 2;
    }
    for value in "$envs" "$iterations" "$rollouts" "$steps" "$max_events" "$sustain"; do
        [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Expected positive integer: $value" >&2; exit 2; }
    done
    [[ "$cooldown" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid cooldown" >&2; exit 2; }
}

activate() {
    source "$official_root/wbt_training/training_runs/setup_env.sh"
    export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
    php_launcher_init
    php_activate_sim
}

arm_run() {
    local phase="$1" arm="$2" output="$root/$1/$2"
    [[ ! -e "$output" ]] || { echo "[ERROR] Refusing to reuse $output" >&2; exit 2; }
    "$HSSIM_PYTHON" "$repo_dir/scripts/php_student/student_recovery_ab.py" \
      --mode "$phase" --arm "$arm" --checkpoint "$checkpoint" \
      --output-dir "$output" --num-envs "$envs" --iterations "$iterations" \
      --rollouts "$rollouts" --steps "$steps" --max-events "$max_events" \
      --cooldown "$cooldown" --sustain "$sustain" --threshold "$threshold"
    [[ -s "$output/coverage.json" ]] || { echo "[ERROR] Missing $output/coverage.json" >&2; exit 1; }
    if [[ "$phase" == train ]]; then
        [[ -s "$output/checkpoint.txt" ]] || { echo "[ERROR] Missing $output/checkpoint.txt" >&2; exit 1; }
    fi
}

gate() {
    "$python" - "$root" "$checkpoint" <<'PY'
import json, pathlib, sys
root, checkpoint = pathlib.Path(sys.argv[1]), str(pathlib.Path(sys.argv[2]).resolve())
rows = {}
for arm in ('student', 'teacher'):
    folder = root / 'probe' / arm
    contract = json.loads((folder / 'contract.json').read_text())
    coverage = json.loads((folder / 'coverage.json').read_text())['recovery']
    assert contract['source_checkpoint'] == checkpoint, 'Source checkpoint mismatch'
    assert contract['arm'] == arm and contract['ppo_excluded_on_recovery_active'], 'Contract mismatch'
    assert coverage['event_count'] > 0 and coverage['active_valid_step_count'] > 0, 'No valid recovery events'
    if arm == 'student': assert coverage['teacher_control_step_count'] == 0
    else: assert coverage['teacher_control_step_count'] == coverage['active_valid_step_count']
    rows[arm] = coverage
summary = {'pass': True, 'source_checkpoint': checkpoint,
           'control_events': rows['student']['event_count'],
           'teacher_events': rows['teacher']['event_count'],
           'teacher_control_steps': rows['teacher']['teacher_control_step_count'],
           'teacher_invalid_aborts': rows['teacher']['invalid_abort_count']}
(root / 'probe' / 'gate.json').write_text(json.dumps(summary, indent=2) + '\n')
print('[PASS] Recovery A/B probe gate:', summary)
PY
}

eval_arm() {
    local arm="$1" seed="$2" ckpt folder
    ckpt="$(<"$root/train/$arm/checkpoint.txt")"
    [[ -f "$ckpt" ]] || { echo "[ERROR] Missing $arm checkpoint" >&2; exit 2; }
    folder="$root/eval/$arm/seed$seed"
    [[ ! -s "$folder/full_clip_seed${seed}.json" ]] || { echo "[SKIP] $arm seed$seed"; return; }
    STUDENT_CHECKPOINT="$ckpt" STUDENT_EVAL_REPORT_DIR="$folder" \
    STUDENT_EVAL_EPISODES_PER_MOTION="${STUDENT_RECOVERY_AB_EVAL_EPISODES:-3}" \
    STUDENT_EVAL_NUM_ENVS=40 STUDENT_EVAL_MAX_STEPS=12000 \
    STUDENT_EVAL_SEED="$seed" STUDENT_EVAL_OBSTACLE_SEED="$seed" \
    STUDENT_EVAL_DIAGNOSTICS=1 STUDENT_EVAL_TAKEOVER_MODE=none \
    PHP_EVAL_PROFILE=clean bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}

summary() {
    "$python" - "$root" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for arm in ('student','teacher'):
    print(arm)
    for p in sorted((root/'eval'/arm).glob('seed*/full_clip_seed*.json')):
        if not p.stat().st_size: continue
        r=json.loads(p.read_text())
        assert r['complete'] and r['intervention']['mode']=='none', p
        rows=r['motions']; s=sum(x['successes'] for x in rows); n=sum(x['episodes'] for x in rows)
        print(' ',p.parent.name,f'{s}/{n} ({s/n:.1%})')
PY
}

case "$mode" in
    preflight) check; activate; arm_run probe student; arm_run probe teacher; gate ;;
    train) check; [[ -s "$root/probe/gate.json" ]] || { echo "[ERROR] Run preflight first" >&2; exit 2; }; \
           activate; arm_run train student; arm_run train teacher ;;
    eval-gate) check; eval_arm student 43; eval_arm teacher 43; summary ;;
    eval) check; for seed in 43 44 45 46; do eval_arm student "$seed"; eval_arm teacher "$seed"; done; summary ;;
    summary) summary ;;
    *) echo "Usage: bash $0 preflight|train|eval-gate|eval|summary"; [[ "$mode" == help ]] || exit 2 ;;
esac
