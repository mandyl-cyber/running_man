#!/usr/bin/env bash
# One entry point: paired probe -> coverage gate -> short continuation -> no-takeover eval.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
checkpoint="${STUDENT_ROLLIN_SOURCE:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
root="${STUDENT_ROLLIN_ROOT:-$repo_dir/reports/php_student/teacher_rollin_ab_v3}"
mode="${1:-help}"
envs="${STUDENT_ROLLIN_NUM_ENVS:-256}"
rollouts="${STUDENT_ROLLIN_PROBE_ROLLOUTS:-21}"
iterations="${STUDENT_ROLLIN_ITERATIONS:-100}"
start="${STUDENT_ROLLIN_WINDOW_START:-0.25}"
end="${STUDENT_ROLLIN_WINDOW_END:-0.55}"

usage() {
    echo "Usage: bash $0 smoke|preflight|diagnose|train|eval|all|summary"
    echo "Set STUDENT_ROLLIN_ROOT to a fresh output directory for each experiment."
}
check() {
    [[ -f "$checkpoint" && -x "$python" && -f "$official_root/wbt_training/training_runs/setup_env.sh" ]] || {
        echo "[ERROR] Missing source checkpoint, Python, or official setup" >&2; exit 2;
    }
    [[ "$envs" =~ ^[1-9][0-9]*$ && "$rollouts" =~ ^[1-9][0-9]*$ && "$iterations" =~ ^[1-9][0-9]*$ ]] || {
        echo "[ERROR] envs, rollouts, iterations must be positive integers" >&2; exit 2;
    }
}
activate() {
    source "$official_root/wbt_training/training_runs/setup_env.sh"
    export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
    php_launcher_init
    php_activate_sim
}
run_arm() {
    local phase="$1" arm="$2" count="$3" folder="${4:-$1}" output
    output="$root/$folder/$arm"
    [[ ! -e "$output" ]] || { echo "[ERROR] Refusing to reuse $output" >&2; exit 2; }
    "$HSSIM_PYTHON" "$repo_dir/scripts/php_student/teacher_rollin_ab.py" \
        --mode "$phase" --arm "$arm" --checkpoint "$checkpoint" \
        --output-dir "$output" --num-envs "$envs" \
        --rollouts "$count" --iterations "$iterations" \
        --window-start "$start" --window-end "$end"
    [[ -s "$output/coverage.json" ]] || {
        echo "[ERROR] $phase $arm produced no coverage report. See $output/error.txt" >&2; exit 1;
    }
    if [[ "$phase" == train ]]; then
        [[ -s "$output/checkpoint.txt" ]] || { echo "[ERROR] Missing trained checkpoint path" >&2; exit 1; }
    fi
}
gate() {
    "$python" "$repo_dir/scripts/php_student/gate_teacher_rollin_ab.py" \
        --student "$root/probe/student/coverage.json" \
        --teacher "$root/probe/teacher/coverage.json" \
        --output "$root/probe/gate.json"
}
train() {
    [[ -f "$root/probe/gate.json" ]] || { echo "[ERROR] Run preflight first" >&2; exit 2; }
    "$python" - "$root/probe/gate.json" "$root/probe/student/contract.json" "$root/probe/teacher/contract.json" "$checkpoint" "$envs" "$start" "$end" <<'PY'
import json,sys,pathlib
assert json.load(open(sys.argv[1]))['gate_pass'], 'Coverage gate failed'
for path in sys.argv[2:4]:
    c=json.load(open(path))
    assert c['source_checkpoint']==str(pathlib.Path(sys.argv[4]).resolve()), 'Probe source differs from training source'
    assert c['num_envs']==int(sys.argv[5]) and c['window']==[float(sys.argv[6]),float(sys.argv[7])], 'Probe settings differ from training settings'
PY
    activate
    run_arm train student "$rollouts"
    run_arm train teacher "$rollouts"
}
eval_arm() {
    local arm="$1" ckpt
    if [[ "$arm" == source ]]; then
        ckpt="$checkpoint"
    else
        ckpt="$(<"$root/train/$arm/checkpoint.txt")"
    fi
    [[ -f "$ckpt" ]] || { echo "[ERROR] Missing $arm checkpoint: $ckpt" >&2; exit 2; }
    STUDENT_CHECKPOINT="$ckpt" STUDENT_EVAL_REPORT_DIR="$root/eval/$arm" \
    STUDENT_EVAL_EPISODES_PER_MOTION="${STUDENT_ROLLIN_EVAL_EPISODES:-3}" \
    STUDENT_EVAL_NUM_ENVS="${STUDENT_ROLLIN_EVAL_ENVS:-40}" \
    STUDENT_EVAL_MAX_STEPS="${STUDENT_ROLLIN_EVAL_MAX_STEPS:-6000}" \
    STUDENT_EVAL_SEED="${STUDENT_ROLLIN_EVAL_SEED:-43}" \
    STUDENT_EVAL_OBSTACLE_SEED="${STUDENT_ROLLIN_EVAL_SEED:-43}" \
    STUDENT_EVAL_TAKEOVER_MODE=none STUDENT_EVAL_DIAGNOSTICS=1 PHP_EVAL_PROFILE=clean \
    bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}
summary() {
    "$python" - "$root" "${STUDENT_ROLLIN_EVAL_SEED:-43}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); seed=sys.argv[2]
results={}
for arm in ('source','student','teacher'):
    path=root/'eval'/arm/f'full_clip_seed{seed}.json'
    if not path.is_file():
        print(f'[PENDING] {arm}: {path}'); continue
    data=json.loads(path.read_text()); rows=data['motions']
    n=sum(x['episodes'] for x in rows); s=sum(x['successes'] for x in rows)
    print(f'[{arm}] complete={data["complete"]}, {s}/{n} ({s/n:.1%}), clips={len(rows)}, takeover={data["intervention"]["mode"]}')
    if not data['complete'] or data['intervention']['mode']!='none':
        raise SystemExit('[ERROR] Incomplete or intervention-contaminated evaluation')
    results[arm] = {r['motion']:r['successes'] for r in rows}
    if data.get('diagnostics_enabled'):
        failures=[(r['motion'],r['diagnostic']['failure_frame_mean'],r['diagnostic']['failure_progress_mean']) for r in rows if r.get('diagnostic',{}).get('failure_frame_mean') is not None]
        for name,frame,progress in sorted(failures,key=lambda item:item[2])[:5]:
            print(f'  earliest failure: {name} frame={frame:.1f}, progress={progress:.1%}')
if 'student' in results and 'teacher' in results:
    a,b=results['student'],results['teacher']
    if set(a)!=set(b): raise SystemExit('[ERROR] Evaluated motion sets differ')
    wins=sum(b[m]>a[m] for m in a); losses=sum(b[m]<a[m] for m in a)
    print(f'[A/B] teacher-roll-in wins={wins}, losses={losses}, ties={len(a)-wins-losses} clips')
PY
}
case "$mode" in
    smoke) check; activate; run_arm probe student 2 smoke; run_arm probe teacher 2 smoke ;;
    preflight) check; activate; run_arm probe student "$rollouts"; run_arm probe teacher "$rollouts"; gate ;;
    diagnose) check; eval_arm source; summary ;;
    train) check; train ;;
    eval) check; eval_arm student; eval_arm teacher; summary ;;
    all) check; activate; run_arm probe student "$rollouts"; run_arm probe teacher "$rollouts"; gate; train; eval_arm student; eval_arm teacher; summary ;;
    summary) summary ;;
    *) usage; [[ "$mode" == help ]] || exit 2 ;;
esac
