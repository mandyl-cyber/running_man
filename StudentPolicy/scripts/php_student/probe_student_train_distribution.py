"""Measure phase-wise teacher-label coverage under the saved training recipe.

This is an inference-only rollout, not a resumed optimization run. It preserves
the saved Student environment and preprocessing recipe, changing only env count.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

from wbt_training.utils.checkpoint import apply_preprocess_hook, load_saved_experiment_config
from wbt_training.utils.isaacsim_clock import use_php_isaacsim


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or not args.checkpoint.is_file():
        parser.error("checkpoint must exist; num-envs and steps must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint)))
    config = dataclasses.replace(
        saved,
        training=dataclasses.replace(saved.training, num_envs=args.num_envs, headless=True, export_onnx=False),
    )
    config = apply_preprocess_hook(use_php_isaacsim(config))
    env, device, app = setup_simulation_environment(config)
    try:
        algo = get_class(config.algo._target_)(
            device=device, env=env, config=config.algo.config, log_dir=str(args.output.parent), multi_gpu_cfg=None
        )
        algo.setup()
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        algo.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        algo.policy.set_loaded_teachers([True] * int(checkpoint["num_teachers"]))
        algo.policy.eval()
        obs = env.reset_all()
        cmd = env.command_manager.get_state("motion_command")
        ends = torch.nonzero(cmd.motion.motion_ends, as_tuple=False).squeeze(-1)
        starts = torch.cat((torch.zeros_like(ends[:1]), ends[:-1] + 1))
        bins = [dict(samples=0, valid=0, action_l2_sum=0.0, ref_pos_error_sum=0.0) for _ in range(5)]
        reset_count = 0
        for step in range(args.steps):
            actor, teacher, _, depth = algo._extract_groups(obs)
            ids = torch.searchsorted(ends, cmd.time_steps, right=False)
            phase = (cmd.time_steps - starts[ids]).float() / (ends[ids] - starts[ids]).clamp_min(1).float()
            labels = algo.policy.teacher_act(teacher)
            valid = torch.any(labels != 0, dim=1)
            actions = algo.policy.act({"actor_obs": actor, "depth_obs": depth})
            imitation_error = torch.linalg.vector_norm(algo.policy.action_mean - labels, dim=1)
            ref_error = torch.linalg.vector_norm(cmd.ref_pos_w - cmd.robot_ref_pos_w, dim=1)
            bucket = torch.clamp((phase * 5).long(), 0, 4)
            for idx, row in enumerate(bins):
                mask = bucket == idx
                keep = mask & valid
                row["samples"] += int(mask.sum().item())
                row["valid"] += int(keep.sum().item())
                row["action_l2_sum"] += float(imitation_error[keep].sum().item())
                row["ref_pos_error_sum"] += float(ref_error[mask].sum().item())
            obs, _, dones, _ = env.step({"actions": actions})
            reset_count += int(dones.sum().item())
            if step and step % 100 == 0:
                print(f"[PHASE PROBE] step={step}/{args.steps} resets={reset_count}", flush=True)
        rows = []
        for idx, row in enumerate(bins):
            rows.append({
                "phase": f"{idx * 20}-{(idx + 1) * 20}%",
                "samples": row["samples"],
                "teacher_valid_samples": row["valid"],
                "teacher_valid_rate": row["valid"] / row["samples"] if row["samples"] else None,
                "valid_action_l2_mean": row["action_l2_sum"] / row["valid"] if row["valid"] else None,
                "ref_pos_error_mean_m": row["ref_pos_error_sum"] / row["samples"] if row["samples"] else None,
            })
        result = {
            "contract": "saved_student_training_config_inference_rollout",
            "checkpoint": str(args.checkpoint),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "reset_count": reset_count,
            "preprocess_hook_kwargs": saved.training.preprocess_hook_kwargs,
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[PASS] {args.output}", flush=True)
    finally:
        if app:
            close_simulation_app(app)


if __name__ == "__main__":
    main()
