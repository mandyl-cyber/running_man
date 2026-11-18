"""Controlled continuation: single-frame versus 3-frame proprioceptive history.

Both arms use the same source Student, fresh optimizer, learning rate, and
Teacher roll-in. Only actor_obs history length differs. Historical columns
are zero-initialized in the migrated first layer, preserving initial outputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import traceback
from pathlib import Path

import torch
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from wbt_training.utils.checkpoint import apply_preprocess_hook, load_saved_experiment_config
from wbt_training.utils.isaacsim_clock import use_php_isaacsim

from teacher_rollin_ab import RollinPPO


# Alphabetical observation-term order, verified against the saved 108-D actor.
ACTOR_TERM_DIMS = (
    ("actions", 29), ("base_ang_vel", 3), ("dof_pos", 29),
    ("dof_vel", 29), ("robot_anchor_projected_gravity", 3),
    ("velocity_command", 15),
)


def migrated_first_layer(old: torch.Tensor, history: int) -> torch.Tensor:
    actor_dim = sum(dim for _, dim in ACTOR_TERM_DIMS)
    if old.shape[1] != actor_dim + 32:
        raise ValueError(f"Expected source first layer to have {actor_dim + 32} columns, got {old.shape}")
    output = old.new_zeros(old.shape[0], actor_dim * history + 32)
    old_pos = new_pos = 0
    for _, dim in ACTOR_TERM_DIMS:
        output[:, new_pos + (history - 1) * dim:new_pos + history * dim] = old[:, old_pos:old_pos + dim]
        old_pos += dim
        new_pos += history * dim
    output[:, actor_dim * history:] = old[:, actor_dim:]
    return output


def verify_migration(old: torch.Tensor, expanded: torch.Tensor, history: int) -> float:
    torch.manual_seed(82)
    current = torch.randn(32, 108, device=old.device)
    latent = torch.randn(32, 32, device=old.device)
    history_actor = torch.randn(32, 108 * history, device=old.device)
    old_pos = new_pos = 0
    for _, dim in ACTOR_TERM_DIMS:
        history_actor[:, new_pos + (history - 1) * dim:new_pos + history * dim] = current[:, old_pos:old_pos + dim]
        old_pos += dim
        new_pos += history * dim
    before = torch.nn.functional.linear(torch.cat((current, latent), dim=1), old)
    after = torch.nn.functional.linear(torch.cat((history_actor, latent), dim=1), expanded)
    error = float((before - after).abs().max())
    if error > 1e-4:
        raise RuntimeError(f"First-layer migration changed initial output: max_abs={error}")
    return error


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--arm", choices=("single", "history3"), required=True)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--rollin-start", type=float, default=0.25)
    p.add_argument("--rollin-end", type=float, default=0.55)
    p.add_argument("--rollin-probability", type=float, default=1.0)
    args = p.parse_args()
    if not args.checkpoint.is_file() or args.num_envs < 1 or args.iterations < 1 or args.learning_rate <= 0:
        p.error("Missing checkpoint or invalid budget")
    if not 0 <= args.rollin_start < args.rollin_end <= 1:
        p.error("Roll-in window must satisfy 0 <= start < end <= 1")
    if not 0 <= args.rollin_probability <= 1:
        p.error("Roll-in probability must lie in [0,1]")
    history = 3 if args.arm == "history3" else 1
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old = source["model_state_dict"]["student.0.weight"]
    expanded = migrated_first_layer(old, history)
    error = verify_migration(old, expanded, history)
    print(f"[VERIFY] {args.arm} first-layer output max_abs={error:.2e}", flush=True)
    if args.verify_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the depth-camera preflight/training; no CUDA device is visible")
    if args.output_dir is None or args.output_dir.exists():
        p.error("Training requires a fresh --output-dir")
    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint)))
    expected_terms = tuple(sorted(saved.observation.groups["actor_obs"].terms))
    if expected_terms != tuple(name for name, _ in ACTOR_TERM_DIMS):
        raise RuntimeError(f"Actor observation order changed: {expected_terms}")
    groups = dict(saved.observation.groups)
    groups["actor_obs"] = dataclasses.replace(groups["actor_obs"], history_length=history)
    config = dataclasses.replace(
        saved,
        observation=dataclasses.replace(saved.observation, groups=groups),
        training=dataclasses.replace(saved.training, num_envs=args.num_envs,
                                     headless=True, export_onnx=False),
        algo=dataclasses.replace(saved.algo, config=dataclasses.replace(
            saved.algo.config, learning_rate=args.learning_rate,
            save_interval=args.iterations + 1,
        )),
    )
    config = apply_preprocess_hook(use_php_isaacsim(config))
    args.output_dir.mkdir(parents=True)
    config.save_config(str(args.output_dir / "holosoma_config.yaml"))
    (args.output_dir / "contract.json").write_text(json.dumps({
        "source_checkpoint": str(args.checkpoint.resolve()), "arm": args.arm,
        "actor_history_length": history, "depth_history_length": 1,
        "rollin_window": [args.rollin_start, args.rollin_end], "num_envs": args.num_envs,
        "rollin_probability": args.rollin_probability,
        "iterations": args.iterations, "fresh_optimizer": True,
        "learning_rate": args.learning_rate, "first_layer_max_abs_error": error,
    }, indent=2) + "\n")
    env, device, app = setup_simulation_environment(config)
    try:
        algo = RollinPPO(device=device, env=env, config=config.algo.config,
                         log_dir=str(args.output_dir), multi_gpu_cfg=None)
        algo.num_teachers = 1
        algo.setup()
        state = dict(source["model_state_dict"])
        state["student.0.weight"] = expanded
        algo.policy.load_training_state_dict(state, strict=True)
        algo.policy.set_loaded_teachers(source["teacher_loaded"])
        algo.policy.teachers.eval()
        for param in algo.policy.teachers.parameters():
            param.requires_grad_(False)
        algo.current_learning_iteration = int(source["iter"])
        algo.learning_rate = args.learning_rate
        algo.ppo_coef = float(source["ppo_coef"])
        for group in algo.optimizer.param_groups:
            group["lr"] = args.learning_rate
        algo.policy.project_action_std()
        algo.attach_checkpoint_metadata(config, None)
        algo.configure_rollin("teacher", args.rollin_start, args.rollin_end,
                              args.rollin_probability)
        algo.install_action_swap()
        algo.learn(num_learning_iterations=args.iterations)
        ckpt = args.output_dir / f"model_{algo.current_learning_iteration:05d}.pt"
        if not ckpt.is_file():
            raise RuntimeError(f"Missing final checkpoint: {ckpt}")
        (args.output_dir / "checkpoint.txt").write_text(str(ckpt.resolve()) + "\n")
        (args.output_dir / "coverage.json").write_text(json.dumps(algo.summary(), indent=2) + "\n")
        print(f"[PASS] {args.arm}: {ckpt}", flush=True)
    except BaseException:
        (args.output_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if app:
            close_simulation_app(app)


if __name__ == "__main__":
    main()
