"""Run the official PHP teacher full-clip evaluator with typed, fixed-seed overrides.

This bypasses Tyro's parser for heterogeneous saved randomization dictionaries;
the official policy loading and report logic remain unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

from holosoma.utils.eval_utils import CheckpointConfig
from wbt_training.training_runs.eval_teacher import _apply_eval_profile, run_eval
from wbt_training.utils.checkpoint import load_saved_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obstacle-seed", type=int, default=42)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.bundle.is_dir():
        parser.error("checkpoint and bundle must exist")
    if min(args.episodes, args.num_envs, args.max_steps) <= 0:
        parser.error("episodes, num-envs and max-steps must be positive")
    names = sorted(p.name.removesuffix("_motion.npz") for p in args.bundle.glob("*_motion.npz"))
    if not names:
        parser.error("bundle has no *_motion.npz files")
    if args.report.exists() and args.report.stat().st_size:
        parser.error(f"report already exists: {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    names_file = args.report.parent / "motion_names.txt"
    names_file.write_text("\n".join(names) + "\n")
    os.environ.update({
        "PHP_EVAL_PROFILE": "clean",
        "PHP_EVAL_REPORT_OUTPUT": str(args.report),
        "PHP_EVAL_EPISODES_PER_MOTION": str(args.episodes),
        "PHP_EVAL_MOTION_NAMES_FILE": str(names_file),
    })
    checkpoint = CheckpointConfig(checkpoint=str(args.checkpoint))
    saved, wandb_path = load_saved_experiment_config(checkpoint)
    config = saved.get_eval_config()
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            registry_name=args.bundle.as_uri(),
            preprocess_hook_kwargs=json.dumps({
                "add_onpath_obstacle": True,
                "num_variants": 1,
                "obstacle_seed": args.obstacle_seed,
            }),
            num_envs=args.num_envs,
            max_eval_steps=args.max_steps,
            seed=args.seed,
            headless=True,
            export_onnx=False,
        ),
        terrain=dataclasses.replace(
            config.terrain,
            terrain_term=dataclasses.replace(
                config.terrain.terrain_term,
                spawn=dataclasses.replace(config.terrain.terrain_term.spawn, randomize_tiles=True),
            ),
        ),
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                sim=dataclasses.replace(config.simulator.config.sim, max_episode_length_s=20.0),
            ),
        ),
        logger=dataclasses.replace(
            config.logger,
            video=dataclasses.replace(config.logger.video, enabled=False),
        ),
    )
    config = _apply_eval_profile(config)
    run_eval(config, checkpoint, saved, wandb_path)
    if not args.report.is_file() or not args.report.stat().st_size:
        raise RuntimeError(f"Teacher evaluator produced no report: {args.report}")


if __name__ == "__main__":
    main()
