"""Paired Student continuation: the only A/B difference is who drives a short phase window."""

from __future__ import annotations

import argparse
import dataclasses
import json
import traceback
from pathlib import Path

import torch
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationPPO
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from wbt_training.utils.checkpoint import apply_preprocess_hook, load_saved_experiment_config
from wbt_training.utils.isaacsim_clock import use_php_isaacsim
from wbt_training.utils.student_teacher import PhpDistillationPPO


class RollinPPO(PhpDistillationPPO):
    def configure_rollin(self, arm: str, start: float, end: float,
                         probability: float = 1.0) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Teacher roll-in probability must be in [0, 1]")
        self.rollin_arm = arm
        self.rollin_start = start
        self.rollin_end = end
        self.rollin_probability = probability
        self._pending = None
        self._window = None
        self._gate = None
        self._counts = torch.zeros(5, 3, device=self.device, dtype=torch.long)
        self._early_counts = torch.zeros(3, device=self.device, dtype=torch.long)
        self._action_gap_sum = torch.zeros(5, device=self.device)
        self._action_gap_count = torch.zeros(5, device=self.device, dtype=torch.long)
        cmd = self.env.command_manager.get_state("motion_command")
        self._clip_ends = torch.nonzero(cmd.motion.motion_ends, as_tuple=False).flatten()
        self._clip_starts = torch.cat((torch.zeros_like(self._clip_ends[:1]), self._clip_ends[:-1] + 1))
        if len(self._clip_ends) == 0:
            raise RuntimeError("Motion bundle contains no clips")
        self._clip_phase_counts = torch.zeros(len(self._clip_ends), 5, 3, device=self.device, dtype=torch.long)

    def _phase(self) -> torch.Tensor:
        cmd = self.env.command_manager.get_state("motion_command")
        ids = torch.searchsorted(self._clip_ends, cmd.time_steps, right=False).clamp_max(len(self._clip_ends) - 1)
        self._clip_ids = ids
        return (cmd.time_steps - self._clip_starts[ids]).float() / (
            self._clip_ends[ids] - self._clip_starts[ids]
        ).clamp_min(1).float()

    def _get_ppo_gate(self) -> torch.Tensor:
        gate = super()._get_ppo_gate()
        phase = self._phase()
        self._window = (phase >= self.rollin_start) & (phase < self.rollin_end)
        self._phase_bins = torch.bucketize(phase, torch.tensor([0.2, 0.4, 0.6, 0.8], device=phase.device))
        self._gate = gate.clone()
        self._gate[self._window] = 0.0  # Both arms use the same DAgger-only window.
        return self._gate

    def _update_ppo_gate_stats(self, student_mean: torch.Tensor, teacher_actions: torch.Tensor) -> None:
        super()._update_ppo_gate_stats(student_mean, teacher_actions)
        if self._window is None or self._gate is None:
            raise RuntimeError("Roll-in window must be sampled before the PPO gate statistics")
        valid = torch.any(teacher_actions != 0, dim=1)
        if self.rollin_arm == "teacher":
            sampled = (torch.ones_like(valid) if self.rollin_probability == 1.0
                       else torch.rand(valid.shape, device=valid.device) < self.rollin_probability)
            drive = self._window & valid & sampled
        else:
            drive = torch.zeros_like(valid)
        if torch.any(drive & (self._gate.squeeze(-1) != 0)):
            raise RuntimeError("Teacher-driven transitions would enter the on-policy PPO loss")
        self._pending = (drive, teacher_actions)
        early = (self._phase() >= 0.15) & (self._phase() < 0.25)
        self._early_counts += torch.stack((early.sum(), (early & valid).sum(), (early & drive).sum()))
        self._counts[:, 0].scatter_add_(0, self._phase_bins, torch.ones_like(self._phase_bins, dtype=torch.long))
        self._counts[:, 1].scatter_add_(0, self._phase_bins, valid.long())
        self._counts[:, 2].scatter_add_(0, self._phase_bins, drive.long())
        cell = self._clip_ids * 5 + self._phase_bins
        flat = self._clip_phase_counts.view(-1, 3)
        flat[:, 0].scatter_add_(0, cell, torch.ones_like(cell, dtype=torch.long))
        flat[:, 1].scatter_add_(0, cell, valid.long())
        flat[:, 2].scatter_add_(0, cell, drive.long())
        gap = torch.linalg.vector_norm(student_mean - teacher_actions, dim=1)
        self._action_gap_sum.scatter_add_(0, self._phase_bins, torch.where(valid, gap, 0.0))
        self._action_gap_count.scatter_add_(0, self._phase_bins, valid.long())

    def install_action_swap(self) -> None:
        original_step = self.env.step

        def step_with_rollin(actor_state):
            pending, self._pending = self._pending, None
            if pending is None:
                return original_step(actor_state)
            drive, teacher_actions = pending
            if torch.any(drive):
                if self._warmup_active:
                    raise RuntimeError("This experiment requires resumed Student training, not warm-up")
                actor_state = dict(actor_state)
                actor_state["actions"] = torch.where(drive[:, None], teacher_actions, actor_state["actions"])
            return original_step(actor_state)

        self.env.step = step_with_rollin

    def summary(self) -> dict:
        counts = self._counts.cpu().tolist()
        gaps = self._action_gap_sum.cpu().tolist()
        gap_counts = self._action_gap_count.cpu().tolist()
        per_clip = self._clip_phase_counts.cpu().tolist()
        def row_stats(row):
            return {"samples": row[0], "valid_labels": row[1], "teacher_control": row[2],
                    "valid_fraction": row[1] / row[0] if row[0] else None}
        return {
            "arm": self.rollin_arm,
            "window": [self.rollin_start, self.rollin_end],
            "teacher_control_probability": self.rollin_probability,
            "early_15_25": {"samples": int(self._early_counts[0].item()),
                            "valid_labels": int(self._early_counts[1].item()),
                            "teacher_control": int(self._early_counts[2].item())},
            "phase_bins": ["0-20", "20-40", "40-60", "60-80", "80-100"],
            "counts": [
                {**row_stats(row), "valid_action_l2": gaps[i] / gap_counts[i] if gap_counts[i] else None}
                for i, row in enumerate(counts)
            ],
            "clip_phase_counts": [
                {"clip_id": clip_id, "phase_bins": [row_stats(row) for row in clip_rows]}
                for clip_id, clip_rows in enumerate(per_clip)
            ],
        }

    def save(self, path: str | None = None, name: str = "last.ckpt") -> None:
        DistillationPPO.save(self, path=path, name=name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("probe", "train"), required=True)
    p.add_argument("--arm", choices=("student", "teacher"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--rollouts", type=int, default=21)
    p.add_argument("--window-start", type=float, default=0.25)
    p.add_argument("--window-end", type=float, default=0.55)
    args = p.parse_args()
    if not (args.checkpoint.is_file() and args.num_envs > 0 and args.iterations > 0 and args.rollouts > 0):
        p.error("Checkpoint must exist and counts must be positive")
    if not 0 <= args.window_start < args.window_end <= 1:
        p.error("Window must lie in [0,1] with start < end")
    if args.output_dir.exists():
        p.error(f"Refusing to overwrite {args.output_dir}")
    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint)))
    config = dataclasses.replace(
        saved,
        training=dataclasses.replace(saved.training, num_envs=args.num_envs, headless=True, export_onnx=False),
        algo=dataclasses.replace(saved.algo, config=dataclasses.replace(
            saved.algo.config, save_interval=args.iterations + 1,
        )),
    )
    config = apply_preprocess_hook(use_php_isaacsim(config))
    args.output_dir.mkdir(parents=True)
    config.save_config(str(args.output_dir / "holosoma_config.yaml"))
    (args.output_dir / "contract.json").write_text(json.dumps({
        "source_checkpoint": str(args.checkpoint.resolve()), "arm": args.arm,
        "mode": args.mode, "num_envs": args.num_envs,
        "iterations": args.iterations if args.mode == "train" else None,
        "rollouts": args.rollouts if args.mode == "probe" else None,
        "window": [args.window_start, args.window_end],
        "ppo_window_gate": 0, "teacher_control_requires_valid_label": True,
        "evaluation_teacher_takeover": False,
    }, indent=2) + "\n")
    env, device, app = setup_simulation_environment(config)
    try:
        (args.output_dir / "stage.txt").write_text("environment_ready\n")
        algo = RollinPPO(device=device, env=env, config=config.algo.config,
                         log_dir=str(args.output_dir), multi_gpu_cfg=None)
        algo.num_teachers = 1
        algo.setup()
        (args.output_dir / "stage.txt").write_text("algorithm_ready\n")
        algo.attach_checkpoint_metadata(config, None)
        algo.load(str(args.checkpoint))
        (args.output_dir / "stage.txt").write_text("checkpoint_loaded\n")
        algo.configure_rollin(args.arm, args.window_start, args.window_end)
        algo.install_action_swap()
        if args.mode == "train":
            algo.learn(num_learning_iterations=args.iterations)
            (args.output_dir / "checkpoint.txt").write_text(
                str(args.output_dir / f"model_{algo.current_learning_iteration:05d}.pt") + "\n")
        else:
            algo._train_mode()
            obs = env.reset_all()
            for _ in range(args.rollouts):
                obs = algo._rollout_step(obs)
                algo.storage.clear()
        (args.output_dir / "coverage.json").write_text(json.dumps(algo.summary(), indent=2) + "\n")
        print(f"[PASS] {args.mode} {args.arm}: {args.output_dir}", flush=True)
    except BaseException:
        (args.output_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if app:
            close_simulation_app(app)


if __name__ == "__main__":
    main()
