"""Short Student continuation probes using the PHP/Holosoma trainer."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationPPO
from holosoma.agents.distillation_ppo.rollout_storage import RolloutStorageDistillation
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

from wbt_training.utils.checkpoint import apply_preprocess_hook, load_saved_experiment_config
from wbt_training.utils.isaacsim_clock import use_php_isaacsim
from wbt_training.utils.student_teacher import PhpDistillationPPO


class PhaseWeightStorage(RolloutStorageDistillation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register("phase_weight", shape=(1,), dtype=torch.float)
        self.register("anchor_prefix_mask", shape=(1,), dtype=torch.float)
        self.pending_weight: torch.Tensor | None = None
        self.pending_prefix_mask: torch.Tensor | None = None

    def add(self, **data: torch.Tensor) -> None:
        if self.pending_weight is None or self.pending_prefix_mask is None:
            raise RuntimeError("Phase weight or prefix mask was not captured before the environment step")
        super().add(**data, phase_weight=self.pending_weight, anchor_prefix_mask=self.pending_prefix_mask)
        self.pending_weight = None
        self.pending_prefix_mask = None


class PhaseWeightedPhpDistillationPPO(PhpDistillationPPO):
    focus_multiplier = 1.0
    source_count = 0
    support_onsets: torch.Tensor | None = None
    sweep_start_iteration: int | None = None
    sweep_milestones: set[int] = set()
    cold_gate_fallback: float | None = None
    anchor_prefix_frames = 0
    anchor_coef = 0.0
    source_policy = None
    early_ppo_frames = 0
    early_ppo_scale = 1.0
    replay_actor_obs: torch.Tensor | None = None
    replay_depth_obs: torch.Tensor | None = None
    replay_targets: torch.Tensor | None = None
    replay_weights: torch.Tensor | None = None
    replay_batch_size = 0
    replay_coef = 0.0

    def _build_storage(self) -> None:
        self.storage = PhaseWeightStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.config.num_steps_per_env,
            actor_obs_dim=self.actor_obs_dim,
            teacher_obs_dim=self.teacher_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            depth_shape=self.depth_shape,
            num_actions=self.num_act,
            device=self.device,
        )

    def _motion_frame_and_id(self) -> tuple[torch.Tensor, torch.Tensor]:
        cmd = self.env.command_manager.get_state("motion_command")
        if not hasattr(self, "_clip_ends"):
            self._clip_ends = torch.nonzero(cmd.motion.motion_ends, as_tuple=False).squeeze(-1)
            self._clip_starts = torch.cat((torch.zeros_like(self._clip_ends[:1]), self._clip_ends[:-1] + 1))
        ids = torch.searchsorted(self._clip_ends, cmd.time_steps, right=False)
        frame = cmd.time_steps - self._clip_starts[ids]
        return frame, ids

    def _update_ppo_gate_stats(self, student_mean: torch.Tensor, teacher_actions: torch.Tensor) -> None:
        super()._update_ppo_gate_stats(student_mean, teacher_actions)
        frame, ids = self._motion_frame_and_id()
        source_ids = ids % self.source_count
        onset = self.support_onsets[source_ids]
        valid = torch.any(teacher_actions != 0, dim=1)
        focus = valid & (onset >= 0) & (frame >= onset - 110) & (frame < onset - 40)
        weight = torch.where(focus, self.focus_multiplier, 1.0).view(-1, 1)
        self.storage.pending_weight = weight.detach()
        self.storage.pending_prefix_mask = (frame < self.anchor_prefix_frames).float().view(-1, 1).detach()

    def _update_step(self, mini_batch):
        self._current_phase_weight = mini_batch["phase_weight"].squeeze(-1)
        self._anchor_actor_obs = mini_batch["actor_obs"]
        self._anchor_depth_obs = mini_batch["depth_obs"]
        self._anchor_prefix_mask = mini_batch["anchor_prefix_mask"].squeeze(-1)
        result = super()._update_step(mini_batch)
        if self.anchor_coef > 0:
            result["anchor_prefix_loss"] = self._last_anchor_loss.detach()
            result["anchor_prefix_fraction"] = self._anchor_prefix_mask.mean().detach()
        if self.replay_coef > 0:
            result["fixed_replay_loss"] = self._last_replay_loss.detach()
            result["fixed_replay_weighted_loss"] = (self.replay_coef * self._last_replay_loss).detach()
        return result

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        if self.anchor_coef > 0:
            if self.source_policy is None:
                raise RuntimeError("Anchor enabled but frozen source policy is missing")
            with torch.no_grad():
                target = self.source_policy.act_inference({
                    "actor_obs": self._anchor_actor_obs,
                    "depth_obs": self._anchor_depth_obs,
                })
            current = self.policy.student(self.policy._student_input(
                self._anchor_actor_obs, self._anchor_depth_obs))
            per_sample = (current - target).square().mean(dim=-1)
            mask = self._anchor_prefix_mask
            anchor_loss = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
            self._last_anchor_loss = anchor_loss
            loss = loss + self.anchor_coef * anchor_loss
        if self.replay_coef > 0:
            if any(value is None for value in (self.replay_actor_obs, self.replay_depth_obs,
                                               self.replay_targets, self.replay_weights)):
                raise RuntimeError("Fixed replay enabled without its dataset")
            indices = torch.multinomial(self.replay_weights, self.replay_batch_size, replacement=True)
            current = self.policy.student(self.policy._student_input(
                self.replay_actor_obs[indices], self.replay_depth_obs[indices]))
            replay_loss = (current - self.replay_targets[indices]).square().mean()
            self._last_replay_loss = replay_loss
            loss = loss + self.replay_coef * replay_loss
        super()._optimizer_step(loss)

    def _dagger_loss_per_sample(self, student_mean: torch.Tensor, teacher_actions: torch.Tensor) -> torch.Tensor:
        return super()._dagger_loss_per_sample(student_mean, teacher_actions) * self._current_phase_weight

    def _get_ppo_gate(self) -> torch.Tensor:
        gate = super()._get_ppo_gate()
        cmd = self.env.command_manager.get_state("motion_command")
        if (self.cold_gate_fallback is not None
                and getattr(cmd, "_gate_seen", float("inf")) < getattr(cmd, "_gate_warmup", 0.0)):
            gate = torch.full_like(gate, self.cold_gate_fallback)
        if self.early_ppo_frames > 0:
            frame, _ids = self._motion_frame_and_id()
            gate = torch.where((frame < self.early_ppo_frames).view(-1, 1),
                               gate * self.early_ppo_scale, gate)
        return gate

    def save(self, path: str | None = None, name: str = "last.ckpt") -> None:
        # The A/B checkpoint is for evaluation, not deployment export.
        DistillationPPO.save(self, path=path, name=name)

    def _post_epoch_logging(self, it, loss_dict):
        super()._post_epoch_logging(it, loss_dict)
        if self.sweep_start_iteration is not None:
            step = it - self.sweep_start_iteration + 1
            if step in self.sweep_milestones:
                self.save(str(Path(self.log_dir) / f"model_step_{step:03d}.pt"))
                print(f"[SWEEP CHECKPOINT] step={step}", flush=True)


def rebuild_gate_without_updates(algo: PhaseWeightedPhpDistillationPPO, target_valid_samples: int) -> dict:
    """Collect source-policy rollouts to reconstruct the unsaved phase gate EMA.

    The resulting state can be saved once and loaded by both A/B arms. No
    rollout is stored for optimisation and no optimizer step is taken.
    """
    cmd = algo.env.command_manager.get_state("motion_command")
    if not getattr(cmd.motion, "_ppo_gate_on", False):
        raise RuntimeError("The source configuration has no residual-driven PPO gate")
    if target_valid_samples < int(cmd._gate_warmup):
        raise RuntimeError(f"Need at least gate warmup={cmd._gate_warmup:g} valid samples")
    algo.policy.eval()
    obs_dict = algo.env.reset_all()
    steps = 0
    max_steps = max(10000, 20 * target_valid_samples // algo.env.num_envs)
    # The simulator mutates state tensors across resets. inference_mode would
    # turn newly created tensors into inference tensors and break the next reset.
    with torch.no_grad():
        while cmd._gate_seen < target_valid_samples and steps < max_steps:
            actor_obs, teacher_obs, _critic_obs, depth_obs = algo._extract_groups(obs_dict)
            actions = algo.policy.act({"actor_obs": actor_obs, "depth_obs": depth_obs})
            teacher_actions = algo.policy.teacher_act(teacher_obs).detach()
            algo._update_ppo_gate_stats(algo.policy.action_mean.detach(), teacher_actions)
            obs_dict, _rewards, dones, _infos = algo.env.step({"actions": actions})
            algo.policy.reset(dones)
            steps += 1
            if steps % 256 == 0:
                print(f"[GATE REBUILD] steps={steps} valid={cmd._gate_seen:.0f}/{target_valid_samples}", flush=True)
    if cmd._gate_seen < target_valid_samples:
        raise RuntimeError(f"Gate rebuild exhausted {steps} steps with only {cmd._gate_seen:.0f} valid samples")
    table = cmd._adaptive_gate_table()
    if table is None:
        raise RuntimeError("Gate rebuild did not engage the per-frame table")
    result = {
        "steps": steps,
        "valid_samples": cmd._gate_seen,
        "bins_seen_fraction": float((cmd._gate_bin_cnt > 1e-8).float().mean()),
        "table_mean": float(table.mean()),
        "table_std": float(table.std()),
        "table_min": float(table.min()),
        "table_max": float(table.max()),
        "optimizer_steps": 0,
    }
    print(f"[GATE REBUILD COMPLETE] {result}", flush=True)
    return result


def save_gate_state(cmd, path: Path) -> None:
    torch.save({
        "bin_sum": cmd._gate_bin_sum.detach().cpu(),
        "bin_cnt": cmd._gate_bin_cnt.detach().cpu(),
        "seen": float(cmd._gate_seen),
        "nbins": int(cmd._gate_nbins),
    }, path)


def load_gate_state(cmd, path: Path) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if int(saved["nbins"]) != int(cmd._gate_nbins):
        raise RuntimeError("Gate state bin count does not match the current motion bundle")
    for name, attr in (("bin_sum", "_gate_bin_sum"), ("bin_cnt", "_gate_bin_cnt")):
        tensor = saved[name]
        current = getattr(cmd, attr)
        if tensor.shape != current.shape or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Invalid gate state tensor: {name}")
        current.copy_(tensor.to(current.device))
    cmd._gate_seen = float(saved["seen"])
    table = cmd._adaptive_gate_table()
    if table is None:
        raise RuntimeError("Loaded gate state did not engage")
    return {
        "valid_samples": cmd._gate_seen,
        "bins_seen_fraction": float((cmd._gate_bin_cnt > 1e-8).float().mean()),
        "table_mean": float(table.mean()),
        "table_std": float(table.std()),
        "table_min": float(table.min()),
        "table_max": float(table.max()),
        "source_state": str(path.resolve()),
    }


def attach_fixed_replay(algo: PhaseWeightedPhpDistillationPPO, traces_dir: Path,
                        mode: str, batch_size: int, coefficient: float) -> dict:
    """Attach exact-depth source-Student states, balanced uniformly over clips."""
    paths = sorted(traces_dir.glob("motion_??.npz"))
    if len(paths) != 40:
        raise RuntimeError(f"Expected 40 source traces, got {len(paths)} in {traces_dir}")
    actor_rows, depth_rows, target_rows, weight_rows = [], [], [], []
    per_clip, replay_max_abs = {}, 0.0
    teacher_disagreement = []
    for path in paths:
        with np.load(path) as data:
            frame = np.asarray(data["frame"], dtype=np.int64)
            depth_frame = np.asarray(data["depth_frame"], dtype=np.int64)
            if np.any(np.diff(frame) <= 0) or np.any(np.diff(depth_frame) <= 0):
                raise RuntimeError(f"Non-increasing trace frames: {path}")
            row = np.searchsorted(frame, depth_frame)
            if np.any(row >= len(frame)) or not np.array_equal(frame[row], depth_frame):
                raise RuntimeError(f"Depth frames do not exactly align: {path}")
            valid = np.asarray(data["teacher_valid"][row], dtype=bool)
            mask = depth_frame < 100 if mode == "prefix_source" else (
                (depth_frame >= 100) & (depth_frame < 250) & valid)
            count = int(mask.sum())
            if count < 10:
                raise RuntimeError(f"Insufficient {mode} rows in {path}: {count}")
            actor = torch.as_tensor(np.asarray(data["actor_obs"][row][mask], dtype=np.float32), device=algo.device)
            depth = torch.as_tensor(np.asarray(data["depth"][mask], dtype=np.float32), device=algo.device)
            recorded = torch.as_tensor(np.asarray(data["student_action"][row][mask], dtype=np.float32), device=algo.device)
            teacher = torch.as_tensor(np.asarray(data["teacher_action"][row][mask], dtype=np.float32), device=algo.device)
            with torch.no_grad():
                replay = algo.policy.act_inference({"actor_obs": actor, "depth_obs": depth})
            error = float((replay - recorded).abs().max())
            if error > 1e-3:
                raise RuntimeError(f"Source checkpoint does not reproduce {path}: {error}")
            replay_max_abs = max(replay_max_abs, error)
            actor_rows.append(actor)
            depth_rows.append(depth)
            target_rows.append(recorded if mode == "prefix_source" else teacher)
            weight_rows.append(torch.full((count,), 1.0 / count, device=algo.device))
            per_clip[path.stem] = count
            if mode == "transition_teacher":
                teacher_disagreement.extend(torch.linalg.vector_norm(recorded - teacher, dim=1).cpu().tolist())
    algo.replay_actor_obs = torch.cat(actor_rows)
    algo.replay_depth_obs = torch.cat(depth_rows)
    algo.replay_targets = torch.cat(target_rows)
    algo.replay_weights = torch.cat(weight_rows)
    algo.replay_batch_size = batch_size
    algo.replay_coef = coefficient
    result = {
        "mode": mode, "total_rows": int(algo.replay_targets.shape[0]),
        "clips": len(paths), "rows_per_clip": per_clip,
        "source_replay_max_abs": replay_max_abs,
        "teacher_disagreement_l2_mean": float(np.mean(teacher_disagreement)) if teacher_disagreement else None,
        "batch_size_per_optimizer_step": batch_size, "coefficient": coefficient,
        "sampling": "uniform_over_clips_then_uniform_over_valid_aligned_rows",
    }
    print(f"[FIXED REPLAY READY] mode={mode} rows={result['total_rows']} "
          f"max_source_error={replay_max_abs:.2e} coef={coefficient}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "focus"), required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--focus-multiplier", type=float, default=2.0)
    parser.add_argument("--event-traces", type=Path, required=True)
    parser.add_argument("--save-loaded-only", action="store_true")
    parser.add_argument("--save-milestones", default="")
    parser.add_argument("--cold-gate-fallback", type=float)
    parser.add_argument("--anchor-prefix-frames", type=int, default=0)
    parser.add_argument("--anchor-coef", type=float, default=0.0)
    parser.add_argument("--gate-rebuild-valid-samples", type=int, default=0)
    parser.add_argument("--gate-rebuild-mode", choices=("keep", "discard"))
    parser.add_argument("--gate-rebuild-state", type=Path)
    parser.add_argument("--early-ppo-frames", type=int, default=0)
    parser.add_argument("--early-ppo-scale", type=float, default=1.0)
    parser.add_argument("--replay-traces", type=Path)
    parser.add_argument("--replay-mode", choices=("prefix_source", "transition_teacher"))
    parser.add_argument("--replay-batch-size", type=int, default=256)
    parser.add_argument("--replay-coef", type=float, default=0.0)
    args = parser.parse_args()
    if args.num_envs < 96 or args.iterations < 0 or args.focus_multiplier < 1:
        parser.error("num-envs must be >=96; iterations nonnegative; multiplier >=1")
    if args.iterations == 0 and not args.save_loaded_only:
        parser.error("Zero iterations requires --save-loaded-only")
    milestones = [int(item) for item in args.save_milestones.split(",") if item]
    if milestones and (args.save_loaded_only or any(step < 1 or step > args.iterations for step in milestones)):
        parser.error("Milestones must be within the training run")
    if args.cold_gate_fallback is not None and not 0.0 < args.cold_gate_fallback <= 1.0:
        parser.error("Cold gate fallback must be in (0, 1]")
    if args.anchor_prefix_frames < 0 or args.anchor_coef < 0 or (args.anchor_coef > 0 and args.anchor_prefix_frames == 0):
        parser.error("Anchor requires positive prefix frames and nonnegative coefficient")
    if args.early_ppo_frames < 0 or not 0.0 <= args.early_ppo_scale <= 1.0:
        parser.error("Early PPO requires nonnegative frames and scale in [0, 1]")
    if (args.early_ppo_frames > 0) != (args.early_ppo_scale < 1.0):
        parser.error("Early PPO frames and a reduced scale must be set together")
    if (args.replay_mode is not None) != (args.replay_traces is not None and args.replay_coef > 0):
        parser.error("Fixed replay needs traces, mode, and a positive coefficient together")
    if args.replay_batch_size < 1 or args.replay_coef < 0:
        parser.error("Fixed replay batch size must be positive and coefficient nonnegative")
    if args.replay_traces is not None and not args.replay_traces.is_dir():
        parser.error(f"Missing replay traces: {args.replay_traces}")
    has_gate_samples = args.gate_rebuild_valid_samples > 0
    has_gate_state = args.gate_rebuild_state is not None
    if has_gate_samples and has_gate_state:
        parser.error("Choose gate rebuild samples or a saved gate state, not both")
    if (has_gate_samples or has_gate_state) != (args.gate_rebuild_mode is not None):
        parser.error("Gate rebuild requires keep/discard mode")
    if args.gate_rebuild_state is not None and not args.gate_rebuild_state.is_file():
        parser.error(f"Missing gate state: {args.gate_rebuild_state}")
    if args.output_dir.exists():
        parser.error(f"Refusing to reuse output directory: {args.output_dir}")
    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint)))
    config = dataclasses.replace(
        saved,
        training=dataclasses.replace(saved.training, num_envs=args.num_envs, headless=True, export_onnx=False),
        algo=dataclasses.replace(
            saved.algo,
            config=dataclasses.replace(saved.algo.config, save_interval=args.iterations + 1),
        ),
    )
    config = apply_preprocess_hook(use_php_isaacsim(config))
    args.output_dir.mkdir(parents=True)
    config.save_config(str(args.output_dir / "holosoma_config.yaml"))
    (args.output_dir / "ab_contract.json").write_text(json.dumps({
        "source_checkpoint": str(args.checkpoint),
        "arm": args.arm,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "focus_multiplier": args.focus_multiplier if args.arm == "focus" else 1.0,
        "focus_event_window_frames": [-110, -40],
        "event_traces": str(args.event_traces.resolve()),
        "teacher_valid_only": True,
        "save_loaded_only": args.save_loaded_only,
        "save_milestones": milestones,
        "cold_gate_fallback": args.cold_gate_fallback,
        "anchor_prefix_frames": args.anchor_prefix_frames,
        "anchor_coef": args.anchor_coef,
        "gate_rebuild_valid_samples": args.gate_rebuild_valid_samples,
        "gate_rebuild_mode": args.gate_rebuild_mode,
        "gate_rebuild_state": str(args.gate_rebuild_state.resolve()) if args.gate_rebuild_state else None,
        "early_ppo_frames": args.early_ppo_frames,
        "early_ppo_scale": args.early_ppo_scale,
        "replay_traces": str(args.replay_traces.resolve()) if args.replay_traces else None,
        "replay_mode": args.replay_mode,
        "replay_batch_size": args.replay_batch_size if args.replay_mode else None,
        "replay_coef": args.replay_coef,
    }, indent=2) + "\n")
    env, device, app = setup_simulation_environment(config)
    try:
        from wbt_training.utils.registry import bundle_pairs
        names_path = args.event_traces.parent / "motion_names.txt"
        names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
        source_paths = [p for p, _terrain, _obj in bundle_pairs(
            Path(saved.training.registry_name.removeprefix("file://")))]
        if len(source_paths) != len(names) or [p.name.removesuffix("_motion.npz") for p in source_paths] != names:
            raise RuntimeError("Training source bundle does not match event trace names/order")
        lengths, onsets = [], []
        for i, path in enumerate(source_paths):
            with np.load(path) as motion:
                lengths.append(len(motion["joint_pos"]))
            with np.load(args.event_traces / f"motion_{i:02d}.npz") as trace:
                frame = np.asarray(trace["frame"])
                ankle_z = np.asarray(trace["tracked_ankle_wrist_pos_w"][:, :2, 2])
                foot_force = np.linalg.norm(trace["foot_contact_force"], axis=-1)
                ground = np.median(ankle_z[:min(40, len(ankle_z))], axis=0)
                elevated = np.any((ankle_z >= ground + 0.15) & (foot_force >= 20.0), axis=1)
                onsets.append(int(frame[np.flatnonzero(elevated)[0]]) if elevated.any() else -1)
        cmd = env.command_manager.get_state("motion_command")
        ends = torch.nonzero(cmd.motion.motion_ends, as_tuple=False).squeeze(-1).cpu().numpy()
        observed_lengths = np.diff(np.r_[-1, ends])
        if len(observed_lengths) % len(lengths) or not np.array_equal(
            observed_lengths, np.tile(lengths, len(observed_lengths) // len(lengths))
        ):
            raise RuntimeError("Training segments do not match repeated source motion lengths/order")
        PhaseWeightedPhpDistillationPPO.source_count = len(lengths)
        PhaseWeightedPhpDistillationPPO.support_onsets = torch.as_tensor(onsets, device=device)
        print(f"[EVENT MAP PASS] {len(lengths)} source clips x {len(observed_lengths)//len(lengths)} variants; "
              "pre-contact window=[onset-110,onset-40)", flush=True)
        algo = PhaseWeightedPhpDistillationPPO(
            device=device, env=env, config=config.algo.config,
            log_dir=str(args.output_dir), multi_gpu_cfg=None,
        )
        algo.num_teachers = 1
        algo.focus_multiplier = args.focus_multiplier if args.arm == "focus" else 1.0
        algo.cold_gate_fallback = args.cold_gate_fallback
        algo.anchor_prefix_frames = args.anchor_prefix_frames
        algo.anchor_coef = args.anchor_coef
        algo.early_ppo_frames = args.early_ppo_frames
        algo.early_ppo_scale = args.early_ppo_scale
        algo.setup()
        algo.attach_checkpoint_metadata(config, None)
        algo.load(str(args.checkpoint))
        if args.replay_mode is not None:
            stats = attach_fixed_replay(algo, args.replay_traces, args.replay_mode,
                                        args.replay_batch_size, args.replay_coef)
            (args.output_dir / "fixed_replay.json").write_text(json.dumps(stats, indent=2) + "\n")
        if args.gate_rebuild_mode is not None:
            cmd = env.command_manager.get_state("motion_command")
            if args.gate_rebuild_state is None:
                stats = rebuild_gate_without_updates(algo, args.gate_rebuild_valid_samples)
                save_gate_state(cmd, args.output_dir / "gate_rebuild_state.pt")
            else:
                stats = load_gate_state(cmd, args.gate_rebuild_state)
            (args.output_dir / "gate_rebuild.json").write_text(json.dumps(stats, indent=2) + "\n")
            if args.gate_rebuild_mode == "discard":
                cmd._gate_bin_sum.zero_()
                cmd._gate_bin_cnt.zero_()
                cmd._gate_seen = 0.0
                print("[GATE CONTROL] Discarded the reconstructed gate before training", flush=True)
            else:
                print("[GATE TREATMENT] Kept the reconstructed gate for training", flush=True)
        if args.anchor_coef > 0:
            algo.source_policy = copy.deepcopy(algo.policy)
            algo.source_policy.eval()
            for parameter in algo.source_policy.parameters():
                parameter.requires_grad_(False)
            print(f"[ANCHOR READY] prefix_frames={args.anchor_prefix_frames} coef={args.anchor_coef}", flush=True)
        if args.save_loaded_only:
            algo.save(str(args.output_dir / "model_loaded.pt"))
            print(f"[PASS] loaded checkpoint saved without parameter updates: {args.output_dir}", flush=True)
            return
        algo.sweep_start_iteration = algo.current_learning_iteration
        algo.sweep_milestones = set(milestones)
        algo.learn(num_learning_iterations=args.iterations)
        if milestones and not all((args.output_dir / f"model_step_{step:03d}.pt").is_file() for step in milestones):
            raise RuntimeError("Training returned without writing all requested milestone checkpoints")
        print(f"[PASS] {args.arm} continuation finished in {args.output_dir}", flush=True)
    except BaseException:
        traceback.print_exc()
        (args.output_dir / "run_failed.marker").write_text("Training raised an exception; see terminal traceback.\n")
        raise
    finally:
        if app:
            close_simulation_app(app)


if __name__ == "__main__":
    main()
