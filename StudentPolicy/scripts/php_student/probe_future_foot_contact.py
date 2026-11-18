"""Grouped-motion offline probe for near-future contact or pelvis ascent.

This is an observability diagnostic on Teacher-controlled traces, not a
Student rollout or a deployable contact predictor. The existing trace schema
has foot forces but no hand contact sensor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def load_data(trace_dir: Path, bundle: Path, start: float, end: float, horizon: int,
              force: float, force_key: str, target_mode: str, rise_delta: float,
              elevation_delta: float):
    actors, depths, targets, clips = [], [], [], []
    per_clip = []
    names_path = trace_dir.parent / "motion_names.txt"
    if not names_path.is_file() or not bundle.is_dir():
        raise FileNotFoundError(f"Need {names_path} and motion bundle {bundle}")
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    for path in sorted(trace_dir.glob("motion_*.npz")):
        clip = int(path.stem.rsplit("_", 1)[1])
        if clip >= len(names):
            raise ValueError(f"No source motion name for clip {clip}")
        with np.load(bundle / f"{names[clip]}_motion.npz") as motion:
            source_duration = max(len(motion["joint_pos"]) - 1, 1)
        with np.load(path) as data:
            frame = np.asarray(data["frame"], dtype=np.int64)
            if len(frame) < horizon + 10 or not np.all(np.diff(frame) == 1):
                raise ValueError(f"Noncontiguous or short trace: {path}")
            if target_mode == "contact_transition":
                if force_key not in data:
                    raise KeyError(f"{path} lacks {force_key}; collect a new trace before this probe")
                force_norm = np.linalg.norm(data[force_key], axis=-1)
                if force_norm.shape != (len(frame), 2) or not np.isfinite(force_norm).all():
                    raise ValueError(f"Invalid {force_key} in {path}: {force_norm.shape}")
                contact = force_norm >= force
                # Five-frame majority vote reduces single-step force chatter.
                smoothed = np.stack(
                    [np.convolve(contact[:, foot].astype(float), np.ones(5) / 5, mode="same") >= 0.6
                     for foot in range(2)], axis=1
                )
            elif target_mode == "pelvis_rise":
                root_z = np.asarray(data["root_state"][:, 2], dtype=np.float32)
                if not np.isfinite(root_z).all():
                    raise ValueError(f"Nonfinite root z in {path}")
            else:
                ankle_z = np.asarray(data["tracked_ankle_wrist_pos_w"][:, :2, 2], dtype=np.float32)
                foot_force = np.linalg.norm(data["foot_contact_force"], axis=-1)
                if ankle_z.shape != foot_force.shape or not np.isfinite(ankle_z).all():
                    raise ValueError(f"Invalid ankle/contact arrays in {path}")
                ground_level = np.median(ankle_z[:min(40, len(ankle_z))], axis=0)
                elevated_support = np.any((ankle_z >= ground_level + elevation_delta)
                                          & (foot_force >= force), axis=1)
                first_support_row = (int(np.flatnonzero(elevated_support)[0]) if elevated_support.any()
                                     else len(frame) + horizon)
            row_of_frame = {int(value): idx for idx, value in enumerate(frame)}
            count = positive = 0
            for image_idx, image_frame in enumerate(data["depth_frame"]):
                row = row_of_frame.get(int(image_frame))
                if row is None or row < 5 or row + horizon >= len(frame):
                    continue
                if not start <= int(image_frame) / source_duration < end or not bool(data["teacher_valid"][row]):
                    continue
                if target_mode == "first_elevated_foot_contact" and row >= first_support_row:
                    continue
                if target_mode == "contact_transition":
                    future_change = bool(np.any(smoothed[row + 1:row + horizon + 1] != smoothed[row]))
                elif target_mode == "pelvis_rise":
                    future_change = bool(root_z[row + horizon] - root_z[row] >= rise_delta)
                else:
                    future_change = bool(row < first_support_row <= row + horizon)
                actors.append(np.asarray(data["actor_obs"][row], dtype=np.float32))
                depths.append(np.asarray(data["depth"][image_idx], dtype=np.float32))
                targets.append(float(future_change))
                clips.append(clip)
                count += 1
                positive += int(future_change)
            per_clip.append({"clip": clip, "samples": count, "positives": positive})
    if not actors or len(set(clips)) < 10:
        raise RuntimeError("Need valid aligned samples from at least 10 clips")
    return (np.stack(actors), np.stack(depths), np.asarray(targets, dtype=np.float32),
            np.asarray(clips, dtype=np.int32), per_clip)


def auc(scores: np.ndarray, target: np.ndarray) -> float:
    positive = target > 0.5
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    # Exact tie-aware Mann–Whitney AUC; no sklearn dependency.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        ranks[lo:hi] = (lo + hi + 1) / 2
    rank_sum = ranks[np.argsort(order)][positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class Predictor(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 128), nn.ELU(),
                                 nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit(features: torch.Tensor, target: torch.Tensor, train: torch.Tensor,
        test: torch.Tensor, steps: int, seed: int):
    mean = features[train].mean(0)
    std = features[train].std(0).clamp_min(1e-3)
    x = ((features - mean) / std).clamp(-10, 10)
    torch.manual_seed(seed)
    model = Predictor(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ids = torch.where(train)[0]
    positive_rate = target[train].mean().clamp(0.05, 0.95)
    positive_weight = ((1 - positive_rate) / positive_rate).detach()
    for _ in range(steps):
        batch = ids[torch.randint(len(ids), (min(128, len(ids)),))]
        loss = F.binary_cross_entropy_with_logits(model(x[batch]), target[batch],
                                                   pos_weight=positive_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = torch.sigmoid(model(x[test])).numpy()
        labels = target[test].numpy()
    pred = scores >= 0.5
    pos = labels > 0.5
    return {"auc": auc(scores, labels),
            "balanced_accuracy": float(0.5 * (pred[pos].mean() + (~pred[~pos]).mean()))
            if pos.any() and (~pos).any() else float("nan"),
            "test_samples": int(test.sum()), "test_positives": int(pos.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path,
                        default=Path(os.environ.get("PHP_BUNDLE", "/home/sjtu/.cache/php-parkour/motions-low-climb-76")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-start", type=float, default=0.2)
    parser.add_argument("--phase-end", type=float, default=0.6)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--force-threshold", type=float, default=20.0)
    parser.add_argument("--force-key", choices=("foot_contact_force", "hand_contact_force_proxy"),
                        default="foot_contact_force")
    parser.add_argument("--target-mode", choices=("contact_transition", "pelvis_rise",
                                                  "first_elevated_foot_contact"),
                        default="contact_transition")
    parser.add_argument("--rise-delta", type=float, default=0.08,
                        help="Minimum root/pelvis height increase in metres for pelvis_rise")
    parser.add_argument("--elevation-delta", type=float, default=0.15,
                        help="Foot/ankle height above initial median for elevated support")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.output.exists() or not args.trace_dir.is_dir() or args.folds < 2
            or args.steps < 1 or args.horizon < 1 or args.force_threshold <= 0
            or args.rise_delta <= 0 or args.elevation_delta <= 0
            or not 0 <= args.phase_start < args.phase_end <= 1):
        parser.error("Invalid input, or output already exists")
    torch.set_num_threads(2)
    actor_np, depth_np, target_np, clip_np, per_clip = load_data(
        args.trace_dir, args.bundle, args.phase_start, args.phase_end, args.horizon,
        args.force_threshold, args.force_key, args.target_mode, args.rise_delta,
        args.elevation_delta)
    unique_clips = np.unique(clip_np)
    if len(unique_clips) < args.folds * 2:
        parser.error("Need at least two clips per held-out fold")
    actor = torch.from_numpy(actor_np)
    depth = torch.from_numpy(depth_np)[:, None]
    # Fixed spatial averaging preserves raw image geometry and trains no encoder.
    pooled = F.adaptive_avg_pool2d(depth, (12, 20)).flatten(1)
    target = torch.from_numpy(target_np)
    rng = np.random.default_rng(args.seed)
    shuffled_clips = unique_clips.copy()
    rng.shuffle(shuffled_clips)
    fold_of_clip = {int(c): i % args.folds for i, c in enumerate(shuffled_clips)}
    fold_ids = torch.tensor([fold_of_clip[int(c)] for c in clip_np])
    # The shuffled-image control retains image and clip distributions but breaks
    # within-clip alignment. Shuffling is fixed before any model fitting.
    shuffled_pooled = pooled.clone()
    for clip in unique_clips:
        indices = np.flatnonzero(clip_np == clip)
        shuffled_pooled[indices] = pooled[rng.permutation(indices)]
    features = {
        "actor_only": torch.cat((actor, torch.zeros_like(pooled)), dim=1),
        "actor_aligned_depth": torch.cat((actor, pooled), dim=1),
        "actor_shuffled_depth": torch.cat((actor, shuffled_pooled), dim=1),
    }
    fold_rows = []
    for fold in range(args.folds):
        test = fold_ids == fold
        train = ~test
        row = {"fold": fold, "heldout_clips": sorted(int(c) for c in shuffled_clips
                                                   if fold_of_clip[int(c)] == fold)}
        for name, value in features.items():
            row[name] = fit(value, target, train, test, args.steps, args.seed + fold)
        fold_rows.append(row)
        print(f"[FOLD {fold}] " + " ".join(f"{name}={row[name]['auc']:.3f}" for name in features),
              flush=True)
    means = {name: float(np.nanmean([row[name]["auc"] for row in fold_rows])) for name in features}
    result = {"contract": "teacher_driven_contact_transition_grouped_motion_probe",
              "warning": "Wrist net force is only a hand-contact proxy; foot transitions include ordinary walking. No online Student success is measured.",
              "trace_dir": str(args.trace_dir.resolve()), "seed": args.seed,
              "bundle": str(args.bundle.resolve()),
              "phase_window": [args.phase_start, args.phase_end], "horizon_steps": args.horizon,
              "target_mode": args.target_mode, "rise_delta_m": args.rise_delta,
              "elevation_delta_m": args.elevation_delta,
              "force_key": args.force_key, "force_threshold_N": args.force_threshold,
              "samples": len(target),
              "positive_rate": float(target.mean()), "per_clip": per_clip,
              "fold_rows": fold_rows, "mean_auc": means,
              "aligned_vs_actor_auc": means["actor_aligned_depth"] - means["actor_only"],
              "aligned_vs_shuffled_auc": means["actor_aligned_depth"] - means["actor_shuffled_depth"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("[RESULT] " + json.dumps(means), flush=True)
    print(f"[PASS] {args.output}", flush=True)


if __name__ == "__main__":
    main()
