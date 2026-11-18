"""Read-only probes for cross-clip action ambiguity and retained depth spatial cues.

All labels are computed from existing paired Teacher-controlled traces. The
near-region centroid is an image-derived proxy, not obstacle ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from probe_observation_vs_state import load_samples
from probe_trainable_depth import load_backbone


def standardized(x: torch.Tensor) -> torch.Tensor:
    return ((x - x.mean(0)) / x.std(0).clamp_min(0.05)).clamp(-10, 10)


def nearest_cross_clip(feature: torch.Tensor, action: torch.Tensor,
                       clip: torch.Tensor, command: torch.Tensor) -> dict:
    x = standardized(feature)
    distance = torch.cdist(x, x) / x.shape[1] ** 0.5
    same_command = command[:, None] == command[None, :]
    same_clip = clip[:, None] == clip[None, :]
    allowed = same_command & ~same_clip
    distance[~allowed] = float("inf")
    best_distance, nearest = distance.min(1)
    found = torch.isfinite(best_distance)
    gap = torch.linalg.vector_norm(action - action[nearest], dim=1)
    # Same-clip neighbours give a scale for how far states move in a few frames.
    within = torch.cdist(x, x) / x.shape[1] ** 0.5
    within[~(same_command & same_clip)] = float("inf")
    within.fill_diagonal_(float("inf"))
    within_distance = within.min(1).values
    radius = float(torch.quantile(within_distance[torch.isfinite(within_distance)], 0.75))
    close = found & (best_distance <= radius)
    return {"matched": int(found.sum()), "median_cross_distance": float(best_distance[found].median()),
            "within_clip_q75_radius": radius, "mean_action_gap": float(gap[found].mean()),
            "close_pairs": int(close.sum()),
            "close_pair_mean_action_gap": float(gap[close].mean()) if close.any() else None,
            "close_pair_conflict_gt_1_l2": float((gap[close] > 1.0).float().mean()) if close.any() else None}


def ridge_mae(x: torch.Tensor, y: torch.Tensor, clip: np.ndarray,
              valid: torch.Tensor, folds: int = 5) -> dict:
    # Standardize using training clips only; no adjacent-frame random split.
    values = []
    clip_ids = np.asarray(sorted(set(map(int, clip))))
    for fold in range(folds):
        held = set(map(int, clip_ids[fold::folds]))
        test = valid & torch.from_numpy(np.isin(clip, list(held)))
        train = valid & ~test
        if int(train.sum()) < 100 or int(test.sum()) < 20:
            raise RuntimeError("Insufficient per-fold spatial-proxy samples")
        mu, sd = x[train].mean(0), x[train].std(0).clamp_min(0.05)
        a = ((x[train] - mu) / sd).double()
        b = ((x[test] - mu) / sd).double()
        a = torch.cat((a, torch.ones(len(a), 1, dtype=torch.float64)), dim=1)
        b = torch.cat((b, torch.ones(len(b), 1, dtype=torch.float64)), dim=1)
        target = y[train].double()
        reg = torch.eye(a.shape[1], dtype=torch.float64) * 10.0
        reg[-1, -1] = 0.0
        coef = torch.linalg.solve(a.T @ a + reg, a.T @ target)
        pred = b @ coef
        baseline = torch.full_like(y[test].double(), float(target.mean()))
        values.append({"fold": fold, "test_n": int(test.sum()),
                       "mae": float((pred - y[test]).abs().mean()),
                       "mean_baseline_mae": float((baseline - y[test]).abs().mean())})
    return {"folds": values, "mae_mean": float(np.mean([v["mae"] for v in values])),
            "baseline_mae_mean": float(np.mean([v["mean_baseline_mae"] for v in values]))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--teacher-traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(2)
    samples = load_samples(args.teacher_traces, 0.25, 0.65)
    actor = torch.from_numpy(samples["actor"])
    depth = torch.from_numpy(samples["depth"])
    action = torch.from_numpy(samples["action"])
    clip = torch.from_numpy(samples["clip"]).long()
    command = actor[:, -15:].argmax(1)
    backbone = load_backbone(args.checkpoint, samples, torch.device("cpu")).eval()
    with torch.no_grad():
        latent = torch.cat([backbone(batch) for batch in depth.split(128)])
        grid = F.adaptive_avg_pool2d(depth.unsqueeze(1), (8, 12)).flatten(1)
    features = {"actor_only": actor, "actor_plus_32d_latent": torch.cat((actor, latent), dim=1),
                "actor_plus_spatial_grid": torch.cat((actor, grid), dim=1),
                "privileged_teacher_obs": torch.from_numpy(samples["privileged"])}
    neighbors = {name: nearest_cross_clip(x, action, clip, command) for name, x in features.items()}
    # Image-derived spatial proxy: normalized horizontal centroid of pixels
    # with depth < -0.4; omit frames with no nearby region or near-total fill.
    near = depth < -0.4
    fraction = near.float().mean((1, 2))
    valid = (fraction >= 0.02) & (fraction <= 0.8)
    columns = torch.linspace(-1.0, 1.0, depth.shape[-1])
    centroid = (near.float() * columns[None, None, :]).sum((1, 2)) / near.float().sum((1, 2)).clamp_min(1)
    spatial = {name: ridge_mae(x, centroid, samples["clip"], valid)
               for name, x in {"actor_only": actor, "32d_latent": latent,
                               "raw_8x12_grid": grid}.items()}
    result = {"contract": "teacher_driven_40clip_same_frame_structural_diagnostic",
              "warning": "Nearest-neighbour similarity does not prove physical state identity; near-region image centroid is not an obstacle-pose label.",
              "samples": len(actor), "clips": int(clip.unique().numel()),
              "phase_window": [0.25, 0.65], "near_region_proxy_valid_samples": int(valid.sum()),
              "cross_clip_same_command_nearest": neighbors,
              "near_region_horizontal_centroid": spatial}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, stats in neighbors.items():
        print(f"[NEIGHBOUR] {name}: action_gap={stats['mean_action_gap']:.3f}, "
              f"close_pairs={stats['close_pairs']}, close_conflict={stats['close_pair_conflict_gt_1_l2']}")
    for name, stats in spatial.items():
        print(f"[SPATIAL] {name}: MAE={stats['mae_mean']:.3f}, mean-baseline={stats['baseline_mae_mean']:.3f}")
    print(f"[PASS] {args.output}")


if __name__ == "__main__":
    main()
