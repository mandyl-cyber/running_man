#!/usr/bin/env python3
"""Validate the pinned official PHP/Holosoma student-distillation contract.

This is intentionally a source-level, fail-closed audit.  StudentPolicy owns
the experiment entry point, while the authoritative implementation remains in
the pinned official checkout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


EXPECTED_PHP_COMMIT = "c7c9ce3fc75a8754a7e403c58de3b6933254d01b"
EXPECTED_HOLOSOMA_COMMIT = "4a0bf04cde5930af5ed59da797568d32be972525"

REQUIRED_SOURCE_FRAGMENTS = {
    "wbt_training/config_values/experiment.py": [
        "student_hidden_dims=[2048, 1024, 512, 256, 128]",
        "teacher_hidden_dims=[",
        "critic_hidden_dims=[512, 256, 128]",
        "DepthOnlyFCBackbone58x87Small",
        "depth_output_dim=32",
        "num_learning_epochs=2",
        "num_mini_batches=96",
        "learning_rate=3e-4",
        "num_steps_per_env=24",
        "num_learning_iterations=30000",
        "empirical_normalization=False",
        "dagger_end_epoch=10000",
        "dagger_loss_coef=10.0",
        "num_envs=2048",
        '"num_variants": 3',
    ],
    "wbt_training/config_values/depth_distillation.py": [
        '"latency_frame": (3, 4)',
        '"resize": (58, 87)',
        '"buffer_len": 6',
        '"enable_holes": True',
        '"curriculum_start_step": 24_000',
        '"curriculum_end_step": 240_000',
        '"target_multiplier": 2.0',
        "use_adaptive_timesteps_sampler=False",
        '"velocity_command"',
        '"which_motion"',
    ],
    "wbt_training/config_values/motion_command.py": [
        "One-hot velocity command from the motion data",
        "return cmd.vel_cmd.view(env.num_envs, -1)",
        "idx = torch.where(expert_terminate, torch.full_like(idx, -1.0), idx)",
    ],
    "wbt_training/utils/student_teacher.py": [
        "class PhpDistillationPPO",
        "RoutingColumnTeacherMixin",
    ],
}


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args()

    root = Path(args.official_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    holosoma = root / "thirdparty/holosoma"
    if not root.is_dir() or not holosoma.is_dir():
        raise SystemExit(f"[ERROR] Invalid official PHP checkout: {root}")

    php_commit = git_head(root)
    holosoma_commit = git_head(holosoma)
    commit_ok = php_commit == EXPECTED_PHP_COMMIT and holosoma_commit == EXPECTED_HOLOSOMA_COMMIT
    if not commit_ok and not args.allow_unpinned:
        raise SystemExit(
            "[ERROR] Official backend revision drift detected. "
            f"PHP={php_commit} (expected {EXPECTED_PHP_COMMIT}), "
            f"Holosoma={holosoma_commit} (expected {EXPECTED_HOLOSOMA_COMMIT}). "
            "Set PHP_ALLOW_UNPINNED=1 only for an intentional re-audit."
        )

    checked: dict[str, list[str]] = {}
    for relative, fragments in REQUIRED_SOURCE_FRAGMENTS.items():
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"[ERROR] Missing official source file: {path}")
        source = path.read_text(encoding="utf-8")
        missing = [fragment for fragment in fragments if fragment not in source]
        if missing:
            raise SystemExit(f"[ERROR] Official contract drift in {relative}; missing: {missing}")
        checked[relative] = fragments

    report = {
        "compatible": True,
        "official_root": str(root),
        "php_commit": php_commit,
        "holosoma_commit": holosoma_commit,
        "pinned_commits_match": commit_ok,
        "allow_unpinned": bool(args.allow_unpinned),
        "checked_source_contract": checked,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] Official contract: PHP={php_commit} Holosoma={holosoma_commit}")
    print(f"[PASS] Contract report: {output}")


if __name__ == "__main__":
    main()
