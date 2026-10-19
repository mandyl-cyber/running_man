from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetLayout:
    dataset_dir: str
    tracking_dir: str
    manifest_file: str
    terrain_file: str
    motion_files: list[str]


def add_dataset_args(parser: argparse.ArgumentParser):
    """Add arguments for the structured teacher dataset layout."""
    group = parser.add_argument_group("dataset", description="Arguments for structured local datasets.")
    group.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Dataset directory. It may point to one dataset or to a root used with --dataset_name.",
    )
    group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset folder name under --dataset_dir, for example climb_15_z_scale_1.0.",
    )
    group.add_argument(
        "--dataset_motion_glob",
        type=str,
        default="*.npz",
        help="Glob used under the dataset tracking directory.",
    )
    group.add_argument(
        "--dataset_motion_index",
        type=int,
        default=None,
        help="Optional sorted motion index from the dataset tracking directory.",
    )


def _find_dataset_dir(dataset_dir: str, dataset_name: str | None) -> str:
    root = os.path.abspath(dataset_dir)
    candidate = os.path.join(root, dataset_name) if dataset_name else root
    if os.path.isdir(os.path.join(candidate, "tracking")) and os.path.isdir(os.path.join(candidate, "terrain")):
        return candidate

    if dataset_name is None:
        children = [
            name
            for name in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, name, "tracking"))
            and os.path.isdir(os.path.join(root, name, "terrain"))
        ]
        if children:
            preview = ", ".join(children[:10])
            if len(children) > 10:
                preview += ", ..."
            raise ValueError(
                "--dataset_dir points to a dataset root with multiple datasets. "
                f"Pass --dataset_name. Available examples: {preview}"
            )

    raise FileNotFoundError(f"Could not find a dataset containing tracking/ and terrain/: {candidate}")


def resolve_dataset_layout(args_cli: argparse.Namespace) -> DatasetLayout | None:
    """Resolve a structured dataset into explicit motion, manifest, and terrain files."""
    if args_cli.dataset_dir is None:
        return None

    dataset_dir = _find_dataset_dir(args_cli.dataset_dir, args_cli.dataset_name)
    tracking_dir = os.path.join(dataset_dir, "tracking")
    terrain_dir = os.path.join(dataset_dir, "terrain")
    manifest_file = os.path.join(tracking_dir, "batch_manifest.json")
    if not os.path.isfile(manifest_file):
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_file}")

    terrain_files = sorted(glob.glob(os.path.join(terrain_dir, "*.usd")))
    if not terrain_files:
        raise FileNotFoundError(f"Dataset terrain USD not found under: {terrain_dir}")
    terrain_file = terrain_files[0]

    motion_files = sorted(glob.glob(os.path.join(tracking_dir, args_cli.dataset_motion_glob)))
    if not motion_files:
        raise FileNotFoundError(
            f"No motion files matched {args_cli.dataset_motion_glob!r} under: {tracking_dir}"
        )

    if args_cli.dataset_motion_index is not None:
        index = args_cli.dataset_motion_index
        if index < 0 or index >= len(motion_files):
            raise IndexError(
                f"--dataset_motion_index={index} is outside available motion range [0, {len(motion_files) - 1}]"
            )
        motion_files = [motion_files[index]]

    return DatasetLayout(
        dataset_dir=dataset_dir,
        tracking_dir=tracking_dir,
        manifest_file=manifest_file,
        terrain_file=terrain_file,
        motion_files=motion_files,
    )


def apply_dataset_defaults(args_cli: argparse.Namespace, *, mode: str) -> DatasetLayout | None:
    """Fill explicit CLI paths from --dataset_dir without overriding user-provided paths."""
    layout = resolve_dataset_layout(args_cli)
    if layout is None:
        return None

    if mode != "train":
        raise ValueError(f"Unknown dataset mode: {mode}")

    if args_cli.motion_file is None and args_cli.motion_files is None and args_cli.motion_dir is None:
        args_cli.motion_files = layout.motion_files
    if args_cli.manifest_file is None:
        args_cli.manifest_file = layout.manifest_file
    if args_cli.terrain_file is None:
        args_cli.terrain_file = layout.terrain_file

    args_cli.terrain_use_manifest_pose = True

    print(f"[INFO]: Using dataset: {layout.dataset_dir}")
    print(f"[INFO]: Dataset manifest: {args_cli.manifest_file}")
    print(f"[INFO]: Dataset terrain: {args_cli.terrain_file}")
    print(f"[INFO]: Dataset motions: {len(layout.motion_files)} file(s)")
    return layout
