"""Render contact-approach depth samples from paired Student traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = sorted(args.trace_dir.glob("motion_*.npz"))
    if not files:
        raise SystemExit("No trace files")
    targets = (100, 150, 200, 250)
    fig, axes = plt.subplots(len(files), len(targets), figsize=(12, 2.5 * len(files)))
    for row, file in enumerate(files):
        with np.load(file) as data:
            frames = data["depth_frame"]
            for col, target in enumerate(targets):
                idx = int(np.argmin(np.abs(frames - target)))
                axes[row, col].imshow(data["depth"][idx], cmap="viridis", vmin=-0.6, vmax=0.6)
                axes[row, col].set_title(f"{file.stem}, frame {frames[idx]}")
                axes[row, col].axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(args.output)


if __name__ == "__main__":
    main()
