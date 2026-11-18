"""Check paired trace command labels against their exact source motion frames."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--names", type=Path, default=None)
    args = parser.parse_args()
    names_path = args.names or args.trace_dir.parent / "motion_names.txt"
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    files = sorted(args.trace_dir.glob("motion_*.npz"))
    if not files:
        raise SystemExit(f"No traces in {args.trace_dir}")

    print("motion frames direct_mae best_offset best_mae max_abs cmd_classes fps depth_meta")
    global_direct: list[float] = []
    global_best: list[float] = []
    for trace_path in files:
        motion_id = int(trace_path.stem.split("_")[-1])
        motion_name = names[motion_id]
        with np.load(trace_path) as trace, np.load(args.bundle / f"{motion_name}_motion.npz") as motion:
            frame = trace["frame"].astype(np.int64)
            command = trace["velocity_command"].astype(np.float64)
            source = motion["vel_cmd"].astype(np.float64)
            direct = float(np.abs(command - source[frame]).mean())
            offset_errors = {
                offset: float(np.abs(command - source[frame + offset]).mean())
                for offset in (-2, -1, 0, 1, 2)
                if frame.min() + offset >= 0 and frame.max() + offset < len(source)
            }
            best_offset = min(offset_errors, key=offset_errors.get)
            best_mae = offset_errors[best_offset]
            max_abs = float(np.abs(command - source[frame]).max())
            # Command rows are 15-way one-hot or zero; this catches accidental
            # frame drift while avoiding a fragile assumption about class IDs.
            classes = Counter(np.argmax(command, axis=1).tolist())
            depth_meta = "not_recorded"
            if "depth_latency_frames" in trace:
                latency = trace["depth_latency_frames"].astype(int)
                match = trace["depth_buffer_match_error"]
                depth_meta = (
                    f"latency={dict(sorted(Counter(latency.tolist()).items()))},"
                    f"match_max={float(match.max()):.2g}"
                )
                if "depth_camera_pos_w" in trace:
                    camera_pos = trace["depth_camera_pos_w"]
                    depth_meta += f",camera_z={camera_pos[:, 2].min():.3f}..{camera_pos[:, 2].max():.3f}m"
            global_direct.append(direct)
            global_best.append(best_mae)
            print(
                f"{motion_name} {len(frame)} {direct:.8f} {best_offset:+d} {best_mae:.8f} "
                f"{max_abs:.8f} {dict(sorted(classes.items()))} {int(motion['fps'][0])} {depth_meta}"
            )
    print(
        f"OVERALL traces={len(files)} direct_command_mae={np.mean(global_direct):.9f} "
        f"best_offset_mae={np.mean(global_best):.9f}"
    )
    if any("depth_latency_frames" not in np.load(path) for path in files):
        print("DEPTH_ALIGNMENT=INCOMPLETE: existing traces lack timestamped camera pose/latency metadata")


if __name__ == "__main__":
    main()
