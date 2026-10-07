#!/usr/bin/env python3
"""Export trainer11 TensorBoard scalar logs to CSV and summary PNG figures."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None


DEFAULT_TAG_GROUPS = [
    (
        "Total loss",
        "loss",
        ["train_loss_step", "train_loss_epoch", "val/RL_loss", "val/DS_loss"],
    ),
    (
        "Camera PSNR",
        "PSNR (dB)",
        ["val/RL_rgb_psnr", "val/DS_rgb_psnr", "val/RL_future_rgb_psnr", "val/DS_future_rgb_psnr"],
    ),
    (
        "LiDAR Chamfer distance",
        "Chamfer distance",
        [
            "val/RL_lidar_chamfer_xyz",
            "val/DS_lidar_chamfer_xyz",
            "val/RL_future_lidar_chamfer_xyz",
            "val/DS_future_lidar_chamfer_xyz",
        ],
    ),
    (
        "LiDAR XYZ Euclidean error",
        "meters",
        [
            "val/RL_lidar_xyz_euclidean",
            "val/DS_lidar_xyz_euclidean",
            "val/RL_future_lidar_xyz_euclidean",
            "val/DS_future_lidar_xyz_euclidean",
        ],
    ),
    (
        "LiDAR range MAE",
        "meters",
        [
            "val/RL_lidar_range_mae",
            "val/DS_lidar_range_mae",
            "val/RL_future_lidar_range_mae",
            "val/DS_future_lidar_range_mae",
        ],
    ),
    (
        "KL loss",
        "weighted loss",
        ["train_probabilistic_step", "train_probabilistic_epoch", "val/RL_probabilistic", "val/DS_probabilistic"],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar logs from a trainer11 run.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path(os.environ.get("ALPHA26_LOG_ROOT", "logs")),
        help="Lightning log root. The run is expected at <log-root>/<run-name>/version_*/.",
    )
    parser.add_argument(
        "--run-name",
        default="trainer11_server_full_h256_z128_lidar_re10_empty005",
        help="Run name under log-root.",
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        default=None,
        help="Optional explicit events.out.tfevents.* file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "figures",
        help="Directory for exported PNG/CSV files.",
    )
    parser.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing weight in [0, 1).")
    parser.add_argument("--no-csv", action="store_true", help="Only write the PNG figure.")
    return parser.parse_args()


def latest_event_file(log_root: Path, run_name: str) -> Path:
    search_root = log_root.expanduser().resolve() / run_name
    event_files = glob.glob(str(search_root / "**" / "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found under: {search_root}")
    return Path(max(event_files, key=os.path.getmtime))


def load_scalars(event_file: Path) -> dict[str, dict[str, list[float]]]:
    if EventAccumulator is None:
        raise ImportError(
            "TensorBoard is required to read Lightning event files. "
            "Install it in the training environment, for example: pip install tensorboard"
        )
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        scalars[tag] = {
            "step": [event.step for event in events],
            "wall_time": [event.wall_time for event in events],
            "value": [event.value for event in events],
        }
    return scalars


def write_csv(scalars: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "step", "wall_time", "value"])
        writer.writeheader()
        for tag in sorted(scalars):
            values = scalars[tag]
            for step, wall_time, value in zip(values["step"], values["wall_time"], values["value"]):
                writer.writerow(
                    {
                        "tag": tag,
                        "step": step,
                        "wall_time": f"{wall_time:.6f}",
                        "value": f"{value:.10g}",
                    }
                )


def smooth_values(values: list[float], weight: float) -> list[float]:
    if weight <= 0.0 or not values:
        return values
    if weight >= 1.0:
        raise ValueError("--smooth must be less than 1.0")

    smoothed = []
    last = values[0]
    for value in values:
        last = last * weight + (1.0 - weight) * value
        smoothed.append(last)
    return smoothed


def plot_tags(ax, scalars, tags, title, ylabel, smooth):
    found = False
    for tag in tags:
        if tag in scalars:
            ax.plot(
                scalars[tag]["step"],
                smooth_values(scalars[tag]["value"], smooth),
                label=tag,
                linewidth=1.5,
            )
            found = True

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if found:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No matching scalar", ha="center", va="center", transform=ax.transAxes)


def write_summary_figure(
    scalars: dict[str, dict[str, list[float]]],
    run_name: str,
    out_path: Path,
    smooth: float = 0.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (title, ylabel, tags) in zip(axes.ravel(), DEFAULT_TAG_GROUPS):
        plot_tags(ax, scalars, tags, title, ylabel, smooth)
    fig.suptitle(run_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_metrics(
    log_root: Path,
    run_name: str,
    out_dir: Path,
    event_file: Path | None = None,
    smooth: float = 0.0,
    write_scalar_csv: bool = True,
) -> tuple[Path, Path | None, Path]:
    event_file = event_file.expanduser().resolve() if event_file else latest_event_file(log_root, run_name)
    scalars = load_scalars(event_file)
    if not scalars:
        raise RuntimeError(f"No scalar tags found in TensorBoard event file: {event_file}")

    out_dir = out_dir.expanduser().resolve()
    png_path = out_dir / f"{run_name}_loss_metrics.png"
    csv_path = out_dir / f"{run_name}_tensorboard_scalars.csv" if write_scalar_csv else None

    write_summary_figure(scalars, run_name, png_path, smooth=smooth)
    if csv_path is not None:
        write_csv(scalars, csv_path)
    return event_file, csv_path, png_path


def main() -> None:
    args = parse_args()
    event_file, csv_path, png_path = export_metrics(
        log_root=args.log_root,
        run_name=args.run_name,
        out_dir=args.out_dir,
        event_file=args.event_file,
        smooth=args.smooth,
        write_scalar_csv=not args.no_csv,
    )
    print(f"Loaded: {event_file}")
    if csv_path is not None:
        print(f"Wrote CSV: {csv_path}")
    print(f"Wrote figure: {png_path}")


if __name__ == "__main__":
    main()
