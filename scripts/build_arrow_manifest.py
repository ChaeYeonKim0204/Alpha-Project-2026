#!/usr/bin/env python3
"""Build arrow_manifest.pkl for the Alpha26 CARLA Arrow dataset."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path


def default_data_root() -> Path:
    env_root = os.environ.get("CARLA_ARROW_ROOT")
    if env_root:
        return Path(env_root).expanduser()

    return Path("/home/user/chaeyeon-kim/processed")


def build_manifest(root: Path) -> dict[str, object]:
    files = sorted(str(path) for path in root.glob("*.arrow"))
    manifest = {
        "root": str(root),
        "train": [path for path in files if Path(path).name.startswith("train_")],
        "validation": [path for path in files if Path(path).name.startswith("validation_")],
        "test": [path for path in files if Path(path).name.startswith("test_")],
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan processed/*.arrow and write arrow_manifest.pkl.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_data_root(),
        help="Directory containing *.arrow files. Defaults to CARLA_ARROW_ROOT or /home/user/chaeyeon-kim/processed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output pickle path. Defaults to <root>/arrow_manifest.pkl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_path = (args.out.expanduser().resolve() if args.out else root / "arrow_manifest.pkl")

    if not root.exists():
        raise FileNotFoundError(f"Arrow root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Arrow root is not a directory: {root}")

    manifest = build_manifest(root)
    total = sum(len(manifest[key]) for key in ("train", "validation", "test"))
    if total == 0:
        raise FileNotFoundError(f"No *.arrow files found in: {root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(manifest, f)

    print(f"Wrote: {out_path}")
    print(f"root: {manifest['root']}")
    print(f"train: {len(manifest['train'])}")
    print(f"validation: {len(manifest['validation'])}")
    print(f"test: {len(manifest['test'])}")


if __name__ == "__main__":
    main()
