#!/usr/bin/env python3
"""Build cached valid-window indexes for Alpha26 Arrow files."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pyarrow as pa


def default_data_root() -> Path:
    env_root = os.environ.get("CARLA_ARROW_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path("/home/user/chaeyeon-kim/processed")


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_or_build_manifest(root: Path, manifest_path: Path) -> dict[str, object]:
    if manifest_path.exists():
        with manifest_path.open("rb") as f:
            return pickle.load(f)

    files = sorted(str(path) for path in root.glob("*.arrow"))
    return {
        "root": str(root),
        "train": [path for path in files if Path(path).name.startswith("train_")],
        "validation": [path for path in files if Path(path).name.startswith("validation_")],
        "test": [path for path in files if Path(path).name.startswith("test_")],
    }


def build_file_index(
    arrow_file: Path,
    seq_len: int,
    stride: int,
    sample_every_n: int,
    frame_step: int,
    require_contiguous_frames: bool,
) -> dict[str, object]:
    mmap = pa.memory_map(str(arrow_file), "r")
    reader = pa.ipc.open_stream(mmap)
    table = reader.read_all()
    n = table.num_rows

    run_ids = table.column("run_id").to_pylist()
    frames = table.column("frame").to_pylist() if "frame" in table.column_names else None

    offsets = [j * sample_every_n for j in range(seq_len)]
    max_offset = (seq_len - 1) * sample_every_n
    expected_frame_delta = frame_step * sample_every_n

    index = []
    for i in range(0, n - max_offset, stride):
        if not all(run_ids[i + off] == run_ids[i] for off in offsets):
            continue
        if require_contiguous_frames and frames is not None:
            if not all(
                frames[i + offsets[j]] - frames[i + offsets[j - 1]] == expected_frame_delta
                for j in range(1, seq_len)
            ):
                continue
        index.append(i)

    return {
        **file_signature(arrow_file),
        "num_rows": n,
        "index": index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build arrow_window_index.pkl for trainer11 server runs.",
    )
    parser.add_argument("--root", type=Path, default=default_data_root())
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--sample-every-n", type=int, default=2)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--no-contiguous-check", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else root / "arrow_manifest.pkl"
    )
    out_path = (
        args.out.expanduser().resolve()
        if args.out
        else root / "arrow_window_index.pkl"
    )

    if not root.exists():
        raise FileNotFoundError(f"Arrow root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Arrow root is not a directory: {root}")

    manifest = load_or_build_manifest(root, manifest_path)
    params = {
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "sample_every_n": int(args.sample_every_n),
        "frame_step": int(args.frame_step),
        "require_contiguous_frames": not args.no_contiguous_check,
    }

    selected_files = []
    splits = {}
    for split in args.splits:
        split_files = [str(Path(path).expanduser().resolve()) for path in manifest.get(split, [])]
        splits[split] = split_files
        selected_files.extend(split_files)

    cache = {
        "version": 1,
        "root": str(root),
        "params": params,
        "splits": splits,
        "files": {},
    }

    for idx, file_name in enumerate(dict.fromkeys(selected_files), start=1):
        arrow_file = Path(file_name)
        print(f"[{idx}/{len(set(selected_files))}] indexing {arrow_file.name}", flush=True)
        cache["files"][str(arrow_file)] = build_file_index(arrow_file, **params)
        print(
            f"  rows={cache['files'][str(arrow_file)]['num_rows']} "
            f"windows={len(cache['files'][str(arrow_file)]['index'])}",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    total = sum(len(entry["index"]) for entry in cache["files"].values())
    print(f"Wrote: {out_path}")
    print(f"files: {len(cache['files'])}")
    print(f"windows: {total}")
    print(f"params: {params}")


if __name__ == "__main__":
    main()
