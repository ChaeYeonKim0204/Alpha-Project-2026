#!/usr/bin/env python3
"""Server full-data trainer11 run with restored LiDAR reconstruction weights."""

from __future__ import annotations

import argparse
import bisect
import copy
import os
import pickle
import warnings
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer11_server_defs import (
    LOG_ROOT,
    PROJECT_ROOT,
    cfg,
    _as_arrow_path,
    _make_img_transform,
    _make_img_transform_raw,
    ExperimentCSVLogger,
    MUVODataset,
    MultiArrowStreamDataset,
    StreamWindowDataset,
    WorldModelTrainer,
    load_arrow_manifest,
    pa,
)

from export_tensorboard_metrics import export_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train trainer11 on the full server Arrow dataset.",
    )
    parser.add_argument("--run-name", default="trainer11_server_full_h256_z128_lidar_re10_empty005")
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ALPHA26_BATCH_SIZE", "2")))
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--devices", default=os.environ.get("ALPHA26_DEVICES", "4"))
    parser.add_argument("--strategy", default=os.environ.get("ALPHA26_STRATEGY", None))
    parser.add_argument("--precision", default=os.environ.get("ALPHA26_PRECISION", "16-mixed"))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("ALPHA26_NUM_WORKERS", "2")))
    parser.add_argument("--val-check-interval", type=int, default=600)
    parser.add_argument("--limit-val-batches", type=int, default=20)
    parser.add_argument("--save-top-k", type=int, default=5)
    parser.add_argument("--monitor", default="val/DS_loss")
    parser.add_argument("--monitor-mode", default="min", choices=["min", "max"])
    parser.add_argument("--train-run", default=None, help="Optional single train_run_XXX.arrow for debugging.")
    parser.add_argument("--val-rl-run", default=None)
    parser.add_argument("--val-ds-run", default=None)
    parser.add_argument(
        "--window-index",
        type=Path,
        default=None,
        help="Cached raw window index pkl. Defaults to <DATA_ROOT>/arrow_window_index.pkl.",
    )
    parser.add_argument("--no-window-index-cache", action="store_true")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--no-metric-export", action="store_true")
    parser.add_argument(
        "--metric-smooth",
        type=float,
        default=0.0,
        help="EMA smoothing weight for the exported loss/metric PNG.",
    )
    return parser.parse_args()


def resolve_runtime(requested_devices: str, requested_strategy: str | None, precision: str):
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested = str(requested_devices).strip().lower()

    if cuda_count == 0:
        return "auto", "auto", "auto", precision, 1

    if requested in {"auto", "all"}:
        world_size = cuda_count
    else:
        world_size = max(1, int(requested))
        if world_size > cuda_count:
            warnings.warn(
                f"Requested {world_size} GPUs but only {cuda_count} are visible; using {cuda_count}."
            )
            world_size = cuda_count

    strategy = requested_strategy or ("ddp" if world_size > 1 else "auto")
    return "gpu", world_size, strategy, precision, world_size


def select_train_files(manifest: dict[str, object], train_run: str | None) -> list[str]:
    if train_run:
        path = _as_arrow_path(train_run)
        if path not in manifest["train"] and not os.path.exists(path):
            raise FileNotFoundError(f"Could not find train run: {path}")
        return [path]
    return list(manifest["train"])


class RankZeroExperimentCSVLogger(ExperimentCSVLogger):
    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            super().on_fit_end(trainer, pl_module)


def file_signature(path: str) -> dict[str, int]:
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def expected_window_index_params(run_cfg) -> dict[str, object]:
    return {
        "seq_len": int(run_cfg.RECEPTIVE_FIELD + run_cfg.FUTURE_HORIZON),
        "stride": int(run_cfg.RECEPTIVE_FIELD * run_cfg.DATA.SAMPLE_EVERY_N),
        "sample_every_n": int(run_cfg.DATA.SAMPLE_EVERY_N),
        "frame_step": int(run_cfg.DATA.FRAME_STEP),
        "require_contiguous_frames": True,
    }


def load_window_index_cache(cache_path: Path, run_cfg):
    if not cache_path.exists():
        print(f"Window index cache not found: {cache_path}", flush=True)
        return None

    with cache_path.open("rb") as f:
        cache = pickle.load(f)

    expected = expected_window_index_params(run_cfg)
    if cache.get("version") != 1:
        warnings.warn(f"Ignoring unsupported window index cache version: {cache.get('version')}")
        return None
    if cache.get("params") != expected:
        warnings.warn(
            "Ignoring window index cache because params do not match. "
            f"cache={cache.get('params')} expected={expected}"
        )
        return None

    print(f"Loaded window index cache: {cache_path}", flush=True)
    return cache


def cached_raw_index(cache, arrow_file: str):
    if cache is None:
        return None
    path = str(Path(arrow_file).expanduser().resolve())
    entry = cache.get("files", {}).get(path)
    if entry is None:
        warnings.warn(f"Window index cache missing file; falling back to scan: {path}")
        return None

    current = file_signature(path)
    cached = {"size": entry.get("size"), "mtime_ns": entry.get("mtime_ns")}
    if cached != current:
        warnings.warn(
            "Window index cache file signature mismatch; falling back to scan: "
            f"{path} cache={cached} current={current}"
        )
        return None
    return entry["index"]


class CachedWindowDataset(MUVODataset):
    def __init__(
        self,
        arrow_file,
        raw_index,
        seq_len=4,
        stride=None,
        sample_every_n=None,
        image_input_size=None,
        rgb_recon_size=None,
        lidar_size=None,
        frame_step=None,
        require_contiguous_frames=True,
    ):
        self.seq_len = int(seq_len)
        self.sample_every_n = int(sample_every_n or cfg.DATA.SAMPLE_EVERY_N)
        self.stride = self.seq_len if stride is None else int(stride)
        self.arrow_file = _as_arrow_path(arrow_file)
        self.lidar_size = tuple(lidar_size or cfg.DATA.LIDAR_RANGE_VIEW_SIZE)
        self.img_transform = _make_img_transform(image_input_size or cfg.DATA.IMAGE_INPUT_SIZE)
        self.img_transform_raw = _make_img_transform_raw(rgb_recon_size or cfg.DATA.RGB_RECON_SIZE)
        self.frame_step = int(frame_step or cfg.DATA.FRAME_STEP)
        self.require_contiguous_frames = require_contiguous_frames

        self._mmap = pa.memory_map(self.arrow_file, "r")
        reader = pa.ipc.open_stream(self._mmap)
        self.table = reader.read_all()
        self.n = self.table.num_rows
        self.index = list(raw_index)


class CachedStreamWindowDataset(CachedWindowDataset):
    def __init__(self, arrow_file, raw_index, seq_len=4, num_streams=1, stride=None, **kwargs):
        super().__init__(arrow_file, raw_index, seq_len=seq_len, stride=stride, **kwargs)
        self.num_streams = max(1, int(num_streams))
        self.index = StreamWindowDataset._make_stream_order(self.index, self.num_streams)


class CachedMultiArrowStreamDataset(torch.utils.data.Dataset):
    def __init__(self, arrow_files, cache, seq_len=4, num_streams=1, stride=None, **kwargs):
        self.datasets = []
        cached_count = 0
        scanned_count = 0
        for path in arrow_files:
            raw_index = cached_raw_index(cache, path)
            if raw_index is None:
                ds = StreamWindowDataset(path, seq_len=seq_len, num_streams=num_streams, stride=stride, **kwargs)
                scanned_count += 1
            else:
                ds = CachedStreamWindowDataset(
                    path,
                    raw_index,
                    seq_len=seq_len,
                    num_streams=num_streams,
                    stride=stride,
                    **kwargs,
                )
                cached_count += 1
            if len(ds) > 0:
                self.datasets.append(ds)

        if not self.datasets:
            raise ValueError("No non-empty Arrow datasets were found.")

        lengths = [len(ds) for ds in self.datasets]
        self.cumulative_lengths = np.cumsum(lengths).tolist()
        print(
            f"Window index source: cached_files={cached_count} scanned_files={scanned_count}",
            flush=True,
        )

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        ds_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        prev = 0 if ds_idx == 0 else self.cumulative_lengths[ds_idx - 1]
        return self.datasets[ds_idx][idx - prev]


def build_stream_dataset(arrow_files, dataset_kwargs, cache):
    if cache is None:
        return MultiArrowStreamDataset(arrow_files, **dataset_kwargs)
    return CachedMultiArrowStreamDataset(arrow_files, cache, **dataset_kwargs)


def save_reconstruction_figure(model, dataset, run_cfg, run_name, sample_idx=0):
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    sample = dataset[sample_idx]
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }
    batch = model.prepare_custom_batch(batch)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            _, posterior_output, _, _, future_output = model._observe_and_imagine(batch)

    rf = run_cfg.RECEPTIVE_FIELD
    obs_t = rf - 1
    fut_t = 0
    fut_batch_t = rf

    def rgb(tensor):
        return tensor.detach().float().cpu().permute(1, 2, 0).clamp(0, 1)

    def depth(tensor):
        return tensor.detach().float().cpu()

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = np.asarray(axes)

    axes[0, 0].imshow(rgb(batch["image_raw"][0, obs_t]))
    axes[0, 0].set_title("Observed RGB")
    axes[0, 1].imshow(rgb(posterior_output["rgb_2"][0, obs_t]))
    axes[0, 1].set_title("Posterior RGB")
    axes[0, 2].imshow(rgb(batch["image_raw"][0, fut_batch_t]))
    axes[0, 2].set_title("Future target RGB")
    axes[0, 3].imshow(rgb(future_output["rgb_2"][0, fut_t]))
    axes[0, 3].set_title("Prior future RGB")

    axes[1, 0].imshow(depth(batch["lidar"][0, obs_t, 3]), cmap="magma", vmin=0, vmax=80)
    axes[1, 0].set_title("Observed depth")
    axes[1, 1].imshow(depth(posterior_output["lidar_reconstruction_2"][0, obs_t, 3]), cmap="magma", vmin=0, vmax=80)
    axes[1, 1].set_title("Posterior depth")
    axes[1, 2].imshow(depth(batch["lidar"][0, fut_batch_t, 3]), cmap="magma", vmin=0, vmax=80)
    axes[1, 2].set_title("Future target depth")
    axes[1, 3].imshow(depth(future_output["lidar_reconstruction_2"][0, fut_t, 3]), cmap="magma", vmin=0, vmax=80)
    axes[1, 3].set_title("Prior future depth")

    for ax in axes.ravel():
        ax.axis("off")

    out_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_reconstruction_s{sample_idx}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reconstruction figure: {out_path}")


def main():
    args = parse_args()

    run_cfg = copy.deepcopy(cfg)
    run_cfg.DATA.USE_ALL_TRAIN_RUNS = args.train_run is None
    run_cfg.DATA.NUM_WORKERS = args.num_workers
    run_cfg.STEPS = args.steps
    run_cfg.OPTIMIZER.LR = args.lr if args.lr is not None else run_cfg.OPTIMIZER.LR
    run_cfg.LOGGING.RUN_NAME = args.run_name
    run_cfg.LOGGING.BASE_FILE = "trainer11_reviewed_alldata_40epoch_server.ipynb"
    run_cfg.LOGGING.BEST_CHECKPOINT_PATH = None
    run_cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM = 256
    run_cfg.MODEL.TRANSITION.STATE_DIM = 128

    run_cfg.LOSSES.WEIGHT_PROBABILISTIC = 1e-2
    run_cfg.LOSSES.KL_FREE_BITS = 1.0
    run_cfg.LOSSES.KL_BALANCING_ALPHA = 0.75
    run_cfg.LOSSES.WEIGHT_LIDAR_RE = 1.0
    run_cfg.LOSSES.WEIGHT_LIDAR_EMPTY = 0.05
    run_cfg.LOSSES.WEIGHT_RGB = 1.0
    run_cfg.LOSSES.WEIGHT_FUTURE = 1.0

    manifest = load_arrow_manifest()
    train_files = select_train_files(manifest, args.train_run)
    val_rl_files = [_as_arrow_path(args.val_rl_run or run_cfg.DATA.VAL_RL_RUN)]
    val_ds_files = [_as_arrow_path(args.val_ds_run or run_cfg.DATA.VAL_DS_RUN)]

    accelerator, devices, strategy, precision, world_size = resolve_runtime(
        args.devices,
        args.strategy,
        args.precision,
    )
    global_batch_size = args.batch_size * world_size

    seq_len = run_cfg.RECEPTIVE_FIELD + run_cfg.FUTURE_HORIZON
    stride = run_cfg.RECEPTIVE_FIELD * run_cfg.DATA.SAMPLE_EVERY_N
    dataset_kwargs = dict(
        seq_len=seq_len,
        stride=stride,
        num_streams=global_batch_size,
        sample_every_n=run_cfg.DATA.SAMPLE_EVERY_N,
        image_input_size=run_cfg.DATA.IMAGE_INPUT_SIZE,
        rgb_recon_size=run_cfg.DATA.RGB_RECON_SIZE,
        lidar_size=run_cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
        frame_step=run_cfg.DATA.FRAME_STEP,
    )

    window_index_path = (
        args.window_index.expanduser().resolve()
        if args.window_index
        else Path(run_cfg.DATA.ROOT).expanduser().resolve() / "arrow_window_index.pkl"
    )
    window_index_cache = (
        None
        if args.no_window_index_cache
        else load_window_index_cache(window_index_path, run_cfg)
    )

    print(f"PROJECT_ROOT={PROJECT_ROOT}", flush=True)
    print(f"LOG_ROOT={LOG_ROOT}", flush=True)
    print(f"run_name={run_cfg.LOGGING.RUN_NAME}", flush=True)
    print(f"train files: {len(train_files)}", flush=True)
    print(f"val files: RL={val_rl_files} DS={val_ds_files}", flush=True)
    print(f"devices={devices} strategy={strategy} precision={precision}", flush=True)
    print(f"per_gpu_batch_size={args.batch_size} global_batch_size={global_batch_size} num_streams={global_batch_size}", flush=True)
    print(
        "latent: "
        f"h={run_cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM} "
        f"z={run_cfg.MODEL.TRANSITION.STATE_DIM}",
        flush=True,
    )
    print(
        "losses: "
        f"prob={run_cfg.LOSSES.WEIGHT_PROBABILISTIC} "
        f"kl_free_bits={run_cfg.LOSSES.KL_FREE_BITS} "
        f"lidar_re={run_cfg.LOSSES.WEIGHT_LIDAR_RE} "
        f"lidar_empty={run_cfg.LOSSES.WEIGHT_LIDAR_EMPTY} "
        f"rgb={run_cfg.LOSSES.WEIGHT_RGB} "
        f"future={run_cfg.LOSSES.WEIGHT_FUTURE}",
        flush=True,
    )
    print("Building train dataset object...", flush=True)
    train_ds = build_stream_dataset(train_files, dataset_kwargs, window_index_cache)
    print(f"Built train_ds: {len(train_ds)} windows", flush=True)
    print("Building validation dataset objects...", flush=True)
    val_rl_ds = build_stream_dataset(val_rl_files, dataset_kwargs, window_index_cache)
    val_ds_ds = build_stream_dataset(val_ds_files, dataset_kwargs, window_index_cache)
    print(f"Built validation: val_rl={len(val_rl_ds)} windows, val_ds={len(val_ds_ds)} windows", flush=True)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_rl_loader = DataLoader(val_rl_ds, **loader_kwargs)
    val_ds_loader = DataLoader(val_ds_ds, **loader_kwargs)

    model = WorldModelTrainer(cfg=run_cfg, lr=run_cfg.OPTIMIZER.LR, embedding_n_channels=128)
    model.val_dataset_names = ["RL", "DS"]

    ckpt_dir = LOG_ROOT / run_cfg.LOGGING.RUN_NAME / "checkpoints"
    callbacks = [
        pl.callbacks.ModelSummary(),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch={epoch:02d}-step={step:06d}",
            save_top_k=args.save_top_k,
            monitor=args.monitor,
            mode=args.monitor_mode,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        RankZeroExperimentCSVLogger(
            cfg=run_cfg,
            train_files=train_files,
            val_files=val_rl_files + val_ds_files,
            batch_size=global_batch_size,
            change_summary=(
                "server full data: latent h=256 z=128; restored LiDAR loss weights "
                "WEIGHT_LIDAR_RE=1.0, WEIGHT_LIDAR_EMPTY=0.05; "
                "kept RGB=1.0, KL free-bits=1.0, OneCycleLR"
            ),
        ),
    ]

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_steps=run_cfg.STEPS,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(save_dir=str(LOG_ROOT), name=run_cfg.LOGGING.RUN_NAME),
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=run_cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES,
        num_sanity_val_steps=1,
        use_distributed_sampler=(world_size > 1),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_rl_loader, val_ds_loader])

    if not trainer.is_global_zero:
        return

    best_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
    run_cfg.LOGGING.BEST_CHECKPOINT_PATH = best_path
    if best_path:
        print(f"Best checkpoint: {best_path}")

    print("Key metrics:")
    for key in [
        "val/RL_loss",
        "val/DS_loss",
        "val/RL_rgb_psnr",
        "val/DS_rgb_psnr",
        "val/RL_lidar_chamfer_xyz",
        "val/DS_lidar_chamfer_xyz",
        "val/RL_future_lidar_chamfer_xyz",
        "val/DS_future_lidar_chamfer_xyz",
    ]:
        if key in trainer.callback_metrics:
            value = trainer.callback_metrics[key]
            if torch.is_tensor(value):
                value = value.detach().float().cpu().item()
            print(f"  {key}: {value:.6g}")

    if not args.no_metric_export:
        try:
            event_file, csv_path, png_path = export_metrics(
                log_root=LOG_ROOT,
                run_name=run_cfg.LOGGING.RUN_NAME,
                out_dir=PROJECT_ROOT / "results" / "figures",
                smooth=args.metric_smooth,
            )
            print(f"Exported TensorBoard metrics from: {event_file}")
            print(f"Saved metric CSV: {csv_path}")
            print(f"Saved metric figure: {png_path}")
        except Exception as exc:
            warnings.warn(f"Could not export TensorBoard metrics: {exc}")

    if best_path and not args.no_reconstruction:
        best_model = WorldModelTrainer.load_from_checkpoint(
            best_path,
            cfg=run_cfg,
            lr=run_cfg.OPTIMIZER.LR,
            embedding_n_channels=128,
        )
        viz_ds = MUVODataset(
            val_rl_files[0],
            seq_len=seq_len,
            stride=stride,
            sample_every_n=run_cfg.DATA.SAMPLE_EVERY_N,
            image_input_size=run_cfg.DATA.IMAGE_INPUT_SIZE,
            rgb_recon_size=run_cfg.DATA.RGB_RECON_SIZE,
            lidar_size=run_cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
            frame_step=run_cfg.DATA.FRAME_STEP,
        )
        save_reconstruction_figure(best_model, viz_ds, run_cfg, run_cfg.LOGGING.RUN_NAME)


if __name__ == "__main__":
    main()
