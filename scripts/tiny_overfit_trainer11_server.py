#!/usr/bin/env python3
"""Server-oriented tiny-overfit probe for trainer11.

This version imports definitions from
trainer11_reviewed_alldata_40epoch_server.ipynb so it follows the server
path/log-root resolution used by that notebook. It defaults to one GPU because
tiny overfit is primarily a capacity/pipeline sanity check, but it can run with
DDP via --devices or ALPHA26_TINY_DEVICES.
"""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = SCRIPT_DIR.parent / "notebooks" / "training" / "trainer11_reviewed_alldata_40epoch_server.ipynb"
DEFS_PATH = SCRIPT_DIR / "trainer11_server_defs.py"


def load_notebook_definitions():
    """Load server definitions without requiring Jupyter/ipykernel."""
    if DEFS_PATH.exists():
        spec = importlib.util.spec_from_file_location("trainer11_server_defs_runtime", DEFS_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        globals().update({
            name: value
            for name, value in vars(module).items()
            if not name.startswith("__")
        })
        return

    # Fallback for development worktrees that have the notebook but not the
    # exported defs module. This reads the .ipynb as plain JSON; no ipykernel.
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    ns = globals()
    for cell_idx in range(4, 13):
        cell = nb["cells"][cell_idx]
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        exec(compile(code, f"{NOTEBOOK_PATH}:cell{cell_idx}", "exec"), ns)


def make_subset(dataset, n):
    from torch.utils.data import Subset

    n = min(int(n), len(dataset))
    if n <= 0:
        raise ValueError("tiny subset must contain at least one sample")
    return Subset(dataset, list(range(n)))


def resolve_tiny_runtime(requested_devices):
    import torch
    import warnings

    requested = str(requested_devices).strip().lower()
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if cuda_count == 0:
        return "auto", "auto", "auto", os.environ.get("ALPHA26_PRECISION", "32-true"), 1

    if requested in {"auto", "all"}:
        world_size = cuda_count
    else:
        world_size = max(1, int(requested))
        if world_size > cuda_count:
            warnings.warn(
                f"Requested {world_size} GPUs but only {cuda_count} are visible; using {cuda_count}."
            )
            world_size = cuda_count

    devices = world_size
    accelerator = "gpu"
    strategy = os.environ.get("ALPHA26_TINY_STRATEGY", "ddp" if world_size > 1 else "auto")
    precision = os.environ.get("ALPHA26_PRECISION", "16-mixed")
    return accelerator, devices, strategy, precision, world_size


def save_reconstruction_figure(model, dataset, cfg, run_name, sample_idx=0):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

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
        amp = device == "cuda"
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            _, posterior_output, _, _, future_output = model._observe_and_imagine(batch)

    rf = cfg.RECEPTIVE_FIELD
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
    parser = argparse.ArgumentParser(description="Run server tiny-overfit probe for trainer11.")
    parser.add_argument("--samples", type=int, default=8, help="Number of fixed train windows to overfit.")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps.")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ALPHA26_BATCH_SIZE", "2")))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-run", default="train_run_002.arrow")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("ALPHA26_NUM_WORKERS", "2")))
    parser.add_argument("--devices", default=os.environ.get("ALPHA26_TINY_DEVICES", "1"))
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)
    load_notebook_definitions()

    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader

    accelerator, devices, strategy, precision, world_size = resolve_tiny_runtime(args.devices)
    global_batch_size = args.batch_size * world_size

    tiny_cfg = copy.deepcopy(cfg)
    tiny_cfg.DATA.USE_ALL_TRAIN_RUNS = False
    tiny_cfg.DATA.TRAIN_RUN = args.train_run
    tiny_cfg.DATA.NUM_WORKERS = args.num_workers
    tiny_cfg.OPTIMIZER.LR = args.lr
    tiny_cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES = 1
    tiny_cfg.SCHEDULER.NAME = "none"
    tiny_cfg.STEPS = args.steps
    tiny_cfg.LOGGING.RUN_NAME = args.run_name or f"trainer11_server_tiny_overfit_n{args.samples}_s{args.steps}"
    tiny_cfg.LOGGING.BEST_CHECKPOINT_PATH = None

    manifest = load_arrow_manifest()
    train_file = _as_arrow_path(args.train_run)
    if train_file not in manifest["train"] and not os.path.exists(train_file):
        raise FileNotFoundError(f"Could not find train run: {train_file}")

    seq_len = tiny_cfg.RECEPTIVE_FIELD + tiny_cfg.FUTURE_HORIZON
    stride = tiny_cfg.RECEPTIVE_FIELD * tiny_cfg.DATA.SAMPLE_EVERY_N
    base_ds = MultiArrowStreamDataset(
        [train_file],
        seq_len=seq_len,
        stride=stride,
        num_streams=global_batch_size,
        sample_every_n=tiny_cfg.DATA.SAMPLE_EVERY_N,
        image_input_size=tiny_cfg.DATA.IMAGE_INPUT_SIZE,
        rgb_recon_size=tiny_cfg.DATA.RGB_RECON_SIZE,
        lidar_size=tiny_cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
        frame_step=tiny_cfg.DATA.FRAME_STEP,
    )
    tiny_ds = make_subset(base_ds, args.samples)
    print(f"Tiny overfit dataset: {len(tiny_ds)} windows from {train_file}")
    print(f"LOG_ROOT={LOG_ROOT}")
    print(f"devices={devices} strategy={strategy} precision={precision}")
    print(f"per_gpu_batch_size={args.batch_size} global_batch_size={global_batch_size} num_streams={global_batch_size}")

    loader = DataLoader(
        tiny_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = WorldModelTrainer(cfg=tiny_cfg, lr=tiny_cfg.OPTIMIZER.LR, embedding_n_channels=128)
    model.val_dataset_names = ["tiny"]

    ckpt_dir = LOG_ROOT / tiny_cfg.LOGGING.RUN_NAME / "checkpoints"
    callbacks = [
        pl.callbacks.ModelSummary(max_depth=2),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="step={step:06d}-psnr={val/tiny_rgb_psnr:.3f}",
            save_top_k=3,
            monitor="val/tiny_rgb_psnr",
            mode="max",
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_steps=tiny_cfg.STEPS,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(save_dir=str(LOG_ROOT), name=tiny_cfg.LOGGING.RUN_NAME),
        log_every_n_steps=1,
        val_check_interval=1,
        check_val_every_n_epoch=None,
        limit_val_batches=1.0,
        accumulate_grad_batches=tiny_cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES,
        num_sanity_val_steps=0,
        use_distributed_sampler=(world_size > 1),
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=[loader])

    if not trainer.is_global_zero:
        return

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_path}")
    print("Key metrics:")
    for key in ["val/tiny_rgb_psnr", "val/tiny_loss", "val/tiny_lidar_chamfer_xyz", "train_loss"]:
        if key in trainer.callback_metrics:
            value = trainer.callback_metrics[key]
            if torch.is_tensor(value):
                value = value.detach().float().cpu().item()
            print(f"  {key}: {value:.6g}")

    if best_path:
        best_model = WorldModelTrainer.load_from_checkpoint(
            best_path,
            cfg=tiny_cfg,
            lr=tiny_cfg.OPTIMIZER.LR,
            embedding_n_channels=128,
        )
        save_reconstruction_figure(best_model, tiny_ds, tiny_cfg, tiny_cfg.LOGGING.RUN_NAME)


if __name__ == "__main__":
    main()
