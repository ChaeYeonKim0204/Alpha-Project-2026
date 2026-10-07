"""
train_muvo.py — MUVO-style alpha training entrypoint
Usage:
    python scripts/model_variants/train_muvo.py                         # config_muvo.py 기본값 사용
    ALPHA26_ROOT=/your/root python scripts/model_variants/train_muvo.py # 환경변수로 경로 override
"""

import argparse
import copy
import math
import os
import warnings

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset

from config_muvo import cfg, PROJECT_ROOT
from data_muvo import (
    load_arrow_manifest,
    _select_from_manifest,
    MultiArrowStreamDataset,
)
from trainer_muvo import (
    WorldModelTrainer,
    ExperimentCSVLogger,
    PeriodicReconstructionCallback,
    save_loss_metric_figure,
    save_reconstruction_figure,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train alpha26 MUVO-style world model.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--series", default=None,
                        help="Set experiment series label (e.g. 'muvo_branch_post_migration'). "
                             "Used to populate the 'series' column in results/experiment_log.csv.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--h-dim", type=int, default=None,
                        help="Override cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM (e.g. 256, 512).")
    parser.add_argument("--z-dim", type=int, default=None,
                        help="Override cfg.MODEL.TRANSITION.STATE_DIM (e.g. 128, 256).")
    parser.add_argument("--weight-ssim", type=float, default=None,
                        help="Override cfg.LOSSES.WEIGHT_SSIM (0.0 = off, 0.06 = upstream-effective).")
    parser.add_argument("--devices", default=os.environ.get("ALPHA26_DEVICES", "auto"))
    parser.add_argument("--strategy", default=os.environ.get("ALPHA26_STRATEGY", None))
    parser.add_argument("--precision", default=os.environ.get("ALPHA26_PRECISION", "16-mixed"))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=20)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--save-top-k", type=int, default=3)
    parser.add_argument("--monitor", default="val/RL_loss")
    parser.add_argument("--monitor-mode", default="min", choices=["min", "max"])
    parser.add_argument("--train-run", default=None)
    parser.add_argument("--val-rl-run", default=None)
    parser.add_argument("--val-ds-run", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--overfit-samples", type=int, default=8,
                        help="Use only the first N train windows for train/val overfit checking. Set 0 to disable.")
    parser.add_argument("--one-window-overfit", action="store_true",
                        help="Train on exactly one 8-frame window from one train Arrow file and visualize that same sample.")
    parser.add_argument("--validate-overfit", action="store_true",
                        help="Also run validation on the same overfit subset.")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--no-metric-export", action="store_true")
    parser.add_argument("--recon-every-n-epochs", type=int, default=0,
                        help="N > 0이면 매 N epoch마다 reconstruction figure 저장 "
                             "(overfit 검증용). 0=비활성.")
    parser.add_argument("--recon-sample-idx", type=int, default=0,
                        help="periodic reconstruction에 사용할 dataset sample index.")
    return parser.parse_args()


def resolve_runtime(requested_devices, requested_strategy, precision):
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


def apply_overrides(run_cfg, args):
    if args.run_name is not None:
        run_cfg.LOGGING.RUN_NAME = args.run_name
    if args.series is not None:
        run_cfg.LOGGING.SERIES = args.series
    if args.epochs is not None:
        run_cfg.EPOCHS = args.epochs
    if args.lr is not None:
        run_cfg.OPTIMIZER.LR = args.lr
    if args.h_dim is not None:
        run_cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM = args.h_dim
    if args.z_dim is not None:
        run_cfg.MODEL.TRANSITION.STATE_DIM = args.z_dim
    if args.weight_ssim is not None:
        run_cfg.LOSSES.WEIGHT_SSIM = args.weight_ssim
    if args.num_workers is not None:
        run_cfg.DATA.NUM_WORKERS = args.num_workers
    if args.train_run is not None:
        run_cfg.DATA.TRAIN_RUN = args.train_run
        run_cfg.DATA.USE_ALL_TRAIN_RUNS = False
    if args.one_window_overfit:
        run_cfg.DATA.USE_ALL_TRAIN_RUNS = False
    if args.val_rl_run is not None:
        run_cfg.DATA.VAL_RL_RUN = args.val_rl_run
    if args.val_ds_run is not None:
        run_cfg.DATA.VAL_DS_RUN = args.val_ds_run
    return run_cfg


def build_dataloaders(cfg, batch_size, num_streams=None, overfit_samples=8):
    manifest = load_arrow_manifest()
    num_streams = batch_size if num_streams is None else num_streams
    seq_len    = cfg.RECEPTIVE_FIELD + cfg.FUTURE_HORIZON
    # SAMPLE_EVERY_N을 곱해야 겹치는 window 간 hidden carry가 없어진다.
    stride     = cfg.RECEPTIVE_FIELD * cfg.DATA.SAMPLE_EVERY_N

    train_files  = _select_from_manifest(
        manifest, "train",
        preferred=cfg.DATA.TRAIN_RUN,
        use_all=cfg.DATA.USE_ALL_TRAIN_RUNS,
    )
    val_rl_files = train_files
    val_ds_files = train_files

    dataset_kwargs = dict(
        seq_len=seq_len,
        stride=stride,
        num_streams=num_streams,
        sample_every_n=cfg.DATA.SAMPLE_EVERY_N,
        image_input_size=cfg.DATA.IMAGE_INPUT_SIZE,
        rgb_recon_size=cfg.DATA.RGB_RECON_SIZE,
        lidar_size=cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
        frame_step=cfg.DATA.FRAME_STEP,
    )
    train_ds_full = MultiArrowStreamDataset(train_files, **dataset_kwargs)
    if overfit_samples and overfit_samples > 0:
        subset_size = min(int(overfit_samples), len(train_ds_full))
        if subset_size <= 0:
            raise ValueError("overfit subset must contain at least one sample")
        train_ds = Subset(train_ds_full, list(range(subset_size)))
    else:
        train_ds = train_ds_full
    val_rl_ds = train_ds
    val_ds_ds = train_ds

    print(f"train files  : {len(train_files)}")
    if overfit_samples and overfit_samples > 0:
        print(f"overfit samples: {len(train_ds)} windows from train dataset")
    print("validation uses the same dataset as train for overfit checking")
    print(f"train_ds size: {len(train_ds)} windows "
          f"(val_rl={len(val_rl_ds)}, val_ds={len(val_ds_ds)})")

    num_workers   = getattr(cfg.DATA, "NUM_WORKERS", 0)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    train_loader   = DataLoader(train_ds,   **loader_kwargs)
    val_rl_loader  = DataLoader(val_rl_ds,  **loader_kwargs)
    val_ds_loader  = DataLoader(val_ds_ds,  **loader_kwargs)

    return (train_loader, val_rl_loader, val_ds_loader,
            train_files, val_rl_files, val_ds_files, batch_size,
            train_ds, val_rl_ds, val_ds_ds)


def build_callbacks(cfg, train_files, val_rl_files, val_ds_files, batch_size, save_dir, args):
    return [
        pl.callbacks.ModelSummary(),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(save_dir, cfg.LOGGING.RUN_NAME, "checkpoints"),
            filename="epoch={epoch:02d}-step={step:06d}",
            save_top_k=args.save_top_k,
            monitor=args.monitor,
            mode=args.monitor_mode,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ExperimentCSVLogger(
            cfg=cfg,
            train_files=train_files,
            val_files=val_rl_files + val_ds_files,
            batch_size=batch_size,
            change_summary=(
                f"MUVO-style alpha: {cfg.DATA.IMAGE_INPUT_SIZE[0]}x{cfg.DATA.IMAGE_INPUT_SIZE[1]} "
                f"center-cropped image input, "
                f"{cfg.DATA.LIDAR_RANGE_VIEW_SIZE[0]}x{cfg.DATA.LIDAR_RANGE_VIEW_SIZE[1]} LiDAR range-view "
                f"with {cfg.DATA.LIDAR_SCALE:g}m scaling, ResNet18 multi-scale FPN "
                f"to {cfg.MODEL.EMBEDDING_DIM} channels, "
                f"{cfg.MODEL.FUSION.TRANSFORMER_LAYERS}-layer sensor fusion transformer, "
                f"RSSM h{cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM}/z{cfg.MODEL.TRANSITION.STATE_DIM}, "
                f"WEIGHT_SSIM={getattr(cfg.LOSSES, 'WEIGHT_SSIM', 0.0)}."
            ),
        ),
    ]


def build_trainer(cfg, callbacks, save_dir, args, accelerator, devices, strategy, precision, world_size):
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=cfg.EPOCHS,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(save_dir=save_dir, name=cfg.LOGGING.RUN_NAME),
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES,
        num_sanity_val_steps=1,
        use_distributed_sampler=(world_size > 1),
    )


def maybe_extend_onecycle_resume_checkpoint(ckpt_path, cfg, train_loader):
    if not ckpt_path or getattr(cfg.SCHEDULER, "NAME", "none") != "OneCycleLR":
        return ckpt_path

    total_steps = math.ceil(
        len(train_loader) / max(1, int(cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES))
    ) * int(cfg.EPOCHS)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    changed = False
    for scheduler_state in checkpoint.get("lr_schedulers", []):
        old_total_steps = scheduler_state.get("total_steps")
        if not isinstance(old_total_steps, int) or old_total_steps >= total_steps:
            continue

        scheduler_state["total_steps"] = total_steps
        phases = scheduler_state.get("_schedule_phases")
        if isinstance(phases, list) and len(phases) == 2:
            phases[0]["end_step"] = float(getattr(cfg.SCHEDULER, "PCT_START", 0.3) * total_steps) - 1
            phases[1]["end_step"] = total_steps - 1
        changed = True
        print(f"Extended OneCycleLR resume total_steps: {old_total_steps} -> {total_steps}")

    if not changed:
        return ckpt_path

    patched_path = os.path.join(
        os.path.dirname(ckpt_path),
        f"resume_total_steps_{total_steps}.ckpt",
    )
    torch.save(checkpoint, patched_path)
    print(f"Using patched resume checkpoint: {patched_path}")
    return patched_path


def main():
    args = parse_args()
    run_cfg = apply_overrides(copy.deepcopy(cfg), args)
    save_dir = "./logs"
    if args.one_window_overfit:
        args.overfit_samples = 1
        if args.batch_size != 1:
            warnings.warn("--one-window-overfit forces batch_size=1 so the single window is not dropped.")
            args.batch_size = 1

    overfit_mode = args.overfit_samples and args.overfit_samples > 0
    run_validation = (not overfit_mode) or args.validate_overfit
    if overfit_mode and not run_validation and args.monitor == "val/RL_loss":
        args.monitor = "train_loss_epoch"
        args.monitor_mode = "min"

    accelerator, devices, strategy, precision, world_size = resolve_runtime(
        args.devices, args.strategy, args.precision,
    )
    global_batch_size = args.batch_size * world_size

    (train_loader, val_rl_loader, val_ds_loader,
     train_files, val_rl_files, val_ds_files, batch_size,
     train_ds, val_rl_ds, val_ds_ds) = build_dataloaders(
        run_cfg, batch_size=args.batch_size, num_streams=global_batch_size,
        overfit_samples=args.overfit_samples,
    )

    print(f"run_name={run_cfg.LOGGING.RUN_NAME}")
    print(f"max_epochs={run_cfg.EPOCHS}")
    print(f"devices={devices} strategy={strategy} precision={precision}")
    print(f"per_device_batch_size={args.batch_size} global_batch_size={global_batch_size}")
    print(f"validation_enabled={run_validation}")
    print(f"checkpoint_monitor={args.monitor} mode={args.monitor_mode}")

    model = WorldModelTrainer(cfg=run_cfg, lr=run_cfg.OPTIMIZER.LR)
    model.val_dataset_names = ["RL", "DS"]

    callbacks = build_callbacks(run_cfg, train_files, val_rl_files, val_ds_files, global_batch_size, save_dir, args)
    if args.recon_every_n_epochs and args.recon_every_n_epochs > 0:
        callbacks.append(PeriodicReconstructionCallback(
            dataset=train_ds,
            run_cfg=run_cfg,
            run_name=run_cfg.LOGGING.RUN_NAME,
            every_n_epochs=args.recon_every_n_epochs,
            sample_idx=args.recon_sample_idx,
        ))
        print(f"PeriodicReconstructionCallback: save figure every {args.recon_every_n_epochs} epoch(s).")
    trainer   = build_trainer(run_cfg, callbacks, save_dir, args, accelerator, devices, strategy, precision, world_size)
    resume_ckpt_path = maybe_extend_onecycle_resume_checkpoint(
        args.resume_from_checkpoint, run_cfg, train_loader,
    )

    torch.set_float32_matmul_precision("medium")
    if run_validation:
        trainer.fit(model, train_dataloaders=train_loader,
                    val_dataloaders=[val_rl_loader, val_ds_loader],
                    ckpt_path=resume_ckpt_path)
    else:
        trainer.fit(model, train_dataloaders=train_loader,
                    ckpt_path=resume_ckpt_path)

    if not trainer.is_global_zero:
        return model, train_ds, val_rl_ds, val_ds_ds

    figure_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    if trainer.logger is not None and not args.no_metric_export:
        save_loss_metric_figure(
            run_cfg, trainer.logger.log_dir, figure_dir,
            filename=f"{run_cfg.LOGGING.RUN_NAME}_loss_metrics.png",
        )

    if not args.no_reconstruction:
        best_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
        last_path = getattr(trainer.checkpoint_callback, "last_model_path", "")
        fallback_last = os.path.join(save_dir, run_cfg.LOGGING.RUN_NAME, "checkpoints", "last.ckpt")
        ckpt_path = best_path or last_path or (fallback_last if os.path.exists(fallback_last) else "")
        if ckpt_path:
            label = "Best" if ckpt_path == best_path else "Last"
            print(f"{label} checkpoint for reconstruction: {ckpt_path}")
            viz_model = WorldModelTrainer.load_from_checkpoint(
                ckpt_path,
                cfg=run_cfg,
                lr=run_cfg.OPTIMIZER.LR,
            )
        else:
            print("No checkpoint found for reconstruction; using in-memory final model.")
            viz_model = model

        save_reconstruction_figure(viz_model, train_ds, run_cfg, run_cfg.LOGGING.RUN_NAME)

    return model, train_ds, val_rl_ds, val_ds_ds


if __name__ == "__main__":
    trained_model, train_ds, val_rl_ds, val_ds_ds = main()
