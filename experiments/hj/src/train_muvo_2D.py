"""train_muvo_2D.py — MUVO 2D-token RSSM (RSSMTD) training entrypoint

`train_muvo.py` 베이스 + 다음 추가:
- ClearML `Task.init(...)` (v3, Q14/Q15) — primary 실험 트래킹
- `--no-clearml`, `--cml-project`, `--cml-task` CLI flags
- `--no-augmentation` CLI flag
- `--effective-hz N` CLI flag → cfg.DATA.EFFECTIVE_HZ 갱신 + SAMPLE_EVERY_N 자동 재계산
- `--h-dim != --z-dim` 시 warning + z-dim으로 강제 (RSSMTD constraint)
- change_summary 갱신
"""

import argparse
import copy
import math
import os
import warnings

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset

from config_muvo_2D import cfg, PROJECT_ROOT, _derive_sample_every_n
from data_muvo_2D import (
    load_arrow_manifest,
    _select_from_manifest,
    MultiArrowStreamDataset,
)
from trainer_muvo_2D import (
    WorldModelTrainer,
    ExperimentCSVLogger,
    PeriodicReconstructionCallback,
    save_loss_metric_figure,
    save_reconstruction_figure,
    save_future_frames_per_frame,
    save_counterfactual_figure,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train alpha26 MUVO 2D-token RSSM (RSSMTD).")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--series", default=None,
                        help="experiment_log.csv의 'series' 컬럼 값.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler", default=None,
                        choices=["OneCycleLR", "none"],
                        help="LR scheduler override. 'none'이면 constant LR (overfit 디버깅용).")
    parser.add_argument("--h-dim", type=int, default=None,
                        help="cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM override. "
                             "RSSMTD는 h-dim=z-dim 필수 — 다르게 지정하면 z-dim으로 강제.")
    parser.add_argument("--z-dim", type=int, default=None,
                        help="cfg.MODEL.TRANSITION.STATE_DIM override (also overrides h-dim).")
    parser.add_argument("--weight-ssim", type=float, default=None)
    parser.add_argument("--effective-hz", type=int, default=None,
                        help="cfg.DATA.EFFECTIVE_HZ override. SAMPLE_EVERY_N 자동 재계산.")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable image augmentation (cfg.DATA.AUGMENTATION.ENABLED=False).")
    # ClearML flags (v3)
    parser.add_argument("--no-clearml", action="store_true",
                        help="ClearML 비활성. 디버깅용 빠른 run에 사용.")
    parser.add_argument("--cml-project", default=None, help="ClearML project name override.")
    parser.add_argument("--cml-task", default=None, help="ClearML task name override.")

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
    parser.add_argument("--window-index", default=None, help="(reserved) window index pkl 경로.")
    parser.add_argument("--steps", type=int, default=None, help="max training steps override.")
    parser.add_argument("--val-check-interval", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--overfit-samples", type=int, default=8,
                        help="overfit subset window count. 0이면 비활성.")
    parser.add_argument("--one-window-overfit", action="store_true")
    parser.add_argument("--validate-overfit", action="store_true")
    parser.add_argument("--no-diffusion", action="store_true",
                        help="미래 diffusion(타깃 계산 + DDIM 생성)을 끄고 reconstruction(autoencoder)만 "
                             "학습/시각화. reconstruction-only overfit 디버깅용.")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--no-metric-export", action="store_true")
    parser.add_argument("--recon-every-n-epochs", type=int, default=0,
                        help="N > 0이면 마지막(최종) 에폭에서 reconstruction figure 1회 저장 "
                             "(overfit 검증용). 중간 에폭은 저장 안 함. 0=비활성.")
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

    # 기본 multi-GPU strategy는 ddp_find_unused_parameters_true.
    # SensorFusionTransformer.type_embedding 슬롯 2/3(policy/action)은 forward에서
    # 인덱싱되지 않아 grad를 못 받는 unused parameter라, 평범한 "ddp"는
    # "did not receive grad" RuntimeError로 죽는다 (CLAUDE.md 게이팅 이슈와 동일).
    strategy = requested_strategy or (
        "ddp_find_unused_parameters_true" if world_size > 1 else "auto"
    )
    return "gpu", world_size, strategy, precision, world_size


def apply_overrides(run_cfg, args):
    if args.run_name is not None:
        run_cfg.LOGGING.RUN_NAME = args.run_name
    if args.series is not None:
        run_cfg.LOGGING.SERIES = args.series
    if args.epochs is not None:
        run_cfg.EPOCHS = args.epochs
    if args.steps is not None:
        run_cfg.STEPS = args.steps
    if args.lr is not None:
        run_cfg.OPTIMIZER.LR = args.lr
    if args.scheduler is not None:
        run_cfg.SCHEDULER.NAME = args.scheduler

    # RSSMTD constraint: h-dim == z-dim. 다르면 z-dim 우선.
    if args.h_dim is not None and args.z_dim is not None and args.h_dim != args.z_dim:
        warnings.warn(
            f"RSSMTD requires h-dim == z-dim. Got h-dim={args.h_dim}, z-dim={args.z_dim}. "
            f"Using {args.z_dim} for both."
        )
        args.h_dim = args.z_dim
    if args.h_dim is not None:
        run_cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM = args.h_dim
        # embedding_dim도 같이 맞춰야 함
        run_cfg.MODEL.EMBEDDING_DIM = args.h_dim
        run_cfg.MODEL.FUSION.TRANSFORMER_CHANNELS = args.h_dim
    if args.z_dim is not None:
        run_cfg.MODEL.TRANSITION.STATE_DIM = args.z_dim
        if args.h_dim is None:
            run_cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM = args.z_dim
            run_cfg.MODEL.EMBEDDING_DIM = args.z_dim
            run_cfg.MODEL.FUSION.TRANSFORMER_CHANNELS = args.z_dim

    if args.weight_ssim is not None:
        run_cfg.LOSSES.WEIGHT_SSIM = args.weight_ssim
    if args.effective_hz is not None:
        run_cfg.DATA.EFFECTIVE_HZ = args.effective_hz
        run_cfg.DATA.SAMPLE_EVERY_N = _derive_sample_every_n(
            run_cfg.DATA.CARLA_HZ, run_cfg.DATA.FRAME_STEP, args.effective_hz,
        )
        print(f"--effective-hz={args.effective_hz} → SAMPLE_EVERY_N={run_cfg.DATA.SAMPLE_EVERY_N}")
    if args.no_augmentation:
        run_cfg.DATA.AUGMENTATION.ENABLED = False
    if args.no_diffusion:
        run_cfg.MODEL.DIFFUSION.TRAIN_DIFFUSION = False
        run_cfg.LOSSES.WEIGHT_DIFFUSION = 0.0
        print("--no-diffusion: 미래 diffusion 비활성 (reconstruction-only).")
    if args.no_clearml:
        run_cfg.CML_ENABLED = False
    if args.cml_project is not None:
        run_cfg.CML_PROJECT = args.cml_project
    if args.cml_task is not None:
        run_cfg.CML_TASK = args.cml_task
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


def maybe_init_clearml(run_cfg, args):
    if not getattr(run_cfg, "CML_ENABLED", False) or args.no_clearml:
        print("ClearML disabled (--no-clearml or cfg.CML_ENABLED=False).")
        return None
    try:
        from clearml import Task
    except ImportError:
        warnings.warn("clearml not installed; skipping ClearML init. `pip install clearml` 후 다시.")
        return None
    task = Task.init(
        project_name=run_cfg.CML_PROJECT,
        task_name=run_cfg.CML_TASK,
        task_type=run_cfg.CML_TYPE,
        tags=list(run_cfg.CML_TAGS),
    )
    task.connect({
        "RECEPTIVE_FIELD": run_cfg.RECEPTIVE_FIELD,
        "FUTURE_HORIZON":  run_cfg.FUTURE_HORIZON,
        "EFFECTIVE_HZ":    run_cfg.DATA.EFFECTIVE_HZ,
        "EMBEDDING_DIM":   run_cfg.MODEL.EMBEDDING_DIM,
        "HIDDEN_STATE_DIM": run_cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
        "WEIGHT_ACTION":   run_cfg.LOSSES.WEIGHT_ACTION,
        "WEIGHT_SSIM":     run_cfg.LOSSES.WEIGHT_SSIM,
        "KL_FREE_BITS_ENABLED": run_cfg.LOSSES.KL_FREE_BITS_ENABLED,
        "AUGMENTATION_ENABLED": run_cfg.DATA.AUGMENTATION.ENABLED,
    })
    print(f"ClearML Task initialised: project={run_cfg.CML_PROJECT} task={run_cfg.CML_TASK}")
    return task


def build_dataloaders(cfg, batch_size, num_streams=None, overfit_samples=8):
    manifest = load_arrow_manifest()
    num_streams = batch_size if num_streams is None else num_streams
    seq_len    = cfg.RECEPTIVE_FIELD + cfg.FUTURE_HORIZON
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
    print(f"seq_len={seq_len} stride={stride} sample_every_n={cfg.DATA.SAMPLE_EVERY_N} "
          f"effective_hz={getattr(cfg.DATA, 'EFFECTIVE_HZ', '?')}Hz")
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
                f"MUVO 2D-token latent-diffusion (image-only): "
                f"{cfg.DATA.IMAGE_INPUT_SIZE[0]}x{cfg.DATA.IMAGE_INPUT_SIZE[1]} image, "
                f"FPN→{cfg.MODEL.EMBEDDING_DIM}ch, "
                f"transformer L={cfg.MODEL.FUSION.TRANSFORMER_LAYERS}/h={cfg.MODEL.FUSION.TRANSFORMER_HEADS}, "
                f"action_loss={cfg.LOSSES.WEIGHT_ACTION}, augment={cfg.DATA.AUGMENTATION.ENABLED}, "
                f"FH={cfg.FUTURE_HORIZON}, effective_hz={cfg.DATA.EFFECTIVE_HZ}."
            ),
        ),
    ]


def build_trainer(cfg, callbacks, save_dir, args, accelerator, devices, strategy, precision, world_size):
    # 학습 길이 제어: cfg.STEPS(>0)가 있으면 optimizer step 수로 학습을 구동한다.
    # (overfit은 작은 데이터셋을 STEPS만큼 반복해야 하므로 epoch만으론 부족 —
    #  예: EPOCHS=1 + 8샘플/배치2 = 4 step에서 멈춰버린다.)
    # --epochs를 명시한 경우에만 epoch cap을 함께 적용하고, 아니면 epoch 무제한(-1).
    steps = int(getattr(cfg, "STEPS", 0) or 0)
    if steps > 0:
        max_steps  = steps
        max_epochs = cfg.EPOCHS if args.epochs is not None else -1
    else:
        max_steps  = -1
        max_epochs = cfg.EPOCHS
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(save_dir=save_dir, name=cfg.LOGGING.RUN_NAME),
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval if args.val_check_interval is not None else None,
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
            warnings.warn("--one-window-overfit forces batch_size=1.")
            args.batch_size = 1

    overfit_mode = args.overfit_samples and args.overfit_samples > 0
    run_validation = (not overfit_mode) or args.validate_overfit
    if overfit_mode and not run_validation and args.monitor == "val/RL_loss":
        args.monitor = "train_loss_epoch"
        args.monitor_mode = "min"

    # ClearML init (이 시점에 cfg가 final이어야 task.connect가 정확)
    cml_task = maybe_init_clearml(run_cfg, args)

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
    _steps = int(getattr(run_cfg, "STEPS", 0) or 0)
    if _steps > 0:
        print(f"max_steps={_steps} (steps-driven; "
              f"max_epochs={'cap '+str(run_cfg.EPOCHS) if args.epochs is not None else 'unbounded'})")
    else:
        print(f"max_epochs={run_cfg.EPOCHS} (epoch-driven; STEPS<=0)")
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
        print("PeriodicReconstructionCallback: save reconstruction figure at the LAST epoch only.")
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

    # hj/figures/ (script는 hj/src/에 있으므로 두 단계 올라간다).
    figure_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
    )
    # 학습이 끝난 epoch(1-indexed). figure 파일명에 _e<epoch>로 표시.
    final_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
    run_tag = f"{run_cfg.LOGGING.RUN_NAME}_e{final_epoch:04d}"

    if trainer.logger is not None and not args.no_metric_export:
        try:
            save_loss_metric_figure(
                run_cfg, trainer.logger.log_dir, figure_dir,
                filename=f"hj_{run_tag}_loss_metrics.png",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"save_loss_metric_figure failed: {exc}")

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

        save_reconstruction_figure(viz_model, train_ds, run_cfg, run_tag)
        # diffusion 끄면 미래 생성 기반 figure(per-frame future, counterfactual)는 건너뛴다.
        if args.no_diffusion:
            print("--no-diffusion: future/counterfactual figure skip (reconstruction figure만 저장).")
        else:
            if getattr(run_cfg.LOGGING, "RECON_FIG_PER_FUTURE_FRAME", False):
                try:
                    save_future_frames_per_frame(viz_model, train_ds, run_cfg, run_tag)
                except Exception as exc:  # noqa: BLE001
                    print(f"save_future_frames_per_frame failed: {exc}")
            try:
                save_counterfactual_figure(viz_model, train_ds, run_cfg, run_tag)
            except Exception as exc:  # noqa: BLE001
                print(f"save_counterfactual_figure failed: {exc}")

    return model, train_ds, val_rl_ds, val_ds_ds


if __name__ == "__main__":
    trained_model, train_ds, val_rl_ds, val_ds_ds = main()
