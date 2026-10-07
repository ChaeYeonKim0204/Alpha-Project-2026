#!/usr/bin/env python3
"""MUVO 2D-token RSSM (RSSMTD) checkpoint에서 reconstruction figure를 저장.

이 스크립트는 `*_muvo_2D` 모듈(config/data/trainer)을 사용한다 (#17). 모델·데이터셋을
**checkpoint에 저장된 config(hparams)** 로 재구성해 module-level cfg의 현재 default가
섞이지 않도록 한다 (#19). dataset은 모든 DATA 값을 saved cfg에서 명시 전달 + augmentation
off로 deterministic하게 만든다.

사용:
    python viz_from_ckpt.py --ckpt logs/<run>/version_N/checkpoints/last.ckpt
"""

import argparse
import copy
import os

import torch

from config_muvo_2D import cfg as default_cfg
from data_muvo_2D import _as_arrow_path, MUVODataset
from trainer_muvo_2D import WorldModelTrainer, save_reconstruction_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a MUVO 2D (RSSMTD) checkpoint without training.")
    parser.add_argument("--ckpt", required=True,
                        help="Checkpoint 경로 (예: logs/<run>/version_N/checkpoints/last.ckpt).")
    parser.add_argument("--run-name", default=None,
                        help="출력 파일 prefix. 기본은 saved cfg.LOGGING.RUN_NAME.")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--val-run", default=None,
                        help="시각화할 arrow file/name. 기본은 saved cfg.DATA.VAL_RL_RUN.")
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def _load_saved_cfg(ckpt_path):
    """checkpoint hparams에서 학습 당시 cfg를 복원. 없으면 module-level default로 fallback."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {}) or {}
    saved = hparams.get("cfg", None)
    if saved is None:
        print("checkpoint에 저장된 cfg가 없어 module-level config_muvo_2D.cfg 사용.")
        return copy.deepcopy(default_cfg)
    print("checkpoint hparams의 saved cfg로 모델·데이터셋 재구성.")
    return saved


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    cfg = _load_saved_cfg(args.ckpt)
    run_name = args.run_name or cfg.LOGGING.RUN_NAME
    val_run = args.val_run or cfg.DATA.VAL_RL_RUN

    seq_len = cfg.RECEPTIVE_FIELD + cfg.FUTURE_HORIZON
    stride = cfg.RECEPTIVE_FIELD * cfg.DATA.SAMPLE_EVERY_N
    # dataset을 saved cfg 값으로 명시 생성 (module-level cfg 혼입 방지), augmentation off.
    viz_ds = MUVODataset(
        _as_arrow_path(val_run),
        seq_len=seq_len,
        stride=stride,
        sample_every_n=cfg.DATA.SAMPLE_EVERY_N,
        image_input_size=cfg.DATA.IMAGE_INPUT_SIZE,
        rgb_recon_size=cfg.DATA.RGB_RECON_SIZE,
        lidar_size=cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
        lidar_fov=cfg.DATA.LIDAR_FOV_DEGREES,
        lidar_scale=cfg.DATA.LIDAR_SCALE,
        frame_step=cfg.DATA.FRAME_STEP,
        augment=False,
    )

    # embedding_n_channels는 cfg.MODEL.EMBEDDING_DIM에서 자동 결정 (하드코딩 제거).
    model = WorldModelTrainer.load_from_checkpoint(
        args.ckpt,
        cfg=cfg,
        lr=args.lr if args.lr is not None else cfg.OPTIMIZER.LR,
        map_location="cpu",
    )

    torch.set_float32_matmul_precision("medium")
    save_reconstruction_figure(model, viz_ds, cfg, run_name, sample_idx=args.sample_idx)


if __name__ == "__main__":
    main()
