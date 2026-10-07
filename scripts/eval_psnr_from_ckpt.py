"""eval_psnr_from_ckpt.py — load a checkpoint and report PSNR on N windows.

Reuses the run's saved cfg from the checkpoint's hyperparameters, so no
hparams.yaml parsing is required.

Usage:
    python scripts/eval_psnr_from_ckpt.py \
        --ckpt logs/<run>/checkpoints/last.ckpt \
        --num-windows 1
"""
import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR / "model_variants"))

from data_muvo_2D import (
    load_arrow_manifest,
    _select_from_manifest,
    MultiArrowStreamDataset,
)
from trainer_muvo_2D import WorldModelTrainer


def build_loader(run_cfg, num_windows):
    manifest = load_arrow_manifest()
    seq_len = run_cfg.RECEPTIVE_FIELD + run_cfg.FUTURE_HORIZON
    stride = run_cfg.RECEPTIVE_FIELD * run_cfg.DATA.SAMPLE_EVERY_N
    train_files = _select_from_manifest(
        manifest, "train",
        preferred=run_cfg.DATA.TRAIN_RUN,
        use_all=run_cfg.DATA.USE_ALL_TRAIN_RUNS,
    )
    ds_full = MultiArrowStreamDataset(
        train_files,
        seq_len=seq_len,
        stride=stride,
        num_streams=1,
        sample_every_n=run_cfg.DATA.SAMPLE_EVERY_N,
        image_input_size=run_cfg.DATA.IMAGE_INPUT_SIZE,
        rgb_recon_size=run_cfg.DATA.RGB_RECON_SIZE,
        lidar_size=run_cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
        frame_step=run_cfg.DATA.FRAME_STEP,
    )
    n = min(num_windows, len(ds_full))
    ds = Subset(ds_full, list(range(n)))
    print(f"dataset: {len(ds)} window(s) from {run_cfg.DATA.TRAIN_RUN}")
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-windows", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_cfg = ckpt["hyper_parameters"]["cfg"]
    saved_lr = ckpt["hyper_parameters"].get("lr", 1e-4)
    run_cfg = copy.deepcopy(saved_cfg)

    loader = build_loader(run_cfg, args.num_windows)

    model = WorldModelTrainer.load_from_checkpoint(
        args.ckpt,
        cfg=run_cfg,
        lr=saved_lr,
        map_location="cpu",
    )
    model.to(args.device).eval()

    per_window_psnr = []
    per_window_future_psnr = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch_pp = model.prepare_custom_batch(batch)

            rf, fh = model.rf, model.fh
            recon_batch = model._slice_batch(batch_pp, 0, rf)
            output, _ = model.model.forward(recon_batch)
            recon_metrics = model.compute_eval_metrics(recon_batch, output)
            psnr_now = recon_metrics.get("rgb_psnr")
            psnr_fut = None  # future PSNR skipped (imagine() has a 4D-tensor bug
                              # never exercised on this run, see notes)
            if psnr_now is not None:
                per_window_psnr.append(psnr_now.item())
            if psnr_fut is not None:
                per_window_future_psnr.append(psnr_fut.item())

            parts = [f"window {i}:"]
            if psnr_now is not None:
                parts.append(f"recon PSNR={psnr_now.item():.2f} dB")
            if psnr_fut is not None:
                parts.append(f"future PSNR={psnr_fut.item():.2f} dB")
            print("  ".join(parts))

    if per_window_psnr:
        mean = sum(per_window_psnr) / len(per_window_psnr)
        print(f"\nmean recon PSNR over {len(per_window_psnr)} window(s): {mean:.2f} dB")
    if per_window_future_psnr:
        mean = sum(per_window_future_psnr) / len(per_window_future_psnr)
        print(f"mean future PSNR over {len(per_window_future_psnr)} window(s): {mean:.2f} dB")


if __name__ == "__main__":
    main()
