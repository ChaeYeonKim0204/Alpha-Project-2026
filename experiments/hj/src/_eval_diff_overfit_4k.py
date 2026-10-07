"""diff_overfit_4k last.ckpt로 inference하여 미래 예측이 풀렸는지 확인.

체크포인트의 saved hparams가 cfg를 복원하므로 model architecture는
런 종료 시점(2026-05-27 14:21) 코드와 일치한다. 단, models_muvo_2D.py가
이후 14:57에 수정되었을 수 있어 load_state_dict는 strict=False로 시도하고
missing/unexpected key 수를 보고한다.

Figure 저장 위치: trainer_muvo_2D._hj_figure_dir() = archive/hj/src/figures/
(원래 figure 폴더 archive/hj/figures/ 와 별도. 실행 후 별도로 이동해도 됨.)
"""
import copy
import sys
from pathlib import Path

# Make sibling .py imports resolve when run as a plain script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

from config_muvo_2D import cfg  # noqa: E402
from trainer_muvo_2D import (  # noqa: E402
    WorldModelTrainer,
    save_reconstruction_figure,
    save_future_frames_per_frame,
)
from train_muvo_2D import build_dataloaders  # noqa: E402


def main():
    ckpt = HERE.parent / "logs" / "diff_overfit_4k" / "checkpoints" / "last.ckpt"
    print(f"[eval] loading checkpoint: {ckpt}  (exists={ckpt.is_file()})")

    # eval에선 augmentation 끄기 (deterministic recon).
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.DATA.AUGMENTATION.ENABLED = False

    # checkpoint에서 hparams로 model 복원. cfg는 saved hparams가 우선.
    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    model = WorldModelTrainer.load_from_checkpoint(
        str(ckpt), map_location=map_loc, strict=False
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Same overfit dataset (8 windows) — sample_idx=0 matches the original
    # training figures.
    bundle = build_dataloaders(eval_cfg, batch_size=2, overfit_samples=8)
    train_ds = bundle[7]
    print(f"[eval] train_ds windows: {len(train_ds)}")

    run_name = "diff_overfit_4k_eval"
    # restored cfg from checkpoint
    run_cfg = model.cfg

    print("[eval] save_reconstruction_figure ...")
    save_reconstruction_figure(model, train_ds, run_cfg, run_name, sample_idx=0)

    print("[eval] save_future_frames_per_frame ...")
    save_future_frames_per_frame(model, train_ds, run_cfg, run_name, sample_idx=0)

    print("[eval] done.")


if __name__ == "__main__":
    main()
