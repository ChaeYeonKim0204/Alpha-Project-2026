"""
train_ssim.py — train.py와 동일한 entrypoint, trainer만 `SSIMWorldModelTrainer`로 교체.

train.py를 그대로 import한 뒤, 모듈 전역 `WorldModelTrainer`를 SSIM-augmented
subclass로 monkey-patch한다. train.py 본문(337줄)은 복제하지 않는다.

Usage:
    python train_ssim.py --one-window-overfit --run-name ssim_one_window_overfit
"""

import train
from trainer_ssim import SSIMWorldModelTrainer

train.WorldModelTrainer = SSIMWorldModelTrainer

if __name__ == "__main__":
    trained_model, train_ds, val_rl_ds, val_ds_ds = train.main()
