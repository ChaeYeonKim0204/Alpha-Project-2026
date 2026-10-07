"""
trainer_ssim.py — non-_muvo `trainer.py`를 그대로 두고, SSIM RGB loss만 추가한 sub-trainer.

흐릿한 RGB reconstruction 가설(`_muvo_comparison.md` §7) 검증용. baseline `WorldModelTrainer`를
상속받아 `__init__`과 `compute_loss`만 오버라이드한다.

SSIMLoss는 `muvo/muvo/losses.py:292-348`을 그대로 인라인.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import WorldModelTrainer


class SSIMLoss(nn.Module):
    """upstream `muvo/muvo/losses.py:SSIMLoss` 이식. (B, S, C, H, W) 입력."""

    def __init__(self, channel=1, window_size=11, sigma=1.5, L=1, non_negative=False):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.sigma = sigma
        self.C1 = (0.01 * L) ** 2
        self.C2 = (0.03 * L) ** 2
        self.window = self.create_window()
        self.non_negative = non_negative

    def gaussian(self, window_size, sigma):
        x = torch.arange(window_size)
        gauss = torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(self):
        _1D_window = self.gaussian(self.window_size, self.sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(self.channel, 1, self.window_size, self.window_size).contiguous()

    def _ssim(self, prediction, target):
        window = torch.as_tensor(self.window, dtype=prediction.dtype, device=prediction.device)
        padd = 0
        mu1 = F.conv2d(target,     window, padding=padd, groups=self.channel)
        mu2 = F.conv2d(prediction, window, padding=padd, groups=self.channel)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(target * target,         window, padding=padd, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(prediction * prediction, window, padding=padd, groups=self.channel) - mu2_sq
        sigma12   = F.conv2d(target * prediction,     window, padding=padd, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        ssim_batch = ssim_map.mean([1, 2, 3])
        if self.non_negative:
            ssim_batch = F.relu(ssim_batch)
        return ssim_batch

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, c, h, w)
        target     = target.view(b * s, c, h, w)
        return self._ssim(prediction, target).mean()


class SSIMWorldModelTrainer(WorldModelTrainer):
    """WorldModelTrainer + SSIM term on RGB reconstruction (per downsampling factor)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default 0.06 == upstream effective (rgb_weight=0.1 × ssim_weight=0.6).
        # 0.6은 alpha의 WEIGHT_RGB=1.0과 곱해져 upstream의 10×가 되어 학습 불안정.
        self.weight_ssim = getattr(self.cfg.LOSSES, "WEIGHT_SSIM", 0.06)
        self.ssim_loss   = SSIMLoss(channel=3)

    def compute_loss(self, batch, output, include_kl=True, prefix=""):
        losses = super().compute_loss(batch, output, include_kl=include_kl, prefix=prefix)

        if self.weight_ssim > 0.0 and "rgb_label_1" in batch:
            for factor in [1, 2, 4]:
                key = f"rgb_{factor}"
                if key not in output:
                    continue
                pred   = output[key]
                target = batch["rgb_label_1"]
                if pred.shape[-2:] != target.shape[-2:]:
                    target = self.resize_sequence_tensor(target, pred.shape[-2:], mode="bilinear")
                pred_clamped   = pred.clamp(0.0, 1.0)
                target_clamped = target.clamp(0.0, 1.0)
                ssim_term = 1.0 - self.ssim_loss(pred_clamped, target_clamped)
                discount  = 1.0 / factor
                losses[f"{prefix}ssim_{factor}"] = (
                    self.weight_rgb * discount * self.weight_ssim * ssim_term
                )

        return losses
