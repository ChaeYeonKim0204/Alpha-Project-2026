"""trainer_muvo_2D.py — 2D-token latent-diffusion 베이스 (image-only).

- action loss 추가 (`F.l1_loss(output["action_pred"], batch["action"]) * WEIGHT_ACTION`)
- reconstruction figure: RF+FH frame 전체 column, RGB GT/Pred 두 row
- KL free-bits on/off cfg gate (`KL_FREE_BITS_ENABLED`, deprecated path)
- ClearML artifact upload optional (`Task.current_task()`)
- LiDAR encoder/decoder/loss/metric은 제거 (RGB-only).
"""
import csv
import os
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from models_muvo_2D import Model


# 모든 figure는 hj/figures/ 폴더에 `hj_` 접두사로 저장.
# __file__은 hj/src/trainer_muvo_2D.py → parent.parent로 hj/까지 올라간다.
HJ_DIR = Path(__file__).resolve().parent.parent
HJ_FIGURE_DIR = HJ_DIR / "figures"


def _hj_figure_dir():
    HJ_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return HJ_FIGURE_DIR


# ─────────────────────────────────────────────
# SSIM loss (alpha _muvo와 동일, default off)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Figure / TensorBoard utilities
# ─────────────────────────────────────────────

def _project_root_from_cfg(run_cfg):
    log_path = getattr(getattr(run_cfg, "LOGGING", object()), "EXPERIMENT_LOG_PATH", None)
    if log_path:
        return Path(log_path).expanduser().resolve().parent.parent
    return Path.cwd()


def save_reconstruction_figure(model, dataset, run_cfg, run_name, sample_idx=0):
    """RF+FH frame 전체를 column으로, RGB GT/Pred 두 row로 출력 (image-only).

    `cfg.LOGGING.RECON_FIG_ALL_FRAMES=True`: 모든 frame 표시 (default).
    `False`로 두면 4-column 2-row 단순 figure (마지막 obs + 첫 future).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    sample_idx = min(int(sample_idx), len(dataset) - 1)
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
    fh = run_cfg.FUTURE_HORIZON
    # diffusion이 꺼져 있으면 future_output=None → 미래 칼럼은 생략하고 obs(reconstruction)만 그린다.
    has_future = future_output is not None
    if not has_future:
        fh = 0
    all_frames = bool(getattr(run_cfg.LOGGING, "RECON_FIG_ALL_FRAMES", True))

    def rgb(tensor):
        return tensor.detach().float().cpu().permute(1, 2, 0).clamp(0, 1)

    if all_frames:
        # Columns: RF frames (gt vs posterior recon) + FH frames (gt vs imagine pred).
        n_cols = rf + fh
        n_rows = 2  # gt_rgb, pred_rgb

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows))
        axes = np.atleast_2d(axes)

        for col in range(n_cols):
            if col < rf:
                t = col
                gt_rgb = batch["image_raw"][0, t]
                pred_rgb_t = posterior_output["rgb_1"][0, t]
                col_label = f"obs t={t}"
            else:
                fut_t = col - rf
                t_batch = rf + fut_t
                gt_rgb = batch["image_raw"][0, t_batch]
                pred_rgb_t = future_output["rgb_1"][0, fut_t]
                col_label = f"fut t={fut_t}"

            axes[0, col].imshow(rgb(gt_rgb))
            axes[0, col].set_title(f"{col_label}\nGT RGB")
            axes[1, col].imshow(rgb(pred_rgb_t))
            axes[1, col].set_title("Pred RGB")

        for ax in axes.ravel():
            ax.axis("off")
    else:
        obs_t = rf - 1
        fut_t = 0
        fut_batch_t = rf
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
        axes = np.asarray(axes)
        axes[0].imshow(rgb(batch["image_raw"][0, obs_t])); axes[0].set_title("Observed RGB")
        axes[1].imshow(rgb(posterior_output["rgb_1"][0, obs_t])); axes[1].set_title("Posterior RGB")
        axes[2].imshow(rgb(batch["image_raw"][0, fut_batch_t])); axes[2].set_title("Future target RGB")
        axes[3].imshow(rgb(future_output["rgb_1"][0, fut_t])); axes[3].set_title("Prior future RGB")
        for ax in axes.ravel():
            ax.axis("off")

    path = _hj_figure_dir() / f"hj_{run_name}_reconstruction_s{sample_idx}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved reconstruction figure: {path}")
    return path


def save_future_frames_per_frame(model, dataset, run_cfg, run_name, sample_idx=0):
    """미래 FH 프레임을 **각각 독립 디코딩**해 시점별로 별도 figure(PNG)로 저장.

    joint diffusion 생성(z_fut: (B,FH,C,N))은 그대로 두고, 각 시점 t의 latent를
    독립적으로 디코딩(decode(z_fut[:, t]))해 GT vs Pred(RGB)를 1장씩 출력한다.
    파일: hj/figures/hj_<run_name>_fut_t{t}_s{sample_idx}.png  (t = 0 .. FH-1)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    sample_idx = min(int(sample_idx), len(dataset) - 1)
    sample = dataset[sample_idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
    batch = model.prepare_custom_batch(batch)

    rf = run_cfg.RECEPTIVE_FIELD
    fh = run_cfg.FUTURE_HORIZON
    ddim = getattr(getattr(run_cfg, "PREDICTION", None), "DDIM_STEPS", 50)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            per_frame = model.model.decode_future_per_frame(batch, guidance_scale=1.0, ddim_steps=ddim)

    def rgb(t):
        return t.detach().float().cpu().permute(1, 2, 0).clamp(0, 1)

    out_dir = _hj_figure_dir()
    paths = []
    for t in range(fh):
        out = per_frame[t]
        t_batch = rf + t
        gt_rgb     = batch["image_raw"][0, t_batch]
        pred_rgb   = out["rgb_1"][0, 0]

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].imshow(rgb(gt_rgb));   axes[0].set_title(f"GT RGB (fut t={t})")
        axes[1].imshow(rgb(pred_rgb)); axes[1].set_title("Pred RGB")
        for ax in axes.ravel():
            ax.axis("off")

        path = out_dir / f"hj_{run_name}_fut_t{t}_s{sample_idx}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        print(f"Saved per-frame future figure: {path}")
    return paths


# ─────────────────────────────────────────────
# Counterfactual ("어떤 행동을 하면 어떤 미래가 되는가") figure
# ─────────────────────────────────────────────

def build_probe_actions(batch_size, future_horizon, device, scale=0.5):
    """probe action 세트: 직진/좌회전/우회전/급제동. action=[throttle, steer] (Tanh 범위 [-1,1]).
    데이터 분포 분위수 기반이 이상적이나 1차는 고정 스케일 (OOD 회피용으로 보수적 0.5)."""
    def mk(throttle, steer):
        a = torch.zeros(batch_size, future_horizon, 2, device=device)
        a[..., 0] = throttle
        a[..., 1] = steer
        return a
    return {
        "straight": mk(scale, 0.0),
        "left":     mk(scale, -scale),
        "right":    mk(scale, +scale),
        "brake":    mk(-scale, 0.0),
    }


def save_counterfactual_figure(model, dataset, run_cfg, run_name, sample_idx=0,
                               guidance_scale=2.0):
    """같은 과거에서 행동만 바꿔(직진/좌/우/급제동) 미래 RGB를 그리드로 — "어떤 행동→어떤 미래".

    model: WorldModelTrainer (pl_module). 동일 noise seed 공유라 행 간 차이가 순수 action 효과.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    sample_idx = min(int(sample_idx), len(dataset) - 1)
    sample = dataset[sample_idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
    batch = model.prepare_custom_batch(batch)

    rf = run_cfg.RECEPTIVE_FIELD
    fh = run_cfg.FUTURE_HORIZON
    ddim = getattr(getattr(run_cfg, "PREDICTION", None), "DDIM_STEPS", 50)

    probes = build_probe_actions(1, fh, device)
    names = list(probes.keys())
    action_list = [probes[n] for n in names]

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            cf_outs = model.model.counterfactual_futures(
                batch, action_list, guidance_scale=guidance_scale, ddim_steps=ddim, seed=0,
            )

    def rgb(t):
        return t.detach().float().cpu().permute(1, 2, 0).clamp(0, 1)

    n_rows = 1 + len(names)  # GT future + actions
    n_cols = max(1, fh)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False)

    for c in range(n_cols):
        axes[0, c].imshow(rgb(batch["image_raw"][0, rf + c]))
        axes[0, c].set_title(f"GT future t={c}")
    for r, name in enumerate(names, start=1):
        for c in range(n_cols):
            axes[r, c].imshow(rgb(cf_outs[r - 1]["rgb_1"][0, c]))
            axes[r, c].set_title(f"{name} t={c}")
    for ax in axes.ravel():
        ax.axis("off")

    path = _hj_figure_dir() / f"hj_{run_name}_counterfactual_s{sample_idx}.png"
    fig.suptitle(f"Counterfactual futures (w={guidance_scale}): action -> future", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved counterfactual figure: {path}")
    return path


def _latest_event_file(log_root):
    import glob
    event_files = glob.glob(os.path.join(log_root, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"TensorBoard event file not found under: {log_root}")
    return max(event_files, key=os.path.getmtime)


def _load_tensorboard_scalars(event_file=None, log_root="./logs"):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise ImportError("TensorBoard is required to save loss metric plots.") from exc

    event_file = event_file or _latest_event_file(log_root)
    accumulator = EventAccumulator(event_file)
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        scalars[tag] = {
            "step": [event.step for event in events],
            "value": [event.value for event in events],
        }
    print(f"Loaded TensorBoard scalars from: {event_file}")
    return scalars


def _plot_scalar_tags(ax, scalars, tags, title, ylabel):
    found = False
    for tag in tags:
        if tag in scalars:
            ax.plot(scalars[tag]["step"], scalars[tag]["value"], label=tag)
            found = True
    ax.set_title(title); ax.set_xlabel("step"); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if found:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No matching scalar", ha="center", va="center", transform=ax.transAxes)


def save_loss_metric_figure(cfg, log_root, output_dir, filename=None, event_file=None):
    """Save loss and evaluation metric plots from TensorBoard scalar logs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scalars = _load_tensorboard_scalars(event_file=event_file, log_root=log_root)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    _plot_scalar_tags(axes[0, 0], scalars,
        ["train_loss_step", "train_loss_epoch", "val/RL_loss", "val/DS_loss"],
        "Total loss", "loss")
    _plot_scalar_tags(axes[0, 1], scalars,
        ["val/RL_rgb_psnr", "val/DS_rgb_psnr", "val/RL_future_rgb_psnr", "val/DS_future_rgb_psnr"],
        "Camera PSNR", "PSNR (dB)")
    _plot_scalar_tags(axes[1, 0], scalars,
        ["train_action", "val/RL_action", "val/DS_action"],
        "Action loss (L1)", "loss")
    _plot_scalar_tags(axes[1, 1], scalars,
        ["train/diffusion", "val/RL_diffusion", "val/DS_diffusion"],
        "Diffusion loss (v-pred MSE)", "loss")

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{cfg.LOGGING.RUN_NAME}_loss_metrics.png"
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved loss metric figure to: {path}")
    return path, scalars


# ─────────────────────────────────────────────
# WorldModelTrainer
# ─────────────────────────────────────────────

class WorldModelTrainer(pl.LightningModule):

    def __init__(self, cfg=None, hparams=None, lr=None, embedding_n_channels=None):
        super().__init__()
        self.save_hyperparameters(ignore=["hparams"])

        self.cfg = cfg or hparams
        self.rf  = self.cfg.RECEPTIVE_FIELD
        self.fh  = self.cfg.FUTURE_HORIZON

        if embedding_n_channels is None:
            embedding_n_channels = getattr(self.cfg.MODEL, "EMBEDDING_DIM", 256)
        self.model = Model(self.cfg, embedding_n_channels=embedding_n_channels)
        self.load_pretrained_weights()

        self.lr                    = lr if lr is not None else getattr(self.cfg.OPTIMIZER, "LR", 1e-4)
        self.weight_probabilistic  = getattr(self.cfg.LOSSES, "WEIGHT_PROBABILISTIC", 1e-3)
        self.kl_balancing_alpha    = getattr(self.cfg.LOSSES, "KL_BALANCING_ALPHA", 0.75)
        self.weight_rgb            = getattr(self.cfg.LOSSES, "WEIGHT_RGB", 0.1)
        #self.weight_future         = getattr(self.cfg.LOSSES, "WEIGHT_FUTURE", 1.0)
        self.weight_ssim           = getattr(self.cfg.LOSSES, "WEIGHT_SSIM", 0.0)
        self.ssim_loss             = SSIMLoss(channel=3) if self.weight_ssim > 0.0 else None
        # KL free-bits cfg gate (v3, Q8). default OFF = upstream과 일치.
        self.kl_free_bits_enabled  = bool(getattr(self.cfg.LOSSES, "KL_FREE_BITS_ENABLED", False))
        self.kl_free_bits_value    = float(getattr(self.cfg.LOSSES, "KL_FREE_BITS", 0.0))
        # Action loss (v2, Q9).
        self.weight_action         = getattr(self.cfg.LOSSES, "WEIGHT_ACTION", 1.0)
        # Diffusion loss (KL probabilistic 대체) + optional future-decode.
        self.weight_diffusion      = getattr(self.cfg.LOSSES, "WEIGHT_DIFFUSION", 1.0)
        self.weight_future_decode  = getattr(self.cfg.LOSSES, "WEIGHT_FUTURE_DECODE", 0.0)
        self.ddim_steps            = getattr(getattr(self.cfg, "PREDICTION", None), "DDIM_STEPS", 50)
        # TRAIN_DIFFUSION=False면 미래 diffusion(타깃 계산 + DDIM 생성)을 전부 끄고
        # reconstruction(autoencoder)만 학습/시각화한다 (reconstruction-only overfit용).
        self.train_diffusion       = bool(getattr(self.cfg.MODEL.DIFFUSION, "TRAIN_DIFFUSION", True))

        self.val_dataset_names  = ["RL", "DS"]
        self.test_dataset_names = ["RL", "DS"]

    # ─── Checkpoint ──────────────────────────────────────────────────────────

    def load_pretrained_weights(self):
        if not hasattr(self.cfg, "PRETRAINED"):
            return
        pretrained_path = getattr(self.cfg.PRETRAINED, "PATH", None)
        if not pretrained_path or not os.path.isfile(pretrained_path):
            return
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        cleaned = {(k[len("model."):] if k.startswith("model.") else k): v
                   for k, v in checkpoint.items()}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(f"Loaded: {pretrained_path}  missing={len(missing)}  unexpected={len(unexpected)}")

    # ─── Batch helpers ───────────────────────────────────────────────────────

    def prepare_custom_batch(self, batch):
        if "action" not in batch and "throttle_brake" in batch and "steering" in batch:
            batch["action"] = torch.cat([batch["throttle_brake"], batch["steering"]], dim=-1)
        if "action" in batch:
            if "throttle_brake" not in batch:
                batch["throttle_brake"] = batch["action"][..., 0:1]
            if "steering" not in batch:
                batch["steering"]       = batch["action"][..., 1:2]
        if "image_raw" in batch and "rgb_label_1" not in batch:
            batch["rgb_label_1"] = batch["image_raw"].float()
        elif "image" in batch and "rgb_label_1" not in batch:
            raise KeyError(
                "rgb_label_1에는 raw reconstruction target이 필요합니다. "
                "현재 batch에는 정규화된 'image'만 있고 'image_raw'가 없습니다."
            )
        for key in ["image", "image_raw", "action", "speed", "rgb_label_1"]:
            if key in batch and torch.is_tensor(batch[key]):
                batch[key] = batch[key].float()
        return batch

    @staticmethod
    def _slice_batch(batch, start, end):
        sliced = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim >= 2:
                sliced[key] = value[:, start:end]
            else:
                sliced[key] = value
        return sliced

    def forward(self, batch):
        batch = self.prepare_custom_batch(batch)
        return self.model.forward(batch)

    # ─── Evaluation metrics ──────────────────────────────────────────────────

    @staticmethod
    def psnr(prediction, target, max_pixel_value=1.0, eps=1e-8):
        prediction = prediction.float().clamp(0.0, max_pixel_value)
        target     = target.float().clamp(0.0, max_pixel_value)
        mse        = torch.mean((prediction - target) ** 2, dim=(2, 3, 4)).clamp_min(eps)
        return 20 * torch.log10(torch.as_tensor(max_pixel_value, device=prediction.device) / torch.sqrt(mse))

    def compute_eval_metrics(self, batch, output, prefix=""):
        metrics = {}
        if "rgb_2" in output and "rgb_label_1" in batch:
            pred   = output["rgb_2"]
            target = batch["rgb_label_1"]
            if pred.shape[-2:] != target.shape[-2:]:
                target = self.resize_sequence_tensor(target, pred.shape[-2:], mode="bilinear")
            metrics[f"{prefix}rgb_psnr"] = self.psnr(pred, target).mean()
        return metrics

    # ─── Losses ──────────────────────────────────────────────────────────────

    @staticmethod
    def gaussian_kl_loss(prior, posterior, eps=1e-6, free_bits=0.0):
        mu_p  = prior["mu"]
        sig_p = prior["sigma"].clamp_min(eps)
        mu_q  = posterior["mu"]
        sig_q = posterior["sigma"].clamp_min(eps)
        kl    = (torch.log(sig_p / sig_q)
                 + (sig_q.pow(2) + (mu_q - mu_p).pow(2)) / (2.0 * sig_p.pow(2))
                 - 0.5)
        # kl shape (B, S, C, N) — sum on the channel dim (last dim of state vector).
        # For RSSMTD output (B, S, C, N), we sum over C; then mean over remaining dims.
        kl_sum = kl.sum(dim=-2)  # (B, S, N) — per-token sum over channel
        if free_bits > 0.0:
            kl_sum = kl_sum.clamp_min(free_bits)
        return kl_sum.mean()

    def balanced_kl_loss(self, prior, posterior):
        alpha = self.kl_balancing_alpha
        free_bits = self.kl_free_bits_value if self.kl_free_bits_enabled else 0.0
        prior_loss     = self.gaussian_kl_loss(prior,
                             {k: v.detach() for k, v in posterior.items()}, free_bits=free_bits)
        posterior_loss = self.gaussian_kl_loss({k: v.detach() for k, v in prior.items()},
                             posterior, free_bits=free_bits)
        return alpha * prior_loss + (1.0 - alpha) * posterior_loss

    @staticmethod
    def resize_sequence_tensor(x, target_hw, mode="bilinear"):
        b, s, c, h, w = x.shape
        x = x.reshape(b * s, c, h, w)
        x = F.interpolate(x, size=target_hw, mode=mode,
                          **({} if mode == "nearest" else {"align_corners": False}))
        return x.reshape(b, s, c, *target_hw)

    def compute_loss(self, batch, output, include_kl=True, prefix=""):
        losses = {}

        # KL은 diffusion 경로에선 output에 prior/posterior가 없어 자동 스킵됨 (호환 유지).
        if include_kl and "prior" in output and "posterior" in output:
            losses[f"{prefix}probabilistic"] = (
                self.weight_probabilistic * self.balanced_kl_loss(output["prior"], output["posterior"])
            )

        # Diffusion loss (KL 대체): "과거+행동→미래" latent를 맞히는 MSE(v).
        if "v_pred" in output and "v_target" in output:
            losses[f"{prefix}diffusion"] = (
                self.weight_diffusion * F.mse_loss(output["v_pred"], output["v_target"])
            )

        if "rgb_label_1" in batch:
            for factor in [1, 2, 4]:
                key = f'rgb_{factor}'
                if key in output:
                    pred     = output[key]
                    target   = batch["rgb_label_1"]
                    discount = 1 / factor
                    if pred.shape[-2:] != target.shape[-2:]:
                        target = self.resize_sequence_tensor(target, pred.shape[-2:], mode="bilinear")
                    losses[f"{prefix}{key}"] = self.weight_rgb * discount * F.l1_loss(pred, target)

                    if self.ssim_loss is not None:
                        pred_clamped   = pred.clamp(0.0, 1.0)
                        target_clamped = target.clamp(0.0, 1.0)
                        ssim_term      = 1.0 - self.ssim_loss(pred_clamped, target_clamped)
                        losses[f"{prefix}ssim_{factor}"] = (
                            self.weight_rgb * discount * self.weight_ssim * ssim_term
                        )

        # Action loss (v2, Q9).
        if "action_pred" in output and "action" in batch and self.weight_action > 0.0:
            action_target = batch["action"][:, :output["action_pred"].shape[1]].float()
            losses[f"{prefix}action"] = self.weight_action * F.l1_loss(output["action_pred"], action_target)

        if not losses:
            raise RuntimeError("No valid loss computed.")
        return losses

    def loss_reducing(self, losses):
        return sum(losses.values())

    # ─── Core forward: observe + imagine ─────────────────────────────────────

    def _observe_full(self, batch):
        output, state_dict = self.model.forward(batch)
        losses = self.compute_loss(batch, output, include_kl=True)
        return losses, output, state_dict

    @staticmethod
    def _slice_output(output, start, end):
        """frame 차원(dim=1)을 slice. diffusion 텐서(v_pred 등)는 제외."""
        skip = {"v_pred", "v_target", "z0_hat"}
        sliced = {}
        for key, value in output.items():
            if key in skip:
                continue
            if torch.is_tensor(value) and value.ndim >= 2:
                sliced[key] = value[:, start:end]
            else:
                sliced[key] = value
        return sliced

    def _observe_and_imagine(self, batch, h_init=None, s_init=None,
                              continuation=False, Prev_action=None, n_samples=1):
        # 전체 시퀀스 forward: 관측 recon(전 프레임) + diffusion 타깃(v_pred/v_target).
        output_full, state_dict = self.model.forward(batch)
        losses = self.compute_loss(batch, output_full, include_kl=True)

        # 관측 프레임(0..RF-1) recon만 metric/figure에 사용.
        output_obs = self._slice_output(output_full, 0, self.rf)

        output_imagine = None
        losses_imagine = {}
        if self.fh > 0 and self.train_diffusion:
            # GT action·w=1로 미래 생성(diffusion DDIM) → metric/figure.
            futs = self.model.generate_futures(
                batch, num_samples=1, guidance_scale=1.0, ddim_steps=self.ddim_steps,
            )
            output_imagine = futs[0]
            batch_fh = self._slice_batch(batch, self.rf, self.rf + self.fh)
            losses_imagine = self.compute_loss(batch_fh, output_imagine,
                                               include_kl=False, prefix="future_")
            losses.update(losses_imagine)

        return losses, output_obs, state_dict, losses_imagine, output_imagine


    # ─── Training ────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        batch = self.prepare_custom_batch(batch)
        losses, output, state_dict = self._observe_full(batch)

        self.log_dict(
          {f"train/{k}": v for k, v in losses.items()},
          on_step=True, on_epoch=False, prog_bar=False)
        total_loss = self.loss_reducing(losses)
        # train_loss_step / train_loss_epoch 로깅.
        # overfit(검증 없음) 모드의 ModelCheckpoint(monitor="train_loss_epoch")와
        # save_loss_metric_figure의 train loss curve가 참조하는 키 — 없으면 monitor 경고 발생.
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    # ─── Validation / Test ───────────────────────────────────────────────────

    def _dataset_name(self, names, dataloader_idx):
        return names[dataloader_idx] if dataloader_idx < len(names) else f"loader{dataloader_idx}"

    def _log_eval_outputs(self, mode, dataset_name, batch, losses, output, output_imagine):
        total_loss = self.loss_reducing(losses)
        log_prefix = f"{mode}/{dataset_name}"
        self.log(f"{log_prefix}_loss", total_loss, prog_bar=(mode == "val"),
                 on_step=False, on_epoch=True, add_dataloader_idx=False)
        for name, value in losses.items():
            self.log(f"{log_prefix}_{name}", value, prog_bar=False,
                     on_step=False, on_epoch=True, add_dataloader_idx=False)

        obs_metrics = self.compute_eval_metrics(self._slice_batch(batch, 0, self.rf), output)
        for name, value in obs_metrics.items():
            self.log(f"{log_prefix}_{name}", value, prog_bar=False,
                     on_step=False, on_epoch=True, add_dataloader_idx=False)

        if output_imagine is not None and self.fh > 0:
            future_batch   = self._slice_batch(batch, self.rf, self.rf + self.fh)
            future_metrics = self.compute_eval_metrics(future_batch, output_imagine, prefix="future_")
            for name, value in future_metrics.items():
                self.log(f"{log_prefix}_{name}", value, prog_bar=False,
                         on_step=False, on_epoch=True, add_dataloader_idx=False)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self.prepare_custom_batch(batch)
        n_samples = getattr(getattr(self.cfg, "PREDICTION", None), "N_SAMPLES", 1)
        losses, output, state_dict, losses_imagine, output_imagine = self._observe_and_imagine(batch, n_samples=n_samples)
        dataset_name = "RL" if dataloader_idx == 0 else "DS"
        total_loss = self._log_eval_outputs("val", dataset_name, batch, losses, output, output_imagine)
        return total_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self.prepare_custom_batch(batch)
        n_samples = getattr(getattr(self.cfg, "PREDICTION", None), "N_SAMPLES", 1)
        losses, output, state_dict, losses_imagine, output_imagine = self._observe_and_imagine(batch, n_samples=n_samples)
        dataset_name = self._dataset_name(self.test_dataset_names, dataloader_idx)
        total_loss   = self._log_eval_outputs("test", dataset_name, batch, losses, output, output_imagine)

        # ClearML artifact dump (optional).
        try:
            from clearml import Task
            task = Task.current_task()
            if task is not None and batch_idx < 4:  # 처음 4 batch만 dump
                pred_dump = {
                    "rgb_pred":   output.get("rgb_1", torch.zeros(0)).detach().float().cpu().numpy(),
                    "action_pred": output.get("action_pred", torch.zeros(0)).detach().float().cpu().numpy(),
                }
                task.upload_artifact(f"pred_{dataset_name}_batch_{batch_idx}", pred_dump)
        except Exception as exc:  # noqa: BLE001
            # ClearML 미설치/비활성 시 silent fail
            pass

        return {f"test_{dataset_name}_loss": total_loss}

    # ─── Optimizer ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # BN/LayerNorm scale/shift and biases (1D params) should not be weight-decayed.
        # Mirrors muvo_2d/muvo/trainer.py::add_weight_decay.
        def add_weight_decay(module, weight_decay, skip_list=()):
            no_decay, decay = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or any(s in name for s in skip_list):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [{"params": no_decay, "weight_decay": 0.0},
                    {"params": decay,    "weight_decay": weight_decay}]

        wd = getattr(self.cfg.OPTIMIZER, "WEIGHT_DECAY", 0.0)
        parameters = add_weight_decay(self, wd, skip_list=("relative_position_bias_table",))
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=0.01)

        scheduler_name = getattr(self.cfg.SCHEDULER, "NAME", "none")
        if scheduler_name == "OneCycleLR":
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
            if total_steps <= 0:
                total_steps = int(getattr(self.cfg, "STEPS", getattr(self.cfg, "EPOCHS", 1)))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr, total_steps=total_steps,
                pct_start=getattr(self.cfg.SCHEDULER, "PCT_START", 0.3),
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


# ─────────────────────────────────────────────
# Periodic reconstruction figure callback
# ─────────────────────────────────────────────

class PeriodicReconstructionCallback(pl.Callback):
    """**마지막(최종) 학습 에폭에서만** `save_reconstruction_figure`를 1회 호출해 figure를 dump.

    overfit 검증용. dataset과 sample_idx를 미리 받아두고, 학습이 끝나는 마지막 에폭의
    end에서 그 한 sample을 decode → hj/figures/<run_name>_e<epoch>_reconstruction_s*.png.

    (이전엔 every_n_epochs마다 중간 figure를 모두 dump했으나, 마지막 에폭만 출력하도록 변경.)
    save 전후로 model의 train/eval 모드를 복원해 학습 흐름이 깨지지 않도록 함.
    """

    def __init__(self, dataset, run_cfg, run_name, every_n_epochs=100, sample_idx=0):
        super().__init__()
        self.dataset = dataset
        self.run_cfg = run_cfg
        self.run_name = run_name
        # every_n_epochs는 콜백 활성 토글로만 유지 — frequency는 미사용(마지막 에폭만 저장).
        self.every_n_epochs = int(every_n_epochs)
        self.sample_idx = int(sample_idx)

    def _is_last_epoch(self, trainer):
        """max_epochs 도달 또는 trainer.should_stop(예: max_steps) 시 마지막 에폭으로 판단."""
        max_epochs = trainer.max_epochs or 0
        epoch_1_indexed = trainer.current_epoch + 1
        reached_max = max_epochs > 0 and epoch_1_indexed >= max_epochs
        return reached_max or bool(getattr(trainer, "should_stop", False))

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not self._is_last_epoch(trainer):
            return
        epoch_1_indexed = trainer.current_epoch + 1
        was_training = pl_module.training
        run_name = f"{self.run_name}_e{epoch_1_indexed:04d}"
        try:
            save_reconstruction_figure(
                pl_module, self.dataset, self.run_cfg,
                run_name=run_name, sample_idx=self.sample_idx,
            )
            if getattr(self.run_cfg.LOGGING, "RECON_FIG_PER_FUTURE_FRAME", False):
                save_future_frames_per_frame(
                    pl_module, self.dataset, self.run_cfg,
                    run_name=run_name, sample_idx=self.sample_idx,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"PeriodicReconstructionCallback (epoch {epoch_1_indexed}) failed: {exc}")
        finally:
            pl_module.train(was_training)


# ─────────────────────────────────────────────
# Experiment CSV Logger callback
# ─────────────────────────────────────────────

class ExperimentCSVLogger(pl.Callback):
    """학습 종료 후 experiment_log.csv에 실험 요약 한 줄을 자동 추가한다."""

    FIELDNAMES = [
        "date", "run_name", "base_file", "train_files", "val_files", "sample_hz",
        "rgb_recon_size", "h_dim", "z_dim", "lr",
        "max_steps", "scheduler", "batch_size", "change_summary", "final_val_metrics", "notes", "series",
    ]

    def __init__(self, cfg, train_files, val_files, batch_size, change_summary=""):
        super().__init__()
        self.cfg            = cfg
        self.train_files    = train_files
        self.val_files      = val_files
        self.batch_size     = batch_size
        self.change_summary = change_summary

    @staticmethod
    def _size_to_str(size):
        return "x".join(str(v) for v in tuple(size))

    @staticmethod
    def _files_to_str(files):
        return ";".join(os.path.basename(p) for p in files)

    @staticmethod
    def _metric_to_float(value):
        if torch.is_tensor(value):
            value = value.detach().float().cpu()
            return float(value.item()) if value.numel() == 1 else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _collect_final_val_metrics(self, trainer):
        preferred = [
            "val/RL_loss", "val/DS_loss",
            "val/RL_rgb_psnr", "val/DS_rgb_psnr",
            "val/RL_action", "val/DS_action",
        ]
        parts = []
        for key in preferred:
            if key in trainer.callback_metrics:
                value = self._metric_to_float(trainer.callback_metrics[key])
                if value is not None:
                    parts.append(f"{key}={value:.6g}")
        return "; ".join(parts)

    def on_fit_end(self, trainer, pl_module):
        log_path   = getattr(self.cfg.LOGGING, "EXPERIMENT_LOG_PATH", "experiment_log.csv")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

        effective_hz = getattr(self.cfg.DATA, "EFFECTIVE_HZ", None)
        if effective_hz is None:
            effective_hz = 4 / max(1, getattr(self.cfg.DATA, "SAMPLE_EVERY_N", 1))

        row = {
            "date":             date.today().isoformat(),
            "run_name":         getattr(self.cfg.LOGGING, "RUN_NAME", "muvo_2D"),
            "base_file":        getattr(self.cfg.LOGGING, "BASE_FILE", "train_muvo_2D.py"),
            "train_files":      self._files_to_str(self.train_files),
            "val_files":        self._files_to_str(self.val_files),
            "sample_hz":        effective_hz,
            "rgb_recon_size":   self._size_to_str(self.cfg.DATA.RGB_RECON_SIZE),
            "h_dim":            self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
            "z_dim":            self.cfg.MODEL.TRANSITION.STATE_DIM,
            "lr":               pl_module.lr,
            "max_steps":        getattr(trainer, "estimated_stepping_batches", getattr(self.cfg, "STEPS", "")),
            "scheduler":        getattr(self.cfg.SCHEDULER, "NAME", "none"),
            "batch_size":       self.batch_size,
            "change_summary":   self.change_summary,
            "final_val_metrics": self._collect_final_val_metrics(trainer),
            "notes":            (f"max_epochs={getattr(self.cfg, 'EPOCHS', '')}; "
                                 f"로그경로={trainer.logger.log_dir}"
                                 if trainer.logger is not None else
                                 f"max_epochs={getattr(self.cfg, 'EPOCHS', '')}"),
            "series":           getattr(self.cfg.LOGGING, "SERIES", ""),
        }

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended experiment summary to: {log_path}")
