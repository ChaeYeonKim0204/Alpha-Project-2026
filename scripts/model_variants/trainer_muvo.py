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

from models_muvo import Model


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


def _project_root_from_cfg(run_cfg):
    log_path = getattr(getattr(run_cfg, "LOGGING", object()), "EXPERIMENT_LOG_PATH", None)
    if log_path:
        return Path(log_path).expanduser().resolve().parent.parent
    return Path.cwd()


def save_reconstruction_figure(model, dataset, run_cfg, run_name, sample_idx=0):
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
    obs_t = rf - 1
    fut_t = 0
    fut_batch_t = rf

    def rgb(tensor):
        return tensor.detach().float().cpu().permute(1, 2, 0).clamp(0, 1)

    lidar_scale = float(getattr(run_cfg.DATA, "LIDAR_SCALE", 1.0))
    lidar_vmax = 2.0 * lidar_scale

    def depth(tensor):
        return tensor.detach().float().cpu() * lidar_scale

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = np.asarray(axes)

    axes[0, 0].imshow(rgb(batch["image_raw"][0, obs_t]))
    axes[0, 0].set_title("Observed RGB")
    axes[0, 1].imshow(rgb(posterior_output["rgb_1"][0, obs_t]))
    axes[0, 1].set_title("Posterior RGB")
    axes[0, 2].imshow(rgb(batch["image_raw"][0, fut_batch_t]))
    axes[0, 2].set_title("Future target RGB")
    axes[0, 3].imshow(rgb(future_output["rgb_1"][0, fut_t]))
    axes[0, 3].set_title("Prior future RGB")

    axes[1, 0].imshow(depth(batch["lidar"][0, obs_t, 3]), cmap="magma", vmin=0, vmax=lidar_vmax)
    axes[1, 0].set_title("Observed depth")
    axes[1, 1].imshow(depth(posterior_output["lidar_reconstruction_1"][0, obs_t, 3]), cmap="magma", vmin=0, vmax=lidar_vmax)
    axes[1, 1].set_title("Posterior depth")
    axes[1, 2].imshow(depth(batch["lidar"][0, fut_batch_t, 3]), cmap="magma", vmin=0, vmax=lidar_vmax)
    axes[1, 2].set_title("Future target depth")
    axes[1, 3].imshow(depth(future_output["lidar_reconstruction_1"][0, fut_t, 3]), cmap="magma", vmin=0, vmax=lidar_vmax)
    axes[1, 3].set_title("Prior future depth")

    for ax in axes.ravel():
        ax.axis("off")

    out_dir = _project_root_from_cfg(run_cfg) / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}_reconstruction_s{sample_idx}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved reconstruction figure: {path}")
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
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    _plot_scalar_tags(
        axes[0, 0], scalars,
        ["train_loss_step", "train_loss_epoch", "val/RL_loss", "val/DS_loss"],
        "Total loss", "loss",
    )
    _plot_scalar_tags(
        axes[0, 1], scalars,
        ["val/RL_rgb_psnr", "val/DS_rgb_psnr", "val/RL_future_rgb_psnr", "val/DS_future_rgb_psnr"],
        "Camera PSNR", "PSNR (dB)",
    )
    _plot_scalar_tags(
        axes[0, 2], scalars,
        ["val/RL_lidar_chamfer_xyz", "val/DS_lidar_chamfer_xyz",
         "val/RL_future_lidar_chamfer_xyz", "val/DS_future_lidar_chamfer_xyz"],
        "LiDAR Chamfer distance", "Chamfer distance",
    )
    _plot_scalar_tags(
        axes[1, 0], scalars,
        ["val/RL_lidar_xyz_euclidean", "val/DS_lidar_xyz_euclidean",
         "val/RL_future_lidar_xyz_euclidean", "val/DS_future_lidar_xyz_euclidean"],
        "LiDAR XYZ Euclidean error", "meters",
    )
    _plot_scalar_tags(
        axes[1, 1], scalars,
        ["val/RL_lidar_range_mae", "val/DS_lidar_range_mae",
         "val/RL_future_lidar_range_mae", "val/DS_future_lidar_range_mae"],
        "LiDAR range MAE", "meters",
    )
    _plot_scalar_tags(
        axes[1, 2], scalars,
        ["train_probabilistic_step", "val/RL_probabilistic", "val/DS_probabilistic"],
        "KL loss (weighted)", "loss",
    )

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{cfg.LOGGING.RUN_NAME}_loss_metrics.png"
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved loss metric figure to: {path}")
    return path, scalars


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
        self.weight_lidar_re       = getattr(self.cfg.LOSSES, "WEIGHT_LIDAR_RE", 1.0)
        self.weight_lidar_empty    = getattr(self.cfg.LOSSES, "WEIGHT_LIDAR_EMPTY", 0.05)
        self.weight_rgb            = getattr(self.cfg.LOSSES, "WEIGHT_RGB", 0.1)
        self.weight_future         = getattr(self.cfg.LOSSES, "WEIGHT_FUTURE", 1.0)
        # Default 0.06 == upstream effective (rgb_weight=0.1 × ssim_weight=0.6).
        # 0.6은 alpha의 WEIGHT_RGB=1.0과 곱해져 upstream의 10×가 되어 학습 불안정.
        self.weight_ssim           = getattr(self.cfg.LOSSES, "WEIGHT_SSIM", 0.0)
        self.ssim_loss             = SSIMLoss(channel=3) if self.weight_ssim > 0.0 else None

        self._tbptt_h:       Optional[torch.Tensor] = None
        self._tbptt_s:       Optional[torch.Tensor] = None
        self._tbptt_action:  Optional[torch.Tensor] = None
        self._prev_run_id    = None
        self._prev_start_row: Optional[List[int]] = None

        self.val_dataset_names  = ["RL", "DS"]
        self.test_dataset_names = ["RL", "DS"]
        self.metric_max_chamfer_points = 2048

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
        if "range_view_pcd_xyzd" in batch and "lidar" not in batch:
            batch["lidar"] = batch["range_view_pcd_xyzd"].float()
        if "lidar" in batch and "range_view_label_1" not in batch:
            batch["range_view_label_1"] = batch["lidar"].float()
        if "image_raw" in batch and "rgb_label_1" not in batch:
            batch["rgb_label_1"] = batch["image_raw"].float()
        elif "image" in batch and "rgb_label_1" not in batch:
            raise KeyError(
                "rgb_label_1에는 raw reconstruction target이 필요합니다. "
                "현재 batch에는 정규화된 'image'만 있고 'image_raw'가 없습니다."
            )
        for key in ["image", "image_raw", "lidar", "action", "speed",
                    "range_view_label_1", "rgb_label_1"]:
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

    @staticmethod
    def _normalise_run_ids(run_id, batch_size):
        if run_id is None:
            return [None] * batch_size
        if torch.is_tensor(run_id):
            values = run_id.detach().cpu().tolist()
            return values if isinstance(values, list) else [values] * batch_size
        if isinstance(run_id, (list, tuple)):
            return list(run_id)
        return [run_id] * batch_size

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

    @staticmethod
    def range_view_valid_mask(range_view):
        return torch.isfinite(range_view[:, :, -1:]) & (range_view[:, :, -1:] > 0)

    @staticmethod
    def xyz_euclidean_distance(prediction, target):
        valid_mask = WorldModelTrainer.range_view_valid_mask(target)
        if not valid_mask.any():
            return prediction.new_zeros(())
        xyz_error = torch.linalg.norm(prediction[:, :, :3] - target[:, :, :3], dim=2, keepdim=True)
        return xyz_error[valid_mask].mean()

    @staticmethod
    def range_mae(prediction, target):
        valid_mask = WorldModelTrainer.range_view_valid_mask(target)
        if not valid_mask.any():
            return prediction.new_zeros(())
        return F.l1_loss(prediction[:, :, -1:][valid_mask], target[:, :, -1:][valid_mask])

    @staticmethod
    def _subsample_points(points, max_points):
        if points.shape[0] <= max_points:
            return points
        idx = torch.linspace(0, points.shape[0] - 1, max_points, device=points.device).long()
        return points[idx]

    @staticmethod
    def chamfer_distance_range_view(prediction, target, max_points=2048):
        prediction   = prediction.float()
        target       = target.float()
        pred_valid   = WorldModelTrainer.range_view_valid_mask(prediction).squeeze(2)
        target_valid = WorldModelTrainer.range_view_valid_mask(target).squeeze(2)
        scores       = []
        b, s         = prediction.shape[:2]
        for bi in range(b):
            for ti in range(s):
                pred_pts   = prediction[bi, ti, :3].permute(1, 2, 0)[pred_valid[bi, ti]]
                target_pts = target[bi,   ti, :3].permute(1, 2, 0)[target_valid[bi, ti]]
                if pred_pts.numel() == 0:
                    pred_pts = prediction[bi, ti, :3].permute(1, 2, 0)[target_valid[bi, ti]]
                if pred_pts.numel() == 0 or target_pts.numel() == 0:
                    continue
                pred_pts   = WorldModelTrainer._subsample_points(pred_pts,   max_points)
                target_pts = WorldModelTrainer._subsample_points(target_pts, max_points)
                dist       = torch.cdist(pred_pts.unsqueeze(0), target_pts.unsqueeze(0), p=2).squeeze(0)
                scores.append(0.5 * (dist.min(dim=0).values.mean() + dist.min(dim=1).values.mean()))
        return torch.stack(scores).mean() if scores else prediction.new_zeros(())

    def compute_eval_metrics(self, batch, output, prefix=""):
        metrics = {}
        if "rgb_2" in output and "rgb_label_1" in batch:
            pred   = output["rgb_2"]
            target = batch["rgb_label_1"]
            if pred.shape[-2:] != target.shape[-2:]:
                target = self.resize_sequence_tensor(target, pred.shape[-2:], mode="bilinear")
            metrics[f"{prefix}rgb_psnr"] = self.psnr(pred, target).mean()

        if "lidar_reconstruction_2" in output and "range_view_label_1" in batch:
            lidar_scale = float(getattr(self.cfg.DATA, "LIDAR_SCALE", 1.0))
            pred   = output["lidar_reconstruction_2"]
            target = batch["range_view_label_1"]
            if pred.shape[-2:] != target.shape[-2:]:
                target = self.resize_sequence_tensor(target, pred.shape[-2:], mode="nearest")
            metrics[f"{prefix}lidar_chamfer_xyz"]   = self.chamfer_distance_range_view(
                pred, target, max_points=self.metric_max_chamfer_points) * lidar_scale
            metrics[f"{prefix}lidar_xyz_euclidean"] = self.xyz_euclidean_distance(pred, target) * lidar_scale
            metrics[f"{prefix}lidar_range_mae"]     = self.range_mae(pred, target) * lidar_scale
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
        kl_sum = kl.sum(dim=-1)  # (B, S)
        if free_bits > 0.0:
            kl_sum = kl_sum.clamp_min(free_bits)
        return kl_sum.mean()

    def balanced_kl_loss(self, prior, posterior):
        alpha     = self.kl_balancing_alpha
        free_bits = getattr(self.cfg.LOSSES, "KL_FREE_BITS", 0.0)
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

    @staticmethod
    def _masked_loss(pred, target, mask, loss_fn):
        if mask.any():
            return loss_fn(pred[mask.expand_as(pred)], target[mask.expand_as(target)])
        return pred.new_zeros(())

    def compute_loss(self, batch, output, include_kl=True, prefix=""):
        losses = {}

        if include_kl and "prior" in output and "posterior" in output:
            losses[f"{prefix}probabilistic"] = (
                self.weight_probabilistic * self.balanced_kl_loss(output["prior"], output["posterior"])
            )

        if "range_view_label_1" in batch:
            for factor in [1, 2, 4]:
                key = f'lidar_reconstruction_{factor}'
                if key in output:
                    pred     = output[key]
                    target   = batch["range_view_label_1"]
                    discount = 1 / factor
                    if pred.shape[-2:] != target.shape[-2:]:
                        target = self.resize_sequence_tensor(target, pred.shape[-2:], mode="nearest")
                    valid_mask   = target[:, :, -1:] > 0
                    invalid_mask = ~valid_mask
                    pred_depth   = pred[:, :, -1:]
                    target_depth = target[:, :, -1:]
                    if pred.shape[2] >= 3:
                        losses[f'{prefix}lidar_xyz_{factor}'] = (
                            self.weight_lidar_re * discount *
                            self._masked_loss(pred[:, :, :3], target[:, :, :3], valid_mask, F.mse_loss)
                        )
                    losses[f'{prefix}lidar_depth_{factor}'] = (
                        self.weight_lidar_re * discount *
                        self._masked_loss(pred_depth, target_depth, valid_mask, F.l1_loss)
                    )
                    losses[f'{prefix}lidar_empty_depth_{factor}'] = (
                        self.weight_lidar_empty * discount *
                        self._masked_loss(pred_depth, torch.zeros_like(pred_depth), invalid_mask, F.smooth_l1_loss)
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

        if not losses:
            raise RuntimeError("No valid loss computed.")
        return losses

    def loss_reducing(self, losses):
        return sum(losses.values())

    # ─── Core forward: observe + imagine ─────────────────────────────────────

    def _observe_and_imagine(self, batch, h_init=None, s_init=None,
                             continuation=False, prev_action=None):
        batch_rf = self._slice_batch(batch, 0, self.rf)
        batch_fh = self._slice_batch(batch, self.rf, self.rf + self.fh)

        if prev_action is not None:
            if torch.is_tensor(continuation):
                prev_action = prev_action.clone()
                prev_action[~continuation] = 0
            elif not continuation:
                prev_action = None

        output, state_dict = self.model.forward(
            batch_rf, h_init=h_init, s_init=s_init,
            continuation=continuation, init_action=prev_action,
        )
        losses = self.compute_loss(batch_rf, output, include_kl=True)

        output_imagine  = None
        losses_imagine  = {}
        if self.fh > 0 and "action" in batch_fh and batch_fh["action"].shape[1] > 0:
            last_obs_action = batch_rf["action"][:, -1:]
            future_action   = torch.cat([last_obs_action, batch_fh["action"][:, :-1]], dim=1)
            state_imagine   = {
                "hidden_state": state_dict["posterior"]["hidden_state"][:, -1],
                "sample":       state_dict["posterior"]["sample"][:, -1],
                "action":       future_action,
            }
            output_imagine, _ = self.model.imagine(state_imagine, future_horizon=self.fh)
            future_losses     = self.compute_loss(batch_fh, output_imagine, include_kl=False, prefix="future_")
            losses_imagine    = {k: self.weight_future * v for k, v in future_losses.items()}
            losses.update(losses_imagine)

        return losses, output, state_dict, losses_imagine, output_imagine

    # ─── Training ────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        batch      = self.prepare_custom_batch(batch)
        B          = batch["image"].shape[0]
        device     = batch["image"].device
        hidden_dim = self.model.rssm.hidden_state_dim
        state_dim  = self.model.rssm.state_dim

        run_ids  = self._normalise_run_ids(batch.get("run_id", None), B)
        prev_ids = self._prev_run_id if isinstance(self._prev_run_id, list) else [self._prev_run_id] * B

        start_rows_t  = batch.get("start_row", None)
        start_rows    = start_rows_t.tolist() if start_rows_t is not None else [None] * B
        prev_start_rows = self._prev_start_row if self._prev_start_row is not None else [None] * B
        window_stride = self.rf * self.cfg.DATA.SAMPLE_EVERY_N

        have_tbptt = self._tbptt_h is not None and self._tbptt_s is not None
        action_dim = batch["action"].shape[-1] if "action" in batch else 2

        if not have_tbptt:
            h_prev       = torch.zeros(B, hidden_dim, device=device)
            s_prev       = torch.zeros(B, state_dim,  device=device)
            prev_act     = torch.zeros(B, action_dim,  device=device)
            continuation = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            h_prev = self._tbptt_h[:B].to(device).detach()
            s_prev = self._tbptt_s[:B].to(device).detach()
            cont_list = []
            for i in range(B):
                run_ok  = (prev_ids[i] if i < len(prev_ids) else None) == run_ids[i]
                prev_sr = prev_start_rows[i] if i < len(prev_start_rows) else None
                curr_sr = start_rows[i]
                frame_ok = (prev_sr is not None and curr_sr is not None
                            and prev_sr + window_stride == curr_sr)
                cont_list.append(run_ok and frame_ok)
            continuation = torch.tensor(cont_list, dtype=torch.bool, device=device)
            for i in range(B):
                if not cont_list[i]:
                    h_prev[i].zero_()
                    s_prev[i].zero_()
            if self._tbptt_action is not None:
                prev_act = self._tbptt_action[:B].to(device).detach().clone()
                for i in range(B):
                    if not cont_list[i]:
                        prev_act[i].zero_()
            else:
                prev_act = torch.zeros(B, action_dim, device=device)

        self._prev_run_id = run_ids

        losses, output, state_dict, losses_imagine, output_imagine = self._observe_and_imagine(
            batch, h_init=h_prev, s_init=s_prev,
            continuation=continuation, prev_action=prev_act,
        )

        self._tbptt_h      = state_dict["posterior"]["hidden_state"][:, -1, :].detach()
        self._tbptt_s      = state_dict["posterior"]["sample"][:, -1, :].detach()
        self._tbptt_action = batch["action"][:, self.rf - 1].detach()
        self._prev_start_row = start_rows

        total_loss = self.loss_reducing(losses)
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        for name, value in losses.items():
            self.log(f"train_{name}", value, prog_bar=False, on_step=True, on_epoch=True)
        return total_loss

    def on_train_epoch_end(self):
        self._tbptt_h        = None
        self._tbptt_s        = None
        self._tbptt_action   = None
        self._prev_run_id    = None
        self._prev_start_row = None

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
        losses, output, state_dict, losses_imagine, output_imagine = self._observe_and_imagine(batch)
        dataset_name = self._dataset_name(self.val_dataset_names, dataloader_idx)
        total_loss   = self._log_eval_outputs("val", dataset_name, batch, losses, output, output_imagine)
        return {f"val_{dataset_name}_loss": total_loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self.prepare_custom_batch(batch)
        losses, output, state_dict, losses_imagine, output_imagine = self._observe_and_imagine(batch)
        dataset_name = self._dataset_name(self.test_dataset_names, dataloader_idx)
        total_loss   = self._log_eval_outputs("test", dataset_name, batch, losses, output, output_imagine)
        return {f"test_{dataset_name}_loss": total_loss}

    # ─── Optimizer ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer      = torch.optim.AdamW(
            self.parameters(), lr=self.lr,
            weight_decay=getattr(self.cfg.OPTIMIZER, "WEIGHT_DECAY", 0.0),
        )
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
    """매 N epoch마다 `save_reconstruction_figure`를 호출해 학습 도중 figure를 dump.

    overfit 검증용. 1D baseline에서도 `_muvo_2D`와 동일한 학습 도중 figure 비교를
    가능하게 하기 위해 추가. save 전후로 model 모드 복원해 학습 흐름 유지.
    """

    def __init__(self, dataset, run_cfg, run_name, every_n_epochs=100, sample_idx=0):
        super().__init__()
        self.dataset = dataset
        self.run_cfg = run_cfg
        self.run_name = run_name
        self.every_n_epochs = int(every_n_epochs)
        self.sample_idx = int(sample_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs <= 0:
            return
        if not trainer.is_global_zero:
            return
        epoch_1_indexed = trainer.current_epoch + 1
        if epoch_1_indexed % self.every_n_epochs != 0:
            return
        was_training = pl_module.training
        try:
            save_reconstruction_figure(
                pl_module, self.dataset, self.run_cfg,
                run_name=f"{self.run_name}_e{epoch_1_indexed:04d}",
                sample_idx=self.sample_idx,
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
        "rgb_recon_size", "lidar_size", "lidar_fov", "h_dim", "z_dim", "lr",
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
            "val/RL_lidar_chamfer_xyz", "val/DS_lidar_chamfer_xyz",
            "val/RL_future_lidar_chamfer_xyz", "val/DS_future_lidar_chamfer_xyz",
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

        row = {
            "date":             date.today().isoformat(),
            "run_name":         getattr(self.cfg.LOGGING, "RUN_NAME", "trainer11"),
            "base_file":        getattr(self.cfg.LOGGING, "BASE_FILE", "train_muvo.py"),
            "train_files":      self._files_to_str(self.train_files),
            "val_files":        self._files_to_str(self.val_files),
            "sample_hz":        4 / getattr(self.cfg.DATA, "SAMPLE_EVERY_N", 1),
            "rgb_recon_size":   self._size_to_str(self.cfg.DATA.RGB_RECON_SIZE),
            "lidar_size":       self._size_to_str(self.cfg.DATA.LIDAR_RANGE_VIEW_SIZE),
            "lidar_fov":        self._size_to_str(self.cfg.DATA.LIDAR_FOV_DEGREES),
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
