"""trainer_muvo_2D.py — `trainer_muvo.py` 베이스 + 2D-token RSSM 변경:

- action loss 추가 (`F.l1_loss(output["action_pred"], batch["action"]) * WEIGHT_ACTION`)
- near-field Chamfer metric 추가 (PC range cfg에서)
- reconstruction figure: 6 frame 전체 column + LiDAR-scale 보정/미보정 두 row
- KL free-bits on/off cfg gate (`KL_FREE_BITS_ENABLED`)
- ClearML artifact uploasd optional (`Task.current_task()`)
- RSSMTD는 TBPTT, h_init/s_init/continuation/init_action 사용하지 않으므로
  `training_step` TBPTT 로직을 간소화 (state carry 제거).
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

# 모든 figure(reconstruction / trajectory / loss-metric)는 이 yj 디렉토리 안의
# `figure/`에 저장한다. cwd와 무관하게 스크립트 위치 기준으로 고정.
FIGURE_DIR = Path(__file__).resolve().parent / "figure"


def _project_root_from_cfg(run_cfg):
    log_path = getattr(getattr(run_cfg, "LOGGING", object()), "EXPERIMENT_LOG_PATH", None)
    if log_path:
        return Path(log_path).expanduser().resolve().parent.parent
    return Path.cwd()


def save_reconstruction_figure(model, dataset, run_cfg, run_name, sample_idx=0):
    """RF+FH = 6 frame 전체 column + LiDAR scale 보정/미보정 두 row 모드 지원.

    `cfg.LOGGING.RECON_FIG_ALL_FRAMES=True`: 모든 frame 표시 (default v3).
    `cfg.LOGGING.RECON_FIG_BOTH_VIEWS=True`: LiDAR depth를 scaled + unscaled 두 row.

    `False`로 두면 alpha _muvo baseline과 동일한 4-column 2-row 단순 figure.
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
    all_frames = bool(getattr(run_cfg.LOGGING, "RECON_FIG_ALL_FRAMES", True))
    both_views = bool(getattr(run_cfg.LOGGING, "RECON_FIG_BOTH_VIEWS", True))

    lidar_scale = float(getattr(run_cfg.DATA, "LIDAR_SCALE", 1.0))
    lidar_vmax_scaled = 2.0 * lidar_scale
    lidar_vmax_unscaled = 2.0  # training-space [0, ~2]

    def rgb(tensor):
        return tensor.detach().float().cpu().permute(1, 2, 0).clamp(0, 1)

    def depth_scaled(tensor):
        return tensor.detach().float().cpu() * lidar_scale

    def depth_unscaled(tensor):
        return tensor.detach().float().cpu()

    if all_frames:
        # Columns: RF frames (gt vs posterior recon) + FH frames (gt vs imagine pred).
        # Each column shows one frame.
        n_cols = rf + fh
        rgb_rows = 2  # gt_rgb, pred_rgb
        depth_rows = 4 if both_views else 2  # (gt scaled, pred scaled) + (gt unscaled, pred unscaled)
        n_rows = rgb_rows + depth_rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows))
        axes = np.atleast_2d(axes)

        for col in range(n_cols):
            if col < rf:
                # observation frame col
                t = col
                gt_rgb = batch["image_raw"][0, t]
                pred_rgb_t = posterior_output["rgb_1"][0, t]
                gt_lidar_t = batch["lidar"][0, t, 3]
                pred_lidar_t = posterior_output["lidar_reconstruction_1"][0, t, 3]
                col_label = f"obs t={t}"
            else:
                # future frame col
                fut_t = col - rf
                t_batch = rf + fut_t
                gt_rgb = batch["image_raw"][0, t_batch]
                pred_rgb_t = future_output["rgb_1"][0, fut_t]
                gt_lidar_t = batch["lidar"][0, t_batch, 3]
                pred_lidar_t = future_output["lidar_reconstruction_1"][0, fut_t, 3]
                col_label = f"fut t={fut_t}"

            axes[0, col].imshow(rgb(gt_rgb))
            axes[0, col].set_title(f"{col_label}\nGT RGB")
            axes[1, col].imshow(rgb(pred_rgb_t))
            axes[1, col].set_title("Pred RGB")

            axes[2, col].imshow(depth_scaled(gt_lidar_t), cmap="magma", vmin=0, vmax=lidar_vmax_scaled)
            axes[2, col].set_title("GT depth (m)")
            axes[3, col].imshow(depth_scaled(pred_lidar_t), cmap="magma", vmin=0, vmax=lidar_vmax_scaled)
            axes[3, col].set_title("Pred depth (m)")

            if both_views:
                axes[4, col].imshow(depth_unscaled(gt_lidar_t), cmap="magma", vmin=0, vmax=lidar_vmax_unscaled)
                axes[4, col].set_title("GT depth (raw)")
                axes[5, col].imshow(depth_unscaled(pred_lidar_t), cmap="magma", vmin=0, vmax=lidar_vmax_unscaled)
                axes[5, col].set_title("Pred depth (raw)")

        for ax in axes.ravel():
            ax.axis("off")
    else:
        # Legacy 2x4 figure (alpha _muvo와 동일).
        obs_t = rf - 1
        fut_t = 0
        fut_batch_t = rf
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        axes = np.asarray(axes)
        axes[0, 0].imshow(rgb(batch["image_raw"][0, obs_t])); axes[0, 0].set_title("Observed RGB")
        axes[0, 1].imshow(rgb(posterior_output["rgb_1"][0, obs_t])); axes[0, 1].set_title("Posterior RGB")
        axes[0, 2].imshow(rgb(batch["image_raw"][0, fut_batch_t])); axes[0, 2].set_title("Future target RGB")
        axes[0, 3].imshow(rgb(future_output["rgb_1"][0, fut_t])); axes[0, 3].set_title("Prior future RGB")
        axes[1, 0].imshow(depth_scaled(batch["lidar"][0, obs_t, 3]), cmap="magma", vmin=0, vmax=lidar_vmax_scaled)
        axes[1, 0].set_title("Observed depth")
        axes[1, 1].imshow(depth_scaled(posterior_output["lidar_reconstruction_1"][0, obs_t, 3]), cmap="magma", vmin=0, vmax=lidar_vmax_scaled)
        axes[1, 1].set_title("Posterior depth")
        axes[1, 2].imshow(depth_scaled(batch["lidar"][0, fut_batch_t, 3]), cmap="magma", vmin=0, vmax=lidar_vmax_scaled)
        axes[1, 2].set_title("Future target depth")
        axes[1, 3].imshow(depth_scaled(future_output["lidar_reconstruction_1"][0, fut_t, 3]), cmap="magma", vmin=0, vmax=lidar_vmax_scaled)
        axes[1, 3].set_title("Prior future depth")
        for ax in axes.ravel():
            ax.axis("off")

    out_dir = FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}_reconstruction_s{sample_idx}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved reconstruction figure: {path}")
    return path


# ─────────────────────────────────────────────
# Trajectory (TransFuser waypoint) → image projection
# ─────────────────────────────────────────────

def _waypoints_to_pixels(waypoints, image_hw, cam_cfg, y_right_positive=True):
    """ego-frame waypoint (N,2)=[x_forward, y_lateral]를 front-camera 픽셀 (N,2)=[u,v]로 투영.

    핀홀 카메라 + 지면(z=0) 가정. cam_cfg: FOV_DEG / HEIGHT_M / X_OFFSET_M / PITCH_DEG.
    실제 CARLA front-camera rig 값과 일치해야 정확하다(기본값은 일반적 가정치).
    반환: pixels (N,2), visible (N,) bool — 카메라 앞쪽(Z>0) 점만 True.
    """
    import numpy as _np
    H, W = image_hw
    fov = _np.deg2rad(float(getattr(cam_cfg, "FOV_DEG", 90.0)))
    f = W / (2.0 * _np.tan(fov / 2.0))
    cx, cy = W / 2.0, H / 2.0
    h_cam = float(getattr(cam_cfg, "HEIGHT_M", 1.6))
    x_off = float(getattr(cam_cfg, "X_OFFSET_M", 1.2))
    pitch = _np.deg2rad(float(getattr(cam_cfg, "PITCH_DEG", 0.0)))

    wp = _np.asarray(waypoints, dtype=_np.float64).reshape(-1, 2)
    fwd   = wp[:, 0] - x_off
    right = wp[:, 1] * (1.0 if y_right_positive else -1.0)

    # camera frame (OpenCV 관례: X=right, Y=down, Z=forward). 지면 점은 카메라보다 h_cam 아래.
    Xc = right
    Yc = h_cam * _np.ones_like(fwd)
    Zc = fwd
    cosp, sinp = _np.cos(pitch), _np.sin(pitch)
    Yc2 = cosp * Yc - sinp * Zc
    Zc2 = sinp * Yc + cosp * Zc
    visible = Zc2 > 1e-3
    denom = _np.clip(Zc2, 1e-3, None)
    u = f * Xc / denom + cx
    v = f * Yc2 / denom + cy
    return _np.stack([u, v], axis=-1), visible


def _densify_ego_path(points, n_out=16):
    """ego [x_fwd, y_lat] waypoint (N,2) 앞에 원점(0,0)을 붙이고
    호 길이 기준 균등 보간으로 n_out개 점을 만든다(궤적 촘촘히 표시용).
    전체 이동거리가 ~0(정지)이면 원점 1점만 반환.
    """
    import numpy as np
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts = np.vstack([[0.0, 0.0], pts])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-4:
        return pts[:1]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0.0, total, n_out)
    return np.stack([np.interp(u, cum, pts[:, 0]),
                     np.interp(u, cum, pts[:, 1])], axis=1)


def save_trajectory_overlay_figure(pl_module, dataset, run_cfg, run_name, sample_idx=0):
    """GT/pred trajectory를 (좌) front-camera 투영 + (우) BEV top-down 두 패널로 저장.

    - 참조 프레임: 관측 구간(0..rf-1) 중 GT 변위가 가장 큰(가장 많이 움직이는) 프레임을
      자동 선택. (마지막 프레임이 정지 시점이면 그릴 게 없어 보이지 않던 문제 회피.)
    - 궤적은 원점(ego)부터 촘촘히 보간해 그린다.
    - 카메라 투영은 rig 가정치(pitch/높이/오프셋)에 민감해 근거리 점이 화면 밖으로
      떨어질 수 있으므로, rig와 무관하게 항상 보이는 BEV(미터 단위)를 함께 제공한다.
    pl_module: WorldModelTrainer 인스턴스. WAYPOINT.ENABLED=True 인 모델에서만 동작.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl_module = pl_module.to(device).eval()
    sample_idx = min(int(sample_idx), len(dataset) - 1)
    sample = dataset[sample_idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
    batch = pl_module.prepare_custom_batch(batch)

    with torch.no_grad():
        output, _ = pl_module.model.forward(batch)
    if "waypoints_pred" not in output:
        print("waypoints_pred 없음 — cfg.MODEL.WAYPOINT.ENABLED 확인.")
        return None

    rf = run_cfg.RECEPTIVE_FIELD
    gt_wp_all, mask = pl_module.compute_gt_waypoints(batch["action"], batch["speed"])

    # 참조 프레임 자동 선택: mask 유효 & GT 마지막 waypoint 변위가 최대인 관측 프레임.
    obs = min(rf, gt_wp_all.shape[1])
    disp = gt_wp_all[0, :obs, -1].norm(dim=-1)               # (obs,)
    disp = torch.where(mask[0, :obs] > 0, disp, torch.zeros_like(disp))
    ego_stopped = float(disp.max()) <= 1e-3
    t_ref = (obs - 1) if ego_stopped else int(disp.argmax().item())

    pred_wp = output["waypoints_pred"][0, t_ref].detach().float().cpu().numpy()
    gt_wp = gt_wp_all[0, t_ref].detach().float().cpu().numpy()
    speed_ref = float(batch["speed"][0, t_ref].detach().float().cpu())

    gt_dense = _densify_ego_path(gt_wp, n_out=16)
    pred_dense = _densify_ego_path(pred_wp, n_out=16)

    img = batch["image_raw"][0, t_ref].detach().float().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    H, W = img.shape[:2]
    cam_cfg = getattr(run_cfg.DATA, "CAMERA", None)
    y_right = bool(getattr(getattr(run_cfg.MODEL, "WAYPOINT", object()), "Y_RIGHT_POSITIVE", True))
    sgn = 1.0 if y_right else -1.0

    fig, (ax_cam, ax_bev) = plt.subplots(
        1, 2, figsize=(max(6.0, W / 100) + 5.0, max(4.0, H / 100)),
        gridspec_kw={"width_ratios": [max(1.0, W / H), 1.0]},
    )

    # ── (좌) front-camera 투영 (rig 가정치 기준; 근거리 점은 화면 밖일 수 있음) ──
    ax_cam.imshow(img)
    for path, color, label in [(gt_dense, "lime", "GT (kinematic)"), (pred_dense, "red", "pred")]:
        px, vis = _waypoints_to_pixels(path, (H, W), cam_cfg, y_right_positive=y_right)
        pv = px[vis]
        if len(pv):
            ax_cam.plot(pv[:, 0], pv[:, 1], "-o", color=color, label=label, markersize=4, linewidth=2)
    ax_cam.set_xlim(0, W); ax_cam.set_ylim(H, 0); ax_cam.axis("off")
    ax_cam.set_title(f"camera overlay (t={t_ref}, v={speed_ref:.1f} km/h)", fontsize=9)
    ax_cam.legend(loc="lower right", fontsize=8)

    # ── (우) BEV top-down: 전방 x=세로(위), 횡 y=가로(우 +) ──
    max_fwd, max_lat = 1.0, 1.0
    for path, color, label, marker in [(gt_dense, "lime", "GT (kinematic)", "o"),
                                       (pred_dense, "red", "pred", "s")]:
        X = sgn * path[:, 1]
        Y = path[:, 0]
        ax_bev.plot(X, Y, "-", color=color, label=label, linewidth=2, marker=marker, markersize=4)
        max_fwd = max(max_fwd, float(np.abs(Y).max()))
        max_lat = max(max_lat, float(np.abs(X).max()))
    ax_bev.scatter([0], [0], c="k", marker="^", s=90, zorder=5, label="ego")
    lim_lat = max(2.0, max_lat * 1.3)
    ax_bev.set_xlim(-lim_lat, lim_lat)
    ax_bev.set_ylim(-1.0, max(5.0, max_fwd * 1.2))
    ax_bev.set_aspect("equal", adjustable="box")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_xlabel("lateral y [m] (right +)"); ax_bev.set_ylabel("forward x [m]")
    ax_bev.set_title("BEV top-down (meters)", fontsize=9)
    ax_bev.legend(loc="upper right", fontsize=8)
    if ego_stopped:
        ax_bev.text(0, 0.4, "ego stopped\n(waypoints ~0)", ha="center", fontsize=9, color="gray")

    out_dir = FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}_trajectory_s{sample_idx}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory overlay figure: {path} (t_ref={t_ref}, v={speed_ref:.1f} km/h)")
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    _plot_scalar_tags(axes[0, 0], scalars,
        ["train_loss_step", "train_loss_epoch", "val/RL_loss", "val/DS_loss"],
        "Total loss", "loss")
    _plot_scalar_tags(axes[0, 1], scalars,
        ["val/RL_rgb_psnr", "val/DS_rgb_psnr", "val/RL_future_rgb_psnr", "val/DS_future_rgb_psnr"],
        "Camera PSNR", "PSNR (dB)")
    _plot_scalar_tags(axes[0, 2], scalars,
        ["val/RL_lidar_chamfer_xyz", "val/DS_lidar_chamfer_xyz",
         "val/RL_lidar_chamfer_xyz_nearfield", "val/DS_lidar_chamfer_xyz_nearfield"],
        "LiDAR Chamfer (global + near-field)", "Chamfer distance")
    _plot_scalar_tags(axes[1, 0], scalars,
        ["val/RL_lidar_xyz_euclidean", "val/DS_lidar_xyz_euclidean"],
        "LiDAR XYZ Euclidean error", "meters")
    _plot_scalar_tags(axes[1, 1], scalars,
        ["train_action", "val/RL_action", "val/DS_action"],
        "Action loss (L1)", "loss")
    _plot_scalar_tags(axes[1, 2], scalars,
        ["train_probabilistic_step", "val/RL_probabilistic", "val/DS_probabilistic"],
        "KL loss (weighted)", "loss")

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
        self.weight_lidar_re       = getattr(self.cfg.LOSSES, "WEIGHT_LIDAR_RE", 1.0)
        self.weight_lidar_empty    = getattr(self.cfg.LOSSES, "WEIGHT_LIDAR_EMPTY", 0.05)
        self.weight_rgb            = getattr(self.cfg.LOSSES, "WEIGHT_RGB", 0.1)
        #self.weight_future         = getattr(self.cfg.LOSSES, "WEIGHT_FUTURE", 1.0)
        self.weight_ssim           = getattr(self.cfg.LOSSES, "WEIGHT_SSIM", 0.0)
        self.ssim_loss             = SSIMLoss(channel=3) if self.weight_ssim > 0.0 else None
        # KL free-bits cfg gate (v3, Q8). default OFF = upstream과 일치.
        self.kl_free_bits_enabled  = bool(getattr(self.cfg.LOSSES, "KL_FREE_BITS_ENABLED", False))
        self.kl_free_bits_value    = float(getattr(self.cfg.LOSSES, "KL_FREE_BITS", 0.0))
        # Action loss (v2, Q9).
        self.weight_action         = getattr(self.cfg.LOSSES, "WEIGHT_ACTION", 1.0)

        # TransFuser trajectory(waypoint) loss + kinematic GT 파라미터.
        # pose 컬럼이 없어 speed_kmh + steer로 미래 waypoint를 근사 생성한다.
        wp_cfg = getattr(self.cfg.MODEL, "WAYPOINT", None)
        self.weight_waypoint = getattr(self.cfg.LOSSES, "WEIGHT_WAYPOINT", 0.0)
        self.n_waypoints  = int(getattr(wp_cfg, "N_WAYPOINTS", self.fh)) if wp_cfg else self.fh
        self.wp_dt        = float(getattr(wp_cfg, "DT", 1.0 / getattr(self.cfg.DATA, "EFFECTIVE_HZ", 2))) if wp_cfg else 0.5
        self.wp_wheelbase = float(getattr(wp_cfg, "WHEELBASE", 2.85)) if wp_cfg else 2.85
        self.wp_max_steer = float(getattr(wp_cfg, "MAX_STEER_RAD", 1.221)) if wp_cfg else 1.221

        # Near-field Chamfer PC range (v3).
        self.nearfield_pc_range = tuple(getattr(self.cfg.MODEL.RSSM_2D, "NEARFIELD_PC_RANGE",
                                                (-20.0, -20.0, -2.0, 20.0, 20.0, 6.0)))

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
    def _chamfer_cost(pred_pts, target_pts):
        if pred_pts.numel() == 0 or target_pts.numel() == 0:
            return None
        dist = torch.cdist(pred_pts.unsqueeze(0), target_pts.unsqueeze(0), p=2).squeeze(0)
        return 0.5 * (dist.min(dim=0).values.mean() + dist.min(dim=1).values.mean())

    def chamfer_distance_range_view(self, prediction, target, max_points=2048, pc_range=None):
        """If pc_range is provided (xmin,ymin,zmin,xmax,ymax,zmax), mask both pred and target
        to points within that box before Chamfer (near-field variant).
        """
        prediction   = prediction.float()
        target       = target.float()
        pred_valid   = self.range_view_valid_mask(prediction).squeeze(2)
        target_valid = self.range_view_valid_mask(target).squeeze(2)
        scores       = []
        b, s         = prediction.shape[:2]
        for bi in range(b):
            for ti in range(s):
                pred_pts   = prediction[bi, ti, :3].permute(1, 2, 0)[pred_valid[bi, ti]]
                target_pts = target[bi,   ti, :3].permute(1, 2, 0)[target_valid[bi, ti]]
                if pred_pts.numel() == 0:
                    pred_pts = prediction[bi, ti, :3].permute(1, 2, 0)[target_valid[bi, ti]]
                if pc_range is not None:
                    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
                    if pred_pts.numel() > 0:
                        m = ((pred_pts[:, 0] >= xmin) & (pred_pts[:, 0] <= xmax)
                             & (pred_pts[:, 1] >= ymin) & (pred_pts[:, 1] <= ymax)
                             & (pred_pts[:, 2] >= zmin) & (pred_pts[:, 2] <= zmax))
                        pred_pts = pred_pts[m]
                    if target_pts.numel() > 0:
                        m = ((target_pts[:, 0] >= xmin) & (target_pts[:, 0] <= xmax)
                             & (target_pts[:, 1] >= ymin) & (target_pts[:, 1] <= ymax)
                             & (target_pts[:, 2] >= zmin) & (target_pts[:, 2] <= zmax))
                        target_pts = target_pts[m]
                pred_pts   = self._subsample_points(pred_pts,   max_points)
                target_pts = self._subsample_points(target_pts, max_points)
                cost = self._chamfer_cost(pred_pts, target_pts)
                if cost is not None:
                    scores.append(cost)
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
            # Chamfer는 meter 단위로 보고 (loss는 training-space 그대로).
            metrics[f"{prefix}lidar_chamfer_xyz"] = self.chamfer_distance_range_view(
                pred, target, max_points=self.metric_max_chamfer_points
            ) * lidar_scale
            # Near-field Chamfer는 PC range가 meter 기준이므로 pred/target을 meter로 scale 후 적용.
            pred_m   = pred   * lidar_scale
            target_m = target * lidar_scale
            metrics[f"{prefix}lidar_chamfer_xyz_nearfield"] = self.chamfer_distance_range_view(
                pred_m, target_m,
                max_points=self.metric_max_chamfer_points,
                pc_range=self.nearfield_pc_range,
            )
            metrics[f"{prefix}lidar_xyz_euclidean"] = self.xyz_euclidean_distance(pred, target) * lidar_scale
            metrics[f"{prefix}lidar_range_mae"]     = self.range_mae(pred, target) * lidar_scale

        # Trajectory metric: ADE(평균)/FDE(최종) displacement error [m].
        if "waypoints_pred" in output and "action" in batch and "speed" in batch:
            gt_wp, wp_mask = self.compute_gt_waypoints(batch["action"], batch["speed"])
            pred_wp = output["waypoints_pred"]
            s_min = min(pred_wp.shape[1], gt_wp.shape[1])
            m = wp_mask[:, :s_min].bool()
            if m.any():
                disp = torch.linalg.norm(pred_wp[:, :s_min][m] - gt_wp[:, :s_min][m], dim=-1)  # (M, n_wp)
                metrics[f"{prefix}waypoint_ade"] = disp.mean()
                metrics[f"{prefix}waypoint_fde"] = disp[:, -1].mean()
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

    @staticmethod
    def _masked_loss(pred, target, mask, loss_fn):
        if mask.any():
            return loss_fn(pred[mask.expand_as(pred)], target[mask.expand_as(target)])
        return pred.new_zeros(())

    def compute_gt_waypoints(self, action, speed):
        """pose 컬럼 부재 → speed_kmh + steer로 kinematic bicycle을 적분해
        미래 waypoint GT를 근사 생성한다 (TransFuser는 기록된 ego pose를 GT로 쓰지만
        본 데이터셋엔 pose가 없다 — 자세한 한계는 TRAJECTORY_TRANSFUSER.md 참조).

        action: (B, S, 2) = [throttle, steer],  speed: (B, S) [km/h].
        반환:
            gt   (B, S, n_wp, 2) — 프레임 t의 ego frame 기준 t+1..t+n_wp 위치
                                   (+x 전방, +y 우측).
            mask (B, S)          — t+n_wp <= S-1 인 유효 프레임만 1.
        """
        B, S, _ = action.shape
        n = self.n_waypoints
        dt, L, max_steer = self.wp_dt, self.wp_wheelbase, self.wp_max_steer
        steer = action[..., 1]
        v = speed.float() / 3.6  # km/h → m/s
        gt   = action.new_zeros(B, S, n, 2)
        mask = action.new_zeros(B, S)
        for t in range(S):
            if t + n > S - 1:           # 미래 프레임 부족 → 무효
                continue
            x  = action.new_zeros(B)
            y  = action.new_zeros(B)
            th = action.new_zeros(B)
            for k in range(1, n + 1):
                vk = v[:, t + k]
                delta = max_steer * steer[:, t + k]
                th = th + vk / L * torch.tan(delta) * dt   # bicycle yaw-rate 적분
                x  = x + vk * torch.cos(th) * dt
                y  = y + vk * torch.sin(th) * dt
                gt[:, t, k - 1, 0] = x
                gt[:, t, k - 1, 1] = y
            mask[:, t] = 1.0
        return gt, mask

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

        # Action loss (v2, Q9).
        if "action_pred" in output and "action" in batch and self.weight_action > 0.0:
            action_target = batch["action"][:, :output["action_pred"].shape[1]].float()
            losses[f"{prefix}action"] = self.weight_action * F.l1_loss(output["action_pred"], action_target)

        # Trajectory waypoint loss (TransFuser). kinematic GT vs GRU 예측, masked L1.
        if ("waypoints_pred" in output and self.weight_waypoint > 0.0
                and "action" in batch and "speed" in batch):
            gt_wp, wp_mask = self.compute_gt_waypoints(batch["action"], batch["speed"])
            pred_wp = output["waypoints_pred"]
            s_min = min(pred_wp.shape[1], gt_wp.shape[1])
            m = wp_mask[:, :s_min].bool()
            if m.any():
                losses[f"{prefix}waypoint"] = self.weight_waypoint * F.l1_loss(
                    pred_wp[:, :s_min][m], gt_wp[:, :s_min][m]
                )

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

    def _observe_and_imagine(self, batch, h_init=None, s_init=None,
                              continuation=False, Prev_action=None, n_samples=1):

        batch_rf = self._slice_batch(batch, 0, self.rf)
        batch_fh = self._slice_batch(batch, self.rf, self.rf + self.fh)
        output, state_dict = self.model.forward(batch_rf)
        losses = self.compute_loss(batch_rf, output, include_kl=True)
        
        output_imagine = None
        losses_imagine = {}

        if self.fh > 0 and "action" in batch_fh and batch_fh["action"].shape[1] > 0:
            last_obs_action = batch_rf["action"][:, -1:]
            future_actions = torch.cat([last_obs_action, batch_fh["action"][:, :-1]], dim=1)

            state_imagine = {
                # imagine()는 시간축 없는 3D (B, C, N)을 기대한다.
                # [:, -1:]은 (B, 1, C, N) 4D를 남기므로 [:, -1]로 시간축을 떨군다.
                "hidden_state": state_dict["posterior"]["hidden_state"][:, -1],
                "sample": state_dict["posterior"]["sample"][:, -1],
                "action": future_actions,
            }

            outs, lss = [], []

            for _ in range(n_samples):
              oi, _ = self.model.imagine(state_imagine, future_horizon=self.fh)
              fl    = self.compute_loss(batch_fh, oi, include_kl=False,prefix="future_")
              outs.append(oi)
              lss.append(fl)

            # 손실은 N샘플 평균
            losses_imagine = {
            k: torch.stack([l[k] for l in lss]).mean() for k in lss[0]
              }
            losses.update(losses_imagine)
            # 시각화/하위 호환: 첫 샘플 반환
            output_imagine = outs[0]

        return losses, output, state_dict, losses_imagine, output_imagine


    # ─── Training ────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        batch = self.prepare_custom_batch(batch)
        losses, output, state_dict = self._observe_full(batch)

        self.log_dict(
          {f"train/{k}": v for k, v in losses.items()},
          on_step=True, on_epoch=False, prog_bar=False)
        return self.loss_reducing(losses)

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
                    "lidar_pred": output.get("lidar_reconstruction_1", torch.zeros(0)).detach().float().cpu().numpy(),
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
    """매 N epoch마다 `save_reconstruction_figure`를 호출해 학습 도중 figure를 dump.

    overfit 검증용. dataset과 sample_idx를 미리 받아두고 epoch end마다 그 한 sample을
    decode → figure/<run_name>_reconstruction_s<idx>.png 에 매번 덮어쓰기(하나만 유지).

    save 전후로 model의 train/eval 모드를 복원해 학습 흐름이 깨지지 않도록 함.
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
        # 에폭마다 새 파일을 쌓지 않고 같은 파일에 덮어써서 결과 사진을 하나만 유지한다.
        # (학습 끝의 최종 figure와 파일명이 같아 자연히 한 장으로 합쳐진다.)
        try:
            save_reconstruction_figure(
                pl_module, self.dataset, self.run_cfg,
                run_name=self.run_name,
                sample_idx=self.sample_idx,
            )
            # Trajectory(TransFuser waypoint) 오버레이도 함께 저장.
            # waypoint head 비활성이면 내부에서 안전하게 skip.
            save_trajectory_overlay_figure(
                pl_module, self.dataset, self.run_cfg,
                run_name=self.run_name,
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
            "val/RL_lidar_chamfer_xyz_nearfield", "val/DS_lidar_chamfer_xyz_nearfield",
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
