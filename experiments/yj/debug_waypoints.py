"""debug_waypoints.py — 학습 없이 체크포인트 추론만 돌려서
waypoint(pred/GT) 값과 카메라 투영 결과(픽셀·visible 마스크)를 출력한다.

사용:
    python debug_waypoints.py \
        --ckpt logs/yj_overfit_1window_1000ep_rssmtd/checkpoints/last.ckpt \
        --sample-idx 0
"""

import argparse

import numpy as np
import torch

from config_muvo_2D import cfg
from data_muvo_2D import load_arrow_manifest, _select_from_manifest, MultiArrowStreamDataset
from trainer_muvo_2D import WorldModelTrainer, _waypoints_to_pixels


def build_dataset(cfg):
    manifest = load_arrow_manifest()
    seq_len = cfg.RECEPTIVE_FIELD + cfg.FUTURE_HORIZON
    stride = cfg.RECEPTIVE_FIELD * cfg.DATA.SAMPLE_EVERY_N
    train_files = _select_from_manifest(
        manifest, "train",
        preferred=cfg.DATA.TRAIN_RUN,
        use_all=cfg.DATA.USE_ALL_TRAIN_RUNS,
    )
    return MultiArrowStreamDataset(
        train_files,
        seq_len=seq_len,
        stride=stride,
        num_streams=1,
        sample_every_n=cfg.DATA.SAMPLE_EVERY_N,
        image_input_size=cfg.DATA.IMAGE_INPUT_SIZE,
        rgb_recon_size=cfg.DATA.RGB_RECON_SIZE,
        lidar_size=cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
        frame_step=cfg.DATA.FRAME_STEP,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sample-idx", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = build_dataset(cfg)
    sample_idx = min(args.sample_idx, len(ds) - 1)
    sample = ds[sample_idx]

    model = WorldModelTrainer.load_from_checkpoint(
        args.ckpt, cfg=cfg, lr=cfg.OPTIMIZER.LR,
    ).to(device).eval()

    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
    batch = model.prepare_custom_batch(batch)

    with torch.no_grad():
        output, _ = model.model.forward(batch)

    if "waypoints_pred" not in output:
        print("waypoints_pred 없음 — cfg.MODEL.WAYPOINT.ENABLED 확인.")
        return

    rf = cfg.RECEPTIVE_FIELD
    t_ref = rf - 1

    pred_wp = output["waypoints_pred"][0, t_ref].detach().float().cpu().numpy()
    gt_wp_all, mask = model.compute_gt_waypoints(batch["action"], batch["speed"])
    gt_wp = gt_wp_all[0, t_ref].detach().float().cpu().numpy()

    img = batch["image_raw"][0, t_ref]
    H, W = img.shape[-2], img.shape[-1]
    cam_cfg = getattr(cfg.DATA, "CAMERA", None)
    y_right = bool(getattr(getattr(cfg.MODEL, "WAYPOINT", object()), "Y_RIGHT_POSITIVE", True))

    # 화면 안에 들어오려면 필요한 최소 전방거리 (지면 점 기준, pitch=0 가정)
    f = W / (2.0 * np.tan(np.deg2rad(float(getattr(cam_cfg, "FOV_DEG", 90.0))) / 2.0))
    h_cam = float(getattr(cam_cfg, "HEIGHT_M", 1.6))
    x_off = float(getattr(cam_cfg, "X_OFFSET_M", 1.2))
    z_min_in_frame = f * h_cam / (H / 2.0)  # v=H 가 되는 Z

    print(f"\n=== sample_idx={sample_idx}  t_ref={t_ref}  image HxW={H}x{W} ===")
    print(f"camera: FOV=90 f={f:.1f}px  H_cam={h_cam}m  X_OFFSET={x_off}m")
    print(f"  → visible 필터 통과 조건: 전방거리 Z > {x_off}m")
    print(f"  → 프레임 안(아래끝)에 들어올 조건: Z >= {z_min_in_frame:.2f}m\n")

    speed = batch["speed"][0].detach().float().cpu().numpy()
    print(f"speed[km/h] (전체 시퀀스): {np.round(speed, 2)}")
    print(f"gt mask[t_ref]={float(mask[0, t_ref]):.0f}\n")

    for name, wp, color in [("GT ", gt_wp, "lime"), ("pred", pred_wp, "red")]:
        px, vis = _waypoints_to_pixels(wp, (H, W), cam_cfg, y_right_positive=y_right)
        print(f"[{name}] ego waypoints [x_fwd, y_lat] (m):")
        for i in range(len(wp)):
            u, vv = px[i]
            visible = bool(vis[i])
            in_frame = visible and (0 <= u < W) and (0 <= vv < H)
            print(f"   wp{i}: x_fwd={wp[i, 0]:+.3f}  y_lat={wp[i, 1]:+.3f}  "
                  f"→ px=({u:7.1f},{vv:7.1f})  visible={visible}  in_frame={bool(in_frame)}")
        n_draw = int((vis & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)).sum())
        print(f"   → 실제로 화면에 그려질 점 수: {n_draw}\n")


if __name__ == "__main__":
    main()
