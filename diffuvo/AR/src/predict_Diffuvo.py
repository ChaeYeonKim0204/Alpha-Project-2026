"""predict_Diffuvo.py — Diffuvo_WM 추론·시각화 스크립트 (계획서 §5·§6·§7).

학습된 checkpoint(AE encoder/decoder + latent diffusion)를 로드해
val held-out 샘플에 대해 미래를 생성하고 실험 1/2/3 figure를 저장한다.

실험 (계획서 §6):
  1) Action intervention  → intervention_s<idx>.png  (past·speed·seed 고정, action 변경)
  2) Seed diversity        → seed_diversity_s<idx>.png (past·speed·action 고정, seed 변경)
  3) Mean / Variance       → mean_var_s<idx>.png       (action별 다수 seed의 mean/var/error)

저장 위치: `train_Diffuvo.resolve_figure_dir` 재사용 → `figures/<run_name>/` (계획서 §6.0).

의존 주의 (계획서 §8 미완 항목):
  - §8-2 `Model.decode_rgb`가 아직 없어 본 스크립트가 `_slice_token_state`+`image_decoder`로
    RGB 디코드를 폴백 구현한다. Model에 `decode_rgb`가 생기면 자동으로 그쪽을 쓴다.
  - §8-4 trainer의 diffusion 배선 전까지는 통합 checkpoint가 없으므로
    `--ae-checkpoint` / `--diffusion-checkpoint`를 따로 받거나, prefix(`model.`/`diffusion.`)가
    섞인 통합 checkpoint(`--checkpoint`)를 라우팅해 로드한다.
"""
import argparse
import math
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_Diffuvo import cfg
from data_Diffuvo import (
    load_arrow_manifest,
    _select_from_manifest,
    MultiArrowStreamDataset,
)
from models_Diffuvo import Model
from diffusion_Diffuvo import LatentARDiffusion
from train_Diffuvo import resolve_figure_dir


# 계획서 §6 Counterfactual action 프리셋. throttle="cur"는 관측 마지막 throttle 유지.
# cfg.DIFFUSION.ACTION_PRESETS가 있으면 그것을 우선 사용(§8-1).
_DEFAULT_PRESETS = [
    ("straight", "cur", 0.0),
    ("left",     "cur", -0.5),
    ("right",    "cur", +0.5),
    ("brake",    0.0,   0.0),
]
ACTION_PRESETS = [
    tuple(p) for p in getattr(getattr(cfg, "DIFFUSION", None), "ACTION_PRESETS", _DEFAULT_PRESETS)
]


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Diffuvo_WM future prediction & visualization")
    # checkpoint 소스 (통합 1개 또는 stage별 2개).
    p.add_argument("--checkpoint", default=None,
                   help="통합 checkpoint (model.* / diffusion.* prefix로 라우팅).")
    p.add_argument("--ae-checkpoint", default=None,
                   help="Stage1 AE checkpoint (encoder/decoder). model.* prefix 자동 strip.")
    p.add_argument("--diffusion-checkpoint", default=None,
                   help="Stage2 diffusion checkpoint. diffusion.* prefix 자동 strip.")
    p.add_argument("--allow-untrained", action="store_true",
                   help="checkpoint 없이 random weight로 figure 파이프라인만 점검(결과는 무의미).")

    # 추론 노브 (계획서 §7).
    p.add_argument("--ddim-steps", type=int, required=True,
                   help="DDIM denoise 스텝 수. **매 run 명시 필요**(default 강제 안 함).")
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="action classifier-free guidance 강도(1.0=guidance off).")
    p.add_argument("--n-seeds", type=int, default=4, help="실험2·3 seed 개수.")
    p.add_argument("--seeds", type=str, default=None,
                   help="쉼표구분 seed 목록(예: 0,1,2,3). 주면 --n-seeds 무시.")

    # 샘플 / 출력.
    p.add_argument("--split", choices=["RL", "DS", "train"], default="RL",
                   help="RL/DS=val held-out. train=학습 overfit subset 재현(앞 N window).")
    p.add_argument("--num-streams", type=int, default=2,
                   help="--split train에서 윈도잉 재현용. 학습 global_batch_size와 같아야 "
                        "sample 인덱스가 diffusion이 본 window와 일치(기본 2).")
    p.add_argument("--sample-indices", type=str, default="0,1,2",
                   help="시각화할 val 샘플 인덱스(쉼표구분). 계획서 기본 = held-out 3개.")
    p.add_argument("--exp3-action", default="straight",
                   choices=[name for name, _, _ in ACTION_PRESETS] + ["all"],
                   help="실험3에서 mean/var를 낼 action(기본 straight, all이면 전체 프리셋).")
    p.add_argument("--run-name", default=None,
                   help="figure 폴더 이름. 미지정 시 cfg.LOGGING.RUN_NAME.")
    p.add_argument("--figure-dir", default=os.environ.get("ALPHA26_FIGURE_DIR"),
                   help="figures base 경로 override(그 아래 <run_name>/ 자동 생성).")
    return p.parse_args()


# ─────────────────────────────────────────────
# Dataset (val held-out)
# ─────────────────────────────────────────────
def build_val_dataset(run_cfg, split="RL"):
    """train_Diffuvo.build_dataloaders의 val 경로만 복제(동일 전처리, augmentation off)."""
    manifest = load_arrow_manifest()
    preferred = run_cfg.DATA.VAL_RL_RUN if split == "RL" else run_cfg.DATA.VAL_DS_RUN
    files = _select_from_manifest(manifest, "validation", preferred=preferred, use_all=False)
    seq_len = run_cfg.RECEPTIVE_FIELD + run_cfg.FUTURE_HORIZON
    stride = run_cfg.RECEPTIVE_FIELD * run_cfg.DATA.SAMPLE_EVERY_N
    ds = MultiArrowStreamDataset(
        files,
        seq_len=seq_len,
        stride=stride,
        num_streams=1,
        sample_every_n=run_cfg.DATA.SAMPLE_EVERY_N,
        image_input_size=run_cfg.DATA.IMAGE_INPUT_SIZE,
        rgb_recon_size=run_cfg.DATA.RGB_RECON_SIZE,
        lidar_size=run_cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
        lidar_fov=run_cfg.DATA.LIDAR_FOV_DEGREES,
        lidar_scale=run_cfg.DATA.LIDAR_SCALE,
        frame_step=run_cfg.DATA.FRAME_STEP,
        augment=False,
    )
    return ds


def build_train_dataset(run_cfg, num_streams=2):
    """학습 overfit subset 재현. train_Diffuvo.build_dataloaders의 train 경로와 동일한
    파일 선택·윈도잉을 써서 sample 인덱스가 학습이 본 window와 일치하도록 한다.

    num_streams는 학습의 global_batch_size와 같아야 윈도 순서가 동일(기본 2).
    """
    manifest = load_arrow_manifest()
    files = _select_from_manifest(
        manifest, "train",
        preferred=run_cfg.DATA.TRAIN_RUN,
        use_all=run_cfg.DATA.USE_ALL_TRAIN_RUNS,
    )
    seq_len = run_cfg.RECEPTIVE_FIELD + run_cfg.FUTURE_HORIZON
    stride = run_cfg.RECEPTIVE_FIELD * run_cfg.DATA.SAMPLE_EVERY_N
    ds = MultiArrowStreamDataset(
        files,
        seq_len=seq_len,
        stride=stride,
        num_streams=num_streams,
        sample_every_n=run_cfg.DATA.SAMPLE_EVERY_N,
        image_input_size=run_cfg.DATA.IMAGE_INPUT_SIZE,
        rgb_recon_size=run_cfg.DATA.RGB_RECON_SIZE,
        lidar_size=run_cfg.DATA.LIDAR_RANGE_VIEW_SIZE,
        lidar_fov=run_cfg.DATA.LIDAR_FOV_DEGREES,
        lidar_scale=run_cfg.DATA.LIDAR_SCALE,
        frame_step=run_cfg.DATA.FRAME_STEP,
        augment=False,
    )
    return ds


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def _strip_prefix(state, prefix):
    out = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _load_ckpt(path):
    ck = torch.load(path, map_location="cpu")
    return ck.get("state_dict", ck)


def load_models(run_cfg, args, device):
    model = Model(run_cfg).to(device).eval()
    diffusion = LatentARDiffusion.from_cfg(run_cfg).to(device).eval()

    loaded_any = False
    if args.checkpoint:
        state = _load_ckpt(args.checkpoint)
        m_state = _strip_prefix(state, "model.")
        d_state = _strip_prefix(state, "diffusion.")
        if m_state:
            mi, mu = model.load_state_dict(m_state, strict=False)
            print(f"[checkpoint] model: loaded missing={len(mi)} unexpected={len(mu)}")
            loaded_any = True
        if d_state:
            di, du = diffusion.load_state_dict(d_state, strict=False)
            print(f"[checkpoint] diffusion: loaded missing={len(di)} unexpected={len(du)}")
            loaded_any = True
    if args.ae_checkpoint:
        state = _load_ckpt(args.ae_checkpoint)
        state = _strip_prefix(state, "model.") or state
        mi, mu = model.load_state_dict(state, strict=False)
        print(f"[ae-checkpoint] model: missing={len(mi)} unexpected={len(mu)}")
        loaded_any = True
    if args.diffusion_checkpoint:
        state = _load_ckpt(args.diffusion_checkpoint)
        state = _strip_prefix(state, "diffusion.") or state
        di, du = diffusion.load_state_dict(state, strict=False)
        print(f"[diffusion-checkpoint] diffusion: missing={len(di)} unexpected={len(du)}")
        loaded_any = True

    if not loaded_any:
        if not args.allow_untrained:
            raise SystemExit(
                "checkpoint가 없습니다. --checkpoint / --ae-checkpoint / --diffusion-checkpoint 중 "
                "하나를 지정하거나, 파이프라인 점검용이면 --allow-untrained를 주세요."
            )
        print("[warn] checkpoint 없이 random weight로 실행 — 결과 이미지는 무의미(파이프라인 점검용).")
    return model, diffusion


# ─────────────────────────────────────────────
# Encode / decode helpers
# ─────────────────────────────────────────────
def decode_rgb(model, z):
    """(B,S,C,N) latent → rgb_1 (B,S,3,H,W). Model.decode_rgb 있으면 그것 사용(§8-2)."""
    if hasattr(model, "decode_rgb"):
        out = model.decode_rgb(z)
        return out["rgb_1"] if isinstance(out, dict) else out
    img_tokens, _ = model._slice_token_state(z)        # (B,S,C,H_img,W_img)
    return model.image_decoder(img_tokens)["rgb_1"]


@torch.no_grad()
def encode_full(model, sample, device, rf, fh):
    """RF+FH GT 프레임 전체 encode → z (1, RF+FH, C, N).

    z_future_clean(= teacher latent) 추출용. latentMSE(sample,teacher) 지표 계산에 쓰인다.
    (ceiling 디코드 그림은 제거됨)
    """
    n = rf + fh
    image = sample["image"][:n].unsqueeze(0).float().to(device)
    lidar = sample["lidar"][:n].unsqueeze(0).float().to(device)
    speed = sample["speed"][:n].unsqueeze(0).float().to(device)
    return model.encode_fuse_sequence(image, lidar, speed=speed)     # (1,n,C,N)


def _psnr(a, b):
    """a,b: (...,3,H,W) in [0,1]. 평균 PSNR(dB)."""
    mse = torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


@torch.no_grad()
def encode_past(model, sample, device, rf):
    """sample dict(S,...) → z_past (1,RF,C,N) + 관측 컨텍스트(GT/speed/action)."""
    image = sample["image"][:rf].unsqueeze(0).float().to(device)
    lidar = sample["lidar"][:rf].unsqueeze(0).float().to(device)
    speed = sample["speed"][:rf].unsqueeze(0).float().to(device)
    z_past = model.encode_fuse_sequence(image, lidar, speed=speed)   # (1,RF,C,N)
    last_speed = sample["speed"][rf - 1].reshape(1).float().to(device)
    last_action = sample["action"][rf - 1].float().to(device)        # (2,)
    return z_past, last_speed, last_action


def make_action(preset, last_action, device):
    """프리셋 (throttle, steer) → (1,2) action 텐서. throttle='cur'면 관측값 유지."""
    _, thr, steer = preset
    throttle = float(last_action[0].item()) if thr == "cur" else float(thr)
    return torch.tensor([[throttle, float(steer)]], dtype=torch.float32, device=device)


def to_hwc(t):
    """(3,H,W) tensor → (H,W,3) numpy in [0,1]."""
    return t.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def to_gray(t):
    """(3,H,W) or (H,W) → (H,W) numpy (채널 평균)."""
    t = t.detach().float().cpu()
    if t.ndim == 3:
        t = t.mean(0)
    return t.numpy()


# ─────────────────────────────────────────────
# Grid plotting
# ─────────────────────────────────────────────
def save_grid(rows, ncols, path, title=None):
    """rows: list of dict(label, imgs[list], kind in {'rgb','heat'}, vmax).

    imgs는 ncols보다 짧으면 왼쪽 정렬 후 나머지 칸은 비움.
    """
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False)
    for r, row in enumerate(rows):
        axes[r, 0].set_ylabel(row["label"], rotation=0, ha="right", va="center", fontsize=9)
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(row["imgs"]):
                img = row["imgs"][c]
                if row["kind"] == "rgb":
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap="magma", vmin=0, vmax=row.get("vmax", None))
                if r == 0 and row.get("col_titles"):
                    ax.set_title(row["col_titles"][c], fontsize=8)
            else:
                ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    size = os.path.getsize(path)
    print(f"saved {path} ({size} bytes)")


# ─────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────
@torch.no_grad()
def sample_future_latent(diffusion, z_past, action, speed, seed, args, device):
    """단일 (action, seed) → 미래 latent z_hat (1,FH,C,N)."""
    g = torch.Generator(device=device).manual_seed(int(seed))
    return diffusion.sample(z_past, action, speed, n_steps=args.ddim_steps,
                            cfg_scale=args.cfg_scale, generator=g)


@torch.no_grad()
def sample_future_rgb(model, diffusion, z_past, action, speed, seed, args, device):
    """단일 (action, seed) → 미래 RGB (FH,3,H,W)."""
    z_hat = sample_future_latent(diffusion, z_past, action, speed, seed, args, device)
    return decode_rgb(model, z_hat)[0]                 # (FH,3,H,W)


def observed_rows(model, sample, z_past, rf, fh):
    """관측 GT / 관측 recon / 미래 GT row 묶음(모든 실험 공통 상단)."""
    obs_gt = [to_hwc(sample["image_raw"][t]) for t in range(rf)]
    recon = decode_rgb(model, z_past)[0]               # (RF,3,H,W)
    obs_recon = [to_hwc(recon[t]) for t in range(rf)]
    fut_gt = [to_hwc(sample["image_raw"][rf + t]) for t in range(fh)]
    return obs_gt, obs_recon, fut_gt


def exp1_intervention(model, diffusion, sample, z_past, speed, last_action, args, device, rf, fh, path):
    obs_gt, obs_recon, fut_gt = observed_rows(model, sample, z_past, rf, fh)

    # teacher latent(GT future) — latent MSE 지표용. (ceiling 디코드 그림은 제거)
    z_future_clean = encode_full(model, sample, device, rf, fh)[:, rf:rf + fh]   # (1,FH,C,N)
    fut_gt_t = sample["image_raw"][rf:rf + fh].float().to(device)               # (FH,3,H,W)

    rows = [
        {"label": "Observed GT", "imgs": obs_gt, "kind": "rgb",
         "col_titles": [f"obs t={t}" for t in range(rf)]},
        {"label": "Observed Recon", "imgs": obs_recon, "kind": "rgb"},
        {"label": "Future GT", "imgs": fut_gt, "kind": "rgb"},
    ]
    seed = 0  # 실험1은 seed 고정.
    metrics = None
    na_metrics = None

    # action을 주지 않은 순수 미래 예측(null-action, diffusion_Diffuvo.py:285).
    # cond==uncond라 CFG는 무효이므로 cfg_scale=1.0(off)로 호출한다. action 조건 행들과
    # 비교하는 baseline — "모델이 action 없이 자유롭게 그리는 미래".
    g_na = torch.Generator(device=device).manual_seed(seed)
    z_hat_na = diffusion.sample(z_past, None, speed, n_steps=args.ddim_steps,
                                cfg_scale=1.0, generator=g_na)
    fut_na = decode_rgb(model, z_hat_na)[0]                                  # (FH,3,H,W)
    rows.append({"label": "no-action", "imgs": [to_hwc(fut_na[t]) for t in range(fh)], "kind": "rgb"})
    na_metrics = (
        torch.mean((z_hat_na - z_future_clean) ** 2).item(),
        _psnr(fut_na, fut_gt_t),
    )

    for preset in ACTION_PRESETS:
        action = make_action(preset, last_action, device)
        z_hat = sample_future_latent(diffusion, z_past, action, speed, seed, args, device)
        fut = decode_rgb(model, z_hat)[0]                                       # (FH,3,H,W)
        rows.append({"label": f"act={preset[0]}", "imgs": [to_hwc(fut[t]) for t in range(fh)], "kind": "rgb"})
        if preset[0] == "straight":
            # 오버핏 판정 지표: 샘플 latent가 teacher latent에 얼마나 붙었나 + RGB PSNR.
            metrics = (
                torch.mean((z_hat - z_future_clean) ** 2).item(),   # latent MSE
                _psnr(fut, fut_gt_t),                               # sample vs GT
            )
    title = "Exp1: Action intervention (seed fixed) + no-action baseline"
    if metrics is not None:
        title += (f"\nstraight: latentMSE(sample,teacher)={metrics[0]:.4f}  "
                  f"PSNR sample/GT={metrics[1]:.1f}dB")
    if na_metrics is not None:
        title += (f"\nno-action: latentMSE={na_metrics[0]:.4f}  "
                  f"PSNR sample/GT={na_metrics[1]:.1f}dB")
    if metrics is not None:
        print(f"[metrics] {os.path.basename(path)}: straight latentMSE={metrics[0]:.4f} "
              f"PSNR={metrics[1]:.2f} | no-action latentMSE={na_metrics[0]:.4f} "
              f"PSNR={na_metrics[1]:.2f}")
    save_grid(rows, max(rf, fh), path, title=title)


def exp2_seed_diversity(model, diffusion, sample, z_past, speed, last_action, seeds, args, device, rf, fh, path):
    obs_gt, obs_recon, fut_gt = observed_rows(model, sample, z_past, rf, fh)
    action = make_action(("straight", "cur", 0.0), last_action, device)  # action 고정(직진).
    rows = [
        {"label": "Observed GT", "imgs": obs_gt, "kind": "rgb",
         "col_titles": [f"obs t={t}" for t in range(rf)]},
        {"label": "Observed Recon", "imgs": obs_recon, "kind": "rgb"},
        {"label": "Future GT", "imgs": fut_gt, "kind": "rgb"},
    ]
    for s in seeds:
        fut = sample_future_rgb(model, diffusion, z_past, action, speed, s, args, device)
        rows.append({"label": f"seed={s}", "imgs": [to_hwc(fut[t]) for t in range(fh)], "kind": "rgb"})
    save_grid(rows, max(rf, fh), path, title="Exp2: Seed diversity (action=straight)")


def exp3_mean_var(model, diffusion, sample, z_past, speed, last_action, seeds, args, device, rf, fh, path):
    obs_gt, obs_recon, fut_gt = observed_rows(model, sample, z_past, rf, fh)
    rows = [{"label": "Future GT", "imgs": fut_gt, "kind": "rgb",
             "col_titles": [f"fut t={t}" for t in range(fh)]}]

    if args.exp3_action == "all":
        presets = ACTION_PRESETS
    else:
        presets = [p for p in ACTION_PRESETS if p[0] == args.exp3_action]

    for preset in presets:
        action = make_action(preset, last_action, device)
        # seeds × (FH,3,H,W) 스택.
        samples = torch.stack([
            sample_future_rgb(model, diffusion, z_past, action, speed, s, args, device)
            for s in seeds
        ], dim=0)                                       # (K,FH,3,H,W)
        mean = samples.mean(0)                          # (FH,3,H,W)
        var = samples.var(0).mean(1)                    # (FH,H,W) 채널평균 분산
        gt = sample["image_raw"][rf:rf + fh].to(mean.device)
        err = (mean - gt).abs().mean(1)                 # (FH,H,W)
        vmax_var = float(var.max().clamp(min=1e-6))
        vmax_err = float(err.max().clamp(min=1e-6))
        rows.append({"label": f"{preset[0]} Mean", "imgs": [to_hwc(mean[t]) for t in range(fh)], "kind": "rgb"})
        rows.append({"label": f"{preset[0]} Var", "imgs": [to_gray(var[t]) for t in range(fh)], "kind": "heat", "vmax": vmax_var})
        rows.append({"label": f"{preset[0]} |mean-GT|", "imgs": [to_gray(err[t]) for t in range(fh)], "kind": "heat", "vmax": vmax_err})
    save_grid(rows, fh, path, title=f"Exp3: Mean/Variance/Error ({len(seeds)} seeds)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    run_cfg = cfg
    if args.run_name:
        run_cfg.LOGGING.RUN_NAME = args.run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rf, fh = run_cfg.RECEPTIVE_FIELD, run_cfg.FUTURE_HORIZON

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seeds = list(range(args.n_seeds))
    sample_indices = [int(s) for s in args.sample_indices.split(",") if s.strip() != ""]

    fig_dir = resolve_figure_dir(args, run_cfg)
    print(f"device={device} run_name={run_cfg.LOGGING.RUN_NAME}")
    print(f"figure_dir={fig_dir}")
    print(f"ddim_steps={args.ddim_steps} cfg_scale={args.cfg_scale} seeds={seeds}")

    model, diffusion = load_models(run_cfg, args, device)
    if args.split == "train":
        ds = build_train_dataset(run_cfg, num_streams=args.num_streams)
    else:
        ds = build_val_dataset(run_cfg, split=args.split)
    print(f"{args.split} size={len(ds)} samples; visualizing indices={sample_indices}")

    for idx in sample_indices:
        if idx >= len(ds):
            print(f"[skip] sample idx {idx} >= dataset size {len(ds)}")
            continue
        sample = ds[idx]
        z_past, speed, last_action = encode_past(model, sample, device, rf)
        suffix = f"_s{idx}"
        exp1_intervention(model, diffusion, sample, z_past, speed, last_action, args, device, rf, fh,
                          os.path.join(fig_dir, f"intervention{suffix}.png"))
        exp2_seed_diversity(model, diffusion, sample, z_past, speed, last_action, seeds, args, device, rf, fh,
                            os.path.join(fig_dir, f"seed_diversity{suffix}.png"))
        exp3_mean_var(model, diffusion, sample, z_past, speed, last_action, seeds, args, device, rf, fh,
                      os.path.join(fig_dir, f"mean_var{suffix}.png"))


if __name__ == "__main__":
    main()
