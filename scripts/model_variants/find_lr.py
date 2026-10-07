import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # 서버 환경 (디스플레이 없음) 대응
import matplotlib.pyplot as plt
from world_model.configs.default import cfg as default_cfg
from world_model.train import build_dataloaders
from world_model.training.lightning_module import WorldModelTrainer

def lr_range_test(
    base_cfg,
    start_lr: float        = 1e-6,
    end_lr: float          = 1e-1,
    num_steps: int         = 500,
    batch_size: int        = 2,
    smooth_window: int     = 10,
    diverge_factor: float  = 4.0,
    output_dir: str        = "./lr_search_results",
):
    """
    Parameters
    ----------
    base_cfg       : configs/default.py 의 cfg
    start_lr       : 탐색 시작 lr
    end_lr         : 탐색 끝 lr
    num_steps      : 탐색 step 수
    batch_size     : DataLoader 배치 크기
    smooth_window  : loss smoothing 이동 평균 윈도우
    diverge_factor : 초반 loss 평균의 N배 초과 시 발산으로 간주하고 조기 종료
    output_dir     : 그래프 / 결과 저장 경로

    Returns
    -------
    lrs     : 각 step의 lr 리스트
    losses  : 각 step의 loss 리스트
    best_lr : 추천 max_lr (loss 기울기가 가장 가파른 지점)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 모델 준비 ────────────────────────────────────────
    exp_cfg = deepcopy(base_cfg)
    exp_cfg.SCHEDULER.NAME = "none"   # OneCycleLR 비활성화
    exp_cfg.STEPS = num_steps

    
    # ── 데이터 로더 ──────────────────────────────────────
    print("[find_lr] 데이터 로더 생성 중...")
    train_loader, *_ = build_dataloaders(exp_cfg, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[find_lr] device: {device}")

    model = WorldModelTrainer(
        cfg=exp_cfg, lr=start_lr, embedding_n_channels=128
    ).to(device)
    model.train()

    model.log = lambda *args, **kwargs: None

    model._tbptt_h        = None
    model._tbptt_s        = None
    model._tbptt_action   = None
    model._prev_run_id    = None
    model._prev_start_row = None

    # ── 옵티마이저 + ExponentialLR ───────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=start_lr,
        weight_decay=base_cfg.OPTIMIZER.WEIGHT_DECAY,
    )
    # num_steps 동안 start_lr → end_lr에 도달하도록 gamma 계산
    gamma = (end_lr / start_lr) ** (1.0 / num_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    print(f"[find_lr] start_lr={start_lr:.1e}  end_lr={end_lr:.1e}  "
          f"num_steps={num_steps}  gamma={gamma:.6f}")
    print("-" * 55)

    # ── 학습 루프 ────────────────────────────────────────
    lrs: list    = []
    losses: list = []
    data_iter    = iter(train_loader)

    for step in range(num_steps):

        # DataLoader 소진 시 재시작
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # 텐서를 device로 이동
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        batch     = model.prepare_custom_batch(batch)
        loss_dict, *_ = model._observe_and_imagine(batch)
        loss      = model.loss_reducing(loss_dict)

        current_lr = scheduler.get_last_lr()[0]["lr"]
        optimizer.zero_grad()
        loss.backward()
        
        # gradient clipping: 탐색 중 gradient explosion 방지
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        lrs.append(current_lr)
        losses.append(loss.item())

        if step % 50 == 0:
            print(f"  step={step:4d}  lr={current_lr:.3e}  loss={loss.item():.4f}")

        # 발산 감지: 초반 20 step 평균의 diverge_factor 배 초과 시 조기 종료
        if step > 20 and not np.isfinite(loss.item()):
            print(f"\n[find_lr] NaN/Inf 감지 → 조기 종료 (step={step})")
            break
        if step > 20 and loss.item() > diverge_factor * np.mean(losses[:20]):
            print(f"\n[find_lr] 발산 감지 → 조기 종료 "
                  f"(step={step}, lr={current_lr:.3e}, loss={loss.item():.2f})")
            break

    # ── ⑤ 결과 분석 ────────────────────────────────────────
    lrs_arr    = np.array(lrs)
    losses_arr = np.array(losses)

    # 이동 평균 smoothing으로 노이즈 제거
    if len(losses_arr) >= smooth_window:
        kernel       = np.ones(smooth_window) / smooth_window
        smoothed     = np.convolve(losses_arr, kernel, mode="valid")
        smoothed_lrs = lrs_arr[smooth_window - 1:]
    else:
        smoothed     = losses_arr
        smoothed_lrs = lrs_arr

    # loss 기울기가 가장 가파른(가장 음수) 지점 = 가장 빠르게 내려가는 lr
    gradients = np.gradient(smoothed)
    best_idx  = int(np.argmin(gradients))
    best_lr   = float(smoothed_lrs[best_idx])

    # ── ⑥ 그래프 저장 ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: loss vs lr
    axes[0].plot(lrs_arr, losses_arr, alpha=0.3, color="steelblue", label="raw loss")
    if len(smoothed_lrs) > 0:
        axes[0].plot(smoothed_lrs, smoothed, linewidth=2,
                     color="steelblue", label="smoothed")
    axes[0].axvline(best_lr, color="red", linestyle="--",
                    linewidth=1.5, label=f"추천 max_lr = {best_lr:.2e}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("LR Range Test — Loss vs LR")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 오른쪽: gradient vs lr
    if len(smoothed_lrs) > 1:
        axes[1].plot(smoothed_lrs, gradients, color="orange", linewidth=2)
        axes[1].axvline(best_lr, color="red", linestyle="--",
                        linewidth=1.5, label=f"min gradient @ {best_lr:.2e}")
        axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Learning Rate")
        axes[1].set_ylabel("d(Loss)/d(step)")
        axes[1].set_title("Loss Gradient (가장 음수 = 가장 빠르게 내려가는 구간)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "lr_range_test.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[find_lr] 그래프 저장: {plot_path}")

    # ── ⑦ 결과 텍스트 저장 ─────────────────────────────────
    result_path = os.path.join(output_dir, "lr_range_test_result.txt")
    with open(result_path, "w") as f:
        f.write(f"start_lr={start_lr:.2e}\n")
        f.write(f"end_lr={end_lr:.2e}\n")
        f.write(f"num_steps={num_steps}\n")
        f.write(f"best_lr={best_lr:.6e}\n")
        f.write(f"conservative_lr={best_lr / 3:.6e}\n")
        f.write("\nstep,lr,loss\n")
        for i, (lr_v, loss_v) in enumerate(zip(lrs, losses)):
            f.write(f"{i},{lr_v:.6e},{loss_v:.6f}\n")
    print(f"[find_lr] 결과 저장: {result_path}")

    # ── ⑧ 터미널 출력 ──────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"[LR Range Test 결과]")
    print(f"  추천 max_lr      : {best_lr:.3e}")
    print(f"  보수적 적용값    : {best_lr / 3:.3e}  (권장 시작점)")
    print(f"\n  cfg에 반영하려면:")
    print(f"    cfg.OPTIMIZER.LR = {best_lr / 3:.3e}  # 보수적")
    print(f"    cfg.OPTIMIZER.LR = {best_lr:.3e}      # 추천값")
    print(f"{'='*50}\n")

    return lrs, losses, best_lr


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LR Range Test for WorldModel")
    p.add_argument("--start_lr",      type=float, default=1e-6,
                   help="탐색 시작 lr  (기본: 1e-6)")
    p.add_argument("--end_lr",        type=float, default=1e-1,
                   help="탐색 끝 lr   (기본: 1e-1)")
    p.add_argument("--num_steps",     type=int,   default=500,
                   help="탐색 step 수 (기본: 500)")
    p.add_argument("--batch_size",    type=int,   default=2,
                   help="배치 크기    (기본: 2)")
    p.add_argument("--smooth_window", type=int,   default=10,
                   help="smoothing 윈도우 크기 (기본: 10)")
    p.add_argument("--diverge_factor",type=float, default=4.0,
                   help="발산 감지 배수 (기본: 4.0)")
    p.add_argument("--output_dir",    type=str,   default="./lr_search_results",
                   help="결과 저장 경로")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _, _, best_lr = lr_range_test(
        base_cfg       = default_cfg,
        start_lr       = args.start_lr,
        end_lr         = args.end_lr,
        num_steps      = args.num_steps,
        batch_size     = args.batch_size,
        smooth_window  = args.smooth_window,
        diverge_factor = args.diverge_factor,
        output_dir     = args.output_dir,
    )

    print("다음 학습에 적용:")
    print(f"  cfg.OPTIMIZER.LR = {best_lr / 3:.3e}  # 보수적 추천")
    print(f"  cfg.OPTIMIZER.LR = {best_lr:.3e}      # 추천값 그대로")
