# 실험: wp_baseline_muvo2d — `weight_probabilistic`(KL 가중치) 스윕

- **날짜:** 2026-05-31
- **목표:** future(미래 예측) 이미지 품질(`future_rgb_psnr`) 개선
- **변경 파일:** `scripts/model_variants/tune_muvo_2D.py` (스윕 정의 + 승자 자동 저장)
- **관련 문서:** `lr_baseline_muvo2d.md` (직전의 OneCycleLR tail fix)

---

## 1. `weight_probabilistic`(wp)가 뭔가

- 설정 위치: `cfg.LOSSES.WEIGHT_PROBABILISTIC` (`config_muvo_2D.py`, 기본값 **1e-3**).
- CLI 플래그: `--weight-probabilistic`.
- 정체: **KL 손실항의 가중치**입니다. 전체 loss에서 KL 항이 차지하는 세기를 정합니다.

코드에서 소비되는 곳 (`trainer_muvo_2D.py`):

```python
total += self.weight_probabilistic * self.balanced_kl_loss(prior, posterior)
```

여기서 `balanced_kl_loss`는 balanced KL입니다:

```
balanced_kl = α·KL(posterior.detach ‖ prior)  +  (1−α)·KL(prior.detach ‖ posterior)
              └ prior를 posterior로 끌어당김 ┘     └ posterior를 prior로 끌어당김 ┘
```
- `α = KL_BALANCING_ALPHA = 0.75` (기본값).
- KL 항은 **관측(observe) 구간에서만** 계산됩니다. future(imagine) 구간은 `include_kl=False`라 KL이 없습니다.

---

## 2. wp를 바꾸면 무슨 효과가 있나

핵심: **future RGB는 prior rollout(미래 상상)으로 생성**되고, KL은 그 prior가 posterior(실제 관측 인코딩)를 얼마나 잘 따라가게 할지를 정합니다.

| wp 변화 | 직접 효과 | 부작용 |
|---|---|---|
| wp **↑** | KL 압력 ↑ → prior가 posterior에 더 잘 정렬 → **future RGB 개선 기대** | 너무 높으면 **posterior collapse**(latent 정보 소실) → obs RGB·전체 recon 저하 |
| wp **↓** | recon에 자유도 ↑ (obs RGB 약간 유리) | prior 정렬이 약해 **future RGB 갭 ↑** |

현재 상태 진단:
- obs RGB ~21.5–22 dB는 약 20k step에서 이미 포화 → decoder 쪽 헤드룸 거의 없음.
- future RGB ~15.4–17.4 dB로 **6 dB 갭** → 이건 decoder가 아니라 **prior(transition) 정렬 문제**.
- 현재 wp=1e-3은 MUVO 재현용으로 일부러 낮춘 값(이전 1e-2). future를 살릴 헤드룸이 여기 있음.

---

## 3. 이번 변경 (Round 1: wp 스윕)

`tune_muvo_2D.py`의 `TRIALS`를 **wp만 한 knob씩 바꾸는** 구성으로 교체했습니다. baseline(wp=1e-3)은 이미 결과가 있어 재실행하지 않습니다.

```python
TRIALS = [
    ("wp3e3",  {"weight_probabilistic": 3e-3}),
    ("wp1e2",  {"weight_probabilistic": 1e-2}),
]
SHARED = dict(epochs=50, batch_size=6, num_workers=4, devices=4,
              strategy="ddp_find_unused_parameters_true", no_clearml=True)
RUN_PREFIX = "sweep_future_rgb_50ep"
```

스윕(50ep screening)이 끝나면 `experiment_log.csv`에서 **`val/RL_future_rgb_psnr`·`val/DS_future_rgb_psnr` 평균이 가장 높은 trial**을 승자로 골라
`<ALPHA26_OUTPUT_ROOT>/experiments/sweep_future_rgb_50ep_winner.md` (예: `/home/user/chaeyeon-kim/alpha26/results/experiments/...`) 에 자동 저장합니다 (승자 설정, future/obs 점수, 전체 랭킹, 나중에 돌릴 200ep 명령어 포함). **이 단계에서 결승 학습은 돌리지 않습니다.**

### 왜 wp를 먼저, alpha(`kl_balancing_alpha`)는 나중에 (Round 2)

prior 학습 세기 ∝ **wp × α** 입니다. wp가 1e-3로 낮은 상태에서 α만 0.75→0.95로 올려도 절대 변화가 작아 **α 효과가 가려질 수 있습니다.** 그래서:

```
Round 1: wp 스윕 (1e-3 baseline ↔ 3e-3, 1e-2) → 이긴 wp 확정   ← 지금
Round 2: 이긴 wp를 고정하고 α(0.8 / 0.9 / 0.95) 스윕
최종    : 이긴 (wp, α)로 200ep 따로 실행
```

Round 2 trial 템플릿은 `tune_muvo_2D.py`에 주석으로 적어뒀습니다. Round 1 승자 wp를 박아 활성화한 뒤 다시 실행하면 됩니다.

---

## 4. 실행 명령 (서버)

```bash
cd ~/chaeyeon-kim/alpha26
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kcy-alpha
export CARLA_ARROW_ROOT=~/chaeyeon-kim/processed
export ALPHA26_OUTPUT_ROOT=~/chaeyeon-kim/alpha26/results

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 \
python scripts/tune_muvo_2D.py
```

(스윕은 trial을 4 GPU로 **순차** 실행. wp 2개 × 50ep ≈ 5시간.)

---

## 5. 결과에서 볼 것

- **타깃:** `val/RL_future_rgb_psnr`, `val/DS_future_rgb_psnr` — wp를 올렸을 때 올라가는가.
- **가드레일:** `val/RL_rgb_psnr`, `val/DS_rgb_psnr` — obs RGB가 무너지지 않는가(posterior collapse 신호).
- **KL 건강성:** `train_probabilistic` / `val/*_probabilistic` — 0으로 붕괴하지 않는가.

판정:
- future ↑ & obs 유지 → wp 인상 성공 → 승자 wp로 Round 2(α) 진행.
- future ↑ but obs 큰 폭 하락 → 과한 정규화 → free-bits(`--kl-free-bits-enabled --kl-free-bits 1.0`)로 복구 고려.
- future 변화 없음 → KL 정렬 문제 아님 → RSSM 용량(hidden/state dim) 쪽으로.
