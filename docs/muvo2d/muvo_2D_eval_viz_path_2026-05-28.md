# MUVO 2D — evaluation, visualization, path resolution fixes + ep200 baseline 분석 (2026-05-28)

## 요약

`muvo2D_full_muvoaligned_lidar01_h256z256_2hz_bs6x4_ep200` 체크포인트로 baseline 학습을 끝내고
서버 운영 흐름·평가 코드·시각화·결과 저장 경로를 함께 정비했다.

- **upstream muvo_2d** (`/home/carol/chaeyeon-kim/muvo_2d`): 평가 코드 5개 이슈 패치 (#1, #2, #3, #6, #10).
  alpha26 포팅 시 reference로 쓰던 코드가 evaluation 측면에서 깨져 있었던 부분을 따로 고침.
- **alpha26 muvo_2D port** (`scripts/model_variants/*_muvo_2D.py`): 서버 실행 경로 환경변수화, figure label/aspect/vmax 정리.
- **결과 분석**: ep200 baseline은 reconstruction은 수렴, **future prediction 5–7 dB 갭**과 **LiDAR mode collapse**가
  주된 약점. 다음 실험 우선순위: `WEIGHT_LIDAR_RE` 증가 → `WEIGHT_PROBABILISTIC` 조정.

---

## 1. upstream `muvo_2d` 평가 코드 패치

서브에이전트 audit으로 발견한 10개 이슈 중 우선순위 높은 **5개**를 반영. (alpha26 포팅 코드에는 이미 해결돼 있던 항목은 제외.)

| # | 파일:라인 | 내용 | Why |
|---|---|---|---|
| 1 | `muvo/trainer.py:472–489` | LiDAR Chamfer 계산 시 **`depth>0` valid mask** 적용. 이전: random 10000 index sample, mask 없음 → empty ray가 `(0,0,0)` dummy로 chamfer에 섞임. | 미반환 ray가 metric을 낙관적으로 만드는 문제 |
| 2 | `muvo/metrics.py:243–296` (`CDMetric`) | `cf()`의 `try/except` 제거. 빈 point cloud는 `None` 반환 → `add_batch`가 skip + count 안 늘림. `count_in` (near-field) 별도 트래킹. | 빈 prediction에서 silent 0 누적되는 문제 |
| 3 | `muvo/trainer.py:253–258, 1142–1145` | imagine 입력 action slice를 `batch['throttle_brake'][:, rf:]` → **`[:, rf-1:rf-1+fh]`**. validation/test 둘 다. | RSSM transition 첫 step은 obs 마지막 시점 action(`rf-1`)을 써야 함 |
| 6 | `muvo/trainer.py:556–647` | `on_fit_end` 추가. rank-0가 TB scalar 로드 → final PSNR/Chamfer 2-panel figure + per-run summary CSV 저장. | upstream에는 학습 종료 후 자동 figure/CSV export 없었음 |
| 10 | `muvo/metrics.py:354–374` (`PSNRMetric`) | batch-mean의 mean → **frame 단위 sum / count**. | 작은 batch outlier가 PSNR에 과대 영향 |

생략한 issue (#4 BatchNorm train 모드 / #5 limit_val_batches 하드코딩 / #7 test_step n_samples=1 / #8 prediction.py NameError / #9 metrics_vals 길이 3 하드코딩)는 alpha26 포팅 코드에 직접 영향 없거나 운영상 별로 안 중요해서 보류.

---

## 2. alpha26 `scripts/model_variants/` 변경

### 2A. 서버 실행 경로 환경변수화

이전: `PROJECT_ROOT`를 로컬 절대경로 default로 하드코딩, `EXPERIMENT_LOG_PATH`도 `PROJECT_ROOT/results/...` 고정.
서버에서 파일을 `/home/user/chaeyeon-kim/scripts/`에 평평히 두면 `Path(__file__).parents[2]`가 `/home/user`로 잘못 잡혀 결과 저장이 의도 안 한 경로로 흘러간 적이 있음.

| 파일:라인 | 변경 |
|---|---|
| `config_muvo_2D.py:12–27` | `PROJECT_ROOT`를 `Path(__file__).resolve().parents[2]` default + `ALPHA26_ROOT` env override. env가 실제 script 위치와 다르면 warn하고 default 사용. |
| `config_muvo_2D.py:48–56` | `_resolve_experiment_log_path` 헬퍼 추가. `ALPHA26_OUTPUT_ROOT` env 있으면 `<root>/experiment_log.csv`, 없으면 fallback `PROJECT_ROOT/results/experiment_log.csv`. |
| `train_muvo_2D.py:388–398` | `resolve_figure_dir(args)`: `--figure-dir` > `ALPHA26_OUTPUT_ROOT/figures` > `./results/figures`. |
| `train_muvo_2D.py:92–94` | 새 CLI 인자 `--figure-dir`. |
| `train_muvo_2D.py:436–437, 450, 504` | `save_loss_metric_figure` / periodic callback / final `save_reconstruction_figure` 모두 `figure_dir` 전달. |

**서버 실행 표준 환경변수 (option B 레이아웃):**
```bash
cd /home/user/chaeyeon-kim/scripts
export CARLA_ARROW_ROOT=/home/user/chaeyeon-kim/processed
export ALPHA26_OUTPUT_ROOT=/home/user/chaeyeon-kim/alpha26/results
```
결과:
```
/home/user/chaeyeon-kim/
├── processed/                            ← dataset (arrow_manifest.pkl + *.arrow)
├── scripts/
│   ├── *_muvo_2D.py, viz_from_ckpt.py
│   └── logs/<run>/version_N/             ← TB + checkpoints (save_dir="./logs" 상대)
└── alpha26/results/                      ← ALPHA26_OUTPUT_ROOT
    ├── figures/<run>_*.png
    └── experiment_log.csv
```

### 2B. Figure visualization 개선 (`trainer_muvo_2D.py:save_reconstruction_figure`)

| 변경 | 위치 | 이전 → 이후 |
|---|---|---|
| LiDAR title 라벨 | `:180–188, 204–210` | `"GT depth (m)"` → `"GT range-view (m)"` (모든 LiDAR title). 실제로 그리는 게 range-view 이미지의 depth 채널인데 "depth"로 잘못 라벨되어 혼동. |
| LiDAR vmax | `:131–134` | `2.0 * lidar_scale` (=100 m) → **`lidar_scale`** (=50 m). 이전엔 vmax가 sensor max range의 2배라서 ground (1–5 m) returns가 colormap의 1–5%로 매핑되어 거의 검정으로만 보였음. |
| LiDAR aspect | `:179–188, 203–210` | `aspect='auto'` 시도 → **default `aspect='equal'`** 로 되돌림. trainer11 baseline 스타일(자연 비율 얇은 가로 strip) 매칭. |
| `_require_nonempty_file` 헬퍼 | `:84–92` | figure save 후 파일 존재·비어있음 검증 (silent failure 방지). |

### 2C. 기타

- `__pycache__/*.pyc`는 빌드 산출물 — repo에 들어있긴 하지만 commit에는 포함 안 함.

---

## 3. baseline (ep200) 학습 결과 분석

체크포인트: `logs/muvo2D_full_muvoaligned_lidar01_h256z256_2hz_bs6x4_ep200/version_0/checkpoints/last.ckpt`
설정: h=z=256, FPN ch=256, FH=2, 2 Hz, batch 6×4 GPU, 200 epoch.

### 3A. 정량 지표 (TB scalar 추출)

| 메트릭 | 값 | 평가 |
|---|---|---|
| `train_loss_epoch` (최종) | ~0.30 | 수렴 |
| `val/RL_loss`, `val/DS_loss` | ~1.05, ~0.90 | **train 대비 3–3.5×, 약한 overfit** |
| `val/RL_rgb_psnr` (obs recon) | ~21.5 dB | 보통 (좋은 모델: 25+ dB) |
| `val/DS_rgb_psnr` (obs recon) | ~22.0 dB | DS가 RL보다 약간 높음 |
| `val/RL_future_rgb_psnr` | ~15.3 dB | **obs 대비 −6 dB ← future 예측 약함** |
| `val/DS_future_rgb_psnr` | ~17.2 dB | 같은 갭 |
| `val/RL_lidar_chamfer_xyz` | 5.2 → 2.2 m | 절반 이하로 감소, 학습 진행 |
| `val/RL_lidar_chamfer_xyz_nearfield` | 3.7 → 2.7 m | 감소폭 작음 — 가까운 영역이 더 어려움 |
| `val/RL_lidar_xyz_euclidean` | 12.2 → **7.6 m** | sensor max 50 m의 ~15% 오차, 여전히 큼 |
| `val/RL_future_lidar_chamfer_xyz` | 6.2 → 2.75 m | Recon Chamfer와 비슷한 수준까지 — rollout 중 공간 구조는 유지 |
| `train_probabilistic_step` | ~0.035 안정 | KL 발산 없음 |

### 3B. 정성 (reconstruction figure)

**RGB**
- Obs Pred (t=0..3): 건물 윤곽 흐림. 도로·하늘 색감은 대략 맞음.
- Future Pred (t=0, 1): **명확히 hallucinate** — 건물이 녹아내리는 듯한 blur, 도로 가장자리 흐려짐.

**LiDAR range-view**
- GT: 상단(원거리 객체)에 보라 점 흩어짐, 하단(ground)은 거의 검정. 데이터 자체는 정상 (vmax 조정 후 일부 detail 보임).
- Pred: **horizontal banded pattern.** scene-specific detail 거의 없고 row-wise 평균 같은 형태로 collapse.

→ **LiDAR head가 mode collapse 상태.** Chamfer는 줄지만 (좌표 분포는 학습) detail은 학습 안 됨.

---

## 4. 주요 문제점 (다음 실험에서 잡아야 할 것)

1. **LiDAR mode collapse**
   현 가설: `WEIGHT_LIDAR_RE=0.1` vs `WEIGHT_RGB=1.0` 의 10× 차이로 LiDAR가 under-supervise → 평균 패턴에 정착.
2. **Future RGB PSNR이 obs 대비 −6 dB**
   RSSM transition이 약하거나 latent z가 정보 부족.
   `WEIGHT_PROBABILISTIC=1e-3` 으로 KL이 매우 작아(0.035) latent가 거의 N(0,1)로 collapse 가능성.
3. **Mild overfit (train/val 3.5×)**
   현재 augmentation prob 0.3, dropout 0.1. 잡으려면 늘릴 수 있으나 1, 2번이 더 큰 ROI.

---

## 5. 다음 실험 우선순위

| 순서 | 변경 | 현재 → 제안 | 이유 |
|---|---|---|---|
| 🥇 1 | `LOSSES.WEIGHT_LIDAR_RE` | 0.1 → **0.5** (효과 보면 1.0까지) | mode collapse 직접 공격. 가장 명확한 실패 모드 |
| 🥈 2 | `LOSSES.WEIGHT_PROBABILISTIC` | 1e-3 → **5e-3** (보수적) 또는 1e-2 | latent z 정보량 증가 → future 개선 가능. **posterior collapse 주의** |
| 🥉 3 | `LOSSES.KL_FREE_BITS_ENABLED + KL_FREE_BITS` | False → True, 1.0 | (2)의 posterior collapse 부작용 대안 |
| 4 | `OPTIMIZER.LR` | 1e-4 → 5e-5 또는 2e-4 | 마지막에 fine tuning |
| 5 | `DATA.AUGMENTATION.*` 강화 | 현재 prob 0.3 | overfit 살짝 있지만 fatal 아님, 후순위 |

**1과 2를 동시에 바꾸지 말 것** — 효과 분리 가능하도록 한 번에 한 변수씩.

CLI override가 가능하면 매번 config 수정 없이 실험 가능 — 현재 `train_muvo_2D.py:parse_args`에는 `--weight-ssim`만 있고 lidar/probabilistic은 없음. 필요 시 추가 검토.

---

## 6. 관련 파일

- `scripts/model_variants/config_muvo_2D.py` — 환경변수 path 해결, EXPERIMENT_LOG_PATH 헬퍼
- `scripts/model_variants/train_muvo_2D.py` — `--figure-dir`, `resolve_figure_dir`
- `scripts/model_variants/trainer_muvo_2D.py` — figure title/vmax/aspect, `_require_nonempty_file`
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/trainer.py` — Chamfer mask, action off-by-one, on_fit_end
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/metrics.py` — CDMetric empty handling, PSNRMetric frame-level

## 7. 참고

- `docs/muvo_2D_review_fixes_2026-05-25.md` — 이전 라운드 audit fix
- `docs/muvo_2D_muvo_alignment_2026-05-25.md` — upstream MUVO 정합 정리
- `docs/muvo_2D_port_plan.md` — 포팅 계획 v3
