# Alpha 1D Improvement Plan

대상: `alpha26/scripts/model_variants/{config,data,models,train,trainer}_muvo.py`

근거: `code_audit/alpha_1D.md`, `code_audit/upstream_1D_vs_alpha_1D.md`, `paper_translation/03_world_model.md`, `paper_translation/04_evaluation.md`, `paper_translation/99_paper_integrated.md`.

---

## 0. TL;DR — paper/upstream 기준 현재 상태

alpha-1D는 paper §III의 1D RSSM baseline에 해당하는 reduced-complexity 포트이다. 구조적으로는 paper와 잘 맞으나, paper §IV-A 식 (1)의 LiDAR 손실 항이 가중치 0으로 꺼져 있고, 데이터 샘플링/시퀀스 길이/이미지 해상도가 paper와 다르며, paper §IV-B 정량 비교의 baseline 역할을 수행할 만큼의 fairness 셋업($`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$ 분리, validation 메트릭 표준화)이 약하다. 우선 P0 4건, P1 3건, P2 4건을 식별.

---

## 1. 갭 분석 (3–5건 핵심)

1. **LiDAR 손실 가중치 0** — paper §IV-A 식 (1)의 `L_{p,xyz}, L_{p,r}, L_{V,scal}`이 alpha에는 거의 무시되고 있어 LiDAR 헤드가 사실상 학습 신호를 못 받음.
2. **이미지 해상도 축소 (600×960 → 320×768)** — paper 절대 PSNR과 비교 불가, alpha 내부 비교만 가능.
3. **샘플링/시퀀스 차이 (paper 0.2s seq 12 vs alpha 0.5s seq 8)** — temporal dynamics 학습 신호량이 다름.
4. **Validation split 분포 분리 부족** — `val_rl_run` / `val_ds_run` 파일명만 다르고 weather/route shift가 명시되지 않음. paper의 $`𝒟^{\text{RL}}`$ vs $`𝒟^{\text{DS}}`$ 의도가 반영되지 않음.
5. **`ACTION_IN_GRU=True`의 효과 미검증** — paper와 다른 디자인 결정. ablation 필요.

---

## 2. 갭별 상세

### G1. LiDAR 손실 활성화 ★

- **증거**:
  - `config_muvo.py:LOSSES.WEIGHT_LIDAR_RE = 0.0`, `WEIGHT_LIDAR_EMPTY = 0.0` (config_muvo.py L57–58 부근).
  - paper §IV-A 식 (1) (line 330–341): `L = Σ λ_i (λ_img L^img + λ_pcd (L^{pxyz} + L^{pr} + L^{pcd}) + λ_V L^{V,scal})`.
  - paper 본문 (line 322–325): "L2 loss $`ℒ^{\text{pxyz}}`$ ... L1 loss $`ℒ^{\text{pr}}`$ ..."
  - 메모리: 이전 분석 `docs/muvo_paper_summary.md` 와 `_muvo_comparison.md`도 "WEIGHT_LIDAR_RE를 0.1로 복원" 권고.
- **제안 변경**:
  - `cfg.LOSSES.WEIGHT_LIDAR_RE = 0.1` (initial), 후속 step에서 0.5~1.0 grid search.
  - `cfg.LOSSES.WEIGHT_LIDAR_EMPTY = 0.05` (현행 코드 주석 권고치).
  - CLI 플래그 `--weight-lidar-re`, `--weight-lidar-empty` 추가 권장.
- **위험**:
  - 학습 초기 loss 폭주 가능 → warmup으로 완화.
  - LiDAR 헤드 학습이 RGB와 균형을 이뤄야 — 너무 큰 가중치는 RGB 품질 손상.

### G2. 이미지 해상도 paper-aligned 옵션

- **증거**: `cfg.DATA.IMAGE_INPUT_SIZE=(320,768)` vs paper 600×960 (paper line 211–212).
- **제안**: 별도 cfg 변형 `cfg.DATA.IMAGE_INPUT_SIZE_PAPER=(384, 960)` (Arrow 데이터 원본 사이즈 확인 후 가능한 최대). 두 해상도로 비교 학습 가능하게.
- **위험**: VRAM 증가 → batch_size 축소 필요. 4-GPU 기준 batch_size 1까지 떨어질 수 있음.

### G3. EFFECTIVE_HZ + seq 길이 paper-aligned

- **증거**: alpha 1D는 `SAMPLE_EVERY_N=2, FRAME_STEP=5, CARLA_HZ=20` → 0.5s 간격, seq 8.
  paper §IV-A line 370–372: 0.2s 간격, seq 12.
- **제안**:
  - alpha-2D처럼 `_derive_sample_every_n` 도입 + `cfg.DATA.EFFECTIVE_HZ` 추가.
  - `EFFECTIVE_HZ=5` (0.2s 간격) + `FUTURE_HORIZON`을 paper와 맞춰 seq 12 옵션.
  - Window index 재빌드 필요 (`scripts/build_arrow_window_index.py` arg 변경).
- **위험**: 데이터 효율(에피소드당 window 수)이 더 빠른 샘플링으로 변동.

### G4. Validation split 명시 분리 ($`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$)

- **증거**: alpha는 `validation_run_008.arrow` (RL) vs `validation_run_024.arrow` (DS) 두 파일을 단순 분리. weather/route 분포가 명시적으로 다른지 확인되지 않음.
- **제안**:
  - 데이터 수집 단계에서 town/weather 메타데이터를 Arrow에 같이 저장하도록 schema 확장.
  - `data_muvo.py`에서 validation set 구성 시 weather=DS, route=random 조건으로 자동 필터.
  - 또는 paper 사양으로 새 validation Arrow 별도 수집.
- **위험**: 데이터 수집 파이프라인 변경이 필요 (재수집 비용).

### G5. `ACTION_IN_GRU` 효과 ablation

- **증거**: alpha 1D는 `ACTION_IN_GRU=True` (line 47–49 주석 명시), upstream MILE는 False.
- **제안**: 2회 학습으로 비교 — `--action-in-gru true/false` CLI 추가. PSNR/Chamfer/KL trace 비교.
- **위험**: ablation 1회당 4-GPU server run 150K step → 시간 비용.

### G6 (P1). SSIM 가중치 안정화 + 비교

- **증거**: `cfg.LOSSES.WEIGHT_SSIM=0.0` (꺼짐). 이전 노트(`docs/ssim_addition_notes.md`) 및 `_muvo_comparison.md`에 따르면 0.6 시도 → blur+artifact, 0.06 재시도 권장.
- **제안**: warmup 후 `WEIGHT_SSIM=0.06`으로 30K step부터 ramp.
- **위험**: 다시 학습 불안정 시 0.0으로 복귀.

### G7 (P1). Multi-scale loss 정확한 가중치 확인

- **증거**: `trainer_muvo.py:compute_loss`는 factors [1, 2, 4]에 `× 1/factor` 디스카운트 적용.
- **제안**: paper §IV-A "1, 2, and 4 ratios"와 일치하지만, paper의 λ_img는 모든 scale에 동일 가중치일 가능성도 있음. 식 (1)을 그대로 따라 `WEIGHT_RGB / WEIGHT_RGB_RE_2 / WEIGHT_RGB_RE_4` 등 per-scale 가중치 노출.
- **위험**: 작음.

### G8 (P1). Optimizer/Scheduler 검토

- **증거**: alpha = AdamW + OneCycleLR (PCT_START=0.1). paper § IV-A 도 AdamW LR=1e-4 WD=0.01. paper는 scheduler 명시 없음.
- **제안**: paper와 동일 (그대로 유지). OneCycleLR은 step 횟수 정확히 알아야 함 → `--steps`와 일관성 검증.
- **위험**: 없음.

### G9 (P2). Voxel head 추가 (paper §III-D + §IV-A SCAL)

- **증거**: alpha에 voxel decoder/loss 부재. paper conclusion: occupancy 추가 시 sensor 예측에 약간의 이득.
- **제안**:
  - Arrow 데이터에 voxel target 포함되어 있는지 확인 (현재 unlikely).
  - 데이터 수집 단계에서 voxel 생성 (upstream `data/generate_voxels.py` 포팅).
  - `models_muvo.py:Model`에 voxel decoder 헤드 추가 (`common.py:VoxelDecoder1` 참조).
  - SCAL loss(`SemScalLoss + GeoScalLoss + VoxelLoss`) 포팅.
- **위험**: 데이터 수집 + 모델 + loss 전 영역 변경 — 큰 작업. 우선순위 후순위.

### G10 (P2). BEV 헤드 비교 (paper baseline 비교군)

- **증거**: paper §IV-B의 BEV baseline은 alpha에 부재. paper는 BEV bottleneck 입증.
- **제안**: 비교 baseline으로만 의미; alpha의 본 목표는 BEV 미사용이므로 skip 가능. 단 paper 정량 비교에 참여하려면 frustum pooling head 한 변형 추가.
- **위험**: 큼 (frustum pooling = camera intrinsics/extrinsics 필요 — Arrow 데이터에 없음 가능성).

### G11 (P2). Reward / Value 헤드 (upstream-only)

- **증거**: upstream `WorldModelTrainer`는 reward/value 학습 가능. alpha는 raw recon만.
- **제안**: 현재 alpha 목표(world model 기본 학습)에서는 불필요. skip.

### G13 (P0). 학습 시 imagine supervise — paper / upstream 셋업과 일치 ★

- **증거**:
  - paper L372: "All 12 frames were treated as known data" — 학습 시 전체 sequence 가 observation, future prediction 은 검증에서만 평가.
  - upstream 1D `transition.py:RSSM.forward`: 학습 forward 전체 seq posterior, prior 는 KL loss 용. `mile.py:observe_and_imagine` 은 검증 전용.
  - alpha-1D `trainer_muvo.py:_observe_and_imagine`: 학습 `training_step` 에서 호출 (TBPTT carry 포함). `model.forward` 전체 seq posterior + `model.imagine()` 별도 호출 → FH 구간도 학습 loss (`future_*` × `WEIGHT_FUTURE=1.0`).
- **제안 (paper 일치 ablation)**:
  - 옵션 A: `WEIGHT_FUTURE=0` 으로 future_* loss 항만 0 → 빠른 시도.
  - 옵션 B: `trainer_muvo.py:training_step` 에서 imagine 부분 제거 — 코드 변경 필요.
  - 검증/추론 시에는 imagine 적용 유지 (metric 측정).
- **위험·트레이드오프**:
  - paper 와 정량 비교 가능해짐.
  - 단 future prediction 정량 성능은 떨어질 가능성 (학습 supervise 줄어드니까). alpha 의 single-window overfit 실험에서 1D 가 2D 보다 좋았던 결과는 학습 imagine supervise 영향일 수 있음 — paper-aligned 로 바꾸면 결과 달라질 수 있음.
  - **ablation 필수**: paper-aligned (`WEIGHT_FUTURE=0`) vs 현재 alpha 비교.

### G12 (P2). Augmentation 도입 (1D 단독)

- **증거**: alpha-2D는 PixelAugmentation 도입. alpha-1D는 미적용.
- **제안**: alpha-2D의 `PixelAugmentation`을 alpha-1D에 포팅 (`data_muvo.py`의 `_make_img_transform`에 옵션 인자 추가). cfg `DATA.AUGMENTATION` namespace 신설.
- **위험**: 작음. alpha-1D와 2D간 controlled 비교가 흐려질 수 있음 — augmentation 효과 분리 ablation 권장.

---

## 3. 우선순위 표

| ID | 갭 | 영향(RGB / LiDAR / Speed) | 난이도 | 우선순위 | 검증법 |
|---|---|---|---|---|---|
| G1 | LiDAR loss 활성화 | – / **+++** / – | 낮음 | **P0** | tiny-overfit + single-arrow 200 step → LiDAR Chamfer 감소 추적 |
| G2 | 이미지 해상도 paper-aligned | **++** / – / VRAM↑ | 중 | P1 | 4-GPU server run, batch 1, 30K step PSNR 비교 |
| G3 | EFFECTIVE_HZ/seq 12 | **+** / **+** / – | 중 | P1 | window index 재빌드 + tiny-overfit |
| G4 | Validation split 명시 | – / – / – (정량 평가) | 중 | P1 | Arrow schema 확장 + val 분리 |
| G5 | `ACTION_IN_GRU` ablation | **±** / **±** / – | 낮음 | **P0** (직접 비교 가치) | 2회 4-GPU run, 동일 step에서 val/RL_loss 비교 |
| G13 | 학습 imagine supervise 제거 (paper 일치) | **±** / **±** / – | 낮음 (옵션 A) ~ 중 (옵션 B) | **P0** (paper 매치) | `WEIGHT_FUTURE=0` vs 현재 4-GPU 50K step 비교, future metric trace |
| G6 | SSIM warmup | **+** / – / – | 낮음 | **P0** | tiny-overfit + 4-GPU 30K step |
| G7 | Per-scale 가중치 노출 | **±** / – / – | 낮음 | P1 | 코드 변경만, 기존과 동일 동작 확인 |
| G8 | Optimizer 일관성 | – / – / – | 매우 낮음 | P1 | sanity check |
| G9 | Voxel head (paper §III-D) | **+** / **+** / – | 매우 큼 | **P2** | 데이터 수집 후 PTO/PTF/NPT 시나리오 비교 |
| G10 | BEV 비교 baseline | – / – / – | 큼 | P2 | 별 의미 없음 (skip 가능) |
| G11 | Reward/Value 헤드 | – / – / – | 큼 | P2 | skip |
| G12 | Augmentation 포팅 | **+** / – / – | 낮음 | **P0** (1D와 2D controlled 비교 일관성) | tiny-overfit + 4-GPU 30K step |

---

## 4. 권장 실행 순서

**Sprint 1 (1주 내 가능)**:
1. G1: LiDAR loss 활성화 (`WEIGHT_LIDAR_RE=0.1`, `WEIGHT_LIDAR_EMPTY=0.05`).
2. G6: SSIM warmup 0.06 도입.
3. G12: Augmentation 포팅 (alpha-2D `PixelAugmentation`을 1D로).
4. G5: `ACTION_IN_GRU` ablation 한 쌍 (true vs false).

검증: tiny-overfit 단일 window 5000 step → val/lidar_chamfer, val/RL_psnr, val/RL_loss 추적; 4-GPU server run 30K step → 풀 metric.

**Sprint 2 (큰 작업)**:
5. G3: `_derive_sample_every_n` 도입 + window index 재빌드.
6. G2: 이미지 해상도 384×960 옵션 (Arrow 원본 확인 필요).
7. G4: validation split 분포 분리 (데이터 수집부터).

**Sprint 3 (선택)**:
8. G9: Voxel head + SCAL loss (가장 큰 작업, 데이터 수집부터 재정비).

---

## 5. 검증법 상세

### tiny-overfit (단일 GPU)
- `python scripts/tiny_overfit_trainer11_server.py --samples 8 --steps 3000`
- 한 window에 의도적 overfit으로 손실 항이 살아 있는지 확인.
- LiDAR 손실 추가 후 `train_lidar_xyz_1, train_lidar_range_1` scalar가 감소하는지 확인.

### single-arrow debug (단일 GPU)
- `python scripts/train_full_trainer11_server.py --train-run train_run_002.arrow --steps 200 --devices 1 --batch-size 2 --val-check-interval 20 --limit-val-batches 1 --window-index ... --run-name <debug>`
- 200 step 안에서 loss, KL, recon figure 확인.

### 4-GPU server run (production)
- `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 python scripts/train_full_trainer11_server.py --steps 150000 --devices 4 --strategy ddp_find_unused_parameters_true --batch-size 2 --num-workers 2 --window-index ... --run-name <run-name>`
- val/RL_loss, val/DS_loss, val/RL_psnr, val/RL_lidar_chamfer 메트릭으로 비교.

### Experiment 로깅
- `ExperimentCSVLogger`에 `change_summary`로 변경 내용 기록.
- TensorBoard `logs/<run-name>/`에서 trace 비교.

---

## 6. 미해결 질문 (사용자 결정 필요)

1. **이미지 해상도**: Arrow 원본이 600×960인가 320×768인가? `processed/` 메타데이터 확인 필요.
2. **Voxel target**: 현재 Arrow에 voxel 데이터 부재 시 데이터 수집부터 재시작 → 비용 대비 이득 판단.
3. **paper 절대 PSNR 비교 의도**: paper와 동일 절대 수치 비교가 필요한지, alpha 내부 ablation만 충분한지.
