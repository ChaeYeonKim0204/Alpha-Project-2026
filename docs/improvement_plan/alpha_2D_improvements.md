# Alpha 2D Improvement Plan

대상: `alpha26/scripts/model_variants/{config,data,models,train,trainer}_muvo_2D.py`

근거: `code_audit/alpha_2D.md`, `code_audit/upstream_2D_vs_alpha_2D.md`, `paper_translation/03_world_model.md`, `paper_translation/04_evaluation.md`, `paper_translation/99_paper_integrated.md`, 기존 `docs/muvo_2D_port_plan.md` (v3).

---

## 0. TL;DR — paper/upstream 기준 현재 상태

alpha-2D는 paper §III-C의 "2D token state RSSM" + §IV-B headline 결론(2D latent이 카메라 예측 최대 향상)을 충실히 반영한 포트이다. 구조적으로는 paper와 가장 잘 맞으며, alpha-1D 대비 추가 기능(action loss, augmentation, ClearML, near-field metric, 풀 컬럼 figure)을 갖춘다. 그러나 (1) **paper §IV-A의 LiDAR 손실 항이 가중치 0으로 꺼져 있음**, (2) **upstream 2D 대비 capacity 축소 (d_model 512→256, 6L/8H→3L/4H, N_token 1093→369)**, (3) **occupancy/BEV/voxel 부재**가 가장 큰 갭. P0 4건, P1 4건, P2 5건 식별.

---

## 1. 갭 분석 (5건 핵심)

1. **LiDAR 손실 가중치 0** — alpha 1D와 동일 이슈. paper §IV-A 식 (1)의 `L_{p,xyz}, L_{p,r}`가 사실상 비활성.
2. **Encoder capacity (d_model 256, 3L/4H)** vs upstream 2D (512, 6L/8H). paper에서 transformer가 fusion 핵심이므로 capacity 영향 ablation 필요.
3. **Decoder channels=512 vs embedding 256**의 의도 명확하나 검증 필요 — v4 RGB plateau fix 주석.
4. **`KL_FREE_BITS_ENABLED=True` (alpha 1D와 controlled 비교용)** vs paper/upstream 기본 OFF — 비교 시 분리 필요.
5. **3D occupancy 미구현** — paper §III-D + §IV-C 핵심 분기 빠짐.

---

## 2. 갭별 상세

### G1. LiDAR 손실 활성화 ★

- **증거**:
  - `config_muvo_2D.py:LOSSES.WEIGHT_LIDAR_RE=0.0, WEIGHT_LIDAR_EMPTY=0.0`.
  - paper §IV-A 식 (1) (L330–341).
  - alpha-1D와 동일한 갭.
- **제안**:
  - `WEIGHT_LIDAR_RE=0.1`, `WEIGHT_LIDAR_EMPTY=0.05`로 시작.
  - alpha-2D의 near-field Chamfer 메트릭으로 효과 정량화.
- **위험**: LiDAR 헤드 학습이 RGB와 균형 깨질 수 있음 → grid search.

### G2. Transformer capacity 점진 확장

- **증거**:
  - `cfg.MODEL.FUSION.TRANSFORMER_CHANNELS=256, LAYERS=3, HEADS=4` vs upstream 512/6/8.
  - paper §III-B "k-layer transformer" k 미명시.
  - alpha-2D run name `muvo_style_alpha_fpn256_lidar128_rssmtd` 자체가 channels=256 명시.
- **제안 (점진)**:
  - 단계 1: `TRANSFORMER_LAYERS=6` (head/channels 유지) — 학습 시간 ~1.5×.
  - 단계 2: `TRANSFORMER_CHANNELS=384` + LAYERS=6 — VRAM 영향 확인.
  - 단계 3: `TRANSFORMER_CHANNELS=512` (upstream 매치) — batch_size 축소 필요.
- **위험**: VRAM 증가, 학습 시간 증가, OneCycleLR step 재조정.

### G3. RSSMTD decoder channels 검증

- **증거**:
  - `MODEL.RSSM_2D.DECODER_CHANNELS=512` (v4 RGB plateau fix 주석).
  - embedding=256, decoder 내부=512 — 비대칭.
- **제안**:
  - ablation: 256 vs 512 vs 768 decoder_channels에서 RGB PSNR / LiDAR Chamfer 비교.
  - decoder_channels=embedding_dim×2 비율 검증.
- **위험**: ablation 비용. 짧은 30K step run 3개로 충분.

### G4. `KL_FREE_BITS_ENABLED` controlled ablation

- **증거**:
  - alpha-2D 기본 `True` (v4-2: alpha-1D와 controlled 비교 목적).
  - paper에는 명시 없음 (upstream MUVO 기본 OFF).
- **제안**: ablation 한 쌍 (True vs False) + KL trace 비교. 결과에 따라 default를 paper-aligned (False) 또는 alpha-aligned (True)로 결정.
- **위험**: 2회 4-GPU run.

### G5. LiDAR encoder out_indices 비교

- **증거**:
  - alpha-2D `LIDAR_OUT_INDICES=(1,2,3)` (v3: upstream MUVO 매치) — stride-16.
  - upstream 2D도 (1,2,3) 사용 (stride 차이는 input 해상도 차이 때문).
- **제안**: ablation `(0,1,2,3)` vs `(1,2,3)` vs `(2,3,4)` — token grid 변화에 따른 LiDAR Chamfer 영향.
- **위험**: 토큰 수 변경 → RSSMTD config 동기화 필요.

### G6 (P1). `decoder_channels` vs `embedding_dim` 결합 정합성

- **증거**: 현재 `embedding=256, decoder_channels=512` — 디코더 내부에서 한 번 expand. paper §III-D "C × Σ T_j" 표기는 C가 단일 값임을 시사 (token channel과 decoder channel 일치 가능).
- **제안**: `decoder_channels=embedding_dim`(256) 변형도 시도. RGB plateau 재현 여부 확인.

### G7 (P1). PerceptualLoss / LPIPS 옵션 도입

- **증거**:
  - upstream 2D `losses.py`에 `PerceptualLoss, LPIPSLoss` 옵션 존재. paper Fig. 4의 PL 비교.
  - paper 결론: 효과 없음.
- **제안**:
  - 옵션 cfg 키 `LOSSES.PERCEPTUAL.ENABLED, WEIGHT_PERCEPTUAL`, `LOSSES.LPIPS.ENABLED, WEIGHT_LPIPS` 추가.
  - 단 default OFF — paper 결론 검증용 ablation만.
- **위험**: 의존성(`torchmetrics.image.lpip`, `timm`) 추가, 학습 속도 영향.

### G8 (P1). EFFECTIVE_HZ=5 (paper-aligned 0.2s) ablation

- **증거**: alpha-2D 기본 `EFFECTIVE_HZ=2` (0.5s). paper § IV-A line 370–371: 0.2s.
- **제안**: `--effective-hz 5` 한 번 run + window index 재빌드. 결과 비교.
- **위험**: window index 재빌드 시간 + 데이터 효율 차이.

### G9 (P1). Validation split 명시 분리 ($`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$)

- **증거**: alpha-2D의 val_rl/val_ds도 1D와 같은 단순 분리 (`validation_run_008.arrow`, `validation_run_024.arrow`).
- **제안**: alpha-1D와 동일하게 weather/route shift 분리 (Arrow schema 확장).
- **위험**: 데이터 수집 변경.

### G10 (P2). 3D Occupancy 헤드 추가 (paper §III-D + §IV-A SCAL)

- **증거**: alpha-2D에 voxel decoder/loss 부재. upstream 2D는 `ConvDecoder3D` + VoxelSemHead. paper §IV-C: cam+lidar pre-training → voxel head fine-tune 권장.
- **제안 (점진)**:
  - 단계 1: Arrow 데이터에 voxel target 추가 (upstream `generate_voxels.py` 포팅).
  - 단계 2: `models_muvo_2D.py`에 voxel 토큰(예: 12×12×4=576) 추가 → N_token 369 → 945.
  - 단계 3: `TokenConvDecoder3D` (또는 `ConvDecoder3D` 포팅) 추가.
  - 단계 4: SCAL loss (`SemScalLoss + GeoScalLoss + VoxelLoss`) 포팅.
  - 단계 5: PTF/PTO/NPT 시나리오 학습.
- **위험**: 매우 큼. 데이터 + 모델 + loss 전 영역.

### G11 (P2). PointPillars LiDAR encoding 옵션

- **증거**: paper §IV-B의 RV vs PP 비교군. alpha는 RV만.
- **제안**: optional `cfg.MODEL.LIDAR.ENCODING={range_view, point_pillars}` 분기 추가. PointPillars 포팅(upstream `common.py:PointPillarNet`).
- **위험**: 큼. paper 결론은 RV 우세이므로 ablation 가치는 있으나 우선순위 낮음.

### G12 (P2). BEV branch 비교 (paper §IV-B "BEV vs WOB")

- **증거**: paper §IV-B의 BEV 분기는 alpha에 없음. paper 결론: BEV worse.
- **제안**: frustum pooling + BEV backbone 분기 추가 (camera intrinsics 필요 — Arrow에 없을 가능성).
- **위험**: 매우 큼. 데이터 수집부터.

### G13 (P2). ViT 백본 분기 (paper §IV-B Fig. 4)

- **증거**: paper 결론: 카메라에 약간 효과, 다른 메트릭에는 효과 없음.
- **제안**: 카메라 backbone 옵션에 `timm.create_model('mobilevit_v2_*')` 추가.
- **위험**: 모델 크기·VRAM 변화.

### G14 (P2). Reward/Value/Policy 헤드 확장 (RL bootstrap용)

- **증거**: alpha-2D는 `PolicyDecoder`만 (action L1). upstream은 reward/value도.
- **제안**: 현재 world model 학습 목표에서는 불필요 → skip.

### G15 (P0). 학습 시 imagine supervise — paper / upstream 셋업과 일치 ★

- **증거**:
  - paper L372: "All 12 frames were treated as known data" — 학습 시 전체 sequence 가 observation (RF=seq, FH=0 효과). future prediction 은 검증에서만 (`observe_and_imagine` 으로) 평가.
  - upstream 2D `transition_td.py:202-225 RSSMTD.forward`: `for t in range(sequence_length): observe_step(...)` — 학습 forward 는 전체 seq posterior, prior 는 KL loss 용으로만 계산.
  - upstream 2D `muvo.py:479 observe_and_imagine`: RF observe + FH imagine, 검증 전용 메서드.
  - alpha-2D `trainer_muvo_2D.py:621-645 _observe_and_imagine`: 학습 `training_step` 에서 호출. `model.forward` 로 전체 seq posterior reconstruction + `model.imagine(state, future_horizon=self.fh)` 별도 호출 → FH 구간도 학습 loss 적용 (`future_*` prefix × `WEIGHT_FUTURE=1.0`).
- **제안 (paper 일치 ablation)**:
  - 옵션 A: `WEIGHT_FUTURE=0` 으로 두면 future_*  loss 항이 0 → 학습 시 imagine 호출은 남지만 supervise X. 빠른 시도.
  - 옵션 B: `trainer_muvo_2D.py:training_step` 에서 `_observe_and_imagine` 대신 `_observe_only` (가칭) 호출 — `model.forward` 만, imagine 부분 제거. 코드 변경 필요하지만 paper 와 정확히 일치.
  - 검증/추론 시에는 imagine 적용 유지 (paper 와 동일하게 metric 만 측정).
- **위험·트레이드오프**:
  - paper 와 정량 비교 가능해짐 (future prediction 메트릭 동일 조건).
  - 단 future prediction 정량 성능은 떨어질 수 있음 (학습 supervise 줄어드니까). alpha 의 현재 셋업이 future 정밀도엔 유리할 수도.
  - **ablation 필수**: 동일 step 에서 paper-aligned (`WEIGHT_FUTURE=0`) vs 현재 alpha (`WEIGHT_FUTURE=1.0`) 비교. 어느 쪽이 alpha 의 데이터·목표에 더 맞는지 결정.

---

## 3. 우선순위 표

| ID | 갭 | 영향(RGB / LiDAR / Speed) | 난이도 | 우선순위 | 검증법 |
|---|---|---|---|---|---|
| G1 | LiDAR loss 활성화 | – / **+++** / – | 낮음 | **P0** | tiny-overfit + 4-GPU 30K step, near-field Chamfer 감소 |
| G2 | Transformer 6L/8H 확장 | **+** / **+** / VRAM↑ | 중 | **P0** (paper 매치) | 4-GPU 50K step, batch 1 조정 |
| G15 | 학습 imagine supervise 제거 (paper 일치) | **±** / **±** / – | 낮음 (옵션 A) ~ 중 (옵션 B) | **P0** (paper 매치) | `WEIGHT_FUTURE=0` vs 현재 4-GPU 50K step 비교, future metric trace |
| G3 | decoder_channels ablation | **±** / – / – | 낮음 | P1 | 3 run grid (256/512/768) 30K step |
| G4 | KL free-bits ablation | **±** / **±** / – | 낮음 | **P0** (default 결정) | 2 run 50K step + KL trace |
| G5 | LiDAR out_indices ablation | – / **+** / – | 중 | P1 | 3 run 30K step |
| G6 | decoder_channels vs embedding 통일 | **±** / – / – | 낮음 | P1 | sanity 30K step |
| G7 | PerceptualLoss/LPIPS optional | **±** / – / – (paper 결론 검증) | 중 | P1 | 1 run with PL on, 1 with off |
| G8 | EFFECTIVE_HZ=5 | **+** / **+** / – | 중 | P1 | window index 재빌드 + 4-GPU 50K |
| G9 | Validation split 분리 | – / – / – (정량 평가 정확도) | 중 | P1 | 데이터 수집 변경 |
| G10 | 3D Occupancy 헤드 + SCAL | **+** / **+** / – | 매우 큼 | **P2** | 데이터 + 모델 + loss 전 영역 |
| G11 | PointPillars 옵션 | – / **±** / – | 큼 | P2 | paper 결론 재현 (RV > PP) |
| G12 | BEV branch | – / – / – | 매우 큼 | P2 | skip 가능 |
| G13 | ViT 백본 | **±** / – / – | 중 | P2 | paper PL/ViT 비교 |
| G14 | Reward/Value 헤드 | – / – / – | 큼 | P2 | skip |

---

## 4. 권장 실행 순서

**Sprint 1 (alpha-1D와 controlled 비교 가능 셋업)**:
1. G1: LiDAR loss 활성화 — alpha-1D와 동시에 적용해야 fair 비교.
2. G4: KL free-bits ablation — default 정함.
3. G2: Transformer 6L 확장 (1 step) — paper-aligned baseline 도달.

→ 검증: alpha-1D Sprint 1과 동일 setting으로 4-GPU 30K step 비교. paper §IV-B headline ("2D latent benefits camera") 재현 시도.

**Sprint 2 (paper-aligned 추가 변형)**:
4. G3: decoder_channels grid.
5. G5: LiDAR out_indices ablation.
6. G6: decoder_channels=embedding 통일 변형.
7. G7: PerceptualLoss optional 도입 + paper "PL no benefit" 결론 재현 시도.

**Sprint 3 (paper §IV-C 재현)**:
8. G8: EFFECTIVE_HZ=5 (paper 0.2s 매치).
9. G9: validation split 명시 분리.

**Sprint 4 (큰 작업)**:
10. G10: 3D occupancy 추가 → PTF/PTO/NPT 시나리오.

---

## 5. 검증법 상세

### tiny-overfit (단일 GPU)
- 단일 window 5000 step → val/recon figure에서 RGB blur 감소, LiDAR depth 패턴 형성 확인.

### single-arrow debug
- 200 step, `--val-check-interval 20 --limit-val-batches 1` → loss/metric trace 빠른 확인.

### 4-GPU server (production)
- 50K step (Sprint 1) / 150K step (Sprint 2-3) — Server SETUP 명령:
  ```bash
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 \
  python scripts/train_full_trainer11_server.py \
    --steps <N> --devices 4 --strategy ddp_find_unused_parameters_true \
    --batch-size 2 --num-workers 2 \
    --window-index ... \
    --run-name muvo_2D_<sprint>_<variant>
  ```
- 단 본 plan은 `train_muvo_2D.py`를 직접 사용 (server entrypoint와 별개) — `python alpha26/scripts/model_variants/train_muvo_2D.py` 형식.

### ClearML Compare
- alpha-2D는 ClearML 통합 — `Task.init` 후 web UI Compare 모드로 sprint 변형 trace 비교 (TB도 병행).

### 메트릭 우선순위
- 카메라: PSNR (paper main metric)
- LiDAR: near-field Chamfer (alpha 신규), full Chamfer, xyz Euclidean, range MAE
- KL: probabilistic loss trace
- Action: L1 (alpha 2D 신규)

---

## 6. 미해결 질문 (사용자 결정 필요)

1. **Capacity 확장 한계**: 4-GPU + batch 2가 paper 6L/8H/512 일치 시 OOM 가능. 적정 타협점은?
2. **Voxel target 데이터**: 현재 Arrow에 부재 시 데이터 수집부터 재정비 → 우선순위 (P2 vs 조기 도입)?
3. **paper와 절대 PSNR 비교 의도**: 데이터 자체가 paper(600×960, 0.2s, 300K)와 다르므로 절대 수치 비교는 어려움 — alpha 내부 ablation 위주 검증.
4. **`KL_FREE_BITS_ENABLED` default**: True (controlled 1D 비교) vs False (paper-aligned). 어느 쪽 우선?
