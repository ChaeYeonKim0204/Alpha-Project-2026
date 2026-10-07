# Alpha 2D Code Audit — `alpha26/scripts/model_variants/*_muvo_2D.py`

알파 26 프로젝트의 2D 분기(`_muvo_2D.py` 파일군) 5개 파일 감사 결과. v3 스펙(2026-05 기준)을 반영한다.

| 파일 | 줄 수 | 책임 |
|---|---|---|
| `config_muvo_2D.py` | 158 | 2D 전용 cfg (RSSM_2D, augmentation, ClearML, EFFECTIVE_HZ) |
| `data_muvo_2D.py` | 326 | data_muvo + PixelAugmentation 통합 |
| `models_muvo_2D.py` | 842 | RSSMTD + TokenConvDecoder2D + PolicyDecoder |
| `train_muvo_2D.py` | 451 | train_muvo + ClearML init + 신규 CLI 플래그 |
| `trainer_muvo_2D.py` | 868 | trainer_muvo + action loss + 6프레임 figure + KL free-bits 게이트 |

---

## 1. `config_muvo_2D.py` — 2D 학습 셋팅

`cfg` 주요 키 (1D 대비 ★는 신규/변경):

| 그룹 | 키 | 값 |
|---|---|---|
| 시퀀스 | `RECEPTIVE_FIELD, FUTURE_HORIZON` | 4, **2** (seq=6) ★ |
| 데이터 | `_CARLA_HZ, _FRAME_STEP, _EFFECTIVE_HZ` | 20, 5, **2** ★ |
| `DATA.SAMPLE_EVERY_N` | `_derive_sample_every_n(20, 5, 2) = 2` (동적) ★ |
| 입력 | `DATA.IMAGE_INPUT_SIZE, LIDAR_RANGE_VIEW_SIZE` | `(320, 768)`, `(32, 1024)` (1D 동일) |
| 증강 | `DATA.AUGMENTATION.{ENABLED, BLUR_PROB, BLUR_WINDOW, BLUR_STD, SHARPEN_PROB, SHARPEN_FACTOR, COLOR_PROB, BRIGHTNESS, CONTRAST, SATURATION, HUE}` | True, 0.3/0.3/0.3 기본 ★ |
| 모델 공통 | `MODEL.EMBEDDING_DIM, SPEED_CHANNELS, ACTION_DIM` | 256, 16, 2 (1D 동일) |
| Fusion | `MODEL.FUSION.TRANSFORMER_{CHANNELS, LAYERS, HEADS, DROPOUT}` | 256, 3, **4**, 0.1 ★ |
| 1D RSSM | `MODEL.TRANSITION.HIDDEN_STATE_DIM, STATE_DIM, ACTION_LATENT_DIM, ACTION_IN_GRU` | **256, 256**, 64, True (compat) |
| 2D RSSM 신규 | `MODEL.RSSM_2D.{IMAGE_TOKEN_HW, LIDAR_TOKEN_HW, LIDAR_OUT_INDICES, POLICY_TOKENS, TRANSFORMER_DECODER_LAYERS, TRANSFORMER_DECODER_HEADS, DECODER_CHANNELS, NEARFIELD_PC_RANGE}` | (10,24), (2,64), (1,2,3), 1, 3, 4, 512, (...) ★ |
| Policy | `MODEL.POLICY.ENABLED` | True ★ |
| Loss | `LOSSES.WEIGHT_PROBABILISTIC` | 1e-2 |
| Loss KL free-bits | `LOSSES.KL_FREE_BITS, KL_FREE_BITS_ENABLED, KL_BALANCING_ALPHA` | 1.0, **True (v4-2)**, 0.75 ★ |
| Loss sensor | `LOSSES.WEIGHT_RGB, WEIGHT_FUTURE, WEIGHT_LIDAR_RE, WEIGHT_LIDAR_EMPTY, WEIGHT_SSIM` | 1.0, 1.0, **0.0**, **0.0**, 0.0 ⚠ |
| Loss action | `LOSSES.WEIGHT_ACTION` | 1.0 ★ |
| Optimizer/Scheduler | (1D 동일) | – |
| 로깅 | `LOGGING.RUN_NAME` | `muvo_style_alpha_fpn256_lidar128_rssmtd` ★ |
| 로깅 (recon) | `LOGGING.RECON_FIG_ALL_FRAMES, RECON_FIG_BOTH_VIEWS` | True, True ★ |
| ClearML | `CML_ENABLED, CML_PROJECT, CML_TASK, CML_TYPE, CML_TAGS` | True, "alpha26_muvo_2D", "muvo_2D_default", ..., ("rssmtd", "token-shape", "alpha26") ★ |

**토큰 산술**:
- 이미지 토큰: 10 × 24 = **240** (FPN stride-32 of 320×768)
- LiDAR 토큰: 2 × 64 = **128** (FPN stride-16 of 32×1024 — upstream MUVO 매치)
- Policy 토큰: 1
- **합계 N_token = 369** (upstream 2D 1093 대비 voxel 576 제거)

**`_derive_sample_every_n(carla_hz, frame_step, effective_hz)` 함수**:
- `denom = frame_step * effective_hz`
- `carla_hz % denom != 0`이면 ValueError
- 반환: `carla_hz // denom`
- 예: `_derive_sample_every_n(20, 5, 2) = 2`

**Mismatch flags**:
- `WEIGHT_LIDAR_RE=0.0`, `WEIGHT_LIDAR_EMPTY=0.0` — 1D와 동일하게 LiDAR 손실 꺼짐 ⚠ → P0.
- `KL_FREE_BITS_ENABLED=True` — upstream MUVO와 다름. alpha-1D 기준 비교를 위한 의도된 차이(v4-2 주석).
- `ACTION_IN_GRU=True` — RSSMTD는 이 키를 소비하지 않음. 호환성 dead config.

---

## 2. `data_muvo_2D.py` — 데이터 + 증강

1D `data_muvo.py`에서 다음만 차이:

| 변경 | 위치 | 내용 |
|---|---|---|
| Import | line 28 | `from config_muvo_2D import cfg` |
| 신규 클래스 | line 52–96 | `PixelAugmentation(aug_cfg)` — PIL → (Blur 또는 Sharpen, mutex) + ColorJitter |
| 변경 함수 | line 99–109 | `_make_img_transform(size, augment=True)` — augment 플래그 + cfg.DATA.AUGMENTATION 조건부 삽입 |
| 데이터셋 시그니처 | line 207 | `MUVODataset(arrow_file, seq_len, stride, sample_every_n, ..., augment=True)` |
| augment 전파 | line 216–217 | `_make_img_transform(..., augment=augment)` |

`PixelAugmentation` 파라미터 (mutex 검증: `blur_prob + sharpen_prob ≤ 1.0`):

| 항목 | 확률 | 강도 |
|---|---|---|
| Gaussian Blur | `BLUR_PROB=0.3` | window=`BLUR_WINDOW`, σ ∈ `BLUR_STD=(0.1, 1.7)` |
| Sharpen | `SHARPEN_PROB=0.3` | factor ∈ `SHARPEN_FACTOR=(1.0, 5.0)` |
| ColorJitter | `COLOR_PROB=0.3` (독립) | brightness/contrast/saturation 0.3, hue 0.1 |

**증강 적용 범위**: `image` 키만. `image_raw`(타깃), `lidar`, `action`, `speed`에는 미적용.

**Sample dict 키**: 1D와 byte-identical. `{image, image_raw, lidar, action, speed, run_id, start_row}`.

**EFFECTIVE_HZ → SAMPLE_EVERY_N 해석**: config 단계에서 이미 `_derive_sample_every_n`로 해결. dataset은 `cfg.DATA.SAMPLE_EVERY_N`만 읽음.

---

## 3. `models_muvo_2D.py` — 2D world model

`models_muvo.py` 대비 클래스 변경:

| 클래스 | 줄 | 상태 | 책임 |
|---|---|---|---|
| `FPNDecoder` | 41–71 | (변경 적음) | 1D와 동일 인터페이스 |
| `PositionEmbeddingSine` | 78–111 | identical 계열 | 2D sinusoidal PE |
| `SensorFusionTransformer(channels, num_layers=3, nhead=4, dropout=0.1)` | 118–169 | 변경 | nhead 8→**4** (64 dim/head), type embedding 5 slots (image/lidar/policy/action/speed) — encoder는 0/1만 사용. speed broadcast-add (concat 대신) |
| `SpeedEncoder(hidden_channels=16, out_channels=256)` | 176–188 | ★NEW | Linear(1→16)→ReLU→Linear(16→256)→ReLU (1D의 inline 시퀀스를 클래스로 추출) |
| `PolicyDecoder(in_channels)` | 195–212 | ★NEW | MLP→Tanh(2) — 행동 회귀 헤드 (paper §III에는 없음; alpha 추가) |
| `ConvGRUCellGlo(input_dim, hidden_dim)` | 219–247 | ★NEW | 토큰-wise Conv1d 게이트 + sigmoid 전역 게이트 (mean over tokens). FC→Conv 교체 (paper §III-C) |
| `RepresentationModelTD(in_channels, latent_dim, image_token_hw=(10,24), lidar_token_hw=(2,64), num_layers=3, nhead=4)` | 250–313 | ★NEW | TransformerDecoder + 학습 가능 query(image+lidar+policy) — paper의 prior/posterior 모델 |
| `RSSMTD(embedding_dim, action_dim, hidden_state_dim, state_dim, image_token_hw, lidar_token_hw, policy_tokens=1, ...)` | 316–481 | ★NEW | 2D 토큰 상태 RSSM — paper §III-C 정확 구현 |
| `_TokenHead(in_channels, out_channels, downsample_factor, sensor_type)` | 488–500 | ★NEW | (B*S,C,H,W) → 1×1 Conv → reshape (B,S,C,H,W) |
| `TokenConvDecoder2D(in_channels, out_channels, base_hw, target_hw, sensor_type, decoder_channels=256, n_basic_conv=None)` | 503–583 | ★NEW | cat(h,sample) → pre_transpose(stride=2 × n) → trans_conv1/2/3 + _TokenHead 사다리 |
| `Model(cfg, embedding_n_channels, transformer_encoder)` | 590–842 | 재작성 | 1D Model 자리 차지. 토큰 기반 인코딩/디코딩 |

**State shape (`RSSMTD`)**: `h: (B,S,C,N_token), s: (B,S,C,N_token)` with `C=256, N_token=369`. 내부적으로는 `(N_token, B, C)` 레이아웃으로 ConvGRUCellGlo 통과.

**Action injection** (RSSMTD):
- `observe_step` (L420–444): action → `posterior_action_module` → latent_action `(1, B, C)` + type_emb[2] → concat with `[h_tokens, embedding_tokens]` → TransformerDecoder query.
- `imagine_step` (L446–461): 같은 패턴으로 prior_action_module.

**Forward 흐름** (`Model.forward(batch)`):
1. `encode_fuse_sequence(image, lidar, speed)` → `(B,S,C,N_in)` 토큰 시퀀스
2. RSSMTD 루프: posterior/prior 토큰 출력
3. `_slice_token_state(state) → (img, lid, pol)` — N_token을 모달리티 범위로 슬라이스
4. `_decode_state(hidden_state, sample)`:
   - image_decoder = TokenConvDecoder2D(img tokens) → `{rgb_4, rgb_2, rgb_1}`
   - lidar_decoder = TokenConvDecoder2D(lid tokens) → `{lidar_reconstruction_*}`
   - PolicyDecoder(pol token) → `action_pred (B, S, 2)`

**`decoder_channels=512`** (v4 RGB plateau fix): 토큰 채널(256) 대비 디코더 내부 채널이 더 크다 — upstream MUVO `latent_n_channels=512` 매치.

**TransformerDecoder for fusion 파라미터** (RepresentationModelTD): num_layers=3, nhead=4, d_model=256.

**관찰점**:
- L86–88 `normalize=True, scale=2π` PE 일관성.
- L129 type_embedding 5-slot (image/lidar/policy/action/speed) — encoder는 0/1만 사용, slot 2~4은 RSSMTD에서.
- L277 RSSMTD type_embeddings는 3-slot으로 분리되어 SensorFusionTransformer의 5-slot과 불일치 — 의도된 분리.
- L666 `decoder_channels = 2*embedding_n_channels` 기본 (upstream 매치).
- L769 일본어 주석 "RSSMTD では h_init/s_init/... を使用しない" — RSSMTD는 h_init/s_init/continuation 인자를 받되 무시(zero-init).

---

## 4. `train_muvo_2D.py` — 학습 진입점

`train_muvo.py` 대비 차이:

| 변경 | 위치 |
|---|---|
| Import: `from clearml import Task` (lazy) | line 178 |
| Import: `config_muvo_2D, data_muvo_2D, trainer_muvo_2D` | – |
| 신규 CLI 플래그 | line 51–59: `--effective-hz, --no-augmentation, --no-clearml, --cml-project, --cml-task` |
| `--h-dim` help: "RSSMTD는 h-dim=z-dim 필수" | line 46–47 |
| `apply_overrides` 신규 블록 | line 125–158: h=z 강제 + z 캐스케이드 + effective-hz 재계산 + no-aug/no-cml/cml-overrides |
| `maybe_init_clearml(run_cfg, args)` | line 173–202: Task.init + hyperparam connect (RECEPTIVE_FIELD, FUTURE_HORIZON, EFFECTIVE_HZ, ...) |
| `main()` ClearML 단계 | line 367 (`cml_task = maybe_init_clearml(...)`) before resolve_runtime |
| `build_callbacks` change_summary | line 283–293: 더 상세한 모델 스펙 (h/z dims, out_indices, transformer L/h, EFFECTIVE_HZ, KL_FREE_BITS) |
| Silent OneCycleLR resume patch | line 346 (print 제거) |
| `save_loss_metric_figure` try/except | line 420–426 (graceful degradation) |

**ClearML 게이트**: `if not run_cfg.CML_ENABLED or args.no_clearml: return None`. Connect 항목 — RECEPTIVE_FIELD, FUTURE_HORIZON, EFFECTIVE_HZ, EMBEDDING_DIM, HIDDEN_STATE_DIM, WEIGHT_LIDAR_RE, WEIGHT_ACTION, WEIGHT_SSIM, KL_FREE_BITS_ENABLED, LIDAR_OUT_INDICES, AUGMENTATION_ENABLED.

**Augmentation 게이트**: `args.no_augmentation` → `run_cfg.DATA.AUGMENTATION.ENABLED = False`. dataset은 config을 직접 읽음.

**EFFECTIVE_HZ override**: `args.effective_hz` → `run_cfg.DATA.SAMPLE_EVERY_N = _derive_sample_every_n(20, 5, effective_hz)`.

---

## 5. `trainer_muvo_2D.py` — LightningModule (2D)

`trainer_muvo.py` 대비 변경:

| 변경 블록 | 위치 | 내용 |
|---|---|---|
| Import | line 23 | `from models_muvo_2D import Model` |
| docstring | line 1–10 | action loss / near-field Chamfer / 6-프레임 figure / KL free-bits / ClearML 업로드 / TBPTT 간소화 |
| `save_reconstruction_figure` | line 87–212 | +88줄. `RECON_FIG_ALL_FRAMES` (RF+FH 풀 컬럼), `RECON_FIG_BOTH_VIEWS` (스케일/raw 4 depth row) |
| `WorldModelTrainer.__init__` | line 327–334 | `kl_free_bits_enabled, kl_free_bits_value, weight_action, nearfield_pc_range` 추가 |
| `_observe_and_imagine` | line 614–640 | RSSMTD ignores h_init/s_init/continuation/prev_action (docstring) |
| `training_step` | line 644–652 | TBPTT 래퍼 없이 직접 `_observe_and_imagine(batch)` |
| `compute_loss` 추가 항 | line 600–603 | action loss `F.l1_loss(output["action_pred"], action_target)` |
| `compute_eval_metrics` 추가 | line 494–500 | near-field Chamfer (scale-corrected, `pc_range=nearfield_pc_range`) |
| `test_step` 추가 | line 694–707 | ClearML artifact 업로드 (`pred_rgb, pred_lidar, pred_action`) — 처음 4 batch만 |
| `ExperimentCSVLogger._collect_final_val_metrics` | line 818–819 | `val/RL_lidar_chamfer_xyz_nearfield, val/DS_lidar_chamfer_xyz_nearfield` 메트릭 추가 |

**Loss 항** (트레이너 측 변경):

| 항 | 가중치 키 | 차이 |
|---|---|---|
| RGB / LiDAR / KL / future | 1D와 동일 | – |
| **Action L1** | `WEIGHT_ACTION (1.0)` | NEW ★ |
| **Near-field Chamfer** | metric only (가중치 없음) | NEW ★ |

**KL free-bits 게이트** (L525):
```
free_bits = kl_free_bits_value if kl_free_bits_enabled else 0.0
```
`gaussian_kl_loss`에서 `kl_sum.clamp_min(free_bits)`로 적용. `kl_free_bits_enabled=True`(alpha-2D 기본)이면 free-bits 활성.

**Reconstruction figure (`save_reconstruction_figure`)**:
- `all_frames=True`: RF(4) + FH(2) = 6 컬럼 풀 시퀀스. 2 RGB row + 4 depth row(gt_scaled, pred_scaled, gt_raw, pred_raw).
- `both_views=True`: depth를 LIDAR_SCALE 곱한 m 단위와 학습 공간 단위 둘 다 출력.
- 호출: `PeriodicReconstructionCallback.on_train_epoch_end()` 매 N epoch.

**ClearML artifact**: `test_step`에서 try/except로 `Task.current_task()` 존재 시 `pred_dump = {rgb_pred, lidar_pred, action_pred}` numpy 변환 후 `task.upload_artifact("pred_{dataset_name}_batch_{batch_idx}", pred_dump)`.

---

## 6. 모듈 의존 관계 (2D)

```
config_muvo_2D ──┐
   ↓             │
data_muvo_2D ──→ train_muvo_2D ←── trainer_muvo_2D ←── models_muvo_2D
                       │                  │
                       │                  └── (옵션) clearml.Task
                       ├── maybe_init_clearml
                       └── DataLoader, runtime, callbacks
```

---

## 7. Paper §III/§IV ↔ alpha-2D 매핑

| Paper | alpha-2D 위치 |
|---|---|
| §III.A image/LiDAR encoder | `models_muvo_2D.py:Model.encode_fuse_sequence` (timm ResNet18 + FPN; LiDAR out_indices [1,2,3] stride-16) |
| §III.B Transformer fusion | `SensorFusionTransformer` 3L/4H (channels=256) + 2D sinusoidal PE + 5-slot type embedding |
| §III.C **2D token state RSSM** | `RSSMTD` `(B,S,256,369)` 토큰 상태 — **paper headline 구조** |
| §III.C "FC→Conv 교체" | `ConvGRUCellGlo` (Conv1d + global gate) |
| §III.C TransformerDecoder prior/posterior | `RepresentationModelTD` |
| §III.D 카메라/LiDAR ConvT 디코더 | `TokenConvDecoder2D` (모달리티별, 3단 ConvT + multi-scale 헤드) |
| §III.D Voxel 3D 디코더 | ✗ (alpha-2D 미구현) |
| §III.D Policy 디코더 (paper에 없음) | `PolicyDecoder` + action loss (alpha 추가) |
| §IV-A 식 (1) | `trainer_muvo_2D.py:compute_loss` — LiDAR 항 가중치 0으로 꺼짐 ⚠ |
| §IV-A multi-scale 1/2/4 | discount factors `[1, 2, 4]` |
| §IV-A 0.2 s 샘플 | alpha EFFECTIVE_HZ=2 (0.5 s) — paper와 차이 |
| §IV-A seq 12 / 6 | alpha seq = 4 + 2 = **6** (paper voxel exp seq=6과 일치) |
| §IV-A AdamW LR=1e-4, WD=0.01 | ✓ |
| §IV-B 2D latent (headline) | ✓ (RSSMTD 토큰 상태) |
| §IV-B Perceptual Loss | ✗ |
| §IV-B ViT 백본 | ✗ |
| §IV-C voxel | ✗ |
| 데이터 증강 (auxiliary) | `PixelAugmentation` (paper에 명시 없음; alpha 추가) |
| ClearML 트래킹 (auxiliary) | alpha 추가 |

---

## 8. 1D vs 2D 알파 핵심 차이 (요약 — 자세한 비교는 `alpha_1D_vs_2D.md`)

- **상태 형태**: 1D `(B,S,512)+(B,S,256)` flat vs 2D `(B,S,256,369)` token
- **RSSM**: 1D MLP-GRUCell vs 2D ConvGRUCellGlo + TransformerDecoder
- **디코더**: 1D Linear→ConvT vs 2D token slice→ConvT
- **시퀀스**: 1D seq=8 vs 2D seq=6
- **증강**: 1D 없음 vs 2D PixelAugmentation
- **Action loss**: 1D 없음 vs 2D 1.0
- **KL free-bits**: 1D 항상 적용 (clamp_min) vs 2D cfg 게이트
- **ClearML**: 1D 없음 vs 2D 통합
- **6프레임 recon figure**: 2D 전용
- **near-field Chamfer**: 2D 전용 메트릭
