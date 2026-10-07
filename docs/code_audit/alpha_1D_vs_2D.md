# Alpha 1D vs Alpha 2D — 비교

`alpha26/scripts/model_variants/`의 `_muvo.py` 5개 vs `_muvo_2D.py` 5개 사이의 차이를 표 중심으로 정리한다. 세부 감사 결과는 `alpha_1D.md`, `alpha_2D.md` 참조.

---

## 1. Config 비교 (`config_muvo.py` vs `config_muvo_2D.py`)

| 키 | 1D | 2D | 메모 |
|---|---|---|---|
| `FUTURE_HORIZON` | 4 | **2** | seq 8 → 6 (paper voxel exp 일치) |
| `DATA.SAMPLE_EVERY_N` | 2 (static) | `_derive_sample_every_n(20,5,2)=2` (동적) | EFFECTIVE_HZ 기반 |
| `DATA._CARLA_HZ, _FRAME_STEP, _EFFECTIVE_HZ` | – | 20, 5, 2 | 신규 explicit |
| `DATA.AUGMENTATION.*` | 없음 | 7파라미터 namespace (Blur/Sharpen/ColorJitter) | 신규 |
| `MODEL.FUSION.TRANSFORMER_HEADS` | 8 | **4** | 64 dim/head (upstream convention) |
| `MODEL.TRANSITION.HIDDEN_STATE_DIM` | 512 | **256** | RSSMTD 단일 채널 |
| `MODEL.TRANSITION.STATE_DIM` | 256 | 256 | 동일 |
| `MODEL.RSSM_2D` namespace | 없음 | `IMAGE_TOKEN_HW=(10,24), LIDAR_TOKEN_HW=(2,64), LIDAR_OUT_INDICES=(1,2,3), POLICY_TOKENS=1, TRANSFORMER_DECODER_LAYERS=3, TRANSFORMER_DECODER_HEADS=4, DECODER_CHANNELS=512, NEARFIELD_PC_RANGE=(...)` | 신규 |
| `MODEL.POLICY.ENABLED` | – | True | 신규 (action loss용) |
| `LOSSES.WEIGHT_ACTION` | – | 1.0 | 신규 |
| `LOSSES.KL_FREE_BITS_ENABLED` | (항상 적용, 키 없음) | True (v4-2) | 신규 게이트 |
| `LOGGING.RUN_NAME` | `muvo_style_alpha_fpn256_lidar32` | `muvo_style_alpha_fpn256_lidar128_rssmtd` | – |
| `LOGGING.RECON_FIG_ALL_FRAMES` | – | True | 신규 (6 컬럼 figure) |
| `LOGGING.RECON_FIG_BOTH_VIEWS` | – | True | 신규 (scaled+raw depth) |
| `CML_ENABLED, CML_PROJECT, CML_TASK, CML_TYPE, CML_TAGS` | – | True, "alpha26_muvo_2D", ... | 신규 |

**공통 (변경 없음)**: `IMAGE_INPUT_SIZE=(320,768), LIDAR_RANGE_VIEW_SIZE=(32,1024), LIDAR_SCALE=40.0, MODEL.EMBEDDING_DIM=256, SPEED_CHANNELS=16, OPTIMIZER.LR=1e-4, WEIGHT_DECAY=0.01, SCHEDULER.NAME=OneCycleLR, ACCUMULATE_GRAD_BATCHES=1, ACTION_DIM=2, KL_BALANCING_ALPHA=0.75, WEIGHT_PROBABILISTIC=1e-2, WEIGHT_LIDAR_RE=0.0, WEIGHT_LIDAR_EMPTY=0.0, WEIGHT_RGB=1.0, WEIGHT_FUTURE=1.0, WEIGHT_SSIM=0.0`.

---

## 2. Data Pipeline 비교 (`data_muvo.py` vs `data_muvo_2D.py`)

| 항목 | 1D | 2D |
|---|---|---|
| Config import | `config_muvo` | `config_muvo_2D` |
| 신규 클래스 | – | `PixelAugmentation(aug_cfg)` (line 52–96) |
| `_make_img_transform` 시그니처 | `(size)` | `(size, augment=True)` |
| 증강 적용 | ✗ | `image` 키만 (`image_raw`, LiDAR 미적용) |
| 데이터셋 시그니처 | `MUVODataset(arrow_file, seq_len, ...)` | `MUVODataset(..., augment=True)` |
| `__getitem__` dict 키 | `{image, image_raw, lidar, action, speed, run_id, start_row}` | identical (호환) |
| Range-view 변환 (`point_cloud_to_range_view`) | identical | identical |
| Stream/Multi 데이터셋 클래스 | identical | identical |

---

## 3. Model 비교 (`models_muvo.py` vs `models_muvo_2D.py`)

### 3.1 클래스 인벤토리

| 클래스 | 1D | 2D | 비고 |
|---|---|---|---|
| `FPNDecoder` | ✓ | ✓ | 동일 인터페이스 |
| `PositionEmbeddingSine` | ✓ | ✓ | 동일 |
| `SensorFusionTransformer` | nhead=8, channels=256, 3L | nhead=4, channels=256, 3L | head 수 ↓ |
| `SpeedEncoder` (별도 클래스) | 없음 (inline) | ★ NEW | 책임 분리 |
| `SensorFeatureConv` | ✓ (1D 임베딩) | ✗ (토큰 그대로) | – |
| `RepresentationModel` (MLP) | ✓ | ✗ | – |
| `RepresentationModelTD` | ✗ | ★ NEW | TransformerDecoder 기반 |
| `RSSM` | ✓ (GRUCell) | ✗ | – |
| `RSSMTD` | ✗ | ★ NEW | 2D 토큰 상태 |
| `ConvGRUCellGlo` | ✗ | ★ NEW | Conv1d + global gate |
| `SensorHead` | ✓ (1×1 Conv) | (`_TokenHead`로 재명명) | 동일 기능 |
| `SensorDecoder` | ✓ | ✗ | – |
| `TokenConvDecoder2D` | ✗ | ★ NEW | 토큰→ConvT 사다리 |
| `PolicyDecoder` | ✗ | ★ NEW | MLP→Tanh(2) |
| `Model` | 1D 파이프라인 | 토큰 파이프라인 (재작성) | – |

### 3.2 상태 형태 비교

| 항목 | 1D | 2D |
|---|---|---|
| `h` (deterministic) | `(B, S, 512)` | `(B, S, 256, 369)` |
| `s` (stochastic) | `(B, S, 256)` | `(B, S, 256, 369)` |
| `N_token` | – | 240 (image) + 128 (lidar) + 1 (policy) = **369** |
| 디코더 입력 | `[h, s]` concat `(B*S, 768)` | 토큰 모달리티별 slice `(B*S, 256, H_base, W_base)` |
| Recurrent 연산 | `nn.GRUCell(512, 512)` flat | `Conv1d` over tokens + global gate `mean(hx, -1)` |
| Prior/Posterior | MLP `RepresentationModel` | TransformerDecoder `RepresentationModelTD` |

### 3.3 Action 주입

| 항목 | 1D | 2D |
|---|---|---|
| `ACTION_IN_GRU` | True (alpha 디자인) — GRU 입력에 latent_action 결합 | True (dead config; RSSMTD가 소비 안 함) |
| Prior 모듈 | `prior_action_module(action) + linear→μ,σ` | `prior_action_module(action)` + token type=2로 concat → TransformerDecoder |
| Posterior 모듈 | `posterior_action_module(action) + [h, o, latent_a] linear→μ,σ` | `posterior_action_module(action)` + concat with [h_tokens, embedding] → TransformerDecoder |

### 3.4 디코더 출력

| 출력 키 | 1D | 2D |
|---|---|---|
| `rgb_1, rgb_2, rgb_4` | ✓ | ✓ |
| `lidar_reconstruction_1, _2, _4` | ✓ | ✓ |
| `action_pred` | – | ✓ |

---

## 4. Train Entrypoint 비교 (`train_muvo.py` vs `train_muvo_2D.py`)

| 항목 | 1D | 2D |
|---|---|---|
| Config import | `config_muvo` | `config_muvo_2D, _derive_sample_every_n` |
| Trainer/Models import | `_muvo` | `_muvo_2D` |
| 신규 CLI | – | `--effective-hz, --no-augmentation, --no-clearml, --cml-project, --cml-task` |
| h-z 제약 | 없음 | `h-dim != z-dim`이면 z-dim으로 강제 (warning) |
| ClearML init | – | `maybe_init_clearml(run_cfg, args)` (Task.init + connect hparams) |
| OneCycleLR resume 메시지 | print 있음 | silent |
| `save_loss_metric_figure` | unguarded | try/except |
| change_summary | `MUVO-style alpha: ...` | 상세 RSSMTD 스펙 |
| Argparse 공통 | 동일 (run_name/series/epochs/batch_size/lr/devices/strategy/precision/num_workers/limit-val-batches/save-top-k/monitor/...) | 동일 |

---

## 5. Trainer 비교 (`trainer_muvo.py` vs `trainer_muvo_2D.py`)

| 항목 | 1D | 2D |
|---|---|---|
| Models import | `from models_muvo import Model` | `from models_muvo_2D import Model` |
| TBPTT 상태 carry | `_tbptt_h, _tbptt_s, _tbptt_action` (`training_step`에서 continuation 분기) | 없음 — `training_step`이 `_observe_and_imagine(batch)` 직접 호출 |
| Action loss | – | ✓ `WEIGHT_ACTION × F.l1_loss(action_pred, target)` |
| Near-field Chamfer 메트릭 | – | ✓ `chamfer_distance_range_view(scale-corrected, pc_range=nearfield_pc_range)` |
| KL free-bits 게이트 | always clamp_min (free_bits 값이 0이면 효과 없음) | `kl_free_bits_enabled` cfg 게이트로 명시 ON/OFF |
| Reconstruction figure 모드 | 2×4 fixed | `RECON_FIG_ALL_FRAMES` (RF+FH 풀 컬럼) + `RECON_FIG_BOTH_VIEWS` (scaled+raw depth) |
| ClearML artifact 업로드 | – | `test_step` try/except로 처음 4 배치 |
| ExperimentCSVLogger 메트릭 추가 | 기본 | + `val/RL_lidar_chamfer_xyz_nearfield`, `val/DS_...` |
| Loss 항 공통 | RGB L1 multi-scale, LiDAR L2 xyz / L1 r / smooth-L1 empty, KL balanced, SSIM, future scaled | 동일 + action |

---

## 6. 종합 요약 표

| 차원 | 알파 1D | 알파 2D |
|---|---|---|
| **시퀀스** | RF=4, FH=4 → seq 8 | RF=4, FH=2 → seq 6 |
| **샘플링** | 0.5 s 간격(`SAMPLE_EVERY_N=2`, `FRAME_STEP=5`, `CARLA_HZ=20`) | 동일 (단 `EFFECTIVE_HZ` 명시) |
| **이미지** | 320×768 | 동일 |
| **LiDAR Range View** | 32×1024 | 동일 |
| **모델 핵심** | MILE-style, MLP-GRUCell RSSM | RSSMTD (token state, ConvGRU + TransformerDecoder) |
| **상태 차원** | h=512, z=256 (flat) | h=256, z=256 (token, N=369) |
| **Fusion** | Transformer 3L/8H, channels 256 | Transformer 3L/4H, channels 256 |
| **Decoder** | SensorDecoder (Linear→ConvT) | TokenConvDecoder2D (token slice→ConvT) |
| **Decoder 내부 채널** | 512 (default) | 512 (DECODER_CHANNELS) |
| **Loss: RGB L1 multi-scale** | ✓ | ✓ |
| **Loss: LiDAR xyz/range/empty** | 가중치 0 (꺼짐) ⚠ | 가중치 0 (꺼짐) ⚠ |
| **Loss: KL balanced** | ✓ (free-bits=1.0 항상 적용) | ✓ (free-bits 게이트로 옵션) |
| **Loss: SSIM** | weight=0 | weight=0 |
| **Loss: Action L1** | – | ✓ (weight=1.0) |
| **Augmentation** | – | PixelAugmentation (Blur/Sharpen/ColorJitter) |
| **TBPTT carry** | ✓ | ✗ (간소화) |
| **Recon figure** | 2×4 (obs/post/target/prior) | RF+FH 6 컬럼 + dual depth view |
| **Near-field Chamfer 메트릭** | – | ✓ |
| **Experiment 트래킹** | CSV + TensorBoard | + ClearML |
| **CLI 신규 플래그** | – | `--effective-hz, --no-augmentation, --no-clearml, --cml-project, --cml-task` |
| **검증** | val_rl/val_ds split (동일 분포) | 동일 |
| **Paper §IV-B latent space 권고** | 1D baseline | ✓ 2D latent state 구현 |
| **Paper §IV-A 손실 부족 (LiDAR off)** | ⚠ | ⚠ |
| **Paper occupancy 부재** | ✗ | ✗ |

---

## 7. 가장 큰 디자인 결정 3가지

1. **상태 형태 (1D flat → 2D token)**: paper §IV-B의 headline 결론("2D latent significantly benefits camera predictions")을 그대로 반영. alpha-2D의 정당성을 이루는 핵심 변경.
2. **Decoder 분리 (Linear→ConvT vs Token slice→ConvT)**: 토큰 형태 유지로 spatial 정보 손실 최소화. paper §III-D "토큰을 모달리티별 분할 후 reshape"와 일치.
3. **보조 학습 신호 추가 (action loss, augmentation, near-field metric)**: paper 본문에는 없으나 alpha의 데이터/도메인에 맞춘 실용적 보강. 단 paper와의 직접 비교 시 분리해 검토 필요.
