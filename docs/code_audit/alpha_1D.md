# Alpha 1D Code Audit — `alpha26/scripts/model_variants/*_muvo.py`

알파 26 프로젝트의 1D 분기(`_muvo.py` 파일군) 5개 파일 감사 결과를 정리한다. 각 파일은 별도 Explore 서브에이전트로 감사하였다.

| 파일 | 줄 수 | 책임 |
|---|---|---|
| `config_muvo.py` | 87 | SimpleNamespace cfg 정의 |
| `data_muvo.py` | 263 | Arrow 데이터셋 + range-view 변환 |
| `models_muvo.py` | 631 | 멀티모달 1D RSSM world model |
| `train_muvo.py` | 367 | 학습 진입점 (PL Trainer) |
| `trainer_muvo.py` | 829 | LightningModule + loss/metric/figure |

---

## 1. `config_muvo.py` — 학습 셋팅

`cfg` (SimpleNamespace) 주요 키:

| 그룹 | 키 | 값 |
|---|---|---|
| 시퀀스 | `RECEPTIVE_FIELD`, `FUTURE_HORIZON` | 4, 4 (seq=8) |
| 데이터 경로 | `DATA.ROOT, MANIFEST_PATH, TRAIN_RUN, VAL_RL_RUN, VAL_DS_RUN, USE_ALL_TRAIN_RUNS` | Arrow 파일 경로 + `train_run_002.arrow`, `validation_run_008.arrow`, `validation_run_024.arrow`, `True` |
| 입력 크기 | `DATA.IMAGE_INPUT_SIZE, RGB_RECON_SIZE` | `(320, 768)` 동일 |
| LiDAR | `DATA.LIDAR_RANGE_VIEW_SIZE, LIDAR_FOV_DEGREES, LIDAR_SCALE` | `(32, 1024)`, `(-30.0, 10.0)`, `40.0` |
| 샘플링 | `DATA.SAMPLE_EVERY_N, FRAME_STEP, NUM_WORKERS` | 2, 5, 2 |
| 모델 | `MODEL.ACTION_DIM, EMBEDDING_DIM, SPEED_CHANNELS, SPEED_NORMALISATION` | 2, 256, 16, 50.0 |
| Fusion | `MODEL.FUSION.TRANSFORMER_{CHANNELS,LAYERS,HEADS,DROPOUT}` | 256, 3, 8, 0.1 |
| RSSM | `MODEL.TRANSITION.{ENABLED, HIDDEN_STATE_DIM, STATE_DIM, ACTION_LATENT_DIM, USE_DROPOUT, DROPOUT_PROBABILITY, ACTION_IN_GRU}` | True, 512, 256, 64, False, 0.0, **True** |
| Optimizer | `OPTIMIZER.LR, WEIGHT_DECAY, ACCUMULATE_GRAD_BATCHES` | 1e-4, 0.01, 1 |
| Scheduler | `SCHEDULER.NAME, PCT_START` | OneCycleLR, 0.1 |
| Loss 가중치 | `LOSSES.WEIGHT_PROBABILISTIC, KL_FREE_BITS, KL_BALANCING_ALPHA, WEIGHT_LIDAR_RE, WEIGHT_LIDAR_EMPTY, WEIGHT_RGB, WEIGHT_SSIM, WEIGHT_FUTURE` | **1e-2, 1.0, 0.75, 0.0, 0.0, 1.0, 0.0, 1.0** |
| 로깅 | `LOGGING.RUN_NAME, BASE_FILE, EXPERIMENT_LOG_PATH, SERIES` | `muvo_style_alpha_fpn256_lidar32`, `train_muvo.py`, `results/experiment_log.csv`, `""` |
| 학습 길이 | `EPOCHS, STEPS` | 1, 400 |
| 사전 학습 | `PRETRAINED.PATH` | None |

**환경 변수**: `PROJECT_ROOT = ALPHA26_ROOT or /home/carol/chaeyeon-kim/alpha26`; `DATA_ROOT = CARLA_ARROW_ROOT or {PROJECT_ROOT}/processed`.

**Mismatch flags**:
- `WEIGHT_LIDAR_RE=0.0`, `WEIGHT_LIDAR_EMPTY=0.0` — LiDAR 재구성 손실이 꺼짐 (paper §IV-A 식 (1)의 $`ℒ^{\text{pxyz}}`$, $`ℒ^{\text{pr}}`$ 이 없는 상태). → improvement plan P0.
- `EPOCHS=1` + `STEPS=400`은 quick test 모드. 본격 학습 시 CLI 오버라이드 필요.
- `USE_DROPOUT=False, DROPOUT_PROBABILITY=0.0` 중복.
- `USE_ALL_TRAIN_RUNS=True`인데 `TRAIN_RUN`도 지정 — CLI에서 `--train-run` 사용 시 자동으로 `USE_ALL_TRAIN_RUNS=False`로 변경.

**한국어 주석 (line 47–49)**: `ACTION_IN_GRU=True`가 alpha의 디자인 선택. upstream MUVO는 `False` (action을 prior/posterior MLP에만 주입).

---

## 2. `data_muvo.py` — Arrow 데이터 파이프라인

핵심 함수/클래스:

| 이름 | 줄 | 역할 |
|---|---|---|
| `CenterCropToSize(size)` | 26–37 | PIL 중앙 크롭, 부족 시 ValueError |
| `_make_img_transform(size)` | 40–46 | 인코더 입력: CenterCrop → ToTensor → ImageNet 정규화 |
| `_make_img_transform_raw(size)` | 49–56 | 디코더 타깃: CenterCrop → ToTensor (정규화 없음) |
| `point_cloud_to_range_view(points, H, W, fov_degrees)` | 63–93 | Nx3 → spherical 변환 → (4, H, W) range view (x, y, z, r) + depth-sort 가림 처리 |
| `build_arrow_manifest`, `load_arrow_manifest`, `_select_from_manifest` | 100–137 | Arrow manifest 빌드/로드 + split 선택 |
| `MUVODataset(arrow_file, seq_len=4, ...)` | 145–213 | 한 Arrow 파일에서 seq_len 윈도우 인덱스 빌드. `__getitem__`은 dict 반환 |
| `StreamWindowDataset` | 219–242 | TBPTT batch-slot 정렬을 위해 인덱스를 `num_streams` chunk로 interleave |
| `MultiArrowStreamDataset` | 246–263 | 여러 Arrow 파일을 cumsum으로 글로벌 idx 라우팅 |

`MUVODataset.__getitem__(idx)` 반환 dict:

| 키 | shape | dtype |
|---|---|---|
| `image` | `(seq_len, 3, H, W)` (320×768) | float32, ImageNet 정규화 |
| `image_raw` | `(seq_len, 3, H, W)` | float32 [0, 1] |
| `lidar` | `(seq_len, 4, H, W)` (32×1024) | float32, `/LIDAR_SCALE` |
| `action` | `(seq_len, 2)` (throttle, steer) | float64 |
| `speed` | `(seq_len,)` (km/h) | float64 |
| `run_id` | string | – |
| `start_row` | int | TBPTT continuity check |

Range-view 변환: spherical `(yaw, pitch)` 격자에 점을 빈닝하고 깊이 정렬로 가림 처리. FOV `(-30°, 10°)`.

**소비하는 cfg 키**: `cfg.DATA.{IMAGE_INPUT_SIZE, RGB_RECON_SIZE, LIDAR_RANGE_VIEW_SIZE, LIDAR_FOV_DEGREES, LIDAR_SCALE, ROOT, MANIFEST_PATH, SAMPLE_EVERY_N, FRAME_STEP}`.

**TODO/주석**: 라인 236 "각 chunk tail (min_len 이후)은 TBPTT batch-slot 정렬을 위해 의도적으로 버린다." — 명시된 설계 선택.

---

## 3. `models_muvo.py` — 1D world model

상위 클래스 인벤토리:

| 클래스 | 줄 | 책임 |
|---|---|---|
| `FPNDecoder(feature_info, out_channels=256)` | 25–50 | 다중 stage feature를 동일 채널로 통합, smallest spatial로 합성 |
| `PositionEmbeddingSine(num_pos_feats=64, ...)` | 57–92 | 2D sinusoidal PE |
| `SensorFusionTransformer(channels, num_layers=3, nhead=8, dropout=0.1)` | 95–131 | cam/LiDAR 토큰 fusion (TransformerEncoder, batch_first=False) |
| `BasicBlock` | 134–164 | ResNet 잔차 블록 |
| `SensorFeatureConv(in_channels, out_channels)` | 167–185 | BasicBlock×2 + AdaptiveAvgPool + Flatten → 1D 임베딩 |
| `RepresentationModel(in_channels, latent_dim)` | 192–207 | Linear×2 + LeakyReLU → (μ, σ) |
| `RSSM(embedding_dim, action_dim, hidden_state_dim, state_dim, action_latent_dim, receptive_field, use_dropout, dropout_probability, action_in_gru)` | 210–342 | 1D 상태 RSSM, GRUCell, prior/posterior MLP |
| `SensorHead(in_channels, out_channels, downsample_factor, sensor_type)` | 375–387 | 1×1 Conv 출력 헤드 |
| `SensorDecoder(latent_n_channels, out_channels, target_size, sensor_type, decoder_channels=512, n_pre_doublings)` | 390–468 | Linear → ConvTranspose 사다리 + SensorHead×3 multi-scale |
| `Model(cfg, embedding_n_channels, transformer_encoder)` | 475–631 | timm ResNet18 ×2 + FPN ×2 + SensorFusionTransformer + SensorFeatureConv ×2 + speed_encoder + RSSM + SensorDecoder ×2 |

**Forward 흐름** (`Model.forward(batch)` ≈ L530–L590):
1. `image (B,S,3,H,W) + lidar (B,S,4,H,W) + speed (B,S)`
2. Image → ResNet18 multi-stage → FPN → `(B*S, 256, Hf, Wf)`
3. LiDAR → ResNet18 in_chans=4 multi-stage → FPN → `(B*S, 256, Hf, Wf)`
4. `SensorFusionTransformer(cam, lidar)` → `(cam_out, lidar_out)` 동일 형태
5. `SensorFeatureConv` ×2로 1D 임베딩 `(B,S,256)` 추출, speed 임베딩 `(B,S,16)` 결합
6. `features_combine(Linear)` → `embedding_seq (B,S,256)`
7. RSSM 루프: posterior `(o,h,a)`, prior `(h,a)` — `h, z ∈ ℝ^{B,S,?}` 출력
8. `image_decoder([h,z]) → {rgb_1, rgb_2, rgb_4}`; `lidar_decoder([h,z]) → {lidar_reconstruction_1, _2, _4}`

**RSSM 핵심 (paper §III-C와 매핑)**:
- `nn.GRUCell(input_size=hidden_state_dim, hidden_size=hidden_state_dim)` (line 243).
- `ACTION_IN_GRU=True` (alpha 디자인): GRU 입력 = `[sample_t, latent_action_t]` → h 진화에 action 반영. upstream MUVO(`False`)와 다름.
- posterior: 같은 h 사용 (GRU step 한 번만). dropout on prior로 일부 posterior를 prior로 교체 가능 (`USE_DROPOUT`).
- 초기화: `h_init=0, s_init=0` (continuation flag로 첫 step action 특별 처리).

**Decoder 헤드**: 모달리티 ∈ {rgb, lidar}, downsample factor ∈ {1, 2, 4} → 총 6 헤드. RGB는 3채널, LiDAR는 4채널 (x, y, z, r).

**소비 cfg 키**: `cfg.RECEPTIVE_FIELD, MODEL.{EMBEDDING_DIM, FUSION.*, SPEED_CHANNELS, SPEED_NORMALISATION, ACTION_DIM, TRANSITION.*}`, `cfg.DATA.{RGB_RECON_SIZE, LIDAR_RANGE_VIEW_SIZE}`.

**관찰점**:
- 라인 1 한국어 주석 `#고친이후`.
- 라인 18 `sigmoid2()`: `/2, *2` 하드코딩.
- 라인 103 `batch_first=False` — 시퀀스 우선 텐서 레이아웃.
- 라인 433/436: ConvTranspose 커널 6 (짝수) + stride 2 + padding 2 — upstream 스타일이지만 흔치 않은 패턴.

---

## 4. `train_muvo.py` — 학습 진입점

**Argparse 플래그** (30+개): `--run-name, --series, --epochs, --batch-size, --lr, --h-dim, --z-dim, --weight-ssim, --devices, --strategy, --precision, --num-workers, --limit-val-batches, --check-val-every-n-epoch, --save-top-k, --monitor, --monitor-mode, --train-run, --val-rl-run, --val-ds-run, --resume-from-checkpoint, --overfit-samples, --one-window-overfit, --validate-overfit, --no-reconstruction, --no-metric-export, --recon-every-n-epochs, --recon-sample-idx`.

**메인 흐름** (`main()`):
1. `parse_args()` → validate `--one-window-overfit`은 batch_size=1 강제
2. `apply_overrides(cfg, args)` — CLI → cfg 키 매핑
3. `resolve_runtime()` — CUDA 디바이스 수 → (accelerator, devices, strategy, precision, world_size)
4. `build_dataloaders()` — Arrow manifest 로드 + `_select_from_manifest` + `MultiArrowStreamDataset` + DataLoader
5. `WorldModelTrainer(cfg, lr)` 인스턴스화 (from `trainer_muvo.py`)
6. `build_callbacks()` — `ModelCheckpoint, ExperimentCSVLogger`, 옵션 `PeriodicReconstructionCallback`
7. `pl.Trainer(...)` (TensorBoard logger + DDP if world_size>1)
8. OneCycleLR 체크포인트 재개 시 total_steps 패치
9. `torch.set_float32_matmul_precision('medium')`
10. `trainer.fit()` (overfit mode이면 validate 옵션 분기)
11. `save_loss_metric_figure()` (rank 0만)
12. 최종 reconstruction 그림 (best checkpoint 로드 → `save_reconstruction_figure`)

**Logging**:
- TensorBoard: `./logs/{RUN_NAME}/`
- CSV: `results/experiment_log.csv` (ExperimentCSVLogger 콜백)
- Metrics PNG: `results/figures/{RUN_NAME}_loss_metrics.png`
- Recon PNG: `results/figures/{RUN_NAME}_reconstruction_s{IDX}.png`

**DDP**: `resolve_runtime`이 자동 감지. world_size>1 시 `use_distributed_sampler=True`. global_batch_size = per_device × world_size.

---

## 5. `trainer_muvo.py` — LightningModule

주요 클래스:

| 클래스 | 줄 | 역할 |
|---|---|---|
| `SSIMLoss` | 16–62 | Gaussian-kernel SSIM, `(B,S,C,H,W)` 입력 |
| `WorldModelTrainer(pl.LightningModule)` | 244–698 | 메인 학습 모듈 |
| `PeriodicReconstructionCallback` | 704–738 | N epoch 마다 2×4 recon 그림 저장 |
| `ExperimentCSVLogger` | 744–830 | on_fit_end에서 한 행 CSV 추가 |

`WorldModelTrainer`:
- `__init__(cfg, hparams, lr, embedding_n_channels)` → `self.model = Model(cfg)` (from models_muvo)
- attrs: `weight_probabilistic, weight_lidar_re, weight_lidar_empty, weight_rgb, weight_future, weight_ssim, kl_balancing_alpha`
- TBPTT 상태 추적: `_tbptt_h, _tbptt_s, _tbptt_action`
- `forward(batch) → model output`
- `training_step(batch, batch_idx)` — TBPTT (`continuation` flag), `_observe_and_imagine()` 호출
- `validation_step(batch, batch_idx, dataloader_idx=0)` — eval 메트릭(PSNR, Chamfer, xyz 거리, range MAE)

**Loss 항** (`compute_loss(batch, output)`, 라인 ~440–560):

| 항 | 공식 | 가중치 키 |
|---|---|---|
| KL | `balanced_kl_loss = α·KL(prior‖posterior) + (1-α)·KL(posterior‖prior)` with optional free-bits clamp | `WEIGHT_PROBABILISTIC` (기본 1e-2) |
| LiDAR xyz | `F.mse_loss(pred[..., :3], target[..., :3])` 유효 점만 | `WEIGHT_LIDAR_RE × 1/factor` |
| LiDAR depth | `F.l1_loss(pred_d, target_d)` 유효 점 | `WEIGHT_LIDAR_RE × 1/factor` |
| LiDAR empty | `F.smooth_l1_loss(pred_d, 0)` 빈 점 | `WEIGHT_LIDAR_EMPTY × 1/factor` |
| RGB L1 | `F.l1_loss(pred, target)` | `WEIGHT_RGB × 1/factor` |
| SSIM | `1 − SSIM(pred, target)` | `WEIGHT_RGB × WEIGHT_SSIM × 1/factor` |
| Future_* | 위 항을 future 부분에 동일 적용 | `× WEIGHT_FUTURE` |

Discount factors: [1, 2, 4] multi-scale (paper §IV-A의 1/2/4 다운샘플과 일치).

**KL 처리**:
- `gaussian_kl_loss()` — free-bits clamp `kl_sum.clamp_min(free_bits)`
- `balanced_kl_loss()` — alpha=0.75, prior_loss(detached posterior) + posterior_loss(detached prior)

**Logging scalars**:
- 학습: `train_loss`, `train_{term}`, `train_probabilistic_step`
- 검증: `val/{RL,DS}_loss`, `val/{RL,DS}_{psnr, chamfer, euclidean, range_mae}`, future 메트릭 포함

**Reconstruction figure** (`save_reconstruction_figure`):
- 2×4 subplot — 관측 RGB, posterior RGB, future target RGB, prior future RGB / 동일 depth/LiDAR row
- 저장 경로: `results/figures/{run_name}_reconstruction_s{sample_idx}.png`
- `PeriodicReconstructionCallback`에서 매 N epoch 호출 (선택)

**ExperimentCSVLogger** 필드: date, run_name, base_file, train_files, val_files, sample_hz, rgb_recon_size, lidar_size, lidar_fov, h_dim, z_dim, lr, max_steps, scheduler, batch_size, change_summary, final_val_metrics, notes, series. on_fit_end에서 `cfg.LOGGING.EXPERIMENT_LOG_PATH`에 한 행 append.

**알려진 주석** (line 267-268): `# 0.6은 alpha의 WEIGHT_RGB=1.0과 곱해져 upstream의 10×가 되어 학습 불안정.` → SSIM 가중치 튜닝 시도 흔적.

---

## 6. 모듈 간 의존 관계

```
config_muvo  ──┐
   ↓           │
data_muvo  ──→ train_muvo ←── trainer_muvo ←── models_muvo
                   │              │
                   │              └── compute_loss, save_reconstruction_figure, ExperimentCSVLogger
                   └── DataLoader, runtime, callbacks
```

---

## 7. Paper §III/§IV ↔ alpha-1D 매핑

| Paper | alpha-1D 위치 |
|---|---|
| §III.A 이미지/LiDAR encoder | `models_muvo.py:Model.encode_fuse_sequence` (timm ResNet18 + FPNDecoder) |
| §III.B Transformer fusion | `models_muvo.py:SensorFusionTransformer` (3L, 8H, channels=256) |
| §III.C 1D RSSM (baseline) | `models_muvo.py:RSSM` (GRUCell, `(B,S,256+512)` 상태) |
| §III.D 카메라/LiDAR ConvT 디코더 | `models_muvo.py:SensorDecoder` + `SensorHead` |
| §IV-A 식 (1) | `trainer_muvo.py:compute_loss` — LiDAR 항 가중치 0으로 꺼짐 ⚠ |
| §IV-A multi-scale | discount factors `[1, 2, 4]` |
| §IV-A 0.2s 샘플 | `cfg.DATA.SAMPLE_EVERY_N=2 × FRAME_STEP=5 / CARLA_HZ=20 = 0.5s` (paper 0.2s와 다름) |
| §IV-A seq 12 | alpha seq = 4 + 4 = 8 (paper voxel exp seq=6도 아님; 1D는 paper baseline 대비 짧음) |
| §IV-A AdamW LR=1e-4, WD=0.01 | ✓ 일치 |
| §IV-B 2D latent | ✗ (alpha-1D는 1D latent) |
| §IV-B PL/ViT | ✗ |
| §IV-C voxel | ✗ |
