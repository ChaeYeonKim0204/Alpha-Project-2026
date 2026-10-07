# 3-Way Comparison: alpha non-muvo × alpha `_muvo` × upstream muvo

작성일: 2026-05-15

`alpha26/scripts/model_variants/`의 두 분기(`*.py` non-muvo / `*_muvo.py`)와 upstream
`muvo/`의 default config 학습 경로를 한눈에 비교. cross-reference로 [[muvo_default_config_audit]],
[[ssim_addition_notes]], [[muvo_style_alpha_model_notes]], [[_muvo_comparison]] 참조.

## 약식 표기

- **A** = `alpha26/scripts/model_variants/*.py` (non-muvo, trainer11 직계). 진입점 `train.py`.
- **B** = `alpha26/scripts/model_variants/*_muvo.py` (alpha의 muvo-style 분기, BEV/3D 제외).
  진입점 `train_muvo.py`.
- **C** = `muvo/muvo/*` upstream default config (`muvo.yml` + `config.py`).
  진입점 `muvo/train.py`.

A와 B의 관계: B는 `_muvo` scaffolding 만들면서 A의 4개 파일(`config/data/train/trainer`)을
사이드카로 복제 후 import만 `_muvo` 버전으로 바꾼 것. `models_muvo.py`만 진짜 큰 변경
(upstream mile.py 이식). 그래서 A↔B 차이는 대부분 **config 수치 + models**에 집중되고,
data/trainer/train 구조는 거의 동일.

## 한 줄 요약

| 축 | A (non-muvo) | B (`_muvo`) | C (upstream) |
|---|---|---|---|
| 입력 image | (300, 400) | (320, 768) | (320, 832) crop after |
| 입력 LiDAR | (32, 256) | (32, 1024) | range-view 또는 PointPillar |
| RSSM h/z | 256/128 | 512/256 | **1024/512** |
| 시퀀스 RF/FH | 4 / 4 | 4 / 4 | 4 / 2 |
| Active heads | RGB + KL + future | RGB + KL + future | **RGB + LIDAR_RE + VOXEL_SEG** + KL + Action |
| RGB loss | L1 | L1 (+ SSIM optional) | L1 (SSIM/Perceptual gated, default off) |
| LiDAR loss | OFF (`WEIGHT_LIDAR_RE=0`) | OFF (`WEIGHT_LIDAR_RE=0`) | **ON** (`WEIGHT_LIDAR_RE=0.1`) |
| 입력 데이터 | Arrow (오프라인) | Arrow (오프라인) | CARLA on-the-fly 또는 disk dataset |
| Config 시스템 | `SimpleNamespace` | `SimpleNamespace` | `yacs` (`CfgNode`) + `.yml` |
| 진입점 CLI | `argparse` | `argparse` | hydra-like config merge |

## 1. Config / Data 사이즈

| | A | B | C (upstream default) |
|---|---|---|---|
| Image input H×W | `(300, 400)` | `(320, 768)` | 원본 `(600, 960)` → asymmetric crop bbox `[left=64, top=138, right=896, bottom=458]` (`muvo/config.py:111-112`) → `(320, 832)`. 위쪽 138/아래 142px (sky+hood 제거), 좌우 각 64px |
| RGB recon target | `(216, 288)` (resize) | `(320, 768)` (input == output) | `(320, 832)` (input == output) |
| LiDAR range-view | `(32, 256)` 4ch | `(32, 1024)` 4ch | `(32 or 64, 1024)` config-dependent |
| LiDAR FOV (°) | `(-30, 10)` | `(-30, 10)` | `[-30, 10]` |
| LiDAR scale | `40.0` (`LIDAR_SCALE`) | `40.0` | upstream은 normalisation 다름 |
| RECEPTIVE_FIELD | 4 | 4 | 4 (yml override) |
| FUTURE_HORIZON | 4 | 4 | **2** |
| sample_every_n | 2 | 2 | 별도 stride 기반 |
| frame_step | 5 | 5 | dataset에 별도 |

## 2. Model 아키텍처

### 인코더 / Fusion

| | A | B | C |
|---|---|---|---|
| Image encoder | `timm.resnet18` `out_indices=[2,3,4]` | 동일 | 동일 |
| LiDAR encoder | `timm.resnet18 in_chans=4` | 동일 | 동일 (또는 `PointPillarNet`, default off) |
| Multi-scale fusion | `FPNDecoder` (out_channels **128**) | `FPNDecoder` (out_channels **256**, FPN index 순서 다름) | `DecoderDS` / `Decoder` (`TRANSFORMER.CHANNELS=384`, base default 256을 `muvo.yml:34`가 384로 override) |
| Pos embedding | 학습형 (sinusoidal) | sinusoidal | `PositionEmbeddingSine` + sensor-type embedding |
| Sensor fusion | `SensorFusionTransformer` (3 layer, 8 head, 128 ch) | 동일 구조, channels=**256** | `nn.TransformerEncoder` (**6 layer**, 8 head, **384 ch** = ViT-Small d_model 관행 + 모달리티 fusion 후 RSSM(512) 직전 capacity 확보) |
| Image feature pool | (없음, 직접 flatten) | (B는 `SensorFeatureConv = BasicBlock × 2` 후 fusion) | `BasicBlock × 2` + `AdaptiveAvgPool` |
| 보조 입력 인코더 | 없음 | 없음 | `route_map`/`measurements`/`speed` 인코더 (route default ON) |

### RSSM

| | A | B | C (FZI main, paper-cited) |
|---|---|---|---|
| HIDDEN_STATE_DIM | 256 | **512** | **1024** |
| STATE_DIM | 128 | **256** | **512** |
| ACTION_LATENT_DIM | 64 | 64 | 64 |
| USE_DROPOUT | False | False | **True** (p=0.15) |
| pre_gru 입력 | `[sample, action]` concat (Dreamer 표준) | flag `ACTION_IN_GRU` (default True = A와 동일) | `sample`만 (action은 GRU 이후 prior에만) |
| **State shape** | 1D flat `(D_s,)` | **1D flat `(D_s,)`** | **1D flat `(D_s,)`** in FZI main; **2D `(Ch, T_total)` in bogdoll/2D 브랜치 (paper §IV-B의 "largest boost" variant)** |

**Note**: paper §III-C가 2D latent state를 central innovation으로 endorse하나, paper-cited FZI repo의 main에는 1D 구현만 포함. 2D 구현은 저자 개인 repo `daniel-bogdoll/MUVO` `2D` 브랜치 (`transition_td.py:RSSMTD`, `ConvGRUCellGlo`, `RepresentationModelTD`) 에 존재. 자세한 매핑: [[muvo_paper_summary]].

### 디코더 / Output heads

| | A | B | C |
|---|---|---|---|
| RGB decoder | `SensorDecoder` 4-doubling + bilinear interpolate `(216, 288)` | `ConvDecoder` (upstream 패턴) 5-doubling (10, 24) → 직접 (320, 768) | `ConvDecoder` (320, 832) 또는 `BevDecoder` |
| LiDAR decoder | 동일 SensorDecoder | 동일 ConvDecoder | `ConvDecoder` 또는 `lidar_re` head |
| 출력 스케일 | rgb_1 / rgb_2 / rgb_4 | 동일 | 동일 |
| BEV decoder | 없음 | 없음 (의도적 제거) | `BevDecoder` (default off) |
| Voxel decoder | 없음 | 없음 (의도적 제거) | **`VOXEL_SEG` default ON** |
| Semantic seg | 없음 | 없음 | gated (default off) |
| Depth (mono) | 없음 | 없음 | gated (default off) |
| Policy / action | 없음 | 없음 | `Policy` head + action loss (default ON) |

## 3. Data 파이프라인

| | A | B | C |
|---|---|---|---|
| 입력 소스 | `train_run_*.arrow` (오프라인 dump) | 동일 Arrow | CARLA on-the-fly 또는 disk dataset |
| Image 전처리 | `Resize(300, 400)` + ImageNet norm | `CenterCropToSize(320, 768)` + ImageNet norm | 고정 bbox crop `[64, 138, 896, 458]` (좌우 대칭, 상하 비대칭) + **default ON augmentation** (학습 step에만): 30% blur OR 30% sharpen, 30% ColorJitter (B/C/S ±0.3, H ±0.1) — `muvo/models/preprocess.py:201-215`, gated by `self.training` |
| LiDAR 전처리 | `point_cloud_to_range_view` (xyz+depth → 32×256) | 동일 projection, 32×1024 | range-view 또는 PointPillar feature 또는 histogram |
| Action/Speed | `throttle_brake` + `steering` → 2D action, speed/50.0 | 동일 | throttle/steer + speed/5.0 |
| 윈도 구성 | `StreamWindowDataset` (TBPTT-align) → `MultiArrowStreamDataset` (concat) | 동일 클래스 (data_muvo는 A의 사본) | `CarlaDataset` 단순 sequential |
| 증강 | **없음** | **없음** | **학습 step default ON**: image blur 30% / sharpen 30% / ColorJitter 30%; route_map dropout/rotation/translate/scale/shear 각 2.5% — val/test는 bypass |

## 4. Trainer / Loss

| | A | B | C |
|---|---|---|---|
| 프레임워크 | `pl.LightningModule` | 동일 | 동일 |
| 활성 RGB loss | L1만 (`WEIGHT_RGB=1.0`) | L1만 (`WEIGHT_RGB=1.0`) | L1 (`rgb_weight=0.1` hardcoded) |
| SSIM | `config.WEIGHT_SSIM=0.06`이지만 base `trainer.py`는 안 읽음. `trainer_ssim.py` (subclass)에서만 사용 | `cfg.LOSSES.WEIGHT_SSIM` flag로 정식 통합, default 0.0 | `LOSSES.SSIM: False` default, gated `if cfg.LOSSES.SSIM:` |
| LiDAR loss | OFF (weight 0) | OFF (weight 0) | **ON** `WEIGHT_LIDAR_RE=0.1` |
| KL weight | `1e-2` | `1e-2` | **`1e-3`** |
| KL free-bits | 1.0 | 1.0 | (없음 — upstream은 free-bits 미사용) |
| KL balancing α | 0.75 | 0.75 | 0.75 |
| Future weight | 1.0 | 1.0 | (별도 future loss 없음; imagine은 학습 안 함, eval에만) |
| RGB 다중 스케일 | rgb_1/2/4 discount=1/factor | 동일 | 동일 |
| TBPTT | 수동 (`_tbptt_h`, `_tbptt_s`, run_id/start_row 정합) | 동일 (B는 A의 사본) | sequence 안에서만, 명시적 TBPTT carry는 별도 없음 |
| Auxiliary losses | (없음) | (없음) | **Voxel** (CE + class weights), **Route** (regression), KL/probabilistic |

## 5. Training entrypoint

| | A | B | C |
|---|---|---|---|
| CLI | argparse (`train.py`) | argparse (`train_muvo.py`, B는 `--h-dim/--z-dim/--weight-ssim` 추가) | yacs/hydra-like — `--config-file muvo.yml` + CLI override |
| DDP | `--strategy=ddp` `find_unused_parameters_true` | 동일 | Lightning auto |
| Val datasets | 2 (RL=`validation_run_008`, DS=`validation_run_024`) | 동일 (data_muvo가 A의 사본이라) | 3 (val0/val1/val2 separate loaders) |
| limit-val-batches | 20 (CLI default) | 20 | `VAL_CHECK_INTERVAL=5000` step 기반 |
| Checkpointing | `save_top_k=3` monitor=`val/RL_loss` (overfit이면 `train_loss_epoch`로 자동) | 동일 | `MyModelCheckpoint` (custom) |
| Logger | TensorBoard (`./logs/<run-name>/`) | 동일 | TensorBoard + ClearML (commented) |
| Reconstruction figure | 학습 후 `save_reconstruction_figure` 자동 2×4 (RGB+depth × 4 conditions) | 동일 | BEV/voxel/seg 시각화 별도 함수 |

## 6. Validation / Eval

| | A | B | C |
|---|---|---|---|
| PSNR | ✓ rgb_2 기준 | ✓ 동일 | ✓ + SSIM metric |
| Chamfer distance | ✓ subsample 2048 | ✓ 동일 | ✓ `CDMetric` |
| LiDAR range MAE / XYZ Euclidean | ✓ | ✓ | (별도 metric set) |
| Voxel IoU | (없음) | (없음) | ✓ (`SSCMetrics`, 3D occupancy) |
| Segmentation IoU | (없음) | (없음) | (gated) |
| Eval reconstruction figure | 학습 끝나면 1장 | 동일 | per-validation-interval video logging 가능 |

## 핵심 진화 패턴

### A → B (alpha 자체 evolution)

- **모델 capacity 증가**: 입력 (300×400 → 320×768), LiDAR (256→1024 width), RSSM (256/128 → 512/256), FPN channel (128 → 256), fusion transformer channel (128 → 256).
- **Decoder 교체**: SensorDecoder (k=4/s=2, bilinear interpolate) → ConvDecoder (k=5,6/s=2, output_padding, interpolate 없음) — commit `7099f87`.
- **Encoder fusion 보강**: SensorFeatureConv plain conv → BasicBlock × 2 residual — commit `7bf7df9`.
- **RSSM action 처리 토글 가능화**: `ACTION_IN_GRU` flag로 alpha 디자인/upstream 디자인 둘 다 — commit `0be4871`.
- **SSIM 정식 옵션화**: 별도 trainer_ssim.py 패턴 → trainer_muvo에 inline 통합 — commit `08336bc`. (default off)
- **Image preprocessing**: Resize → CenterCrop.
- **유지된 것**: TBPTT 메커니즘, Arrow 데이터 소스, 보조 head 부재, LiDAR loss 비활성 (`WEIGHT_LIDAR_RE=0`).

### B → C (`_muvo` ↛ upstream — 의도적 차이)

- **C의 추가 보조 head**: VOXEL_SEG, LIDAR_RE, Route, Policy 모두 default ON. B는 BEV/3D 제거 정책으로 다 OFF + LiDAR_RE도 weight 0.
- **C의 RSSM 2배 크기**: 1024/512 vs B의 512/256. + Dropout 활성 (p=0.15).
- **C의 transformer 더 깊음**: 6-layer vs B의 3-layer.
- **C의 RGB weight 1/10**: 0.1 vs B의 1.0. C는 multi-task balance, B는 RGB-only 단일 task.
- **C의 KL weight 1/10**: 1e-3 vs B의 1e-2. free-bits 없음.
- **C의 데이터**: CARLA on-the-fly / disk dataset, image/route augmentation. B는 Arrow 오프라인 dump 고정.
- **C의 시퀀스**: RF=4, FH=2 (B는 4/4).
- **C의 진입점**: yacs config merge. B는 argparse + SimpleNamespace.

### A ↛ C 직접 비교

B를 거치지 않고 A를 C에 직접 비교하면:
- 거의 모든 축에서 C가 더 크고 무거움 (model, data pipeline, loss heads, eval metrics).
- A에 RGB+LiDAR fusion world model이라는 컨셉만 공유, 구현은 거의 모든 부분 다름.
- A는 trainer11 직계로 alpha 데이터셋과 alpha 학습 인프라(Arrow, TBPTT, RL/DS 검증 dataset 구조)에 최적화됨. C는 CARLA 환경 직접 통합에 최적화.

## 파일별 참조 매핑

| 비교 축 | A 파일:라인 | B 파일:라인 | C 파일:라인 |
|---|---|---|---|
| Config | `config.py` 전체 | `config_muvo.py` 전체 | `muvo/muvo/config.py` + `muvo/configs/muvo.yml` |
| Model 정의 | `models.py` | `models_muvo.py` | `muvo/muvo/models/mile.py` |
| Trainer | `trainer.py` | `trainer_muvo.py` | `muvo/trainer.py` |
| Data | `data.py` | `data_muvo.py` (A 사본) | `muvo/muvo/data/dataset.py` |
| Entrypoint | `train.py` | `train_muvo.py` (A 사본 + B 전용 CLI) | `muvo/train.py` |
| SSIM 별도 | `trainer_ssim.py` (A의 subclass + monkey-patch entry `train_ssim.py`) | (없음 — B는 정식 통합) | (없음 — flag gated) |
