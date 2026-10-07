# Upstream 1D vs Alpha 1D — 비교

`/muvo/` (upstream MILE) vs `alpha26/scripts/model_variants/*_muvo.py` (alpha 1D 포트) 사이의 차이를 정리한다.

---

## 1. 스코프 차이

| 차원 | upstream 1D | alpha 1D |
|---|---|---|
| 데이터 형식 | CARLA에서 직접 수집(pickle DataFrame, PNG/NPY) | Arrow 파일 (`processed/train_run_*.arrow`)로 사전 변환 |
| 데이터 모달리티 | RGB + depth + LiDAR + voxel + BEV seg + instance + route map + semantic image + action + speed + reward + value | RGB + LiDAR(range view) + action + speed (subset) |
| 모델 출력 | 8 헤드 (RGB, Depth, LidarRe, LidarSeg, SemImage, BevSeg, BevCenter/Offset, VoxelSem) + Policy | 2 헤드 (RGB, LidarRe) + (TBPTT) |
| 환경 코드 | `carla_gym/` 전체 + `rl_birdview/` RL agent | 없음 (Arrow 데이터만 사용) |
| 라이브러리 | fvcore CfgNode, ClearML, optional Open3D | SimpleNamespace cfg, optional ClearML(2D만), no Open3D |

---

## 2. 데이터 파이프라인 비교

| 항목 | upstream 1D (`muvo/muvo/data/dataset.py`) | alpha 1D (`data_muvo.py`) |
|---|---|---|
| Dataset 클래스 | `CarlaDataset` (Pandas DataFrame backed) | `MUVODataset` + `StreamWindowDataset` + `MultiArrowStreamDataset` (Arrow backed) |
| Sample 키 | image, route_map, semantic_image, range_view_pcd_xyzd, range_view_pcd_seg, points_raw, num_points, voxel, birdview, birdview_label, instance_label, image_instance_mask, depth, depth_color, steering, throttle_brake, speed, reward, value_function, intrinsics, extrinsics | image, image_raw, lidar, action, speed, run_id, start_row |
| Image 사이즈 | 600×960 (paper 사양) | 320×768 (alpha 축소) |
| LiDAR range view | `PointCloud.do_range_projection` (`muvo/muvo/utils/geometry_utils.py`) | `point_cloud_to_range_view` (alpha 자체 구현, 4채널 xyz+r) |
| 시퀀스 | DataModule + Sampler 기반 | StreamWindowDataset + TBPTT-aligned interleave |
| Augmentation | `PreProcess + PixelAugmentation + RouteAugmentation` (`muvo/muvo/models/preprocess.py`) | ✗ (alpha 1D는 augmentation 없음) |
| 가져오는 cfg 키 | fvcore `CfgNode.DATAROOT, VERSION, STRIDE_SEC, FILTER_BEGINNING_OF_RUN_SEC, FILTER_NORM_REWARD` 등 | SimpleNamespace `cfg.DATA.*` |

---

## 3. 모델 비교

### 3.1 Encoder

| 항목 | upstream 1D (`mile.py`) | alpha 1D (`models_muvo.py`) |
|---|---|---|
| Backbone | ResNet18 (cam, multi-stage); ResNet18 in_chans=4 (LiDAR); 또는 frustum pooling/PointPillars 분기 | timm ResNet18 (cam) + timm ResNet18 in_chans=4 (LiDAR) |
| FPN | ✗ (`mile.py`는 backbone feature 직접 사용 + 별도 BEV backbone) | ✓ `FPNDecoder` (alpha 자체) |
| BEV branch | ✓ (frustum_pooling.py + chauffeurnet) | ✗ (BEV 미사용) |
| PointPillars 분기 | ✓ | ✗ |
| Route encoder | ✓ (`common.py:RouteEncode`) | ✗ |

### 3.2 Fusion

| 항목 | upstream 1D | alpha 1D |
|---|---|---|
| Transformer | 선택적 (config 분기) | 항상 사용 (3L, 8H, channels=256, batch_first=False) |
| PE | 2D sinusoidal | 동일 |
| Sensor type embedding | 2-slot (image/lidar) | 동일 |
| Speed handling | 1D vector concat | 동일 |
| 행동(action) | prior/posterior MLP 입력 | (선택) GRU 입력 (`ACTION_IN_GRU=True` 시) |

### 3.3 RSSM

| 항목 | upstream 1D (`transition.py:RSSM`) | alpha 1D (`models_muvo.py:RSSM`) |
|---|---|---|
| State shape | `h: (B,S,H)`, `s: (B,S,Z)` | 동일 |
| Recurrent | `nn.GRUCell(H, H)` | `nn.GRUCell(H, H)` |
| `ACTION_IN_GRU` | False (default) — GRU 입력 = `pre_gru_net(sample)` | **True** (alpha 디자인) — GRU 입력 = `[sample, latent_action]` |
| Prior `N_θ` | MLP on `[h, latent_a]` | MLP on `[h, latent_a]` |
| Posterior `N_φ` | MLP on `[h, embedding, latent_a]` | MLP on `[h, embedding, latent_a]` |
| Dropout (prior 대체) | `use_dropout` | `USE_DROPOUT=False` (alpha 비활성) |
| 초기화 | h=0, s=0 | h=0, s=0 |

### 3.4 Decoder

| 항목 | upstream 1D | alpha 1D |
|---|---|---|
| 헤드 | 8개 (RGB/Depth/LidarRe/LidarSeg/SemImage/BevSeg/BevCenter+Offset/VoxelSem) + Policy | 2개 (RGB/LidarRe) |
| ConvDecoder 구조 | `common.py:ConvDecoder` (Linear→ConvT 사다리, 모달리티별 인스턴스) | `models_muvo.py:SensorDecoder` (Linear→ConvT 사다리, 모달리티별 인스턴스) — 거의 동일 패턴 |
| Multi-scale | 1/2/4 | 1/2/4 |
| BEV decoder | ✓ (`BevDecoder`, adaptive instance norm) | ✗ |
| Voxel decoder | ✓ (`VoxelDecoder1`, 3D ConvT) | ✗ |
| 디코더 입력 | `[hidden_state, sample]` concat | 동일 |

---

## 4. Loss / Trainer 비교

### 4.1 Loss 항

| 항 | upstream 1D | alpha 1D |
|---|---|---|
| $`ℒ^{\text{img}}`$ (RGB L1 multi-scale) | ✓ `WEIGHT_*_RGB` per-scale | ✓ `WEIGHT_RGB=1.0`, factors [1,2,4] |
| $`ℒ^{\text{pxyz}}`$ (LiDAR L2) | ✓ `WEIGHT_LIDAR_RE` (`muvo/muvo/losses.py:RegressionLoss`) | ✓ 동일 (mse_loss) but `WEIGHT_LIDAR_RE=0.0` ⚠ |
| $`ℒ^{\text{pr}}`$ (LiDAR L1 range) | ✓ | ✓ but 같이 꺼짐 ⚠ |
| L_pcd | ✓ `WEIGHT_LIDAR_RE` (Chamfer 또는 SpatialRegressionLoss) | partial (smooth-L1 empty만, weight 0) |
| $`ℒ^{V,\text{scal}}`$ (Voxel SCAL) | ✓ (`VoxelLoss + SemScalLoss + GeoScalLoss`) | ✗ |
| L_seg (BEV/Sem image) | ✓ | ✗ |
| L_instance (center/offset) | ✓ | ✗ |
| L_depth | ✓ (`SpatialRegressionLoss`) | ✗ |
| L_reward | ✓ | ✗ |
| L_action | ✓ (throttle_brake, steering 회귀) | ✗ (alpha 1D는 행동 헤드 없음) |
| L_probabilistic (KL balanced) | ✓ `KLLoss(α=0.75)` | ✓ 동일 (custom balanced_kl_loss with free-bits clamp) |
| SSIM | ✓ optional | ✓ optional (weight 0 기본) |
| Future scale | ✓ | ✓ |

### 4.2 Trainer 구조

| 항목 | upstream 1D | alpha 1D |
|---|---|---|
| LightningModule | `WorldModelTrainer` (`muvo/muvo/trainer.py`) | `WorldModelTrainer` (`trainer_muvo.py`) |
| shared_step | ✓ — imagination sampling 포함 | 분리된 `_observe_and_imagine()` + training_step에서 TBPTT carry |
| TBPTT carry | ✗ (paper는 seq 12 한 번에) | ✓ (`_tbptt_h, _tbptt_s, _tbptt_action`) |
| 메트릭 | SSCMetrics, SSIMMetric, CDMetric, PSNRMetric, IoU 등 | PSNR, Chamfer, xyz euclidean, range MAE |
| Visualization | `visualise()` BEV+RGB+LiDAR+voxel+trajectory | `save_reconstruction_figure` 2×4 RGB+depth |
| Checkpoint | `MyModelCheckpoint` + `SaveGitDiffHashCallback` | `ModelCheckpoint` + `ExperimentCSVLogger` |
| Optimizer | AdamW + OneCycleLR + freeze list 지원 | AdamW + OneCycleLR (freeze 미지원) |

---

## 5. Validation 셋업

| 항목 | upstream 1D | alpha 1D |
|---|---|---|
| $`𝒟_{\text{val}}^{\text{RL}}`$ | ✓ — 동일 도시+날씨, 무작위 경로 | partial — `validation_run_008.arrow` (분포 분리 명시되지 않음) |
| $`𝒟_{\text{val}}^{\text{DS}}`$ | ✓ — 다른 날씨, 동일 도시 | partial — `validation_run_024.arrow` (분포 분리 명시되지 않음) |
| Sampler stride 분리 | 3 test dataset 별도 (2D는 더 정밀) | val_rl/val_ds 두 DataLoader |

---

## 6. CARLA 환경 비교

upstream 1D는 데이터 수집부터 학습까지 통합:

| 컴포넌트 | upstream | alpha |
|---|---|---|
| CARLA env | `carla_gym/` (envs/suites, obs_manager, task_actor, zombie 등) | ✗ |
| Data collection | `data_collect.py` + `DataWriter` | ✗ (Arrow 사전 변환된 데이터 사용) |
| Voxel offline 생성 | `data/generate_voxels.py` + `voxel_filter` | ✗ |
| RL expert | `rl_birdview/` PPO agent | ✗ |

---

## 7. "upstream에 있고 alpha에 빠진 것"

1. **3D occupancy 디코더 + SCAL loss** — paper §IV-A의 voxel grid 학습.
2. **BEV segmentation/center/offset 헤드** — paper §IV-B의 BEV-based baseline.
3. **Depth head** — paper §IV-A 입력은 stereo depth지만 supervision으로도 사용.
4. **Semantic image head** — auxiliary task.
5. **Instance segmentation (center/offset)** — auxiliary task.
6. **Reward / value function 출력** — RL bootstrap용.
7. **Route map encoder** — 계획 경로 정보 통합.
8. **Frustum pooling (BEV mapping)** — paper §IV-B "BEV" 분기.
9. **PointPillars 분기** — paper §IV-B "PP" 분기.
10. **Action 출력 (policy)** — alpha 1D에는 없음.
11. **Perceptual / LPIPS loss** — paper §IV-B Fig. 4의 PL.
12. **ViT 백본 분기** — paper §IV-B Fig. 4의 ViT.

---

## 8. "alpha에 추가된 것 (upstream에 없는)"

1. **`ACTION_IN_GRU=True`** — GRU 입력에 latent action concat (alpha 1D 디자인). upstream은 False.
2. **TBPTT carry** — `_tbptt_h, _tbptt_s, _tbptt_action`로 sequence window 사이 hidden 전달. upstream은 단일 시퀀스 학습.
3. **`StreamWindowDataset` interleave 패턴** — TBPTT batch-slot 정렬용 chunk interleave.
4. **`ExperimentCSVLogger`** — on_fit_end 한 행 추가, run/sample/loss/metric 요약 CSV 누적.
5. **`PeriodicReconstructionCallback`** — 매 N epoch마다 2×4 figure 저장.
6. **Arrow 데이터셋 + manifest** — 데이터 형식 자체가 다름.

---

## 9. Paper §III/§IV 일관성 검증

| Paper 요소 | upstream 1D | alpha 1D | 일치도 |
|---|---|---|---|
| Image 600×960 | ✓ | ✗ (320×768) | 부분 |
| LiDAR ≤60K pts → range view | ✓ | ✓ (Arrow에 이미 변환) | ✓ |
| ResNet18 백본 | ✓ | ✓ | ✓ |
| 2D sinusoidal PE + sensor embedding | ✓ | ✓ | ✓ |
| k-layer transformer encoder | ✓ | ✓ (3L/8H) | ✓ |
| 1D RSSM (baseline) | ✓ (paper Fig. 4 비교군) | ✓ | ✓ |
| 식 (1) multi-scale $`ℒ^{\text{img}}`$+$`ℒ^{\text{pxyz}}`$+$`ℒ^{\text{pr}}`$+$`ℒ^{V,\text{scal}}`$ | ✓ (모든 항) | ⚠ ($`ℒ^{\text{pxyz}}`$, $`ℒ^{\text{pr}}`$ 가중치 0; $`ℒ^{V,\text{scal}}`$ 없음) | 부분 |
| AdamW LR=1e-4, WD=0.01 | ✓ | ✓ | ✓ |
| Seq 12 | ✓ | ✗ (seq 8) | 부분 |
| 0.2 s 샘플링 | ✓ | ✗ (0.5 s) | 부분 |
| $`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$ 분리 | ✓ (명시 분리) | partial (val_rl/val_ds 파일명만) | 부분 |
