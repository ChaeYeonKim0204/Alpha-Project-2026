# Upstream 2D vs Alpha 2D — 비교

`/muvo_2d/` (upstream 2D-token MUVO) vs `alpha26/scripts/model_variants/*_muvo_2D.py` (alpha 2D 포트) 사이의 차이를 정리한다.

---

## 1. 스코프 차이

| 차원 | upstream 2D | alpha 2D |
|---|---|---|
| 데이터 형식 | CARLA + Pandas + Arrow 가능 | Arrow 파일만 |
| 데이터 모달리티 | 1D 동일 + voxel/BEV/route | RGB + LiDAR(range view) + action + speed |
| 모델 출력 | 8 헤드 + Policy + Voxel | 2 헤드 + Policy |
| 잠재 토큰 수 | **1093** (image 260 + lidar 256 + voxel 576 + policy 1) | **369** (image 240 + lidar 128 + policy 1) — voxel 제거 |
| Decoder | `ConvDecoder2D / ConvDecoder3D / PolicyDecoder` 별도 모듈 | `TokenConvDecoder2D / PolicyDecoder` (Conv3D 없음) |
| PerceptualLoss/LPIPS | ✓ (옵션) | ✗ |
| ClearML | ✓ (`prediction.py`) | ✓ (`train_muvo_2D.py` + `maybe_init_clearml`) — alpha가 더 깔끔 통합 |
| Augmentation | `preprocess.py:PixelAugmentation` | `data_muvo_2D.py:PixelAugmentation` (port) |
| Action loss | ✓ | ✓ |

---

## 2. Config 비교

| 키 | upstream 2D | alpha 2D |
|---|---|---|
| `MODEL.TRANSFORMER_TRANSITION.ENABLED` | 분기 스위치 (True → MUVO else Mile) | (없음) — 항상 RSSMTD |
| `MODEL.TRANSFORMER.CHANNELS` | 512 (`hidden_state_dim = state_dim = 512`) | 256 (`HIDDEN_STATE_DIM = STATE_DIM = 256`) |
| 이미지 토큰 grid | 10×26 = 260 | 10×24 = **240** |
| LiDAR 토큰 grid | 4×64 = 256 (out_indices [1,2,3]) | 2×64 = **128** (out_indices (1,2,3) 동일하나 다운 비율 다름) |
| Voxel 토큰 grid | 12×12×4 = 576 | – |
| Policy 토큰 | 1 | 1 |
| **총 N_token** | **1093** | **369** |
| Transformer encoder | 6L/8H (paper와 일치) | 3L/4H (alpha는 더 가벼움) |
| RSSMTD TransformerDecoder | 3L/4H | 3L/4H |
| Decoder channels | 512 (matches latent) | 512 (DECODER_CHANNELS, even though embedding=256) |
| PerceptualLoss | optional (`LOSSES.PERCEPTUAL.ENABLED`) | ✗ |
| Augmentation toggle | `LOSSES.PERCEPTUAL.ENABLED`와 별도 — preprocess에 항상 적용 | `DATA.AUGMENTATION.ENABLED` (`--no-augmentation`로 OFF) |
| EFFECTIVE_HZ 동적 derivation | paper 0.2s 고정 | `_derive_sample_every_n(CARLA_HZ, FRAME_STEP, EFFECTIVE_HZ)` 동적 |
| KL free-bits | paper 명시 없음 | `KL_FREE_BITS_ENABLED=True` (v4-2) |
| ClearML cfg | – | `CML_ENABLED, CML_PROJECT="alpha26_muvo_2D", CML_TASK, CML_TYPE, CML_TAGS` |

---

## 3. 모델 비교

### 3.1 Encoder

| 항목 | upstream 2D (`muvo.py`) | alpha 2D (`models_muvo_2D.py`) |
|---|---|---|
| Backbone | timm ResNet18 (cam, out_indices [2,3,4]) + timm ResNet18 (LiDAR, in_chans=4, out_indices [1,2,3]) + PointPillar 분기 | 동일 (PointPillar 분기 없음) |
| FPN | ✓ | ✓ `FPNDecoder` |
| BEV branch | ✓ (frustum_pooling) | ✗ |
| Voxel branch | ✓ (PointPillarNet 또는 직접) | ✗ |
| Route encoder | ✓ | ✗ |

### 3.2 Fusion

| 항목 | upstream 2D | alpha 2D |
|---|---|---|
| `SensorFusionTransformer` | 6L/8H, d_model=512 | 3L/4H, d_model=256 |
| Position embedding | 2D sinusoidal + 3D for voxel | 2D sinusoidal (voxel 없음) |
| Type embedding slots | 5 slots (image/lidar/voxel/route/aux) | 5 slots (image/lidar/policy/action/speed) — encoder는 0/1만 사용 |
| Speed handling | broadcast-add | broadcast-add (동일) |

### 3.3 RSSM (RSSMTD)

| 항목 | upstream 2D (`transition_td.py:RSSMTD`) | alpha 2D (`models_muvo_2D.py:RSSMTD`) |
|---|---|---|
| State shape (B,S,C,N_token) | `(B,S,512,1093)` | `(B,S,256,369)` |
| ConvGRUCellGlo 시그니처 | `(input_dim=512, hidden_dim=512)` Conv1d + global gate (mean over tokens) | `(input_dim=256, hidden_dim=256)` 동일 패턴 |
| RepresentationModelTD | TransformerDecoder + 학습 가능 query (image 10×26, lidar 4×64, voxel 12×12×4, policy 1) | TransformerDecoder + 학습 가능 query (image 10×24, lidar 2×64, policy 1) — voxel 제거 |
| TransformerDecoder 파라미터 | 3L/4H, d_model=512 | 3L/4H, d_model=256 |
| Type embeddings 내부 | 3-slot (h, embedding, action) | 동일 패턴 |
| Action injection | observe_step: latent_action → type_emb[2] → concat with [h, embedding] | 동일 |
| State 초기화 | h=0, s=0 (ignore h_init/s_init) | 동일 (docstring 명시) |

### 3.4 Decoder

| 항목 | upstream 2D | alpha 2D |
|---|---|---|
| 카메라 디코더 | `ConvDecoder2D` (5단 ConvT) + RGBHead/SegmentationHead/DepthHead/SemHead | `TokenConvDecoder2D` (3단 ConvT + multi-scale 헤드, RGB만) |
| LiDAR 디코더 | `ConvDecoder2D` + LidarReHead/LidarSegHead | `TokenConvDecoder2D` + LidarRe |
| Voxel 디코더 | `ConvDecoder3D` (3단 ConvTranspose3d) + VoxelSemHead | ✗ |
| Policy 디코더 | `PolicyDecoder` (MLP→Tanh(2)) | 동일 |
| 디코더 입력 | `(B*S, C, H_base, W_base)` (이미지: 10×26; LiDAR: 4×64; voxel: 12×12×4) | 동일 패턴 (단 토큰 grid 차이) |
| Multi-scale | 1/2/4 | 1/2/4 |
| Style decoder (AdaIN) | commented out | – |

### 3.5 모델 클래스 비교 인벤토리

| 클래스 | upstream 2D | alpha 2D |
|---|---|---|
| `Model`/`MUVO` 메인 | `muvo.py:MUVO(nn.Module)` — 1093 토큰 통합 fusion | `models_muvo_2D.py:Model` — 369 토큰 통합 fusion |
| `FPNDecoder` | ✓ | ✓ |
| `PositionEmbeddingSine` | ✓ (+ 3D variant) | ✓ (2D only) |
| `SensorFusionTransformer` | ✓ | ✓ |
| `SpeedEncoder` | inline / Sequential | ✓ 별도 클래스 |
| `PolicyDecoder` | `decoder.py` | `models_muvo_2D.py` |
| `ConvGRUCellGlo` | `transition_td.py` | `models_muvo_2D.py` |
| `RepresentationModelTD` | `transition_td.py` | `models_muvo_2D.py` |
| `RSSMTD` | `transition_td.py` | `models_muvo_2D.py` |
| `_TokenHead` | – (`common.py`의 헤드 사용) | ✓ |
| `TokenConvDecoder2D` | – (`ConvDecoder2D + Head` 결합) | ✓ |
| `ConvDecoder3D` | ✓ | ✗ |
| `PointPillarNet` | ✓ | ✗ |
| `RouteEncode` | ✓ | ✗ |

---

## 4. Loss / Trainer 비교

### 4.1 Loss 항

| 항 | upstream 2D | alpha 2D |
|---|---|---|
| $`ℒ^{\text{img}}`$ (RGB L1 multi-scale) | ✓ | ✓ |
| $`ℒ^{\text{pxyz}}`$ | ✓ | ✓ but `WEIGHT_LIDAR_RE=0.0` ⚠ |
| $`ℒ^{\text{pr}}`$ | ✓ | ✓ but 같이 꺼짐 ⚠ |
| $`ℒ^{\text{pcd}}`$ (Chamfer / empty) | ✓ Chamfer Distance | partial — empty smooth-L1 (weight 0) |
| $`ℒ^{V,\text{scal}}`$ (Voxel SCAL) | ✓ | ✗ |
| $`ℒ^{\text{seg}}`$ (BEV/SemImage) | ✓ | ✗ |
| $`ℒ^{\text{depth}}`$ | ✓ | ✗ |
| $`ℒ^{\text{reward}}`$ | ✓ | ✗ |
| $`ℒ^{\text{action}}`$ (throttle/steering) | ✓ | ✓ `WEIGHT_ACTION=1.0` |
| $`ℒ^{\text{prob}}`$ (KL balanced) | ✓ | ✓ + free-bits 옵션 게이트 |
| SSIM | optional | optional (weight 0) |
| **PerceptualLoss / LPIPS** | ✓ optional | ✗ |
| Near-field Chamfer 메트릭 | – | ✓ (alpha 추가) |

#### 4.1.1 paper $`ℒ^{\text{pcd}}`$ 항 — 정의 모호성과 두 구현의 해석 차이

paper 식 (1)에 등장하는 $`ℒ^{\text{pcd}}`$ 의 **정확한 정의가 paper 본문에 명시되어 있지 않음**. paper §IV-A L323-325 는 $`ℒ^{p,\text{xyz}}`$ (L2) 와 $`ℒ^{p,r}`$ (L1) 만 정의 후 식 (1) 에 $`ℒ^{\text{pcd}}`$ 가 추가 등장. 두 가지 해석 가능:

**해석 A — Chamfer Distance (PCD 정합 손실)**: 일반적인 ML 관습. 예측 점 분포 ↔ GT 점 분포 거리. upstream 저자도 이 해석 따라 `losses.py:354 CDLoss` 클래스 정의.

**해석 B — Empty smooth-L1 (빈 픽셀 hallucinate 방지)**: alpha 의 자체 해석. 빈 픽셀에서 `pred_depth → 0` 강제.

→ paper 본문이 정의를 안 했으니 **두 해석 모두 가능**. upstream 저자가 Chamfer 로 해석했으니 일반 관습에 더 가깝긴 하지만, 절대적 정답은 아님.

| 항목 | upstream 2D | alpha 2D |
|---|---|---|
| 손실 함수 | `CDLoss` 클래스 정의됨 (`losses.py:354`) | `F.smooth_l1_loss(pred_depth, zeros_like)` |
| **학습에서 실제 사용?** | ❌ **`trainer.py:124` 주석처리** (`# self.lidar_cd_loss = CDLoss()`) | ❌ `WEIGHT_LIDAR_EMPTY=0.0` 으로 꺼짐 |
| 검증/평가 사용 | ✓ `CDMetric` (`trainer.py:133-137`) — validation/test 메트릭으로만 | ✓ `chamfer_distance_range_view` (alpha 자체 메트릭, eval 만) |
| 적용 마스크 (만약 켰다면) | **valid 점만** | **invalid 점만** (`~valid_mask`) |
| 의도 | 예측 점 분포 ↔ GT 점 분포 거리 | 빈 픽셀에서 `pred_depth → 0` 강제 |
| cfg 키 | `LOSSES.WEIGHT_LIDAR_RE` 와 묶임 (코드는 주석처리) | `LOSSES.WEIGHT_LIDAR_EMPTY` (별도) |
| 코드 위치 | `muvo_2d/muvo/losses.py:354 CDLoss`, `trainer.py:124` (주석) | `trainer_muvo_2D.py:564, 576-578` |

**핵심 관찰**:
- **paper 식 (1) $`ℒ^{\text{pcd}}`$ 항은 upstream / alpha 두 구현 모두에서 학습에 사용되지 않음** — upstream 은 Chamfer 로 의도했으나 주석처리, alpha 는 empty 로 해석했으나 weight 0.
- paper-aligned 학습을 진짜 하려면 두 구현 모두 갭이 있는 셈. **alpha 가 empty 항을 켜는 것** 도 paper $`ℒ^{\text{pcd}}`$ 의 가능한 해석 (해석 B) 이므로 유효한 옵션.
- upstream 의 `CDLoss` 주석처리 이유는 paper 본문에 추가 단서가 없으므로 추정만 가능: (a) 계산 비용 부담, (b) 학습 안정성, (c) 다른 loss 항만으로 충분하다고 판단, (d) 단순 미완성 코드.

→ **improvement_plan 측면**: 두 가지 옵션 모두 합리적:
1. **해석 A 따라 alpha 에 Chamfer 학습 loss 신규 추가** — upstream 저자 의도에 맞춤
2. **해석 B 따라 alpha 의 `WEIGHT_LIDAR_EMPTY` 를 0 → 양수로** — 현재 코드 구조 그대로 단순히 가중치만 켜기
어느 쪽이 paper 진짜 의도인지 paper 만으로는 결정 X. 둘 다 ablation 가치 있음.

#### 4.1.2 SpatialRegressionLoss (upstream) vs `_masked_loss` (alpha) — 본질은 동일

upstream `losses.py:76-101 SpatialRegressionLoss` 와 alpha `trainer_muvo_2D.py:541 _masked_loss` 는 **같은 paper 항** ($`ℒ^{p,\text{xyz}}`$, $`ℒ^{p,r}`$) 을 다른 wrapping 으로 구현한 것. 본질은 동일.

```python
# upstream (losses.py:76-101)
class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255):
        self.loss_fn = F.l1_loss if norm==1 else F.mse_loss
    def forward(self, prediction, target, instance_mask=None):
        mask = target[:, :, :1] != self.ignore_index  # 255 = CARLA sentinel
        loss = self.loss_fn(prediction, target, reduction='none')
        loss = torch.sum(loss, dim=-3, keepdims=True)  # 채널 합
        return loss[mask].mean()

# alpha (trainer_muvo_2D.py:541, 563-578)
valid_mask = target[:, :, -1:] > 0  # depth>0 픽셀
self._masked_loss(pred[:, :, :3], target[:, :, :3], valid_mask, F.mse_loss)  # paper ℒ^{p,xyz}
self._masked_loss(pred_depth, target_depth, valid_mask, F.l1_loss)            # paper ℒ^{p,r}
```

| 항목 | upstream `SpatialRegressionLoss(norm=N)` | alpha `_masked_loss(...,F.X_loss)` |
|---|---|---|
| paper 항 대응 | `ℒ^{p,xyz}` (norm=2 MSE), `ℒ^{p,r}` (norm=1 L1) | 동일 |
| 손실 함수 | `F.mse_loss` / `F.l1_loss` | `F.mse_loss` / `F.l1_loss` (동일) |
| 마스크 기준 | `target[:, :, :1] != 255` (CARLA ignore_index sentinel) | `target[:, :, -1:] > 0` (depth>0 = 점 있음) |
| 마스크 적용 순서 | 채널 합산 → mask → mean | mask → loss_fn → mean (loss_fn 안 reduction) |
| 추상화 | `nn.Module` 클래스 | static method |

**의미**: 둘 다 paper $`ℒ^{p,xyz}`$, $`ℒ^{p,r}`$ 를 다른 wrapping 으로 구현 — 결과 수치는 마스크 정의 차이 (255 ignore vs depth>0) 때문에 약간 다를 수 있지만 paper 항으로서는 동등. alpha 가 paper $`ℒ^{p,xyz/p,r}`$ 를 안 쓰는 이유는 **`WEIGHT_LIDAR_RE=0.0` (가중치 0)** 이지 loss 구현 자체 차이가 아님.

### 4.2 Trainer 비교

| 항목 | upstream 2D (`muvo_2d/muvo/trainer.py`) | alpha 2D (`trainer_muvo_2D.py`) |
|---|---|---|
| 모델 분기 | `MUVO if TRANSFORMER_TRANSITION.ENABLED else Mile` | 항상 RSSMTD Model |
| TBPTT carry | ✗ (paper도 단일 시퀀스) | ✗ (alpha-2D는 1D에서 제거) |
| 시각화 | `visualise()` BEV+RGB+LiDAR+voxel+trajectory | `save_reconstruction_figure` RF+FH 6 컬럼 + dual depth |
| Checkpoint callback | `MyModelCheckpoint` + `SaveGitDiffHashCallback` | `ModelCheckpoint` + `ExperimentCSVLogger` |
| Experiment tracker | ClearML (`prediction.py` 추론용 위주) | ClearML (`train_muvo_2D.py`에서 학습 init) |
| ClearML artifact 업로드 | `prediction.py`에서 inline | `trainer_muvo_2D.test_step`에서 try/except + 처음 4 batch |
| KL free-bits | paper 사양 | cfg 게이트 (`KL_FREE_BITS_ENABLED`) |
| 메트릭 | SSCMetrics + PSNR + SSIM + Chamfer + IoU | PSNR + Chamfer + xyz + range MAE + **near-field Chamfer** |

#### 4.2.1 학습 시 imagine 처리 — paper L372 ("All 12 frames as known data") 와의 정합성

paper §IV-A 본문 "All 12 frames were treated as known data" (L372) 는 학습 단계에서 sequence 전체가 observation 으로 처리됨을 명시. upstream 2D 코드는 이를 그대로 따르고, alpha 2D 는 학습 시에도 imagine supervise → **paper 와 학습 셋업 불일치**.

| 단계 | upstream 2D | alpha 2D |
|---|---|---|
| **trainer 진입점** | `trainer.py:419 training_step` → `forward(batch)` → `self.model.forward(batch)` | `trainer_muvo_2D.py:661 training_step` → `_observe_and_imagine(batch)` |
| **RSSMTD.forward (학습 본체)** | `transition_td.py:202` `for t in range(sequence_length): observe_step(...)` — **전체 seq posterior**. prior 는 KL loss 용으로만 계산 | `models_muvo_2D.py:381 RSSMTD.forward` — 동일 패턴, 전체 seq observe_step |
| **decoder 입력 sample** | observe_step 의 **posterior sample** (use_dropout 분기에서만 일부 prior) | observe_step 의 posterior sample |
| **추가 imagine 학습 루프** | ❌ 없음. `muvo.py:479 observe_and_imagine` 는 **검증/추론 전용** (training_step 에서 호출 X) | ✅ `trainer_muvo_2D.py:645` `oi, _ = self.model.imagine(state_imagine, future_horizon=self.fh)` 학습에서 호출 |
| **학습 loss 구성** | observe (전체 seq posterior) reconstruction × multi-scale | observe reconstruction + **추가 FH imagine reconstruction** (`future_*` prefix × `WEIGHT_FUTURE=1.0`) |
| **검증 시 imagine** | `observe_and_imagine(batch, future_horizon=cfg.FUTURE_HORIZON)` — RF observe + FH imagine, future prediction metric 측정 | 학습과 같은 `_observe_and_imagine` (logic 동일) |
| **paper L372 일치** | ✅ 일치 (학습=전체 posterior, 검증만 RF/FH 분리) | ❌ 불일치 (학습 때부터 FH imagine 도 supervise) |

**의미**:
- alpha 의 현재 셋업은 future prediction 을 학습 단계에서 더 강하게 supervise → future metric 은 좋아질 수 있지만 **paper 와 직접 정량 비교 불가**.
- alpha-1D vs 2D 의 single-window overfit 결과 (1D > 2D, `docs/muvo_4way_1D_2D_comparison.md`) 가 학습 imagine supervise 의 영향일 수 있음 — paper-aligned 로 바꾸면 결과 달라질 가능성.
- paper 일치 ablation: `WEIGHT_FUTURE=0` (옵션 A) 또는 `training_step` 에서 imagine 부분 제거 (옵션 B). 자세한 사항은 `improvement_plan/alpha_{1D,2D}_improvements.md` G13/G15 참조.

---

## 5. CARLA 환경 비교

upstream 2D는 1D처럼 전체 CARLA 환경 코드를 포함; alpha 2D는 Arrow 데이터만 사용.

| 컴포넌트 | upstream 2D | alpha 2D |
|---|---|---|
| `carla_gym/` | ✓ (1D와 거의 동일) | ✗ |
| `agents/rl_birdview/` (1D는 `rl_birdview/`) | ✓ | ✗ |
| `data_collect.py` + `DataWriter` | ✓ | ✗ |
| Voxel 생성 (`data/generate_voxels.py`) | ✓ | ✗ |

---

## 6. "upstream 2D에 있고 alpha 2D에 빠진 것"

1. **3D occupancy 헤드 (`ConvDecoder3D + VoxelSemHead`)** — paper §III-D, §IV-A 핵심 모듈.
2. **BEV 분기 (`FrustumPooling`)** — paper §IV-B "BEV mapping" 비교군.
3. **PointPillarNet** — paper §IV-B "PP" 비교군.
4. **Depth/SemImage/BEV seg 헤드** — upstream auxiliary heads (paper §III-D 는 cam/LiDAR/occupancy 만 명시).
5. **Route encoder** — 계획 경로 토큰.
6. **PerceptualLoss/LPIPS** — paper §IV-B의 PL 비교 (효과 없음으로 결론).
7. **Higher fusion capacity** — upstream Transformer 6L/8H (d=512) vs alpha 3L/4H (d=256).
8. **Voxel-related N_token (576)** — upstream 1093 vs alpha 369.

---

## 7. "alpha 2D에 추가된 것"

1. **`KL_FREE_BITS_ENABLED` 게이트** — alpha 1D와의 controlled 비교용 (v4-2 주석).
2. **Near-field Chamfer 메트릭** — `pc_range=NEARFIELD_PC_RANGE`로 근거리 정밀도 평가.
3. **`save_reconstruction_figure` 풀 컬럼 모드** — RF+FH 모두 시각화 + 스케일/raw depth 동시 표시.
4. **`maybe_init_clearml`** — 학습 단계에 통합된 깔끔한 ClearML init.
5. **`_derive_sample_every_n` 동적 derivation** — EFFECTIVE_HZ 변경 시 자동 SAMPLE_EVERY_N 재계산.
6. **CLI: `--effective-hz, --no-augmentation, --no-clearml`** — 빠른 디버그 토글.
7. **TokenConvDecoder2D `_slice_token_state`** — N_token을 모달리티 범위로 명시 분할.

---

## 8. Paper §III/§IV 일관성

| Paper 요소 | upstream 2D | alpha 2D |
|---|---|---|
| Image 600×960 | ✓ | ✗ (320×768) |
| LiDAR ≤60K pts → range view | ✓ | ✓ |
| ResNet18 백본 | ✓ | ✓ |
| LiDAR out_indices (stride-16) | (1,2,3) | (1,2,3) ✓ |
| 2D sinusoidal PE + sensor embedding | ✓ | ✓ |
| Transformer encoder k=6/8H | ✓ | 3L/4H (alpha 가벼움) |
| **2D token RSSM** | ✓ (1093 tokens) | ✓ (369 tokens) |
| FC→Conv 교체 (ConvGRU) | ✓ | ✓ |
| TransformerDecoder prior/posterior + 학습 가능 query | ✓ | ✓ |
| 식 (1) $`ℒ^{\text{img}}`$ + $`ℒ^{\text{pxyz}}`$ + $`ℒ^{\text{pr}}`$ + $`ℒ^{V,\text{scal}}`$ | ✓ | ⚠ (LiDAR 항 weight 0, voxel 없음) |
| Multi-scale 1/2/4 | ✓ | ✓ |
| Seq 12 / 6 | ✓ | seq 6 ✓ (paper voxel exp 일치) |
| 0.2 s 샘플링 | ✓ | ✗ (0.5 s) |
| $`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$ 분리 | ✓ | partial |
| Perceptual Loss optional | ✓ | ✗ |
| 3D occupancy head | ✓ | ✗ |
| PointPillars/BEV 분기 | ✓ | ✗ |

---

## 9. 핵심 결론

- alpha 2D는 upstream 2D의 **본질적 구조(2D 토큰 RSSM + ConvGRU + TransformerDecoder + ConvT 디코더)를 충실히 재현**한다.
- 큰 차이는 **(a) voxel/BEV 미구현, (b) 토큰 수 축소(1093→369), (c) capacity 축소(d_model 512→256, encoder 6→3 layers), (d) PerceptualLoss 제외**.
- alpha 2D는 **alpha 1D와의 controlled 비교 + 알파 데이터셋 fit + 실용적 보조 신호(action loss, augmentation, near-field metric, ClearML 통합)** 측면에서 upstream보다 잘 정비되어 있다.
- paper §IV-A 식 (1)의 $`ℒ^{\text{pxyz}}`$ / $`ℒ^{\text{pr}}`$ 항이 가중치 0으로 꺼져 있는 것은 alpha 2D의 **가장 큰 paper-vs-code 불일치** (improvement plan P0).
