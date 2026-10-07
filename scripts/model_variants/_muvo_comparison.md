# `_muvo.py` 파일군 비교 정리

작성일: 2026-05-14

## 0. 배경 & 목적

`alpha26/scripts/model_variants/`에 있는 `*_muvo.py` 5개 파일은 **upstream muvo 스타일의 다른 world model을 만들기 위한 alpha26 내부 분기**다. 핵심 설계 결정:

- **3D occupancy / BEV(birdview) / voxel 헤드는 일부러 모두 제거**
- RGB + LiDAR range-view + future prediction만 다루는 가벼운 변형

원래 동기: 기존 `files/*.py`(non-`_muvo`, alpha26 자체 trainer11)로 학습했을 때 **RGB reconstruction과 future imagination이 둘 다 흐릿하게 나오는 문제**. "muvo는 안 흐릿한데 왜 alpha는 흐릿하지?"라는 추론으로 muvo 로직에 맞추는 작업을 시작.

비교 대상 경로:
- 새 파일: `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/{config,data,models,train,trainer}_muvo.py`
- 같은 디렉토리의 alpha26 사본: `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/{config,data,models,train,trainer}.py`
- upstream 원본: `/home/carol/chaeyeon-kim/muvo/muvo/{config.py, trainer.py, models/mile.py, data/dataset.py}`, `/home/carol/chaeyeon-kim/muvo/train.py`

---

## 1. 파일별 한 줄 요약

| 파일 | 진짜 출처 | upstream muvo 포팅 정도 |
|---|---|---|
| `config_muvo.py` | alpha26 자체 cfg 스타일을 모방한 신규 작성 | **거의 없음.** 공통 키 ~10개, 값은 다수 변경 |
| `data_muvo.py` | `data.py` 직접 복제 | **거의 없음.** 차이는 3곳 |
| `models_muvo.py` | `mile.py` + `transition.py` 컨셉 리라이트 | **유일하게 실제 포팅.** 단 뒷부분 미완료 |
| `train_muvo.py` | `train.py` 거의 verbatim 사본 | **없음.** import 슬롯만 `_muvo`로 |
| `trainer_muvo.py` | `trainer.py` 거의 verbatim 사본 | **없음.** loss 구성은 그대로 |

즉 **`models_muvo.py` 하나만 실제로 upstream에서 가져온 코드**가 들어가 있고, 나머지 4개는 사이드카 스캐폴딩(`models_muvo`를 돌리기 위한 cfg/data/train/trainer 변형 사본).

---

## 2. `config_muvo.py` vs `muvo/muvo/config.py`

원본은 `fvcore/yacs CfgNode` 시스템, 새 파일은 `SimpleNamespace`. 공유 키 ~10개에 불과.

### 동일 키, 다른 값 (주요)
- `RECEPTIVE_FIELD`: 1 → **4**
- `FUTURE_HORIZON`: 1 → **4**
- `MODEL.EMBEDDING_DIM`: 512 → **256**
- `MODEL.TRANSITION.HIDDEN_STATE_DIM`: 1024 → **512**
- `MODEL.TRANSITION.STATE_DIM`: 512 → **256**
- `LOSSES.WEIGHT_PROBABILISTIC`: 1e-3 → **1e-2**
- `LOSSES.WEIGHT_LIDAR_RE`: 0.1 → **0.0** (default; 서버 cfg에서 override)
- `SPEED.NORMALISATION`: 5.0 → **50.0**

### 빠진 항목 (upstream에만 있음)
- `MODEL.TRANSFORMER`, `MODEL.ENCODER`, `MODEL.BEV`, `MODEL.LIDAR`, `MODEL.ROUTE`, `MODEL.MEASUREMENTS`, `MODEL.POLICY`, `MODEL.REWARD`
- 손실 헤드: `SEMANTIC_SEG`, `INSTANCE_SEG`, `VOXEL_SEG`, `LIDAR_RE`, `LIDAR_SEG`, `SEMANTIC_IMAGE`, `DEPTH`
- 손실 가중치: `WEIGHT_ACTION/SEGMENTATION/INSTANCE/REWARD/LIDAR_SEG/SEM_IMAGE/DEPTH/VOXEL`, `RGB_INSTANCE`, **`SSIM`**
- `EVAL`, `SAMPLER` 블록 전체

### 추가된 항목 (alpha26 전용)
- `DATA` 블록 (`MANIFEST_PATH`, `IMAGE_INPUT_SIZE=(320,768)`, `RGB_RECON_SIZE=(320,768)`, `LIDAR_RANGE_VIEW_SIZE=(32,1024)`, `LIDAR_SCALE=40.0`, `SAMPLE_EVERY_N=2`, `FRAME_STEP=5`) ← 초기는 `(320,800)/(216,288)`이었으나 commit `7099f87`에서 (320,768)/(320,768)로 변경, ConvDecoder 5-doubling 정합용
- `MODEL.FUSION` (`TRANSFORMER_CHANNELS=256`, `TRANSFORMER_LAYERS=3`, `TRANSFORMER_HEADS=8`)
- `LOSSES.KL_FREE_BITS=1.0`, `WEIGHT_LIDAR_EMPTY`, `WEIGHT_RGB`, `WEIGHT_FUTURE`
- `LOGGING` 블록

---

## 3. `data_muvo.py` vs upstream dataset

**upstream `muvo/muvo/data/dataset.py` 코드는 한 줄도 안 들어옴.** `data.py`를 거의 그대로 복제한 뒤 3곳만 수정:

1. `VerticalCropToSize` (L23-38): 폭 고정 + 높이 center-crop. `data.py`의 `transforms.Resize`를 대체
2. `cfg.DATA.LIDAR_SCALE` range view 정규화 (L198)
3. `from config_muvo import cfg` (L16)

upstream에서 빠진 것: `DataModule`, BEV/route map 로딩, instance label, voxel occupancy, depth/semantic image 분리, PointPillar 점군, intrinsics/extrinsics, reward channel, run filtering, ego masking, LiDAR semantic remap 등 **전부**. 의도된 누락.

---

## 4. `models_muvo.py` vs `muvo/muvo/models/mile.py` (+ transition.py)

**`_muvo` 파일 중 유일한 실제 포팅 작업.** 단 **뒷부분 미완료**.

### 진척도 표

| 구간 | `models.py`(alpha 원본) | `models_muvo.py` 현재 | upstream 대응 | 상태 |
|---|---|---|---|---|
| `FPNDecoder` (L25-50) | top-down, `xs[2]` 시작, `F.interpolate` 업샘플, out=128 | bottom-up, `xs[0]` 시작, `adaptive_max_pool2d` 다운샘플, out=256 | `DecoderDS` (`common.py:102-130`) | ✅ muvo 스타일 |
| `PositionEmbeddingSine` (L57-92) | 동일 | 동일 | `common.py:636+` 동일 | ✅ (원래 동일) |
| `SensorFusionTransformer` (L95-131) | 동일 | 동일 | mile.py 인라인 fusion을 클래스화 | ✅ |
| `SensorFeatureConv` | (없음) `pool+flatten` | `BasicBlock × 2 + AdaptiveAvgPool + Flatten` | `BasicBlock × 2 + pool` (`mile.py:104-115`) | ✅ **muvo 동일** (2026-05-15 마이그레이션, ResNet skip connection 포함) |
| `RepresentationModel` | 동일 | 동일 | `transition.py:5-25` | ✅ |
| `RSSM` | 동일 | 동일 | `transition.py:28-191` | ✅ `action_in_gru` 플래그로 토글. default `True`=alpha(표준 Dreamer), `False`=upstream muvo. 코드 코멘트로 의도 명시 |
| **`SensorHead`** | 동일 | `1×1 Conv only` | `RGBHead`/`LidarReHead` (`common.py:274-303`) | ✅ **muvo 동일** (2026-05-15 bilinear interpolate 제거) |
| **`SensorDecoder`** | (이전) k=4 ConvTranspose | upstream `ConvDecoder` 패턴 (constant_size + N pre_transpose + 3 trans_conv + heads) | `ConvDecoder` (`common.py:549-632`) | ✅ **muvo 동일** (2026-05-15 마이그레이션, n_pre_doublings 파라미터화) |
| **`Model.encode_fuse_sequence`** | 명시적 토큰 그리드 풀(20,20)/(4,16) 사용 | 명시적 풀 제거. FPN 다운샘플에 의존 | mile.py도 명시적 풀 없음 | ✅ muvo 스타일 |
| `Model.forward` / `Model.imagine` | 동일 | 동일 | mile.py와 다른 alpha 스타일 | ⚠️ 인터페이스는 alpha (TBPTT용 `h_init`/`s_init`/`continuation` 인자). 의도된 차이 |

### `SensorDecoder` vs `ConvDecoder` 디테일 차이 (마이그레이션 후 — 2026-05-15)

| 항목 | upstream `ConvDecoder` | alpha `SensorDecoder` (현재) |
|---|---|---|
| `constant_size` | 하드코딩 `(5, 13)` (RGB 320×832 기준) | `_decoder_base_size(target, n_pre_doublings)` 자동. RGB (320, 768) → (5, 12), LiDAR (32, 1024) → (1, 32) |
| pre_transpose kernels | k=5 (op=1) × 2 + k=6 × 1 | 동일 (n_pre_doublings에 따라 갯수 조정, 마지막은 k=6) |
| trans_conv channels | 512 → 256 → 128 → 64 | 512 → 256 → 128 → 64 (동일) |
| Head | 1×1 Conv 만 (해상도는 trans_conv가 결정) | 1×1 Conv 만 (동일) |
| Target divisibility | 64 (=2^6) | RGB 6-doubling(n=3), LiDAR 5-doubling(n=2) — `_auto_n_pre_doublings` |

RGB target은 자세히는 `docs/muvo_decoder_head_explained.md` §6.0~6.3 참고.

### `RSSM` 디테일 차이

| 항목 | upstream | alpha (default) | alpha (`action_in_gru=False`) |
|---|---|---|---|
| `pre_gru_net` 입력 | `sample_t` only | `[sample_t, latent_action_t]` concat | `sample_t` only (upstream과 동일) |
| `pre_gru_net` in_dim | `state_dim` | `state_dim + action_latent_dim` | `state_dim` |
| GRU hidden state에 action 영향 | ❌ 없음 | ✅ 있음 | ❌ 없음 |
| `prior` 입력에 action | ✅ 있음 | ✅ 있음 | ✅ 있음 |

alpha default(`True`)는 표준 Dreamer V1/V2/V3 패턴. `False`는 upstream `transition.py:163`과 일치.
`cfg.MODEL.TRANSITION.ACTION_IN_GRU`로 토글.

---

## 5. `train_muvo.py` vs `muvo/train.py`

**muvo의 hydra/yacs CLI로 갈아탄 게 아님.** alpha26 `train.py`를 거의 verbatim 복제한 사본.

차이:
- import 슬롯이 `_muvo` 모듈을 가리킴
- `WorldModelTrainer(...)` 생성자에서 `embedding_n_channels=128` 인자 제거
- `ExperimentCSVLogger`의 `change_summary` 문구 "MUVO-style alpha:"로 교체

upstream의 hydra config / `SaveGitDiffHashCallback` / `MyModelCheckpoint` / ClearML 통합은 도입 안 됨.

---

## 6. `trainer_muvo.py` vs `muvo/muvo/trainer.py`

**upstream `trainer.py`의 포팅이 아님.** alpha26 `trainer.py`를 거의 verbatim 복제한 사본.

### 차이 (vs alpha26 `trainer.py`)
- `models_muvo` import (L12)
- `LIDAR_SCALE` 적용 (L204-205)
- `EMBEDDING_DIM` 기본값 처리 (L362, L368-370)
- 그 외 거의 동일

### upstream 대비 빠진/다른 항목 (Loss/Optimizer/Logging)

| 항목 | upstream 위치 | 새 파일 상태 |
|---|---|---|
| **SSIM loss on RGB** (weight 0.6) | `trainer.py:97-98, 312-318` | **❌ 없음** (alpha26 코드 전체에 SSIM 한 줄도 없음) |
| **`add_weight_decay` 파라미터 그룹** (bias/1D 파라미터 WD 제외) | `trainer.py:1031, 1054` | ❌ 없음. weight_decay 자체는 `AdamW`에 들어가지만 단일 값을 전체 파라미터에 적용 |
| **N_SAMPLES imagination** (multi-sample prior rollout) | `trainer.py:224, 244-247` | ❌ 단일 샘플만 |
| **Tensorboard 이미지/비디오 푸시** | `trainer.py:528, 646, 754-957` | ❌ `self.log` scalar만. matplotlib PNG 외부 저장 |
| **Action loss** (throttle/steering reconstruction) | `trainer.py:256-259` | ❌ 없음 (Policy 헤드 자체가 없음) |
| **Dropout-only eval** (`self.train()` + Dropout만 eval) | `trainer.py:405-408, 1080-1083` | ❌ 일반 eval |
| **`active_inference` 토글** | `trainer.py:393-397` | ❌ 없음 |

### 의도된 차이 (alpha26 trainer11 고유)
- TBPTT 윈도우 사이 RSSM state 캐리오버 (`_tbptt_h/_tbptt_s/_tbptt_action`, L519-561)
- KL free-bits (`KL_FREE_BITS`, L391, L386)
- `strict=False` pretrained 로드 (L238)
- Future prediction loss 분리 weight (`WEIGHT_FUTURE`)

---

## 7. 흐릿한 RGB 진단

### 원인 후보 (가능성 순) — 2026-05-15 진행 상황 업데이트

1. **SSIM/perceptual loss 부재** ★ 가장 유력 → 🔬 검증 중
   - alpha26 코드 전체에 SSIM 한 줄도 없었음 → `trainer_ssim.py` 추가 (subclass).
   - weight 0.6 (alpha 절대 기준)으로 1000-epoch 1-window 돌렸을 때 학습 불안정 + 체커보드 노출.
   - weight 0.06 (upstream effective, `WEIGHT_RGB(1.0) × ssim_weight(0.06)`)으로 재시도 중.
   - 단독 효과는 `_muvo` decoder 마이그레이션과 직교한 검증.

2. **Decoder 단순화로 인한 checkerboard 가능성** → ✅ `_muvo`에서 해결
   - 이전 alpha `SensorDecoder`의 `k=4 s=2 p=1` → upstream `k=5(op=1)/k=6(p=2)` 혼합으로 변경.
   - `_muvo` 파일에서 진행. non-`_muvo` (baseline) 는 그대로 유지.

3. **`SensorHead`의 bilinear interpolate** → ✅ `_muvo`에서 해결
   - bilinear interpolate 제거, 1×1 conv 만. decoder가 정확한 해상도 책임.
   - `RGB_RECON_SIZE=(320, 768)`로 변경해 32 배수 정렬.

4. **KL weight 과다 (posterior collapse)**
   - `WEIGHT_PROBABILISTIC=1e-2` (alpha) vs `1e-3` (upstream default)
   - 다만 `KL_FREE_BITS=1.0`이 있어서 어느 정도 완화. 미점검.

5. **Decoder 용량 / latent 차원 부족**
   - alpha 비교 기준: `STATE_DIM=128`, `HIDDEN_STATE_DIM=256`
   - `_muvo` (config_muvo.py): `STATE_DIM=256`, `HIDDEN_STATE_DIM=512` → upstream(512/1024)의 절반은 그대로지만 alpha baseline 대비 2배.

6. **Future prediction 손실 비중 (`WEIGHT_FUTURE=1.0`)**
   - future는 본질적으로 불확실 → 그래디언트가 encoder/decoder를 평균쪽으로 끌어당김. 미점검.

---

## 8. 작업 후보 / 진행 상태

### Step 1: SSIM 추가 (non-`_muvo`)
- ✅ `trainer_ssim.py` + `train_ssim.py` 작성 (subclass + monkey-patch, baseline 보존)
- ✅ 1000-epoch 1-window overfit (weight 0.6) — 학습 불안정, 체커보드 아티팩트
  - `docs/ssim_addition_notes.md` 참고
- ✅ weight 0.06 (upstream effective)로 낮추고 재시도 → 결과 대기 중

### Step 2: `models_muvo.py` 마이그레이션 (2026-05-15 완료)
- ✅ **(A) `SensorDecoder` → `ConvDecoder` 패턴**: upstream 패턴 이식. n_pre_doublings 파라미터화로 RGB 6-doubling/LiDAR 5-doubling 자동 선택. `docs/muvo_decoder_head_explained.md` 참고.
- ✅ **(B) `SensorHead` 단순화**: bilinear interpolate 제거, 1×1 conv 만.
- ✅ **(C) `SensorFeatureConv` → `BasicBlock × 2`**: ResNet skip connection 복원.
- ✅ **(D) RSSM `pre_gru` 결정**: alpha의 action concat 유지 (표준 Dreamer). `cfg.MODEL.TRANSITION.ACTION_IN_GRU` 플래그로 upstream 패턴(`False`) 토글 가능.

### Step 3: `trainer_muvo.py` 보완 (선택, 미진행)
- SSIM loss 정식 도입 (현재 trainer_ssim.py는 non-`_muvo`용)
- `add_weight_decay` 파라미터 그룹 분리
- (선택) N_SAMPLES imagination

---

## 9. 의도된 누락 (검토 불필요)

다음은 **사용자가 의도적으로 제거했으므로 다시 추가할 필요 없음**:
- BEV / birdview 관련 일체 (`BevDecoder`, `backbone_bev`, BEV seg/center/offset loss, `MODEL.BEV` cfg)
- 3D occupancy / voxel 일체 (`VoxelDecoder1`, voxel SSC, sem/geo scal loss, `MODEL.VOXEL`)
- Route map (`RouteEncode`, `backbone_route`)
- PointPillar LiDAR (`PointPillarNet`)
- Policy 헤드 / action regression
- LiDAR semantic segmentation, semantic image, depth image 헤드
- Deployment / sim interface (`deployment_forward`, `sim_forward`)
- Hydra/yacs config, `DataModule`, ClearML, `SaveGitDiffHashCallback` 등 upstream 인프라
