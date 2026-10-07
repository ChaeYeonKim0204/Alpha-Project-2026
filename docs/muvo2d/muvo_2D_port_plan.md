# alpha26 `_muvo_2D` 포팅 계획 (v3, 팀 2차 리뷰 반영)

> **v3 변경 요약** (2026-05-18, 팀 2차 피드백):
> - **Q1 사실 정정**: upstream MUVO의 LiDAR encoder는 `out_indices=[1, 2, 3]` (stride-16 종료) ← `muvo_2d/muvo/models/muvo.py:78-82` 직접 확인. mile.py의 [2,3,4]와 다름. alpha `_muvo_2D`도 upstream MUVO 패턴 매칭 위해 **LiDAR encoder만 `out_indices=[1,2,3]`로 변경**. 결과: LiDAR token (1, 32) → **(2, 64) = 128 token**. Total token 273 → **369**.
> - **KL free-bits on/off cfg flag** 추가 (`cfg.LOSSES.KL_FREE_BITS_ENABLED`, default **False** = upstream과 일치).
> - **Effective Hz cfg 변수** 추가 (`cfg.DATA.EFFECTIVE_HZ`, default 2). `SAMPLE_EVERY_N` 자동 도출.
> - **Reconstruction figure**: 1 window의 6 frame 모두 column으로 표시 (RF 4 + FH 2). LiDAR-scale 보정/미보정 두 view는 v2 그대로.
> - **ClearML 채택** (Q14/15): 실험 관리를 upstream 방식으로 통일. `Task.init(...)` 추가, cfg에 `CML_PROJECT/TASK/TYPE/TAG`. TensorBoard는 ClearML이 auto-capture하므로 보조 유지. `clearml_guide.md` 비전공자 친화 신규 작성.
> - PolicyDecoder는 upstream 코드 그대로 (in_channels만 cfg.MODEL.EMBEDDING_DIM=256 자동 적용).

## v2 → v3 변경 핵심 표

| 항목 | v2 | v3 | 이유 |
|---|---|---|---|
| LiDAR encoder out_indices | [2,3,4] | **[1,2,3]** | upstream MUVO와 매칭 (muvo.py:78-82 확인) |
| LiDAR token grid | 1 × 32 = 32 | **2 × 64 = 128** | stride-16 종료 |
| Total tokens | 273 | **369** | 240 image + 128 lidar + 1 policy |
| KL free-bits | 항상 적용 (=1.0) | **cfg flag로 on/off, default OFF** | upstream과 일치 |
| Sampling rate 지정 | `SAMPLE_EVERY_N`/`FRAME_STEP` 직접 | **`EFFECTIVE_HZ` cfg로 자동 도출** | 사용자 가독성 |
| Reconstruction figure frame 수 | 일부 frame | **6 frame 전체 표시** | 한 window 다 보기 |
| 실험 트래킹 | TensorBoard + CSV | **ClearML primary** + TensorBoard auto-capture | upstream 일치 + 팀 사용성 |

## Context

`alpha26/scripts/model_variants/`의 `_muvo` 5개 페어(config/data/models/train/trainer)는 upstream MUVO main 브랜치의 **1D-latent RSSM** 구조를 alpha dataset 조건(32-ch LiDAR, Arrow offline)에 맞춰 축소 이식한 형태다 (`alpha26/docs/muvo_style_alpha_model_notes.md`). 그러나 MUVO 페이퍼 §IV-B의 ablation 결론은 **"2D latent space itself provides the largest boost in performance, while other changes have little effect"** — 즉 카메라/voxel reconstruction의 진짜 ↑는 1D→2D state shape 전환에서 나온다 (`alpha26/docs/muvo_paper_summary.md`).

이 2D 변형은 author personal repo `daniel-bogdoll/MUVO`의 `2D` 브랜치 = 본 워크스페이스의 `/muvo_2d/`에 존재한다. 현재 alpha `_muvo` baseline은 `WEIGHT_LIDAR_RE=0.0` + no augmentation + no aux head 상태에서 blur escape에 갇혀 있고, `muvo_default_config_audit.md`의 priority list 5번이 "`RSSMTD` 포팅"이다.

본 변경은 alpha `_muvo` baseline을 *유지한 채* `*_muvo_2D.py` 5개 파일을 sibling으로 신규 생성, muvo_2d 스타일의 token-shape state RSSM을 적용해 페이퍼 헤드라인 결과를 alpha dataset에서 검증할 수 있게 한다. `_muvo` 정책(BEV/3D drop)은 유지하되 RSSMTD 원형과 가까운 form을 위해 policy(action) token + speed encoder + PolicyDecoder + action loss + image augmentation을 동시 포함하고, LiDAR encoder는 upstream MUVO와 같은 stride-16 종료로 맞춘다 (v3 변경).

## 본 계획 (`_muvo_2D`) vs upstream `muvo_2d` 상세 차이

### A. Token / state 차원 (v3 갱신)

| 항목 | upstream `muvo_2d` (MUVO 경로) | 본 계획 `_muvo_2D` | 비고 |
|---|---|---|---|
| Image token 그리드 | 10 × 26 = 260 | 10 × 24 = 240 | image encoder `out_indices=[2,3,4]` 동일, alpha 320×768 input에서 stride-32 |
| LiDAR token 그리드 | 4 × 64 = 256 | **2 × 64 = 128** | **v3: LiDAR encoder `out_indices=[1,2,3]` 적용** → alpha (32, 1024) stride-16 |
| Voxel token 그리드 | 12 × 12 × 4 = 576 | 없음 | `_muvo` 정책 (BEV/3D drop) + alpha dataset에 voxel GT 없음 |
| Policy token | 1 | 1 | RSSMTD 원형 유지 + PolicyDecoder가 실제 decode |
| Speed (signal) | 별도 token | type embedding modulation add | token 추가 안 함 |
| **Total tokens** | **1093** | **369** (v3) | 240 + 128 + 1 |
| Embedding / channel dim | 512 | 256 | `_muvo` 축소 이식 일관 |
| RSSM hidden / state dim | 512 / 512 | 256 / 256 | RSSMTD hidden=state=embedding 단일 채널 |

### B. 모델 구조 / 클래스 (v3 갱신)

| 항목 | upstream `muvo_2d` | 본 계획 `_muvo_2D` | 비고 |
|---|---|---|---|
| Top-level model | `MUVO(nn.Module)` | 기존 `Model` 클래스 rewrite | 클래스명 유지 |
| RSSM | `RSSMTD` | `RSSMTD` 포팅 (voxel query 제거 버전) | |
| Representation model | `RepresentationModelTD` (4-modality query) | 3-modality query (image / lidar / policy) | `query_embed_voxel` 제거 |
| Recurrent core | `ConvGRUCellGlo` | 동일 포팅 | |
| Image encoder | ResNet18 `out_indices=[2,3,4]` | 동일 | |
| **LiDAR encoder** | **ResNet18 `out_indices=[1,2,3]` `in_chans=4`** | **v3: upstream과 동일 `[1,2,3]`** | v2의 `[2,3,4]`에서 변경 |
| FPN 종료 stride (LiDAR) | stride-16 | stride-16 | upstream과 일치 |
| BEV encoder | 옵션 (default on) | 없음 | `_muvo` 정책 |
| PointPillars LiDAR option | 있음 | 없음 (range-view만) | `_muvo` 정책 |
| Route encoder | 있음 | 없음 | `_muvo` 정책 |
| Speed encoder | `Linear(1,16)→ReLU→Linear(16,256)→ReLU` + 별도 token | upstream 동일 MLP + type embedding modulation add | token 추가 안 함 |
| Measurement encoder (GPS/route_command/route_command_next) | 옵션 (default off) | 없음 | upstream도 default off + alpha dataset 없음 |
| Sensor-type embedding | n_type slot | 5 slot 고정 (image/lidar/policy/action/speed) | |
| Transformer fusion (encoder side) | 6-layer, d=384, nhead=8 | 3-layer, d=256, nhead=4 | alpha `_muvo` baseline 유지 |
| TransformerDecoder (representation) | 6-layer, nhead=8, d=512 | 3-layer, nhead=4, d=256 | encoder와 layer 일치 + 64 dim/head |
| `SensorFeatureConv`, `features_combine` | 없음 | 제거 | spatial 평탄화 충돌 |

### C. Decoder (변경 없음 vs v2)

| 항목 | upstream `muvo_2d` | 본 계획 `_muvo_2D` | 비고 |
|---|---|---|---|
| Image decoder | `ConvDecoder2D` (5 stage) | 신규 `TokenConvDecoder2D` (5 stage, 320×768 target) | |
| LiDAR decoder | `ConvDecoder2D` (range view, 64×1024) | 동일 패턴, **(32×1024 target, 시작 (2,64))** | v3: 시작 grid 변경 |
| Voxel / BEV / semseg / depth decoder | 있음 | 없음 | `_muvo` 정책 |
| Policy decoder | `PolicyDecoder(in_channels=512)` `Linear×4 + Tanh` | **upstream 코드 그대로 복붙, `in_channels=cfg.MODEL.EMBEDDING_DIM=256`** | Q6: 코드 그대로 |
| StyleDecoder (AdaIN) | 주석 보관 | 포팅 안 함 | |

### D. Loss / 메트릭 / 가중치 (v3 갱신)

| 항목 | upstream `muvo_2d` | 본 계획 `_muvo_2D` | 비고 |
|---|---|---|---|
| RGB loss | L1 multi-scale (1,2,4) | 동일 | |
| SSIM | 0.3 (default on) | 0.0 (default off) | `ssim_addition_notes.md` 후순위 |
| PerceptualLoss | optional default off | 포팅 안 함 | |
| LPIPSLoss | dead | 포팅 안 함 | |
| LiDAR loss | xyz L2 + range L1 + empty L1, `WEIGHT_LIDAR_RE=0.1` | 동일 framework, `WEIGHT_LIDAR_RE=0.0` | 별도 ablation |
| LIDAR_SCALE 보정 | 없음 | loss 항 미보정 (footgun) | 본 PR scope 밖 |
| Voxel/Dice/SemScal loss | 코드 있음/주석 | 없음 | head 없음 |
| KL weight | `1e-3` | `1e-2` | alpha `_muvo` baseline 유지 |
| KL free-bits | **없음** | **`cfg.LOSSES.KL_FREE_BITS_ENABLED=False` default, ON 시 1.0 nats** | v3 변경 (Q8): on/off cfg + default OFF로 upstream과 일치 |
| KL balancing α | 0.75 | 0.75 | |
| Action loss | `RegressionLoss(norm=1)` × `WEIGHT_ACTION=1.0` | upstream 동일 포팅 | v2 도입 |
| Near-field Chamfer | 있음 (PC range [-20,-20,-2,20,20,6]) | 포팅 | |
| Near-field voxel IoU | 있음 | 없음 | voxel head 없음 |
| PSNR / SSIM / Chamfer (global) | 있음 | 동일 | |

### E. 데이터 / 입력 (v3 갱신)

| 항목 | upstream `muvo_2d` | 본 계획 `_muvo_2D` | 비고 |
|---|---|---|---|
| Dataset | CARLA on-the-fly + augmentation | Arrow offline (alpha) | 본질적 차이 |
| Image input size | 600×960 → crop 320×832 | 600×800 → center-crop 320×768 | alpha 카메라 spec |
| LiDAR RV size | (64, 1024) | (32, 1024) | alpha 32-ch |
| Image augmentation | ON (각 30%) | ON, `cfg.DATA.AUGMENTATION.ENABLED` 단일 flag | v2 도입 |
| Route map augmentation | ON | 없음 | route 없음 |
| Action vector | throttle_brake + steering | 동일 (raw) | |
| Speed normalization | 5.0 (m/s) | 50.0 (km/h) | alpha unit |
| Receptive field | 4 | 4 | |
| Future horizon | 2 | 2 (v2) | upstream 일치 |
| Sequence length | 6 | 6 | RF+FH |
| **Sampling Hz 지정** | `DATASET.STRIDE_SEC=0.2`초 → 5Hz | **`cfg.DATA.EFFECTIVE_HZ` cfg (default 2)** → SAMPLE_EVERY_N 자동 도출 | v3 신규 (Q13) |
| `sample_every_n` / `frame_step` | knob 없음 | `FRAME_STEP=5` 유지, `SAMPLE_EVERY_N`은 `EFFECTIVE_HZ`에서 자동 | |
| Window index 캐시 | N/A | 새로 생성 필요 (`--seq-len 6 --stride 6`) | FH=2로 줄어 비호환 |

### F. 학습 / 평가 entrypoint (v3 갱신)

| 항목 | upstream `muvo_2d` | 본 계획 `_muvo_2D` | 비고 |
|---|---|---|---|
| Trainer 선택 | flag-driven (`TRANSFORMER_TRANSITION.ENABLED`) | 별도 `_muvo_2D.py` 5개 파일 sibling | |
| Test step | hand-rolled + ClearML upload | alpha 표준 PL `test_step` + ClearML artifact (v3 추가) | |
| **실험 트래킹** | **ClearML primary + TensorBoard auto-capture** | **v3: ClearML 채택 (`Task.init(...)` 추가)** | upstream 일치 (Q14/15) |
| Logging | ClearML + TensorBoard | ClearML primary + TensorBoard 보조 + `experiment_log.csv` 유지 | TensorBoard는 ClearML이 web UI에 자동 mirror |
| Artifact upload | `task.upload_artifact('data_i', ...)` | 동일 패턴, test_step에서 prediction batch dump | 옵션 |
| Eval dataset split | val0/val1/val2 3-way | alpha 단일 val | 변경 없음 |
| **Reconstruction figure** | `model.visualise(...)` (일부 frame) | **6 frame 전체 column 표시 + LiDAR-scale 보정/미보정 두 row** | v3 변경 (Q18) + v2 두 view |
| Checkpoint contract | muvo_2d 호환 | alpha 새 cfg → baseline `_muvo` ckpt와 비호환 | 새 run 시작 |
| CLI flag | upstream argparse | alpha + `--no-augmentation` + `--no-clearml` + `--cml-project`, `--cml-task` | RSSMTD 제약상 h-dim=z-dim 필수 |

### G. 본 계획에서 의도적으로 *제외*

이번 PR scope 밖. 별도 PR로:

- **PerceptualLoss** — `ssim_addition_notes.md` 후순위.
- **`WEIGHT_LIDAR_RE` 0.0→0.1 복원** + LIDAR_SCALE loss 보정 — 별도 PR.
- **Voxel / BEV / semseg / depth heads** — `_muvo` 정책 위배.
- **Route map / measurement encoder** — alpha dataset에 GPS/route 없음.
- **MobileViTv2 backbone option** — upstream ablation marginal.
- **Hydra-style config** — `SimpleNamespace`로 충분.
- **val0/val1/val2 3-way 분할** — alpha Arrow 데이터 분할과 무관.
- 기존 `_muvo*.py` 5개 파일 수정 — baseline 보존 필수.

### H. upstream의 latent bug 회피

- voxel loss commented out (`muvo_2d/muvo/trainer.py:192-212, 405-412`) — voxel head 없음, N/A.
- `LPIPSLoss` 정의-but-미사용 dead — 포팅 안 함.
- `prediction.py`의 `WorldModelTrainer.load_from_checkpoint` 주석 + 비표준 속성 의존 — 본 계획은 alpha 표준 PL `test_step` 사용.

---

## 결정사항 (v3, 사용자 확인됨)

- **Token decomposition** (v3): image 240 (10×24) + lidar 128 (2×64) + policy 1 = **369 tokens × 256ch**. Speed는 type embedding modulation add (token 추가 안 함). LiDAR encoder `out_indices=[1,2,3]`로 upstream MUVO 매칭.
- **KL free-bits**: cfg flag `cfg.LOSSES.KL_FREE_BITS_ENABLED` default False (upstream 일치).
- **Sampling Hz**: 단일 cfg `cfg.DATA.EFFECTIVE_HZ` default 2. `SAMPLE_EVERY_N = CARLA_HZ / (FRAME_STEP * EFFECTIVE_HZ)` 자동 도출.
- **실험 트래킹**: ClearML primary, `Task.init(...)` train_muvo_2D.py에 추가. cfg에 `CML_PROJECT/TASK/TYPE/TAG`. `--no-clearml`로 비활성 가능. TensorBoard 보조 (ClearML이 자동 capture).
- **Reconstruction figure**: RF+FH=6 frame 전체 column + LiDAR-scale 보정/미보정 두 row.
- **PolicyDecoder**: upstream 코드 그대로 (in_channels는 cfg.MODEL.EMBEDDING_DIM=256 자동).
- **Scope (v2+v3 누적)**: (a) RSSMTD + RepresentationModelTD + ConvGRUCellGlo, (b) TokenConvDecoder2D, (c) near-field Chamfer, (d) Speed encoder, (e) PolicyDecoder + action loss, (f) Image augmentation, (g) Recon figure 6 frame + 두 view, (h) ClearML 트래킹, (i) KL free-bits cfg flag, (j) EFFECTIVE_HZ cfg.
- **포함 안 함**: voxel head, PerceptualLoss, `WEIGHT_LIDAR_RE` 복원, measurement encoder.
- **차원 조정**: TransformerDecoder 6→3 layer, nhead 8→4.
- **시퀀스 길이**: FH 4→2 (v2) → seq_len 8→6. window index 캐시 새로 생성.
- **파일 네이밍**: `_muvo_2D.py` (대문자 D). 5개 페어 신규 생성, 기존 `_muvo.py` 무수정.

## 신규 파일 5개

모두 `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/` 아래.

1. `config_muvo_2D.py` — `config_muvo.py` copy + cfg 추가 (RSSM_2D, augmentation, action, CML, FH=2, EFFECTIVE_HZ, KL_FREE_BITS_ENABLED, LOGGING.RECON_FIG_*)
2. `data_muvo_2D.py` — `data_muvo.py` copy + import swap + PixelAugmentation 포팅
3. `models_muvo_2D.py` — `models_muvo.py` copy + RSSMTD/RepresentationModelTD/ConvGRUCellGlo/SpeedEncoder/PolicyDecoder/TokenConvDecoder2D + LiDAR encoder `out_indices=[1,2,3]`로 변경 (이 PR의 ~80%)
4. `trainer_muvo_2D.py` — `trainer_muvo.py` copy + decode shape 적응 + action loss + near-field Chamfer + 6 frame figure 두 view + KL free-bits cfg gate + ClearML artifact upload (옵션)
5. `train_muvo_2D.py` — `train_muvo.py` copy + import swap + `Task.init(...)` + `--no-clearml`, `--no-augmentation`, `--cml-project`, `--cml-task`

기존 `_muvo` 5개는 손대지 않는다.

---

## 파일별 변경 상세 (v3)

### 1. `config_muvo_2D.py`

`config_muvo.py` 그대로 copy 후:

- **신규 `MODEL.RSSM_2D` namespace**:
  ```python
  cfg.MODEL.RSSM_2D = SimpleNamespace(
      IMAGE_TOKEN_HW = (10, 24),   # FPN stride-32 of (320, 768)
      LIDAR_TOKEN_HW = (2, 64),    # v3: stride-16 of (32, 1024)
      LIDAR_OUT_INDICES = [1, 2, 3],   # v3: upstream MUVO 매칭
      POLICY_TOKENS = 1,
      TRANSFORMER_DECODER_LAYERS = 3,
      TRANSFORMER_DECODER_HEADS = 4,
      NEARFIELD_PC_RANGE = [-20, -20, -2, 20, 20, 6],
  )
  ```
- `MODEL.TRANSITION.HIDDEN_STATE_DIM = STATE_DIM = 256` (RSSMTD 가정).
- 신규 `cfg.MODEL.SPEED.ENABLED = True`, `SPEED.CHANNELS = 16`. `SPEED_NORMALISATION=50.0` 유지.
- 신규 `cfg.MODEL.POLICY.ENABLED = True`, `POLICY.HIDDEN_DIM = 256`.
- 신규 `cfg.LOSSES.WEIGHT_ACTION = 1.0`.
- **신규 `cfg.LOSSES.KL_FREE_BITS_ENABLED = False`** (v3, Q8), `KL_FREE_BITS = 1.0` (값은 유지하되 flag로 gate).
- 신규 `cfg.DATA.AUGMENTATION` namespace (v2와 동일):
  ```python
  cfg.DATA.AUGMENTATION = SimpleNamespace(
      ENABLED = True,
      BLUR_PROB=0.3, BLUR_WINDOW=5, BLUR_STD=[0.1, 1.7],
      SHARPEN_PROB=0.3, SHARPEN_FACTOR=[1, 5],
      COLOR_PROB=0.3, BRIGHTNESS=0.3, CONTRAST=0.3, SATURATION=0.3, HUE=0.1,
  )
  ```
- **신규 `cfg.DATA.CARLA_HZ = 20`, `cfg.DATA.EFFECTIVE_HZ = 2`** (v3, Q13).
  - 기존 `SAMPLE_EVERY_N` 키는 두되 init 시 `EFFECTIVE_HZ`에서 자동 도출하는 헬퍼 호출:
    ```python
    def derive_sample_every_n(cfg):
        return cfg.DATA.CARLA_HZ // (cfg.DATA.FRAME_STEP * cfg.DATA.EFFECTIVE_HZ)
    cfg.DATA.SAMPLE_EVERY_N = derive_sample_every_n(cfg)
    ```
  - 예: `CARLA_HZ=20, FRAME_STEP=5, EFFECTIVE_HZ=2` → `SAMPLE_EVERY_N=2`.
  - 명시적으로 override 시 (`--sample-every-n N`) 헬퍼 값 무시.
- **FUTURE_HORIZON 2** (v2), seq_len=6.
- **신규 ClearML cfg** (v3, Q14):
  ```python
  cfg.CML_ENABLED = True
  cfg.CML_PROJECT = 'alpha26_muvo_2D'
  cfg.CML_TASK = 'muvo_2D_v1'   # CLI/--cml-task로 override
  cfg.CML_TYPE = 'training'
  cfg.CML_TAGS = ['rssmtd', 'token-shape', 'alpha26']
  ```
- 신규 `cfg.LOGGING.RECON_FIG_BOTH_VIEWS = True` (v2, 보정/미보정 두 view).
- **신규 `cfg.LOGGING.RECON_FIG_ALL_FRAMES = True`** (v3, Q18) — 6 frame 전체 표시.
- `LOGGING.RUN_NAME` → `muvo_style_alpha_fpn256_lidar128_rssmtd`, `BASE_FILE` → `train_muvo_2D.py`.

유지: `LIDAR_SCALE=40.0`, `EMBEDDING_DIM=256`, `WEIGHT_LIDAR_RE=0.0`, `WEIGHT_SSIM=0.0`, `EPOCHS=1`, KL `1e-2`, `FRAME_STEP=5`.

### 2. `data_muvo_2D.py`

`data_muvo.py` 그대로 copy 후:

- Import swap: `from config_muvo_2D import cfg`.
- **PixelAugmentation 클래스 신규** (`muvo_2d/muvo/models/preprocess.py:295-333` 포팅).
- `_make_img_transform` (입력)에만 PixelAugmentation 적용, `_make_img_transform_raw` (target)은 그대로.
- Dataset 반환 dict key/shape 동일.

### 3. `models_muvo_2D.py` — 핵심 변경 (v3 추가)

`models_muvo.py` 그대로 copy 후:

#### 3-1. 신규 클래스

- **`PositionEmbeddingSine`** (`muvo_2d/muvo/models/common.py:636-678`)
- **`ConvGRUCellGlo`** (`muvo_2d/muvo/models/transition_td.py:28-58`)
- **`RepresentationModelTD`** 축소 포팅:
  - Query embedding 3개: `query_embed_image (1, 256, 10, 24)`, `query_embed_lidar (1, 256, 2, 64)` (v3 갱신), `query_embed_policy (1, 1, 256)`.
  - `type_embedding` 5 slot.
  - **TransformerDecoder d_model=256, nhead=4, num_layers=3** (v2).
- **`RSSMTD`** 축소 포팅:
  - `n_out_tokens = 240 + 128 + 1 = 369` (v3).
  - action_dim=2, embedding_dim=hidden_state_dim=state_dim=256.
- **`SpeedEncoder`** 신규 (upstream `muvo.py:124-130` 포팅): `Linear(1,16)→ReLU→Linear(16,256)→ReLU`. 출력을 type embedding의 speed slot에 broadcast add.
- **`PolicyDecoder`** 신규 (upstream `decoder.py:11-26` **코드 그대로 복붙**, in_channels는 cfg.MODEL.EMBEDDING_DIM=256 자동).

#### 3-2. `Model` 클래스 교체

- `Model.__init__`:
  - `self.rssm = RSSMTD(...)`.
  - `self.speed_encoder = SpeedEncoder(...)`.
  - `self.policy_decoder = PolicyDecoder(in_channels=cfg.MODEL.EMBEDDING_DIM)`.
  - **`self.range_view_encoder = timm.create_model(..., out_indices=cfg.MODEL.RSSM_2D.LIDAR_OUT_INDICES, ...)`** (v3): cfg에서 `[1,2,3]` 읽음.
- `Model.encode_fuse_sequence` rewrite:
  - LiDAR FPN 출력이 stride-16 → `(B*S, 256, 2, 64)` (v3, 이전 (1,32)에서 변경).
  - Speed embedding type embedding slot에 add.
  - `cam_tokens (B*S, 256, 240)`, `lidar_tokens (B*S, 256, 128)` (v3).
- `Model.forward`: state_dict shape `(B, S, 369, 256)`.
- `Model.imagine`: state_imagine dict 동일 구조 (`hidden_state (B, 369, 256)` 등).

#### 3-3. `TokenConvDecoder2D`

- Modality slice (v3 갱신):
  ```python
  img_h = hidden_state[:, :, :240, :].reshape(B, S, 10, 24, 256).permute(0,1,4,2,3)
  lid_h = hidden_state[:, :, 240:368, :].reshape(B, S, 2, 64, 256).permute(0,1,4,2,3)
  pol_h = hidden_state[:, :, 368, :]  # (B, S, 256)
  ```
- Image ladder: `(B*S, 256, 10, 24) → 5× stride-2 → (B*S, 64, 320, 768)`.
- LiDAR ladder (v3): `(B*S, 256, 2, 64) → 4× stride-2 → (B*S, 64, 32, 1024)`. (2→32 = 4× doubling, 64→1024 = 4× doubling). 따라서 5→4 stage로 감소. `head_4/2/1` 분기는 동일.
- Policy: `cat([pol_h, pol_s], dim=-1) → Linear(512→256) → PolicyDecoder`.
- Output dict 키: `rgb_1/2/4`, `lidar_reconstruction_1/2/4`, `action_pred (B, S, 2)`.

### 4. `trainer_muvo_2D.py`

`trainer_muvo.py` 그대로 copy 후:

- Import swap.
- `__init__`: 변경 없음 (cfg 경유).
- **`compute_loss`**:
  - state_dict shape `(B, S, 369, 256)` — KL 계산 시 token 차원 평균 (검증 포인트 #1).
  - **KL free-bits cfg gate** (v3): `if cfg.LOSSES.KL_FREE_BITS_ENABLED: kl_sum = kl_sum.clamp_min(cfg.LOSSES.KL_FREE_BITS)`. Default False → free-bits 미적용.
  - RGB/LiDAR loss 변경 없음.
  - **신규 action loss** (v2): `F.l1_loss(output["action_pred"], batch["action"]) * cfg.LOSSES.WEIGHT_ACTION`. `future_` prefix variant 동일.
- `compute_eval_metrics`: near-field Chamfer 추가 (PC range cfg에서).
- **`save_reconstruction_figure`** (v3):
  - **6 frame 전체 column** 표시 (`cfg.LOGGING.RECON_FIG_ALL_FRAMES=True`):
    - 컬럼: `[RF_0, RF_1, RF_2, RF_3, FH_0, FH_1]` (RF=4 + FH=2).
    - posterior recon이 RF 컬럼에, imagine prediction이 FH 컬럼에.
  - **LiDAR-scale 보정/미보정 두 row** (`cfg.LOGGING.RECON_FIG_BOTH_VIEWS=True`, v2):
    - Row 1: scaled (meter 단위, `× lidar_scale`).
    - Row 2: unscaled (training-space [0,1]).
  - Figure 전체 shape (예): 4 row (gt_rgb / pred_rgb / gt_lidar_meter / pred_lidar_meter / gt_lidar_unscaled / pred_lidar_unscaled) × 6 column.
- `_log_eval_outputs`: `nearfield`, `action` 키 log 추가.
- **ClearML artifact upload** (v3, 옵션): `test_step` 끝에서 batch별 prediction을 `task.upload_artifact(f'pred_batch_{batch_idx}', ...)`로 dump.

### 5. `train_muvo_2D.py`

`train_muvo.py` 그대로 copy 후:

- Import swap + `from clearml import Task`.
- **`Task.init(...)` 추가** (v3):
  ```python
  if cfg.CML_ENABLED and not args.no_clearml:
      task = Task.init(
          project_name=cfg.CML_PROJECT,
          task_name=args.cml_task or cfg.CML_TASK,
          task_type=cfg.CML_TYPE,
          tags=cfg.CML_TAGS,
      )
      task.connect(cfg)
  ```
- CLI flag 추가: `--no-clearml` (override `cfg.CML_ENABLED=False`), `--cml-project`, `--cml-task`, `--no-augmentation`, `--effective-hz`.
- `--effective-hz N` 적용 시 `cfg.DATA.EFFECTIVE_HZ=N` → `SAMPLE_EVERY_N` 자동 재계산.
- `change_summary` 갱신: `"MUVO-style alpha 2D-token RSSM (RSSMTD) + speed + policy + augmentation + ClearML"`.

---

## 변경 순서 (구현 시)

1. `config_muvo_2D.py` 작성 (RSSM_2D, augmentation, CML, EFFECTIVE_HZ, KL_FREE_BITS_ENABLED).
2. `data_muvo_2D.py` 작성 + PixelAugmentation + `EFFECTIVE_HZ→SAMPLE_EVERY_N` resolver.
3. `models_muvo_2D.py`에 ConvGRUCellGlo/RepresentationModelTD/RSSMTD/SpeedEncoder/PolicyDecoder/PositionEmbeddingSine 신규 추가 (standalone test).
4. `models_muvo_2D.Model` rewrite + LiDAR encoder `out_indices=[1,2,3]` 변경 + Speed encoder integration. 임시로 SensorDecoder reduce.
5. `TokenConvDecoder2D` + `PolicyDecoder` 통합. 임시 reduce 제거.
6. `trainer_muvo_2D.py` 작성: action loss + near-field Chamfer + 6 frame figure 두 view + KL free-bits gate + ClearML artifact (옵션).
7. `train_muvo_2D.py` 작성 + `Task.init` + CLI flags.
8. **Window index 캐시 재생성** (`build_arrow_window_index.py --seq-len 6 --stride 6 --sample-every-n 2 --frame-step 5 --out arrow_window_index_seq6.pkl`).
9. **ClearML 환경 설정** (별도 `clearml_guide.md` 참조): `pip install clearml`, `clearml-init` 명령으로 ~/.clearml/clearml.conf 생성.
10. Tiny overfit으로 검증.

---

## 검증 (Verification)

1. **Window index 캐시 재생성**:
   ```bash
   python scripts/build_arrow_window_index.py \
     --root /home/user/chaeyeon-kim/processed \
     --out  /home/user/chaeyeon-kim/processed/arrow_window_index_seq6.pkl \
     --seq-len 6 --stride 6 --sample-every-n 2 --frame-step 5
   ```

2. **ClearML 셋업** (1회):
   ```bash
   pip install clearml
   clearml-init   # web에서 받은 credential 입력
   ```
   상세는 `alpha26/docs/clearml_guide.md`.

3. **Import sanity**:
   ```bash
   python -c "from scripts.files.models_muvo_2D import Model, SpeedEncoder, PolicyDecoder; print('ok')"
   python -c "from scripts.files.trainer_muvo_2D import WorldModelTrainer; print('ok')"
   ```

4. **EFFECTIVE_HZ 자동 도출 확인**:
   ```bash
   python -c "from scripts.files.config_muvo_2D import cfg; print(cfg.DATA.SAMPLE_EVERY_N, cfg.DATA.EFFECTIVE_HZ)"
   # 기대: 2 2 (default)
   ```

5. **Augmentation off ablation**: `--no-augmentation` 플래그로 forward pass 확인.

6. **KL free-bits OFF/ON 비교**: default OFF로 첫 run → ON으로 두 번째 run, KL scalar 자릿수 차이 확인 (검증 포인트 #1).

7. **Tiny overfit** (1 GPU):
   ```bash
   python scripts/model_variants/train_muvo_2D.py \
     --train-run train_run_002.arrow --steps 200 --devices 1 --batch-size 2 \
     --val-check-interval 20 --limit-val-batches 1 \
     --window-index /home/user/chaeyeon-kim/processed/arrow_window_index_seq6.pkl \
     --no-clearml --run-name debug_muvo_2D_tiny
   ```
   기대: forward pass error 없음, 모든 loss 정상 자릿수. 6 frame figure 정상 render. ClearML 없이 TensorBoard만으로 동작.

8. **ClearML tracking 확인** (별건):
   ```bash
   python scripts/model_variants/train_muvo_2D.py \
     --steps 200 --devices 1 --cml-task debug_muvo_2D_clearml ...
   ```
   web UI (app.clear.ml or 내부 server)에서 task 자동 등록 + scalar/figure mirror 확인.

9. **4-GPU 본 학습** (위 셋 통과 후):
   ```bash
   NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 \
   python scripts/model_variants/train_muvo_2D.py \
     --steps 150000 --devices 4 \
     --strategy ddp_find_unused_parameters_true \
     --batch-size 2 --num-workers 2 \
     --window-index /home/user/chaeyeon-kim/processed/arrow_window_index_seq6.pkl \
     --cml-task muvo_2D_full_v1 --run-name muvo_2D_full_v1
   ```
   메모리 확인: token 수 369 + seq_len 6 → v2의 273 대비 35% 증가, baseline `_muvo`(flat) 대비 RSSM ~270× 증가 가능. `--batch-size 1`로 회피 가능.

10. **`experiment_log.csv` 자동 row 추가 확인** — `ExperimentCSVLogger.on_fit_end` 그대로 동작.

11. **ClearML 비교 view** — `_muvo` baseline run과 `_muvo_2D` run을 ClearML UI에서 동시 view (`clearml_guide.md` §run 비교 참조).

---

## 변경하지 않는 것 (Out of scope)

- `WEIGHT_LIDAR_RE` 복원 — 별도 ablation.
- LIDAR_SCALE unit footgun fix — 별도 PR.
- PerceptualLoss — 후순위.
- Voxel head / `ConvDecoder3D` — alpha dataset에 GT 없음.
- Measurement encoder (route_command / GPS) — alpha dataset에 정보 없음.
- 기존 `_muvo*.py` 5개 파일 수정 — baseline 보존 필수.
- BEV / depth / semseg / LiDAR seg head — `_muvo` 정책 위배.

---

## 영향받는 critical 파일 path 목록

신규 생성:
- `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/config_muvo_2D.py`
- `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/data_muvo_2D.py`
- `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/models_muvo_2D.py`
- `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/trainer_muvo_2D.py`
- `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/train_muvo_2D.py`

참조 (수정 안 함):
- `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/{config,data,models,train,trainer}_muvo.py` — copy 베이스
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/models/muvo.py:78-82` — LiDAR encoder `out_indices=[1,2,3]` 포팅 근거 (v3)
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/models/muvo.py:124-130, 415-417` — SpeedEncoder
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/models/transition_td.py` — RSSMTD, RepresentationModelTD, ConvGRUCellGlo
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/models/common.py:636-678` — PositionEmbeddingSine
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/models/decoder.py:11-26` — PolicyDecoder
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/models/preprocess.py:295-333` — PixelAugmentation
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/metrics.py:253-289` — CDMetric near-field
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/losses.py:55-73` — RegressionLoss
- `/home/carol/chaeyeon-kim/muvo_2d/muvo/trainer.py:267-272` — action loss 패턴
- `/home/carol/chaeyeon-kim/muvo_2d/prediction.py:23-25, 98` — Task.init + upload_artifact 패턴

업데이트:
- `/home/carol/chaeyeon-kim/alpha26/docs/clearml_guide.md` (신규, Q14/15)
- `/home/carol/chaeyeon-kim/alpha26/docs/tensorboard_guide.md` (보조용으로 축소 갱신)
- `/home/carol/chaeyeon-kim/alpha26/docs/muvo_full_comparison.md` §4.0의 cross-reference (구현 후).

---

## 위험 / 미해결 항목

1. **RSSM 메모리 — token 369 + seq_len 6** — v2의 273 대비 35% 증가. tiny overfit에서 GPU memory 확인 필수. 필요 시 `query_embed_*` 채널 256→128 또는 batch_size 축소.
2. **KL free-bits OFF default일 때 posterior collapse 위험** — alpha _muvo baseline은 항상 ON 상태로 stable했음. `_muvo_2D` default OFF로 가면 collapse 가능성. 초기 run에서 KL 자릿수 모니터링 필수. 검증 #6에서 OFF/ON 비교 포함.
3. **Action token이 매 frame concat** — RSSMTD의 action 처리 방식이 `_muvo`와 다름. action loss 도입으로 PolicyDecoder 정확도가 우회 검증.
4. **Speed normalization unit** — upstream m/s vs alpha km/h. dimensionless 일관성 위해 encoder는 동일.
5. **Window index 캐시 비호환** — `arrow_window_index.pkl` (seq_len=8) 못 씀. 새 `arrow_window_index_seq6.pkl` 생성 필수.
6. **Image augmentation 효과 vs RSSMTD 효과 confound** — `--no-augmentation` ablation 필수.
7. **ClearML 서버 의존** — SaaS (app.clear.ml) 또는 self-hosted 서버 필요. 첫 셋업 시 `clearml-init`로 credential 등록. `--no-clearml`로 우회 가능하지만 그러면 실험 비교 web UI 못 씀.
8. **LiDAR token 정확도 검증** — v3 변경 후 첫 forward에서 `lidar_features.shape == (B*S, 256, 2, 64)` 확인 (FPN/encoder의 stride-16 종료 검증).

---

## 참고 가이드

- `alpha26/docs/clearml_guide.md` — ClearML 셋업 + 사용법 (비전공자 친화, Q14/15).
- `alpha26/docs/tensorboard_guide.md` — TensorBoard 보조 사용 (ClearML이 auto-capture, 로컬 디버깅용).
- `alpha26/docs/muvo_full_comparison.md` — 4-way 비교 (upstream main / 2D 브랜치 / docs / alpha _muvo 페어).
