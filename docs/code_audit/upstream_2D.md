# Upstream 2D Code Audit — `/home/carol/chaeyeon-kim/muvo_2d/`

본 문서는 upstream MUVO 2D 변형 리포지토리(`/muvo_2d/`)를 모든 leaf 디렉터리 단위로 감사한 결과를 종합한다. 1D(`/muvo/`) 대비 차이점에 집중한다.

---

## 0. 디렉터리 트리 (1D 대비 변경점 강조)

```
muvo_2d/
├── train.py / prediction.py / data_collect.py / sim_run.py / constants.py
│         ※ prediction.py만 1D와 의미 있는 차이 (inline prediction loop, ClearML)
├── muvo/
│   ├── config.py        ※ +MODEL.TRANSFORMER_TRANSITION, VOXEL.EV_POSITION 변경
│   ├── losses.py        ※ +PerceptualLoss, +LPIPSLoss (Paper §IV-B PL 항)
│   ├── metrics.py       ※ import 정리(외부 chamferdist 제거)
│   ├── trainer.py       ※ +MUVO 모델 분기 (Mile vs MUVO 선택)
│   ├── visualisation.py (identical)
│   ├── configs/         ※ YAML만 (debug/muvo/one_frame/predict)
│   ├── data/   (identical except dataset.py test sampler stride)
│   ├── layers/ (identical)
│   ├── models/                ← 2D의 핵심 차이
│   │   ├── common.py          ※ +PositionEmbeddingSine3D, +PointPillarNet
│   │   ├── decoder.py         ★ NEW: ConvDecoder2D / ConvDecoder3D / PolicyDecoder
│   │   ├── frustum_pooling.py (identical 계열)
│   │   ├── mile.py            (1D 참조용으로 잔존, 2D 파이프라인은 미사용)
│   │   ├── muvo.py            ★ NEW: MUVO 메인 2D 모델 (Mile 대체)
│   │   ├── preprocess.py      (identical)
│   │   ├── transition.py      (1D RSSM, 참조용 잔존)
│   │   ├── transition_td.py   ★ NEW: RSSMTD + ConvGRUCellGlo (Paper §III-C 2D)
│   │   └── utils.py
│   └── utils/  (identical)
├── utils/        (identical)
├── agents/                    ← 1D의 rl_birdview/를 이 위치로 이동, 내용은 동일
│   └── rl_birdview/{models,utils}/
└── carla_gym/                 (대부분 identical; LiDAR/sample_stride만 다름)
    ├── core/obs_manager/birdview/maps/   ← Town01~06 .h5 (데이터)
    ├── core/obs_manager/lidar/            (identical)
    └── core/task_actor/scenario_actor/agents/utils/ (identical)
```

총 11 leaf (Python 보유). 1D 대비 신규: `muvo/models/{muvo.py, decoder.py, transition_td.py}` + `agents/` (이름만 이동). 변경된 파일: `config.py, losses.py, metrics.py, trainer.py, dataset.py, prediction.py`. 동일: 나머지 대다수.

---

## 1. World Model Core — `muvo_2d/muvo/`

### 1.1 Top-level (config/losses/metrics/trainer/visualisation)

| 파일 | 1D 대비 변경 |
|---|---|
| `config.py` (+6줄) | `_C.CML_DATASET_VERSION` 추가; `_C.VOXEL.EV_POSITION = [96, 96, 12]` (1D는 `[32, 96, 12]`); **`_C.MODEL.TRANSFORMER_TRANSITION.ENABLED = True`** (2D RSSM 분기 스위치); `_C.MODEL.LIDAR.BACKBONE` 제거 |
| `losses.py` (+113줄) | **`PerceptualLoss`(timm resnet18 기반 multi-layer feature + style)**, **`LPIPSLoss`(torchmetrics)** 신규 — paper Fig. 4의 PL 항 구현 |
| `metrics.py` (+61줄) | 외부 `chamferdist` import 제거, 내부 `SSIMLoss/CDLoss` import로 정리. 메트릭 로직은 동일 |
| `trainer.py` (+61줄) | `from muvo.models.muvo import MUVO` 추가; **`self.model = MUVO(self.cfg) if cfg.MODEL.TRANSFORMER_TRANSITION.ENABLED else Mile(self.cfg)`** 분기; PerceptualLoss 초기화(`cfg.LOSSES.PERCEPTUAL.ENABLED`) |
| `visualisation.py` | identical |

YAML configs(`muvo_2d/muvo/configs/`): `debug.yml, muvo.yml, one_frame.yml, predict.yml`. `predict.yml`은 신규.

### 1.2 `muvo_2d/muvo/data/`

- `carlagym_utils.py`, `dataset_utils.py` — identical with 1D.
- `dataset.py` — 1D 대비 차이:
  - test split 처리: 1D는 `test_dataset` 단일을 sampler만 다르게; 2D는 `test_dataset_0/1/2` 별도 인스턴스 + 더 촘촘한 stride (100, 100, 10).
  - 그 외 RangeView/voxel 처리 등은 동일.

### 1.3 `muvo_2d/muvo/layers/`, `muvo_2d/muvo/utils/`

모두 1D와 byte-identical (32+5 utility 함수/클래스 동일).

### 1.4 `muvo_2d/muvo/models/` ← 2D의 핵심 차이

| 파일 | 상태 | 책임 |
|---|---|---|
| `muvo.py` (786줄) | **NEW** | 메인 2D MUVO 모델. ResNet18(cam, out_indices [2,3,4]) + ResNet18(LiDAR, in_chans=4, out_indices [1,2,3]) → FPN → SensorFusionTransformer(6L/8H) → 토큰 결합(image 260 + lidar 256 + voxel 576 + policy 1 = 1093 tokens) → `RSSMTD` → 모달리티별 토큰 분할 후 `ConvDecoder2D`/`ConvDecoder3D`/`PolicyDecoder`로 디코드. BEV/PointPillar 분기 모두 지원. |
| `decoder.py` (366줄) | **NEW** | `PolicyDecoder` MLP (state→Tanh(2)); `ConvDecoder2D` 5단 ConvTranspose2d 사다리 + multi-scale 헤드(RGB/Seg/Depth/SemImage/LidarRe/LidarSeg); `ConvDecoder3D` 3단 ConvTranspose3d → VoxelSemHead; `StyleDecoder2D/3D` (commented out, adaptive instance norm 변형) |
| `transition_td.py` (301줄) | **NEW** | `RSSMTD` — paper §III-C 2D 상태 구현. 상태 shape `(N_token, B, C)`, 내부적으로 `(B, S, C, N_token)`로 stack. `ConvGRUCellGlo` (Conv1d 게이트 + global gate via mean over tokens) — paper "FC→Conv 교체" 그대로 구현. `RepresentationModelTD` — TransformerDecoder + 학습 가능 query (image 10×26, lidar 4×64, voxel 12×12×4, policy 1) |
| `common.py` (839줄) | 1D + α | `PositionEmbeddingSine3D` (3D voxel용), `PointPillarNet` (range→pillar) 신규. 나머지 디코더/헤드는 1D와 공유 |
| `frustum_pooling.py` | 동일 계열 | Lift-Splat-Shoot frustum→BEV (1D와 동일) |
| `mile.py` | 1D 잔존 | 2D 파이프라인은 미사용 (`TRANSFORMER_TRANSITION.ENABLED=True`이면 MUVO 분기) |
| `preprocess.py` | identical | PreProcess + PixelAugmentation + RouteAugmentation |
| `transition.py` (192줄) | 1D 잔존 | reference로 남아 있음 |
| `utils.py` | identical | – |

RSSMTD 핵심 (paper §III-C와 비교):
- 상태: `(B, S, C, N_token)`, `N_token = Σ_j T_j = image (10×26=260) + lidar (4×64=256) + voxel (12×12×4=576) + policy (1) = 1093`.
- `ConvGRUCellGlo` — Conv1d로 토큰 차원 across recurrence; `glo = sigmoid(w(hx)) * mean(hx, dim=-1)` 전역 게이트.
- `RepresentationModelTD` — TransformerDecoder(num_layers=3, nhead=4, d_model=cfg.MODEL.TRANSFORMER.CHANNELS=512), 학습 가능 query 임베딩 + 2D/3D positional embedding.
- prior `p(s|h,a)` / posterior `q(s|h,e,a)` 모두 같은 RepresentationModelTD 인스턴스(다만 `prior_action_module` / `posterior_action_module`로 action 분기).

### 1.5 Decoder 분기 (paper §III-D)

`ConvDecoder2D` 입력은 `(B*S, C, H_base, W_base)` (예: 카메라용 (10, 26), LiDAR용 (4, 64)). ConvTranspose2d stride=2를 반복해 multi-scale 헤드 1/2/4 비율로 출력 (Paper §IV-A의 multi-scale 손실에 대응).

`ConvDecoder3D` 입력은 `(B*S, C, 12, 12, 4)` → ConvTranspose3d 3단으로 `(B*S, C_out, 192, 192, 64)`까지 업스케일 (Paper §IV-A의 192×192×64 voxel grid와 일치).

`PolicyDecoder` — MLP(in=2C → ... → Tanh(2)). 행동(throttle, steering) 회귀. Paper 본문엔 explicit policy 헤드 언급 없으나 upstream에서는 expert behavior cloning 보조 수단.

---

## 2. Top-level scripts

| 파일 | 상태 |
|---|---|
| `constants.py` | identical with 1D |
| `train.py` | identical with 1D |
| `data_collect.py` | identical with 1D |
| `sim_run.py` | identical with 1D |
| `prediction.py` | **CHANGED** — 1D 대비: git tracking callback 제거, custom checkpoint 로직 제거, ClearML task 로깅 활성화, **inline prediction loop with visualization** 추가, `trainer.fit()` 제거. 2D MUVO 모델의 직접 추론·시각화에 맞춤. |

---

## 3. RL Agent — `muvo_2d/agents/`

1D의 `rl_birdview/`가 `agents/rl_birdview/`로 이동 — 파일 내용은 모두 byte-identical:
- `agents/rl_birdview/models/`: `distributions.py, ppo.py, ppo_buffer.py, ppo_policy.py, torch_layers.py, torch_util.py`
- `agents/rl_birdview/utils/`: `rl_birdview_wrapper.py, wandb_callback.py`

---

## 4. CARLA Gym — `muvo_2d/carla_gym/`

대부분 1D와 동일. 보이는 차이:
- `core/obs_manager/birdview/maps/`: 데이터 디렉토리 (Town01~06 .h5). Python 파일 없음.
- `dataset.py` 외 carla_gym 코드 자체는 byte-identical.

---

## 5. Paper §III/§IV ↔ upstream 2D 트레이서빌리티

| Paper | upstream 2D 위치 |
|---|---|
| §III.A 이미지/LiDAR 인코딩 | `muvo_2d/muvo/models/muvo.py` (timm ResNet18 + FPN) |
| §III.B Transformer fusion + 2D PE + sensor embedding | `muvo_2d/muvo/models/muvo.py:SensorFusionTransformer` + `common.py:PositionEmbeddingSine` (2D/3D variants) |
| §III.C **2D latent state RSSM** | `muvo_2d/muvo/models/transition_td.py:RSSMTD` ← paper headline 구조 |
| §III.C "FC→Conv 교체" GRU | `transition_td.py:ConvGRUCellGlo` (Conv1d 게이트 + global gate) |
| §III.C "TransformerDecoder + 학습 가능 query" prior/posterior | `transition_td.py:RepresentationModelTD` |
| §III.D 카메라/LiDAR 2D conv 디코더 | `muvo_2d/muvo/models/decoder.py:ConvDecoder2D` |
| §III.D Voxel 3D conv 디코더 | `decoder.py:ConvDecoder3D` |
| §IV-A 식 (1) 손실 | `muvo_2d/muvo/losses.py` (PerceptualLoss/LPIPS 추가) + `trainer.py:compute_loss` |
| §IV-B Perceptual Loss (PL) | `losses.py:PerceptualLoss, LPIPSLoss` |
| §IV-B ViT 백본 | upstream 2D는 ResNet18만; ViT 분기는 잔존 `mile.py` 또는 별도 config |

---

## 6. 1D 대비 핵심 변경 5가지

1. **`MUVO` 클래스 신설**: `Mile`을 `cfg.MODEL.TRANSFORMER_TRANSITION.ENABLED` 분기로 대체. `muvo.py`는 2D 토큰 fusion + RSSMTD + Conv2D/3D 디코더 통합.
2. **`RSSMTD` (Token-shaped RSSM)**: `(B, S, C, N_token)` 상태, ConvGRUCellGlo, RepresentationModelTD. paper §III-C의 "2D latent state" 정확한 구현.
3. **`ConvDecoder2D/3D` 신규 디코더**: ConvTranspose 사다리 + 헤드 결합. 1D의 `mile.py` 내 inline ConvDecoder 호출을 별도 모듈로 분리.
4. **PerceptualLoss/LPIPSLoss**: paper Fig. 4의 PL 항 정량 비교용. 결론적으로 효과 없음(paper §IV-B).
5. **데이터 분기 강화**: test split을 분리된 dataset 인스턴스 + 더 촘촘한 stride로. validation에서 $`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$ 분리 명확화.

---

## 7. 관찰

- **`mile.py`, `transition.py`는 잔존** (참조 / 1D 분기용). 실제 2D 학습 경로는 `muvo.py → transition_td.py → decoder.py`.
- **Decoder는 모듈로 분리** — 헤드(RGB/Seg/Depth/Lidar/Voxel/Policy)가 `ConvDecoder2D` / `ConvDecoder3D` 안에서 multi-scale로 정리. 1D보다 구조적이며 정량 비교가 쉬움.
- **Voxel 토큰(12×12×4=576)**: paper §III-D의 (C × X × Y × Z) occupancy 디코더 입력과 일치. RSSMTD의 N_token에 포함.
- **상태 차원 `C = cfg.MODEL.TRANSFORMER.CHANNELS = 512`**: paper에는 Dh/Ds 수치가 명시되지 않으나, upstream 2D는 hidden_state_dim = state_dim = 512로 통일.
- **alpha-2D 비교 시 주의**: alpha-2D는 voxel 토큰을 제거하고 (image 240 + lidar 128 + policy 1 = 369)로 축소. 즉 upstream 2D의 1093 → alpha 369. occupancy 헤드 부재가 토큰 수에 반영됨.
