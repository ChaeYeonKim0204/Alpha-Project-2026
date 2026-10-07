# Upstream 1D vs 2D — 비교

`/muvo/` (1D 레퍼런스, MILE 기반) vs `/muvo_2d/` (2D latent 변형) 사이의 차이를 표 중심으로 정리한다. 세부 감사 결과는 `upstream_1D.md`, `upstream_2D.md` 참조.

---

## 1. 최상위 변경 요약

| 영역 | upstream 1D (`muvo/`) | upstream 2D (`muvo_2d/`) | 핵심 차이 |
|---|---|---|---|
| 메인 모델 | `muvo/muvo/models/mile.py:Mile` | `muvo_2d/muvo/models/muvo.py:MUVO` | 별도 클래스, 토큰 기반 fusion |
| RSSM | `transition.py:RSSM` `(B,S,D)` | `transition_td.py:RSSMTD` `(B,S,C,N_token)` | 1D 벡터 → **2D 토큰** |
| Recurrent cell | `nn.GRUCell` (FC) | `ConvGRUCellGlo` (Conv1d + global gate) | FC → Conv |
| 분포 모델 N_φ, N_θ | MLP `RepresentationModel` | `RepresentationModelTD`(TransformerDecoder) | MLP → cross-attention |
| Decoder | `common.py:ConvDecoder`(inline) | `decoder.py:ConvDecoder2D/3D, PolicyDecoder` | 모듈 분리, multi-scale, voxel은 3D Conv 분리 |
| Position encoding | `PositionEmbeddingSine` (2D) | + `PositionEmbeddingSine3D` (voxel용) | 3D PE 추가 |
| Loss (PL) | – | `losses.py:PerceptualLoss, LPIPSLoss` | paper §IV-B의 PL 항 |
| Trainer 모델 분기 | `Mile` 고정 | `MUVO if TRANSFORMER_TRANSITION.ENABLED else Mile` | 분기 스위치 |
| RL agent 위치 | `rl_birdview/` | `agents/rl_birdview/` | 이름만 이동(내용 동일) |
| `prediction.py` | trainer.fit 미사용 (test only) | 동일 + inline visualization + ClearML | 추론 흐름 |
| YAML configs | `test_base_1d.yml, test_base_1d_without_voxel.yml, test_base_2d.yml, test_mobilevit_2d.yml` 등 다수 | `debug.yml, muvo.yml, one_frame.yml, predict.yml` | 1D는 ablation별 yml, 2D는 단일 muvo.yml |
| Dataset (`dataset.py`) | test 단일 dataset + sampler 분기 | test 3개 dataset 별도 + stride 100/100/10 | test split 정밀화 |

---

## 2. RSSM 구조 비교 (paper §III-C ↔ 코드)

| 항목 | 1D (`muvo/muvo/models/transition.py`) | 2D (`muvo_2d/muvo/models/transition_td.py`) |
|---|---|---|
| State shape | `h: (B,S,H), s: (B,S,Z)` 평탄 | `h: (B,S,C,N_token), s: (B,S,C,N_token)` 토큰 |
| `N_token` | n/a | **1093** = image 260 (10×26) + lidar 256 (4×64) + voxel 576 (12×12×4) + policy 1 |
| Recurrent | `nn.GRUCell(hidden_state_dim, hidden_state_dim)` | `ConvGRUCellGlo`: Conv1d(in→out, ks=1) over tokens + sigmoid global gate `mean(hx, -1)` |
| Pre-GRU net | `pre_gru_net(sample)` → linear | `pre_gru_net([sample, latent_action])` → Linear+LeakyReLU |
| Prior `p(s|h,a)` | MLP `RepresentationModel([h, latent_a])` → (μ, σ) | TransformerDecoder `RepresentationModelTD` — query: 학습 가능 토큰; key/value: h, latent_a |
| Posterior `q(s|o,h,a)` | MLP `RepresentationModel([h, o, latent_a])` | TransformerDecoder — query: 토큰; key/value: h, embedding, latent_a |
| Dropout (prior 대체) | `use_dropout=True`로 posterior를 일부 prior로 교체 | – |
| Action handling | `action_in_gru=False` (기본): action은 prior/posterior 입력만 | observe/imagine step에서 latent_action을 token type=2로 concat |

---

## 3. Decoder 비교

| 항목 | 1D | 2D |
|---|---|---|
| 입력 | `[h, sample]` concat (B*S, H+Z) → Linear → unflatten | (B*S, C, H_base, W_base) — 토큰을 모달리티별 reshape 후 직접 |
| Camera 디코더 | `common.py:ConvDecoder(rgb)` inline in `mile.py` | `decoder.py:ConvDecoder2D(camera_state)` |
| LiDAR 디코더 | `common.py:ConvDecoder(lidar)` | `decoder.py:ConvDecoder2D(lidar_state)` |
| Voxel 디코더 | `common.py:VoxelDecoder1` (3D ConvT 사다리 + 트라이리니어 보간) | `decoder.py:ConvDecoder3D` (3D ConvT 3단) |
| Multi-scale | 1/2/4 multi-scale 헤드 | 1/2/4 multi-scale 헤드 동일 |
| Policy 디코더 | `common.py:Policy` (state→Tanh(2)) | `decoder.py:PolicyDecoder` (state→Tanh(2)) |
| 헤드 종류 | RGB, Depth, LidarRe, LidarSeg, Sem image, BEV seg/center/offset, Voxel sem | 동일 |

---

## 4. Fusion 비교

| 항목 | 1D | 2D |
|---|---|---|
| Image backbone | timm ResNet18 (3 stage features) + FPN | 동일 (out_indices [2,3,4]) |
| LiDAR backbone | timm ResNet18 in_chans=4 (3 stage) + FPN | 동일 (out_indices [1,2,3] — stride-16 추가) |
| Token sequence | Flatten H, W → concatenate + 2D sinusoidal PE + type embedding (image/lidar) | 동일 패턴이지만 type slot 5개 (image/lidar/policy/action/speed). encoder는 0/1만 사용 |
| Transformer encoder | k=? layers | num_layers=6, nhead=8 (config `MODEL.TRANSFORMER.CHANNELS=512`) |
| Speed handling | 1D feature concat | broadcast-add to all tokens (modulation) |

---

## 5. Loss 항 비교 (paper §IV-A 식 1과 매핑)

| Loss 항 | 1D (`muvo/muvo/losses.py`) | 2D (`muvo_2d/muvo/losses.py`) | Paper |
|---|---|---|---|
| RGB L1 multi-scale | ✓ | ✓ | $`ℒ^{\text{img}}`$ |
| LiDAR L2 xyz | ✓ | ✓ | $`ℒ^{\text{pxyz}}`$ |
| LiDAR L1 range | ✓ | ✓ | $`ℒ^{\text{pr}}`$ |
| Voxel SCAL | ✓ (`VoxelLoss + SemScalLoss + GeoScalLoss`) | ✓ | $`ℒ^{V,\text{scal}}`$ |
| KL probabilistic | `KLLoss` 균형 KL | 동일 | – (paper transition 구조 내부) |
| SSIM | `SSIMLoss` (weight=0 기본) | 동일 | – |
| Chamfer | `CDLoss` | 동일 | – |
| Perceptual / LPIPS | ✗ | ✓ (`PerceptualLoss`, `LPIPSLoss`) | PL (paper §IV-B Fig. 4, 효과 미미) |
| Action loss | ✓ (`WEIGHT_ACTION`) | ✓ 동일 | – |

---

## 6. Config 차이

| Config 키 | 1D | 2D |
|---|---|---|
| `MODEL.TRANSFORMER_TRANSITION.ENABLED` | ✗ | `True` (필수 분기 스위치) |
| `VOXEL.EV_POSITION` | `[32, 96, 12]` | `[96, 96, 12]` |
| `CML_DATASET_VERSION` | ✗ | ✓ |
| `LOSSES.PERCEPTUAL` | ✗ | ✓ (옵션 게이트) |
| 기타 (LOSSES.WEIGHT_*, VOXEL.SIZE, BEV.SIZE 등) | 동일 | 동일 |

---

## 7. 파일 단위 byte-diff 요약

| 디렉터리 | 상태 |
|---|---|
| `muvo/muvo/layers/` ↔ `muvo_2d/muvo/layers/` | 100% identical |
| `muvo/muvo/utils/` ↔ `muvo_2d/muvo/utils/` | 100% identical |
| `muvo/muvo/data/{carlagym_utils.py, dataset_utils.py}` | identical |
| `muvo/muvo/data/dataset.py` | 부분 변경 (test sampler) |
| `muvo/muvo/models/{frustum_pooling.py, preprocess.py, transition.py, utils.py}` | identical |
| `muvo/muvo/models/{common.py, mile.py}` | 변경 (PositionEmbeddingSine3D/PointPillarNet 추가; mile 잔존) |
| `muvo/muvo/models/{muvo.py, decoder.py, transition_td.py}` | 2D-only NEW |
| `muvo/muvo/{losses.py, metrics.py, trainer.py, config.py}` | 변경 |
| `muvo/muvo/visualisation.py` | identical |
| `muvo/{utils, carla_gym}/` | 거의 identical (zombie/criteria/agents 모두 동일) |
| `muvo/rl_birdview/` ↔ `muvo_2d/agents/rl_birdview/` | 이름만 다름, 내용 identical |
| 최상위 scripts | `prediction.py`만 차이; 나머지 identical |

---

## 8. Paper 결론 ↔ upstream 구현 매핑

paper §IV-B/§V의 핵심 결론들은 upstream 2D 구현에 다음과 같이 반영되어 있다:

- **"RV-WOB-TR + 2D latent 최적"** → upstream 2D의 디폴트 (range view encoder + no BEV branch + transformer fusion + RSSMTD).
- **"2D latent state benefits camera predictions"** → `RSSMTD`의 `(B,S,C,N_token)` 토큰 상태 + TransformerDecoder + ConvDecoder2D가 그대로 구현.
- **"PL/ViT effect small"** → upstream 2D는 `PerceptualLoss`/`LPIPSLoss` 옵션을 제공하나 default config는 비활성(ENABLED 가드).
- **"Pre-training (cam+lidar) → voxel"** → upstream 2D의 `predict.yml`, `muvo.yml`에서 단계적 학습이 가능 (freeze 인코더 + voxel 디코더만 학습). `trainer.py`의 freeze list 사용.
- **"Occupancy → sensor minor benefit"** → upstream 2D는 occupancy 헤드를 켠 상태로 multi-task 학습 가능 (`LOSSES.WEIGHT_VOXEL > 0`).

따라서 **upstream 2D는 paper §III/§IV의 권고를 거의 그대로 구현한 reference** 이고, upstream 1D는 paper의 "1D baseline" 비교군(Fig. 4의 light blue)에 해당한다.
