# MUVO Paper Summary

작성일: 2026-05-15

원문: *Bogdoll, Yang, Joseph, Yazgan, Zöllner. **"MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations"**. IEEE Intelligent Vehicles Symposium (IV) 2025. DOI 10.1109/IV64158.2025.11097718.*

PDF 위치: `muvo/muvo/MUVO_A_Multimodal_Generative_World_Model_for_Autonomous_Driving_with_Geometric_Representations.pdf`. 본 요약은 5개 섹션을 분담 정독한 후 verbatim quote 위주로 통합. cross-reference 대상: [[muvo_default_config_audit]], [[three_way_comparison]], [[_muvo_comparison]].

## TL;DR (논문 핵심 3줄)

1. MUVO = "MILE 의 fundamental architecture를 reduced complexity로 가져온" 멀티모달 (camera + LiDAR range-view) world model. **BEV 기반 fusion을 피하는** 것이 명시적 differentiator.
2. **가장 큰 RGB recon 품질 boost는 "2D latent state"**. perceptual loss / ViT backbone 둘 다 시도했지만 "little effect" (perceptual은 오히려 더 흐릿).
3. Optimal recipe (paper 결론): **range-view lossless LiDAR + transformer fusion (TR) + 2D latent space**. 3D occupancy 헤드 추가는 RGB/LiDAR 예측에 "minor improvements" 만 줌, 그러나 occupancy 학습 자체는 camera+lidar pre-training으로 큰 도움 받음.

## Paper Recipe ↔ Public Code 매핑 (이전 "Gap" 분석 정정)

> **정정 (2026-05-15, 본 문서 작성 직후)**: 이전 ⚠️ "Paper vs Public Code Gap" 섹션은
> 잘못된 결론이었음. 사용자가 README 하이퍼링크 단서를 짚어준 후 재조사한 결과,
> **2D variant는 실제 public 코드에 존재**. 단지 paper-cited repo가 아니라 **저자 개인
> repo에 있음**.

paper §III-C / §IV-B / §V가 endorse하는 2D latent state 구현이 어디 있는지 정확한 매핑:

### Repo 2개의 분리

| Repo | URL | RSSM | 비고 |
|---|---|---|---|
| **FZI (paper-cited)** | `github.com/fzi-forschungszentrum-informatik/muvo` | **1D만** (`transition.py:RSSM`, `nn.Linear` + `nn.GRUCell`) | main branch가 paper에서 "code available on GitHub"로 명시한 곳. `origin/2D` 브랜치 존재하나 main과 diff는 README 3줄뿐 (transition.py 동일) |
| **Bogdoll (저자 개인)** | `github.com/daniel-bogdoll/MUVO` | **2D 포함** (`transition_td.py:RSSMTD`) | README 모든 하이퍼링크 (weights releases tag/1.0, tag/2.0, config 파일 등) 가 이 repo로 link |

README (양쪽 repo 동일):
> "The main branch includes most of the results presented in the paper. In the 2D branch,
> you can find 2D latent states, perceptual losses, and a new transformer backbone."

즉 FZI main = paper 실험 대부분의 baseline (1D), bogdoll/2D = paper §IV-B의 2D ablation 변형.

### bogdoll/2D 브랜치의 핵심 파일

| 파일 | 핵심 클래스 | paper §III-C 대응 |
|---|---|---|
| `muvo/models/transition_td.py` (300 lines, neuf) | `ConvGRUCell`, **`ConvGRUCellGlo`** | "we replace all fully connected layers with convolutional layers" |
| 동상 | **`RepresentationModelTD`** (`nn.TransformerDecoder` 사용) | "N_ϕ and N_θ are modeled by a transformer decoder" |
| 동상 | **`RSSMTD`** | 2D RSSM 본체. state shape `Ch × (Σj Tj)` |
| `muvo/models/muvo.py` (785 lines, neuf) | mile.py의 2D version. `from transition_td import RSSMTD`, `self.rssm = RSSMTD(...)` 인스턴스화 |
| `muvo/models/decoder.py` (365 lines, neuf) | `ConvDecoder2D`, `ConvDecoder3D`, `StyleDecoder2D`, `StyleDecoder3D` | 2D conv decoder, 3D occupancy decoder |

### `TRANSFORMER_TRANSITION.ENABLED` flag 의미

FZI repo의 `test_base_1d.yml` (`ENABLED: False`) ↔ `test_base_2d.yml` (`ENABLED: True`)의 유일한
실질적 차이. **FZI main은 이 flag를 안 읽지만** (dead in FZI), bogdoll/2D의 muvo.py에선
1D ↔ 2D 분기 trigger로 active할 가능성 높음 (구체적 read site는 별도 확인 필요).

### 알파 _muvo의 위치

알파 _muvo는 1D RSSM 사용 (`models_muvo.py:RSSM`, `nn.Linear` + `nn.GRUCell`). **FZI main과 같은
1D variant**. bogdoll/2D 대비 2D 미포함.

### alpha blur 원인 분석 함의

- 알파 1D vs FZI main 1D = 동일 → "1D vs 2D"는 두 1D 사이 비교에 무의미.
- 그러나 paper §IV-B의 verdict "**The 2D latent state significantly benefits predictions
  for camera images**" 와 "**the 2D latent space itself provides the largest boost in
  performance**" 는 여전히 유효 — 알파에 2D 도입하는 게 paper recipe상 가장 큰 expected boost.
- bogdoll/2D 브랜치의 `RSSMTD` 등이 **참고 가능한 reference 구현** 으로 존재 → 알파 2D 도입의
  구현 비용 크게 감소.

## §I Introduction

### 문제 의식

> "the majority of such world models focus on camera-based inputs [4]–[11] and some that work in lidar-space [12]–[14]. These works neglect typical sensor setups of autonomous vehicles."

기존 AD world model의 3 gap:
1. **단일 모달리티 지배**: 카메라 only [4]-[11] 또는 LiDAR only [12]-[14].
2. **멀티모달 존재하지만 BEV 의존**: "Only two recent works leverage both camera and lidar data [15], [16]. However, they rely on Bird's-Eye-View (BEV) features... an acknowledged bottleneck due to missing height information [15]."
3. **3D occupancy world model 제약**: "rely on visual inputs only [17] or operate in the occupancy space alone [18]."
4. **평가 gap**: "previous works did not evaluate the impact on future predictions."

### Contributions (verbatim)

> - A multimodal world model with geometric representations that does not rely on BEV features as a bottleneck
> - Extensive evaluation of sensor fusion strategies based on the prediction quality of the world model

### 설계 의도 (intro에서 드러난)

- 의도적 "**no BEV**" — 비교 baseline에 BEV-based를 두고 그 대비.
- "**simple model architecture**" — 광범위 fusion sweep을 위한 의식적 단순화.
- Input: camera + LiDAR; Output: action-conditioned future observations.
- **Optional 3D occupancy decoder**.

§I은 후속 finding을 forecast하지 않음 (e.g., "we find that X is the bottleneck" 류 없음).

## §II Related Work

3 subcategory 중 MUVO는 **World Models**에 위치.

### 주요 대비

| 기존 접근 | MUVO 차별점 (quoted) |
|---|---|
| 라벨 기반 (DriveDreamer 등) | "Contrary to these approaches, MUVO does not require labeled training data." |
| 대규모 sequence model (GAIA-1, VISTA) | "MUVO is computationally efficient and does not require hundreds of GPUs for training." |
| LiDAR/3D-occupancy 단독 world model | "MUVO differs from those approaches by leveraging multimodal data." |
| BEV 기반 멀티모달 (BEVWorld, HoloDrive) | "MUVO does not require BEV features." |
| Scene completion (MonoScene 등) | "they focus on data completion rather than world modeling" |
| Forecasting | "not action-conditioned" |

### BEV bottleneck claim (verbatim)

> "In both cases [BEVWorld, HoloDrive], BEV features lack height information and are thus a bottleneck. MUVO does not require BEV features."

MILE [24] 는 이 섹션에서는 "[24]-[26]" 인용 cluster 안에 한 번 등장 (BEV semantic label 학습 예시), **architectural parent로 명시되지 않음**. 그 명시는 §III에서.

## §III World Model

### 아키텍처 origin (verbatim)

> "our experiment setup follows the fundamental architecture of MILE [24], which is much reduced in complexity compared to other approaches."

수정한 점: (a) stereo camera + LiDAR sensor fusion, (b) low-res BEV mask 대신 **raw sensor data prediction**, (c) **3D occupancy 헤드 추가**.

### §III-A Observation Encoder

- Image: `I ∈ ℝ^{3×Hi×Wi}`, **600 × 960 pixels** (Hu et al. [24] 따라감).
- LiDAR: up to **60,000 points**, **2D cylindrical projection**으로 **lossless bijective representation**: `R ∈ ℝ^{4×Hr×Wr}` (4 channels).
- Backbone (pretrained, layers fused): image features `Fc ∈ ℝ^{C×Hc×Wc}`, LiDAR features `FL ∈ ℝ^{C×HL×WL}`.
- Backbone 종류: ResNet18 (default), MobileViT-V2도 §IV에서 평가.

### §III-B Multimodal Fusion

- Token 형식: `tin ∈ ℝ^{Dt×Nt}` (Dt = 토큰 dim, Nt = 토큰 수).
- F의 H×W flatten → `f ∈ ℝ^{C×HW}`.
- **2D sinusoidal positional embedding** `e ∈ ℝ^{C×HW}` (per [21], [78]).
- **Learnable sensor embedding** `s ∈ ℝ^{C×Ns}` (Ns = 센서 수).
- 합성: `t_i(x,y) = f_i(x,y) + e_i(x,y) + s_i`.
- 모든 센서의 토큰을 **concatenate**한 후 **k-layer Transformer encoder** (multi-head self-attention + MLP + layer norm) 통과 → `tnew ∈ ℝ^{C×(Σi Hi·Wi)}`.

### §III-C Transition Model (★ critical)

**Notation:**
- Inputs: `o_{0:t}`, encoded actions `a_{0:t} ∈ ℝ^{T×Da}` (MLP encoded).
- Outputs: stochastic state `s_{0:t} ∈ ℝ^{T×Ds}`, deterministic state `h_{0:t} ∈ ℝ^{T×Dh}`, future predictions.
- GRU [79]: `h_{t+1} = f_θ(h_t, s_t)`.

**Distributions:**
- Posterior: `q(s_t | o_{≤t}, a_{<t}) ∼ N_ϕ(o_t, h_t, a_t)`
- Prior: `p(s_t | h_t, a_{t-1}) ∼ N_θ(h_t, a_{t-1})`
- `N_ϕ, N_θ` 기본은 MLP. 학습 시 q에서 sampling, prediction 시 p에서 sampling.

**1D → 2D 변형 (verbatim, paper의 central architectural contribution):**

> "We utilize the output tokens `tnew` from the sensor fusion in the form of a two-dimensional latent state as the encoded observations `o_t` for the transition model. **Compared to 1D states, we set the stochastic hidden states `s_t`, and deterministic historical states `h_t` to shape `Ch × (Σ_j T_j)`**, where `T_j` is the number of tokens of each output modality."

> "For `f_θ`, we replace all fully connected layers with convolutional layers in order to utilize two-dimensional states."

> "The probability models `N_ϕ` and `N_θ` are modeled by a transformer decoder. Learnable embeddings, which have the same shape as the stochastic hidden states `s_t`, are used as queries, where `h_t, a_t, (o_t)` are concatenated as key-value pairs."

요약: 1D variant는 state가 flat `(D_s,)`, 2D variant는 `(Ch, T_total)` (channel × token 수). GRU의 FC가 Conv로 교체, prior/posterior MLP가 transformer decoder로 교체.

### §III-D Multimodal Decoder

- Input: `(s_t, h_t)` shape `C × Σ_j T_j`.
- 모달리티별 분할 → 각 `C × T_j`를 output shape에 맞게 reshape.
- Camera/LiDAR: `C × H₀ × W₀`로 reshape 후 [80] [81] 스타일 conv upscale → `C_n × H × W`. **2D convolution**.
- 3D occupancy: `C × X × Y × Z`. **3D convolution**.

## §IV Evaluation

### §IV-A Training Setup

**Loss (Eq. 1):**
```
L = Σ_i λ_i ( λ_img L_img^i + λ_pcd (L_p,xyz^i + L_p,r^i + L_pcd^i) + λ_V L_V,scal^i )
```
- `L_img`: **L1** on RGB (multi-scale: factor 1, 2, 4)
- `L_p,xyz`: **L2** on point cloud (Euclidean)
- `L_p,r`: **L1** on range r
- `L_V,scal`: **Scene-Class Affinity Loss (SCAL)** [42] on 192×192×64 voxel grid (0.5m voxels)
- λ 값들은 paper에 명시 없음

**Datasets (CARLA, RL expert agent):**
- D_train: Town01/03/04/06 × 4 weather (Clear Noon, Wet Noon, Hard Rain Noon, Clear Sunset). 10 FPS, 25 runs × 300s per town. **300,000 frames** 총.
- Sensors per frame: RGB (3×600×960), depth (1×600×960), point cloud (≤60,000×3, 64-channel LiDAR), route map (1×64×64), speed, action (2D: acc + steer).
- D^RL_val: same towns/weather, random routes (representation learning).
- D^DS_val: same towns, **different weather**, random routes (domain shift).

**Train params:** sampling 0.2s (5Hz), sequence 12 frames (voxel 실험은 6), batch 16, AdamW, lr 1e-4, weight decay 0.01, ResNet18 baseline backbone.

**Validation split** (paper §IV-A page 4 verbatim):
> "For validation, we used 6/4 frames as given observations, while 6/2 served as ground truth."

해석:
- **§IV-B sensor fusion ablation** (non-voxel, seq=12): validation **RF=6, FH=6** ("6 frames observed, 6 ground truth")
- **§IV-C voxel experiments** (seq=6): validation **RF=4, FH=2** ("4 frames observed, 2 ground truth")

### Paper 실험 cfg ↔ committed yml ↔ alpha 매핑 (2026-05-19 보강)

| 실험 / cfg 소스 | training seq | val RF | val FH | total | 위치 | 비고 |
|---|---|---|---|---|---|---|
| **Paper §IV-B sensor fusion ablation** (Fig 3,4) | 12 | 6 | 6 | 12 | (committed yml 없음) | 페이퍼의 central "2D > 1D" finding이 측정된 cfg |
| **Paper §IV-C voxel experiment** (Fig 5,6) | 6 | 4 | 2 | 6 | `muvo/configs/muvo.yml`과 일치 (RF=4, FH=2) | "to speed up the training" 위해 짧게 함 |
| `muvo/configs/muvo.yml` | — | 4 | 2 | 6 | committed, README "default config" | paper §IV-C voxel cfg와 동일 |
| `muvo/configs/test_base_1d.yml` | — | 6 | 10 | 16 | committed, release된 1D weight (`RV_WOB_TR_1d_Voxel`) 대응 | paper §IV-B cfg(6/6)와 다름. 더 긴 prediction horizon |
| `muvo/configs/test_base_2d.yml` | — | 6 | 10 | 16 | committed, release된 2D weight (`basic_voxel`) 대응 | paper §IV-B cfg와 다름 |
| `muvo/configs/test_mobilevit_2d.yml` | — | 6 | 10 | 16 | committed, release된 `mobilevit` weight 대응 | paper §IV-B cfg와 다름 |
| `muvo/configs/test_base_1d_without_voxel.yml` | — | 6 | 10 | 16 | committed, release된 `RV_WOB_TR_1d_no_Voxel` 대응 | paper §IV-B cfg와 다름 |
| `muvo/configs/debug.yml` | — | 2 | 1 | 3 | 디버그 | — |
| `muvo/configs/one_frame.yml` | — | 1 | 0 | 1 | baseline 단일 frame | — |
| `muvo_2d/muvo/configs/muvo.yml` | — | 2 | 0 | 2 | 2D 브랜치 stub | sequence 2는 world model로서 의미 X. stub |
| `muvo_2d/muvo/configs/predict.yml` | — | 1 | 6 | 7 | prediction.py 평가용 | — |
| `muvo_2d/muvo/configs/debug.yml` | — | 1 | 0 | 1 | 디버그 | — |
| `muvo_2d/muvo/configs/one_frame.yml` | — | 1 | 0 | 1 | baseline | — |
| **alpha 1D `config_muvo.py`** | — | 4 | **4** | 8 | alpha 신규 | paper §IV-B(6/6)도 §IV-C(4/2)도 아님. 두 paper cfg 모두와 불일치 |
| **alpha 2D `config_muvo_2D.py`** | — | 4 | 2 | 6 | alpha 신규 | **paper §IV-C voxel cfg와 정확히 매칭** (단 alpha에 voxel head는 없음) |

**핵심 발견**:
- `muvo.yml` (README "default")는 사실 §IV-C voxel 실험 cfg임. paper의 central "2D > 1D" finding을 만든 §IV-B 실험 cfg(seq=12, 6/6)는 어떤 committed yml에도 없음.
- `test_*.yml` 4종은 release된 pre-trained weight과 매칭되는 cfg이지만 paper §IV-B와 horizon이 다름 (16 vs 12).
- **alpha 2D의 (4, 2) 선택은 voxel paper cfg 모방**이지 sensor fusion ablation cfg 모방이 아님.
- **alpha 1D의 (4, 4) 선택은 어떤 paper/upstream cfg와도 매칭 안 됨** — alpha 자체 디자인.

### §IV-B Sensor Fusion Strategies (★ critical)

**Metrics:** PSNR (RGB), Chamfer Distance (LiDAR), IoU+/IoU− (voxel).

**Encoder 비교:**
- Image: standard (**WOB** = without BEV) vs BEV mapping
- LiDAR: **RV** (range view, lossless) vs **PP** (PointPillars, lossy 2D BEV pseudo-image)

**Fusion 비교:**
- **AVG**: 1D feature averaging
- **FC**: concatenation + fully connected
- **TR**: §III-B의 transformer multi-head self-attention

**A-B-C 네이밍**: A = LiDAR (PP/RV), B = image (BEV/WOB), C = fusion (AVG/FC/TR). 총 2×2×3 = 12 조합 중 paper Fig 3은 8개 평가.

#### Latent space subsection — ★ paper의 central finding

논문 §IV-B 정확한 verdict들:

| 변형 | Verdict (verbatim) |
|---|---|
| **2D latent vs 1D** | "**The 2D latent state significantly benefits predictions for camera images and spatial voxel occupancies, while lidar predictions do not see any benefit.**" (이유: "camera data is much more complex than lidar data") |
| 추가 perceptual loss | "we do not see a strong effect of utilizing a perceptual loss, as it produces visually poorer reconstructions and does not show any significant advantages" |
| ViT backbone | "Using the vision transformer as an encoder does provide advantages for the prediction of camera images but shows no effect on other metrics" |
| **종합** | "**This shows that the 2D latent space itself provides the largest boost in performance, while other changes have little effect.**" |

즉 paper의 ablation 4가지 (2D, perceptual, ViT, ResNet18 baseline) 중 **RGB 품질을 의미있게 끌어올린 단 하나가 2D latent**. 다른 변경 (ViT, perceptual)은 marginal하거나 부정적.

#### Image Prediction (Fig 3 a, b)

- 모든 네트워크가 D^RL_val → D^DS_val에서 성능 하락하나 상대 ordering 유지.
- "**the transformer-based architecture RV-WOB-TR performs on par or better compared to the other combinations**, and **range view-based lidar encodings show clear advantages over PointPillars**."
- "**Methods with an additional BEV mapping of image features perform worse**, and combinations with PointPillars suffer especially."
- TR의 효과는 encoder 의존적: ResNet-18에선 우월, PP + BEV 조합에선 FC < TR < AVG는 아니라 FC 보다 낮고 AVG 보다 높음.

#### Point Cloud Prediction (Fig 3 c, d)

- D^RL_val에서 RV-WOB-TR이 par/better.
- D^DS_val에서는 RV-WOB-TR 성능 하락.
- Range view > PointPillars 동일.
- BEV mapping은 LiDAR task엔 "no clear disadvantage".
- "**transformer-based architectures generally outperform other fusion techniques**."

#### Optimal fusion strategy (paper의 최종 결론 §IV-B)

> "We determine a **transformer-based architecture with a 2D latent space and lossless range-view representations for point clouds** as an optimal fusion strategy, while performance benefits are more pronounced for camera predictions."

### §IV-C 3D Occupancy Prediction

**3 시나리오:**
- **PTF** (Pre-Trained Frozen): camera+lidar로 50,000 step 사전학습 후 freeze, voxel decoder만 학습. → "the pre-trained weights already contain some, however limited, spatial information"
- **PTO** (Pre-Trained Open): 사전학습 가중치 시작, 전체 네트워크 학습.
- **NPT** (No Pre-Training): scratch 학습. "**more conservative strategy**".

**Finding:** PTO가 PTF보다 우수, NPT는 일부 metric에서 후반에 PTO 추월 (precision). IoU−는 D^DS_val에서 NPT가 더 나음 (voxel이 대부분 비어있어서, DS는 더 noisy 예측 → 낮은 IoU−).

**Verdict (verbatim):**

> "As learning 3D occupancy is computationally intensive, **we conclude that pre-training strategies on only camera and lidar data are generally recommendable**, as they both speed up training and show overall superior performance."

**Reverse (occupancy → camera/lidar 성능 개선?, Fig 6):**

> "we observe only **slightly increased performance gains for both modalities** when occupancy prediction is included, with a more pronounced benefit for camera predictions under the D^RL_val setting."

즉 occupancy 추가가 카메라/LiDAR 성능을 약간 (slightly) 개선, 카메라쪽이 더 큼.

## §V Conclusion

### BEV bottleneck verdict

> "Our experiments demonstrate that **lossless lidar representations with a standard transformer-based fusion and an increased 2D latent space are indeed beneficial** in the case of camera-lidar fusion."

### Occupancy verdict

> "**occupancy predictions benefit from more efficient pre-training** with only camera and lidar predictions. In addition, we observed that **occupancy prediction leads to minor improvements for both camera and lidar predictions**."

### Future work

대규모 real-world 데이터 실험, 더 복잡한 sensor setup, computational efficiency 분석.

### Limitations (암시적)

- Simulation-only (CARLA) 학습이라 real-world 일반화 미검증
- 3D occupancy는 "computationally intensive"
- 명시적 limitations 섹션은 없음

## 알파 `_muvo`와의 연결

### Paper recipe ↔ 알파 _muvo 현황 매핑

| Paper recipe 요소 | 알파 _muvo | 비고 |
|---|---|---|
| **Lossless LiDAR range-view** | ✓ (point_cloud_to_range_view 32×1024) | 일치 |
| **Transformer fusion (TR)** | ✓ (SensorFusionTransformer 3-layer) | upstream은 6-layer, 알파는 3. depth 부족 가능성 |
| **2D latent state** | ✗ (1D RSSM) | **paper의 central finding과 어긋남**. FZI repo main도 1D, 알파도 1D, 둘이 같음. 그러나 **bogdoll/2D 브랜치의 `RSSMTD`가 reference 구현으로 존재** → 알파 도입 가능 |
| **Camera + LiDAR + occupancy 동시** | △ RGB ✓, LiDAR loss **off** (weight 0), VOXEL **제거** | upstream default는 RGB + LiDAR + VOXEL 셋 다 켜져있음. 알파는 RGB만 |
| **Multi-scale L1 (factor 1, 2, 4)** | ✓ (rgb_1/2/4 head) | 일치 |
| **L2(xyz) + L1(range) for LiDAR** | code엔 있으나 weight 0으로 비활성 | LIDAR_RE 복원 시 동일 형태 |
| **600×960 input, RGB hashed/centered** | 알파는 600×800 원본 → 320×768 center crop | 데이터셋 origin 다름 (Immanuel vs CARLA dump) |
| **5Hz 샘플링, 시퀀스 12** (§IV-B) 또는 **시퀀스 6** (§IV-C voxel) | 2Hz 효과적 (sample_every_n=2 × frame_step=5 → 4Hz Arrow × 0.5), 알파 1D 시퀀스 8 (RF=4+FH=4), 알파 2D 시퀀스 6 (RF=4+FH=2) | 알파 1D는 paper 어느 cfg와도 불일치. 알파 2D는 paper §IV-C voxel cfg와 매칭. sampling rate는 둘 다 paper보다 sparse |
| **ResNet18 backbone** | ✓ | 일치 |
| **Pretrained backbone** | ✓ | 일치 |
| **Image augmentation (blur/sharpen/ColorJitter)** | ✗ (없음) | paper 본문엔 augmentation 언급 없으나 upstream code엔 default ON |

### Paper가 부정한 후보 (알파 blur 해결 후보 우선순위 재구성)

- **Perceptual loss** — paper IV-B "produces visually poorer reconstructions" → 우선순위 낮춤 또는 제거.
- **ViT backbone** — 카메라엔 marginal하나 다른 metric엔 무효 → 비용 대비 효용 낮음.

### Paper가 가장 강하게 endorse한 것

**2D latent state**. 알파에 도입하려면:
1. `models_muvo.py:RSSM` 재구현 — state shape (D_s,) → (Ch, T_total)
2. GRU의 nn.Linear/nn.GRUCell → conv 기반 (bogdoll/2D `transition_td.py:ConvGRUCellGlo` 참고)
3. Prior/Posterior MLP → transformer decoder (bogdoll/2D `RepresentationModelTD` 참고)
4. RSSM 인터페이스 (h_init, s_init, output 모양) 전반 수정

**비용**: bogdoll/2D `transition_td.py` (300 lines) + `muvo.py` (785 lines) + `decoder.py` (365 lines) 를 reference로 알파 cfg/data interface에 맞춰 포팅. **paper text만으로 재구현보다 비용 작음** (~중간 정도). 우선순위는 LIDAR_RE 복원 / augmentation 추가 다음, 또는 단독으로 가장 큰 expected boost.

## 다음 단계 매핑

paper-aligned alpha blur 해결 후보 (우선순위 갱신):

1. **`WEIGHT_LIDAR_RE=0.1` 복원** — paper recipe의 "LiDAR" 부분을 알파에 복원. 코드 변경 작음.
2. **Image augmentation 도입** — paper 본문엔 없으나 upstream code default ON, 알파엔 부재. 코드 변경 작음.
3. **VOXEL_SEG 대안 보조 head** — paper IV-C는 occupancy가 RGB recon "slightly" 개선. _muvo branch 정책상 3D occupancy 직접 추가는 어색 → LiDAR depth 채널 강화 또는 semantic image 보조 head 등 우회.
4. **Transformer fusion depth 증가** (3 → 6 layer matching upstream) — capacity 격차 해소. 코드 변경 작음.
5. **RSSM 2D state 도입** (paper's central finding, paper §IV-B "largest boost") — **bogdoll/2D 브랜치의 `transition_td.py:RSSMTD` 등 reference 구현 활용** 가능. 알파 cfg/dataset interface에 맞춰 포팅. 비용 중대형이나 reference 코드 있어 reduced.
6. (낮은 우선순위) Perceptual loss — paper가 부정.
7. (낮은 우선순위) ViT backbone — paper가 marginal로 평가.

## 파일/라인 reference

- Paper PDF: `muvo/muvo/MUVO_A_Multimodal_Generative_World_Model_for_Autonomous_Driving_with_Geometric_Representations.pdf`
- Paper text extract: `muvo_paper.txt` (workspace root, 862 lines)
- Section line ranges: §I (22-63), §II (64-170), §III (171-299), §IV (300-592), §V (593-619)
- **FZI repo (paper-cited)** `github.com/fzi-forschungszentrum-informatik/muvo`:
  - main의 1D RSSM: `muvo/muvo/models/transition.py:28-72` (`RSSM` 클래스)
  - `mile.py:286-297` (`RSSM` instantiate)
  - `origin/2D` 브랜치 = main과 README 3줄 차이뿐 (2D 코드 없음)
- **Bogdoll personal repo** `github.com/daniel-bogdoll/MUVO`:
  - `bogdoll/2D` 브랜치 — paper §III-C의 2D variant 실제 구현
  - `muvo/models/transition_td.py`: `ConvGRUCell`, `ConvGRUCellGlo`, `RepresentationModelTD`, `RSSMTD`
  - `muvo/models/muvo.py`: `from transition_td import RSSMTD; self.rssm = RSSMTD(...)`
  - `muvo/models/decoder.py`: `ConvDecoder2D`, `ConvDecoder3D`
- **`TRANSFORMER_TRANSITION.ENABLED` flag**: FZI repo의 `test_base_1d.yml` / `test_base_2d.yml`의 유일한 실질 차이. FZI main은 이 flag를 안 읽음. Bogdoll/2D에선 read site 확인 필요.
