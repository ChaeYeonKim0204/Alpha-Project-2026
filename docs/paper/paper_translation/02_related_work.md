# II. Related Work — 영-한 대역본

- **원문 위치**: `muvo_paper.txt` L64–L170
- **한 줄 요지 (KR)**: World Model 계열(라벨 의존형/자기지도/시퀀스 모델), LiDAR/3D occupancy 기반 모델, BEV 기반 멀티모달 모델(BEVWorld·HoloDrive), Scene Completion, Forecasting 네 갈래의 선행 연구와 MUVO의 차별점(라벨 불필요·BEV 미사용·멀티모달·행동 조건부)을 정리.

---

## A. World Models (L65–L139)

&nbsp;

> World models are generative models that embed observations into latent states, predict future states conditioned on actions, and decode these latent predictions into the observation space [22], [23].

> World model은 관측(observation)을 잠재 상태(latent state)로 임베딩하고, 행동(action)을 조건으로 미래 상태를 예측한 뒤, 그 잠재 예측을 다시 관측 공간으로 디코드하는 생성 모델이다 [22], [23].
- **world model의 3단 구조**: encoder → transition (latent dynamics) → decoder. 잠재 공간에서 미래를 "상상(imagine)"하고 디코딩.
  ```
  o_t ──encode──▶ z_t  ─┐
                          ├──transition(z_t, a_t) ──▶ z_{t+1}
  a_t ──────────────────┘
  z_{t+1} ──decode──▶ o_{t+1} (예측 관측)
  ```

&nbsp;

> Many approaches rely on labels, privileged information, or expert-designed state spaces, limiting their ability to scale.

> 많은 접근은 라벨, 특권 정보(privileged information), 또는 전문가가 설계한 상태 공간(state space)에 의존하기 때문에 확장성(scalability)이 제한된다.

&nbsp;

> A typical category is the prediction of BEV semantic labels based on supervision during training [24]–[26].

> 대표적인 분류로는 학습 중의 지도(supervision)를 기반으로 BEV semantic 라벨을 예측하는 방식 [24]–[26]이 있다.

&nbsp;

> DriveDreamer [27] conditions real-world RGB images on HD maps and labeled 3D bounding boxes.

> DriveDreamer [27]는 실제 RGB 이미지를 HD 지도와 라벨링된 3D bounding box로 조건화한다.

&nbsp;

> Based on a diffusion model [28], future frames and actions are jointly predicted.

> 확산(diffusion) 모델 [28]을 기반으로 미래 프레임과 행동을 동시(jointly)에 예측한다.
- **diffusion 모델**: 노이즈를 점진적으로 제거(denoise)해서 이미지를 생성. forward 과정에서 데이터에 점진적으로 노이즈를 더하고, reverse 과정에서 학습된 네트워크가 단계별로 노이즈를 제거. 고품질·다양성 우수.
  ```
  data x_0 ──+noise── x_1 ──+noise── ... ──+noise── x_T (≈ pure noise)
      ▲              ▲              ▲              │
      └─denoise─────└─denoise─────└─denoise◀──────┘  ← model 학습 방향
  ```

&nbsp;

> The style of predictions is guided by CLIP [29] embeddings, using annotated scenes during training.

> 예측의 스타일은 CLIP [29] 임베딩에 의해 유도되며, 학습 단계에서 라벨링된(annotated) 장면이 사용된다.

&nbsp;

> Contrary to these approaches, MUVO does not require labeled training data.

> 이러한 접근과 달리 MUVO는 라벨링된 학습 데이터를 필요로 하지 않는다.
- **self-supervised (vs supervised) world model**: 라벨(BEV semantic, 3D bbox 등) 없이 raw 입력 자체를 reconstruction target으로 사용. 라벨 수집 비용이 크게 줄고 데이터 확장이 쉬움. MUVO는 이 계열.
  ```
  supervised   : input → model → semantic_label (사람이 만든 라벨로 학습)
  self-supervised: input → model → input' (입력 자체를 재구성, 라벨 불요)
  ```

---

&nbsp;

> There also exist self-supervised world models.

> 자기지도(self-supervised) world model 계열도 존재한다.

&nbsp;

> DreamerV3 is capable of predicting futures in Minecraft [30].

> DreamerV3는 Minecraft에서 미래를 예측할 수 있다 [30].

&nbsp;

> Drive-GAN [31] was trained on real-world data and acts as an action-conditioned neural simulator.

> Drive-GAN [31]은 실제 데이터로 학습되어 행동 조건부(action-conditioned) 신경 시뮬레이터로 동작한다.

&nbsp;

> DayDreamer [32], [33] learns robotic tasks from real-world visual inputs.

> DayDreamer [32], [33]는 실제 시각 입력으로부터 로봇 작업(robotic task)을 학습한다.

&nbsp;

> A world model from Tesla [11], trained on proprietary multi-camera RGB data, was demonstrated to predict future observations and semantic or spatial data based on supervised fine-tuning.

> Tesla가 발표한 world model [11]은 자체 보유한 다중 카메라 RGB 데이터로 학습되었으며, 지도(supervised) fine-tuning을 통해 미래 관측과 semantic·공간(spatial) 데이터를 함께 예측할 수 있음이 시연되었다.

&nbsp;

> Following a recent line of work interpreting world models as a single sequence model [34]–[40], GAIA-1 [10] was trained on proprietary real-world camera data and can be conditioned with both actions and textual inputs.

> World model을 하나의 시퀀스 모델로 해석하는 최근 연구 흐름 [34]–[40]을 따라, GAIA-1 [10]은 자체 보유 실제 카메라 데이터로 학습되었으며 행동과 텍스트 입력 두 가지로 조건화될 수 있다.

&nbsp;

> Vector quantization [41] was used to tokenize the data.

> 데이터를 토큰화(tokenize)하기 위해 벡터 양자화(Vector Quantization) [41]가 사용되었다.
- **Vector Quantization (VQ)**: 연속 잠재 벡터를 코드북(codebook)의 가장 가까운 임베딩으로 양자화해 이산(discrete) 토큰 시퀀스로 변환. NLP의 토큰처럼 다룰 수 있어 Transformer로 모델링 가능. GAIA-1, VQ-VAE 등이 활용.
  ```
  continuous z ──nearest─▶ codebook entry e_k (k ∈ {1..K})
                   │
                   └─▶ token index k (이산 토큰)
  ```

&nbsp;

> Based on a video diffusion decoder, it achieved temporally consistent, high-resolution predictions.

> 영상 확산(diffusion) 디코더를 기반으로 시간적으로 일관된(temporally consistent) 고해상도 예측을 달성하였다.

&nbsp;

> Similarly, VISTA [7] further increased the image resolution.

> 유사하게 VISTA [7]는 이미지 해상도를 한층 더 끌어올렸다.

&nbsp;

> Compared to recent models of these categories, MUVO is computationally efficient and does not require hundreds of GPUs for training.

> 이러한 범주의 최근 모델들과 비교할 때, MUVO는 계산적으로 효율적이며 학습을 위해 수백 개의 GPU를 필요로 하지 않는다.

---

&nbsp;

> In spatial domains, several world models exist for lidar-data [12]–[14] or 3D occupancy grids [17], [18].

> 공간(spatial) 도메인에서는 LiDAR 데이터를 다루는 world model [12]–[14]이나 3D occupancy grid를 다루는 world model [17], [18]이 다수 존재한다.

&nbsp;

> MUVO differs from those approaches by leveraging multimodal data.

> MUVO는 멀티모달(multimodal) 데이터를 활용한다는 점에서 이들과 구별된다.

&nbsp;

> Most similar to our experiment setup, BEVWorld [15] and HoloDrive [16] leverage both camera and lidar data.

> 우리 실험 설정과 가장 유사하게 BEVWorld [15]와 HoloDrive [16]는 카메라와 LiDAR 데이터를 동시에 활용한다.

&nbsp;

> BEVWorld proposed a multi-model encoder that generates a unified BEV representation.

> BEVWorld는 통합된 BEV 표현을 생성하는 멀티모델 인코더를 제안한다.

&nbsp;

> Upsampled voxel features are used to predict camera and lidar data.

> 업샘플링된 voxel 특징을 사용해 카메라·LiDAR 데이터를 예측한다.

&nbsp;

> Differently, HoloDrive has separate models for image and lidar generation and introduces 2D-to-3D and 3D-to-2D structures to improve a joint generation leveraging BEV representations.

> 한편 HoloDrive는 이미지·LiDAR 생성에 각기 다른 모델을 두고, 2D→3D 및 3D→2D 구조를 도입해 BEV 표현을 활용한 결합 생성(joint generation)을 개선한다.

&nbsp;

> In both cases, BEV features lack height information and are thus a bottleneck.

> 그러나 두 경우 모두 BEV 특징은 높이 정보가 부족해 병목(bottleneck)이 된다.
- **BEVWorld vs HoloDrive**: 둘 다 카메라+LiDAR 융합형 world model이지만 구조가 다름. BEVWorld는 통합 BEV 인코더 하나로 압축, HoloDrive는 이미지/LiDAR 모델을 분리하고 2D↔3D 브리지로 연결. 그러나 결국 BEV 표현을 통과하므로 height 정보 손실 공통.
  ```
  BEVWorld : [cam, lidar] ─▶ unified BEV encoder ─▶ predictions
  HoloDrive: cam   ─▶ image model ─┐
                                     ├─[2D↔3D bridge]─▶ joint generation
              lidar ─▶ lidar model ─┘
              (둘 다 BEV 경유 → height 정보 손실)
  ```

&nbsp;

> MUVO does not require BEV features.

> MUVO는 BEV 특징을 요구하지 않는다.

---

## B. Scene Completion (L140–L158)

&nbsp;

> MonoScene [42] was the first work to infer 3D semantic voxels from a single 2D camera image, reconstructing visible areas and hallucinating occluded ones.

> MonoScene [42]은 단일 2D 카메라 이미지로부터 3D semantic voxel을 추론한 최초의 연구로서, 가시 영역(visible)을 복원하고 가려진 영역(occluded)은 환각(hallucinate)으로 추정한다.

&nbsp;

> Transformer-based architectures [43], [44] have improved by using sparse representations or hybrid encoders.

> Transformer 기반 구조 [43], [44]는 희소 표현(sparse representation)이나 하이브리드 인코더를 통해 개선되었다.

&nbsp;

> Symphonies [45] leverages instance queries to model relations between pixels and voxels that belong to the same instance.

> Symphonies [45]는 인스턴스 쿼리(instance query)를 활용해 같은 인스턴스에 속하는 픽셀과 voxel 사이의 관계를 모델링한다.

&nbsp;

> OccDepth [46] uses stereo cameras to derive depth, while many others [47]–[52] utilize surround vision.

> OccDepth [46]는 스테레오 카메라를 사용해 깊이(depth)를 도출하며, 다른 다수 연구 [47]–[52]는 서라운드 비전(surround vision)을 활용한다.

&nbsp;

> Finally, also methods based on lidar point clouds exist [53]–[55].

> 마지막으로 LiDAR 포인트 클라우드에 기반한 방법도 존재한다 [53]–[55].

&nbsp;

> Instead of relying on 3D supervision, recent works [56]–[59] utilize neural rendering to create voxel-based meshes.

> 3D 지도(supervision)에 의존하는 대신 최근 연구 [56]–[59]는 neural rendering을 사용해 voxel 기반 메시(mesh)를 생성한다.
- **neural rendering**: NeRF 계열. 3D 장면을 신경망 파라미터(예: 방향+위치 → 색·밀도 함수)로 표현하고, 미분 가능 렌더링으로 2D 이미지를 합성하면서 3D 구조를 학습. 3D 라벨 없이 multi-view 이미지만으로 voxel 메시를 얻을 수 있음.
  ```
  3D point (x, y, z) + ray dir (θ, φ)
        │
        ▼
  MLP f_θ ─▶ (color RGB, density σ)
        │
        ▼
  volume rendering ─▶ rendered pixel (compare with GT image)
  ```

&nbsp;

> CLONeR [60] uses a single camera and lidar frame to predict occupancy grids based on neural rendering.

> CLONeR [60]은 단일 카메라+LiDAR 프레임을 사용해 neural rendering 기반으로 occupancy grid를 예측한다.

&nbsp;

> S4C [61] predicts semantic occupancy from a single image, relying on supervision.

> S4C [61]는 지도 학습에 의존하여 단일 이미지로부터 semantic occupancy를 예측한다.

&nbsp;

> While these works are related, they focus on data completion rather than world modeling, which can be seen as an additional system component to address occlusions and scene visibility as defined by the sensor setup.

> 이러한 연구는 관련이 있지만 world modeling보다는 데이터 보완(completion)에 초점을 두며, 이는 센서 구성으로 정의되는 가려짐(occlusion)·장면 가시성(scene visibility) 문제를 해결하기 위한 추가 시스템 구성요소로 볼 수 있다.

---

## C. Forecasting (L159–L170)

&nbsp;

> The OpenOcc benchmark [62] was the first to include voxelwise flow information, similar to OpenScene [63].

> OpenOcc 벤치마크 [62]는 OpenScene [63]과 유사하게 voxel별 흐름(flow) 정보를 최초로 포함한 벤치마크였다.

&nbsp;

> An occupancy network by Tesla was demonstrated to predict motion flow vectors for voxels [51].

> Tesla의 occupancy 네트워크는 voxel에 대한 모션 플로우(flow) 벡터를 예측할 수 있음이 시연되었다 [51].

&nbsp;

> Khurana et al. combined lidar data with motion sensors to predict future 3D occupancy [64].

> Khurana 등은 LiDAR 데이터를 모션 센서와 결합해 미래 3D occupancy를 예측하였다 [64].

&nbsp;

> Liu et al. introduced the task of occupancy completion and forecasting [65], whereas others utilize input images to forecast 3D occupancy [19], [66]–[68].

> Liu 등은 occupancy 보완(completion) 및 예측(forecasting) 과제를 새롭게 정의하였고 [65], 다른 연구들은 입력 이미지를 사용해 3D occupancy를 예측한다 [19], [66]–[68].

&nbsp;

> While these works are related, pure forecasting approaches are not action-conditioned and do not take into account how the planned actions of an agent contribute to future predictions.

> 이러한 연구는 관련이 있지만, 순수 forecasting 접근들은 행동 조건부가 아니며, 에이전트가 계획한 행동이 미래 예측에 어떻게 기여하는지를 반영하지 않는다.
- **forecasting vs action-conditioned world model**: forecasting은 과거 관측만 보고 미래를 추정 (action 무관). world model은 행동을 입력으로 받아 "내가 이 행동을 하면" 어떻게 되는지를 예측 → planning에 사용 가능.
  ```
  forecasting             :     past obs   ─▶ future obs (수동적)
  action-conditioned WM   : past obs + action ─▶ future obs (능동적, planning에 직결)
  ```

---

## 핵심 용어/수치 정리표

| 용어/기호 | 영문 정의 | 한국어 의미 | 등장 위치 |
|---|---|---|---|
| World Model | Embed obs → latent → predict future → decode | 관측 임베딩·미래 예측·디코딩 생성 모델 | L65–68 |
| Labeled / privileged supervision | Training that requires labels/HD maps/3D boxes | 라벨·특권 정보 의존 학습 | L69–78 |
| DriveDreamer [27] | Diffusion-based action+frame predictor, CLIP-guided | 확산 기반 행동/프레임 예측 | L72–77 |
| Self-supervised | No labels needed (DreamerV3, Drive-GAN, DayDreamer, GAIA-1) | 자기지도 학습 | L79–90 |
| GAIA-1 [10] | Sequence-model world model with VQ tokenization | 시퀀스 모델, VQ 토큰화 | L88–92 |
| VISTA [7] | Higher-resolution variant | 고해상도 후속 모델 | L123 |
| BEVWorld [15] | Unified BEV encoder predicting cam+LiDAR | 통합 BEV 인코더 | L130–134 |
| HoloDrive [16] | Separate cam/LiDAR models with 2D↔3D bridging | 분리 모델 + 2D↔3D 구조 | L134–137 |
| BEV bottleneck | BEV features lack height info | 높이 정보 부재로 인한 BEV 병목 | L138 |
| MonoScene [42] | Single-image 3D semantic voxel inference | 단일 이미지→3D semantic voxel | L141 |
| Neural rendering completion [56]–[59] | Voxel mesh via neural rendering | neural rendering 기반 voxel 보완 | L150–151 |
| Forecasting (action-free) | Predict future occupancy without action conditioning | 행동 미조건 미래 예측 | L159–170 |
| Khurana et al. [64] | LiDAR + motion → future 3D occupancy | LiDAR+IMU 기반 미래 occupancy | L162–164 |

---

## 알파(alpha26) 코드 관점 메모

- alpha26은 **자기지도 world model** 계열에 가깝다 — 라벨/HD map 미사용, RGB+LiDAR 원본 reconstruction으로 학습. paper의 "no-label" 흐름과 일치.
- alpha는 **VQ 토큰화 미사용** (GAIA-1 계열 아님). 대신 continuous embedding + RSSM 구조 (DreamerV3·MILE 계열).
- alpha는 **BEVWorld/HoloDrive 같은 BEV 사용 멀티모달 모델이 아님** — paper §II의 "BEV bottleneck" 비판과 일치하는 디자인.
- alpha는 **scene completion/forecasting 라인을 다루지 않음** — 즉 sensor 예측(raw RGB + LiDAR range-view)까지만 수행. occupancy는 미구현.
- alpha-2D의 RSSMTD + TransformerDecoder fusion 구조는 paper §II에서 언급한 "sequence model" 흐름과 일부 친화적이지만 VQ 없이 continuous latent.
