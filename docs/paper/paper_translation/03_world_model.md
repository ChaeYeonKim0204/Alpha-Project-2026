# III. World Model — 영-한 대역본

- **원문 위치**: `muvo_paper.txt` L171–L299
- **한 줄 요지 (KR)**: MUVO는 MILE의 단순 구조를 기반으로, 카메라+LiDAR(range view)를 Transformer로 융합하고 RSSM 기반 transition으로 미래 잠재상태를 예측한 뒤 RGB·LiDAR·3D occupancy 디코더로 복원하는 멀티모달 world model이다. 잠재상태는 1D 벡터가 아닌 **2D 토큰 형태**로 설계됐다.

---

## 개요 (L171–L205)

&nbsp;

> In this work, we evaluate sensor fusion strategies for world models for autonomous driving.

> 본 연구에서는 자율주행을 위한 world model의 sensor fusion 전략을 평가한다.

&nbsp;

> To isolate the effect of fusion strategies, our experiment setup follows the fundamental architecture of MILE [24], which is much reduced in complexity compared to other approaches.

> 융합 전략의 효과를 독립적으로 측정하기 위해, 우리의 실험 구조는 다른 접근들에 비해 복잡도가 크게 낮은 MILE [24]의 기본 구조를 따른다.
- **MILE (Hu et al., 2022)**: CARLA용 카메라-only world model. ResNet18 + BEV backbone + RSSM($`h`$, $`s`$) + 멀티헤드 디코더(BEV seg, RGB recon 등). 본 논문은 MILE 구조를 단순화한 채로 LiDAR/transformer fusion을 추가한 비교 기준.
  ```
  RGB → ResNet18 → BEV lift → BEV backbone → embedding
      → RSSM(h, s) → decoders (BEV seg, RGB)
  ```

&nbsp;

> As shown in Fig. 2, we introduce changes in the architecture to allow for sensor fusion of a typical sensor setup of autonomous vehicles, comprising stereo cameras and lidar [74], [75], and predict raw sensor data rather than low-resolution BEV masks based on camera data.

> Fig. 2에서 보이듯, 우리는 스테레오 카메라와 LiDAR로 구성되는 자율주행 차량의 전형적인 센서 구성 [74], [75]을 sensor fusion으로 다룰 수 있도록 구조를 변경하였고, 카메라 기반의 저해상도 BEV 마스크가 아닌 원본 센서 데이터를 직접 예측하도록 한다.

&nbsp;

> In addition, we introduce an additional head to predict 3D occupancy to analyze the effects of introducing a spatial loss in a sensor-independent space.

> 또한 센서에 비의존적인(sensor-independent) 공간에서 spatial loss를 도입했을 때의 효과를 분석하기 위해 3D occupancy를 예측하는 추가 헤드를 둔다.

&nbsp;

> This Multimodal World Model with Geometric Voxel Representations (MUVO) as our setup utilizes image and lidar sensor data from a typical autonomous vehicle setup, as shown in Fig. 1.

> 이 Multimodal World Model with Geometric Voxel Representations(MUVO) 구조는 Fig. 1과 같이 전형적인 자율주행 차량의 이미지·LiDAR 데이터를 입력으로 사용한다.

&nbsp;

> First, we process, encode, and fuse RGB camera data and lidar point clouds.

> 첫째, RGB 카메라 데이터와 LiDAR 포인트 클라우드를 처리·인코딩·융합한다.

&nbsp;

> Second, we feed the latent representations of the sensor data to a transition model to derive a probabilistic model of the current state, followed by sampling, while concurrently predicting the probabilistic model of future states and sampling from it.

> 둘째, 센서 데이터의 잠재 표현(latent representation)을 transition model에 입력해 현재 상태의 확률 모델을 도출하고 그로부터 샘플링하며, 동시에 미래 상태의 확률 모델도 예측하고 거기서 샘플링한다.

&nbsp;

> Lastly, we decode both current and future states from the probabilistic models, forecasting raw RGB images, point clouds, and 3D occupancy.

> 마지막으로, 확률 모델로부터 현재·미래 상태를 모두 디코드하여 원본 RGB 이미지, 포인트 클라우드, 3D occupancy를 예측한다.

---

## A. Observation Encoder (L206–L218)

&nbsp;

> We input RGB images from a front camera and point clouds from a top-mounted lidar.

> 입력은 전방 카메라의 RGB 이미지와 상단 장착 LiDAR의 포인트 클라우드이다.

&nbsp;

> The 3D point cloud, comprising up to 60,000 points, is projected into a 2D cylindrical projection for lossless bijective representations.

> 최대 60,000개의 점으로 구성되는 3D 포인트 클라우드는 손실 없는 일대일 표현(lossless bijective)을 위해 2D 원통 투영(cylindrical projection)으로 변환된다.
- **cylindrical / range-view projection**: LiDAR 점 $`(x, y, z)`$ 를 구면좌표 $`(\text{yaw}, \text{pitch}, r)`$ 로 변환 후 2D 이미지 격자에 빈닝. 각 픽셀은 $`(x, y, z, r)`$ 4채널. 이미지처럼 다룰 수 있어 2D CNN/transformer를 그대로 적용 가능.
  ```
  point (x, y, z)
       │
       ▼
  yaw = atan2(y, x), pitch = asin(z / r), r = √(x²+y²+z²)
       │
       ▼
  pixel (u, v) on (H_r × W_r) grid → 4 channels: (x, y, z, r)
  ```

&nbsp;

> For images $`ℐ \in ℝ^{3 \times H_i \times W_i}`$, we follow Hu et al [24] and use an input size of $`600 \times 960`$ pixels.

> 이미지 $`ℐ \in ℝ^{3 \times H_i \times W_i}`$ 에 대해서는 Hu 등 [24]을 따라 $`600 \times 960`$ 픽셀 입력 크기를 사용한다.

&nbsp;

> For images $`ℐ`$ and point clouds $`ℛ \in ℝ^{4 \times H_r \times W_r}`$ in range view representation, we utilize a pre-trained backbone for feature extraction.

> 이미지 $`ℐ`$ 와 range view 표현의 포인트 클라우드 $`ℛ \in ℝ^{4 \times H_r \times W_r}`$ 에 대해서는 사전 학습된 백본을 사용해 특징을 추출한다.

&nbsp;

> We derive feature maps from different model layers similar to [24] and fuse them, culminating in image features $`ℱ_c \in ℝ^{C \times H_c \times W_c}`$ and point cloud features $`ℱ_L \in ℝ^{C \times H_L \times W_L}`$.

> [24]와 유사하게 백본의 서로 다른 레이어에서 feature map을 추출해 융합하고, 최종적으로 이미지 특징 $`ℱ_c \in ℝ^{C \times H_c \times W_c}`$ 와 포인트 클라우드 특징 $`ℱ_L \in ℝ^{C \times H_L \times W_L}`$ 를 얻는다.

---

## B. Multimodal Fusion (L219–L242)

&nbsp;

> Similar to [21], [76], [77], we employ the self-attention mechanism of a Transformer [78] to fuse features of different sensors.

> [21], [76], [77]과 유사하게, 서로 다른 센서의 특징을 융합하기 위해 Transformer [78]의 self-attention 메커니즘을 사용한다.

&nbsp;

> It takes a sequence of tokens as input, where each token is a $`D_t`$-dimensional feature vector, so the input sequence is $`\mathrm{t}_{\text{in}} \in ℝ^{D_t \times N_t}`$, with $`D_t`$ representing the feature dimension of each token and $`N_t`$ the number of tokens in the sequence.

> Transformer는 토큰 시퀀스를 입력으로 받으며, 각 토큰은 $`D_t`$ 차원의 특징 벡터, 즉 입력 시퀀스는 $`\mathrm{t}_{\text{in}} \in ℝ^{D_t \times N_t}`$ 이고 $`D_t`$ 는 토큰의 특징 차원, $`N_t`$ 는 시퀀스의 토큰 수이다.

&nbsp;

> We flatten the $`H`$ and $`W`$ dimensions of the features $`ℱ`$ obtained from the encoder described in Sec. III-A, resulting in tokens $`\mathrm{f} \in ℝ^{C \times HW}`$.

> §III-A에서 얻은 특징 $`ℱ`$ 의 $`H`$, $`W`$ 차원을 평탄화(flatten)하여 토큰 $`\mathrm{f} \in ℝ^{C \times HW}`$ 을 만든다.

&nbsp;

> Subsequently, we incorporate the 2D sinusoidal positional embedding [21], [78] $`\mathrm{e} \in ℝ^{C \times HW}`$ into each token to introduce spatial inductive biases.

> 이어서 공간 inductive bias를 부여하기 위해 각 토큰에 2D 사인파(sinusoidal) 위치 임베딩 [21], [78] $`\mathrm{e} \in ℝ^{C \times HW}`$ 을 더한다.
- **2D sinusoidal positional embedding**: transformer는 토큰 순서를 모르므로 위치 정보를 따로 더해 줘야 함. 1D는 $`\sin / \cos(\text{pos}/T^{2i/d})`$, 2D는 ($`x`$ 좌표용 채널의 절반, $`y`$ 좌표용 절반)을 각각 $`\sin / \cos`$ 로 생성해 합치는 방식.
  ```
  for position (x, y):
    PE[2i  ] = sin(x / 10000^{2i/d})    ─┐ x 방향
    PE[2i+1] = cos(x / 10000^{2i/d})    ─┘
    PE[d/2+2i  ] = sin(y / 10000^{2i/d}) ─┐ y 방향
    PE[d/2+2i+1] = cos(y / 10000^{2i/d}) ─┘
  ```

&nbsp;

> The learnable sensor embeddings $`\mathrm{s} \in ℝ^{C \times N_s}`$ are added, introducing a sensor category, where $`N_s`$ is the number of sensors.

> 학습 가능한 센서 임베딩 $`\mathrm{s} \in ℝ^{C \times N_s}`$ 를 더해 센서 종류 정보를 부여하며, $`N_s`$ 는 센서 개수이다.

&nbsp;

> The resulting tokens $`\mathrm{t} \in ℝ^{C \times HW}`$ are obtained, with each token $`\mathrm{t}_i(x, y) = \mathrm{f}_i(x, y) + \mathrm{e}_i(x, y) + \mathrm{s}_i`$, where $`i`$ indicates the $`i`$-th sensor, and $`(x, y)`$ denotes the coordinate index of that token within the sensor feature.

> 최종 토큰 $`\mathrm{t} \in ℝ^{C \times HW}`$ 은 $`\mathrm{t}_i(x, y) = \mathrm{f}_i(x, y) + \mathrm{e}_i(x, y) + \mathrm{s}_i`$ 로 정의되며, $`i`$ 는 $`i`$-번째 센서, $`(x, y)`$ 는 해당 센서 특징 내 토큰의 좌표 인덱스이다.

&nbsp;

> These tokens from all sensors are concatenated and fed into a Transformer encoder comprising $`k`$ layers, each consisting of multi-head self-attention, Multilayer Perceptrons (MLP), and layer normalizations, resulting in new tokens $`\mathrm{t}_{\text{new}} \in ℝ^{C \times (\sum_i H_i W_i)}`$.

> 모든 센서의 토큰을 이어 붙여, 다중 헤드 self-attention, MLP, layer norm으로 구성된 $`k`$ 개의 층을 가진 Transformer 인코더에 입력해 새 토큰 $`\mathrm{t}_{\text{new}} \in ℝ^{C \times (\sum_i H_i W_i)}`$ 를 얻는다.

---

## C. Transition Model (L243–L277)

&nbsp;

> The input consists of fused observation features $`\mathrm{o}_{0:t}`$ and encoded actions $`\mathrm{a}_{0:t} \in ℝ^{T \times D_a}`$, based on a simple MLP, assuming access to a policy or motion planner.

> 입력은 융합된 관측 특징 $`\mathrm{o}_{0:t}`$ 와, 간단한 MLP로 인코딩된 행동 $`\mathrm{a}_{0:t} \in ℝ^{T \times D_a}`$ 로 구성되며 policy/motion planner 출력이 주어진다고 가정한다.

&nbsp;

> The output includes stochastic hidden states $`\mathrm{s}_{0:t} \in ℝ^{T \times D_s}`$ and deterministic historical states $`\mathrm{h}_{0:t} \in ℝ^{T \times D_h}`$, predictions for future states $`\mathrm{s}_{t:t+n}`$, and $`\mathrm{h}_{t:t+n}`$, $`T`$ represents the number of frames, also referred to as the sequence length, and $`D_a, D_s, D_h`$ are the dimensions of each vector respectively.

> 출력은 확률적(stochastic) 은닉 상태 $`\mathrm{s}_{0:t} \in ℝ^{T \times D_s}`$ 와 결정적(deterministic) 과거 상태 $`\mathrm{h}_{0:t} \in ℝ^{T \times D_h}`$, 그리고 미래 상태 예측 $`\mathrm{s}_{t:t+n}`$, $`\mathrm{h}_{t:t+n}`$ 을 포함하며, $`T`$ 는 프레임 수(시퀀스 길이), $`D_a`$, $`D_s`$, $`D_h`$ 는 각 벡터의 차원이다.

&nbsp;

> The deterministic historical variable $`\mathrm{h}_{t+1} = f_{\theta}(\mathrm{h}_t, \mathrm{s}_t)`$ is modelled by a Gated Recurrent Unit (GRU) [79] $`f_{\theta}`$, enabling the model to remember past states.

> 결정적 과거 변수 $`\mathrm{h}_{t+1} = f_{\theta}(\mathrm{h}_t, \mathrm{s}_t)`$ 는 Gated Recurrent Unit(GRU) [79] $`f_{\theta}`$ 로 모델링되어 과거 상태를 기억할 수 있게 한다.
- **GRU (Gated Recurrent Unit)**: LSTM의 단순화. update gate $`z_t`$ 와 reset gate $`r_t`$ 로 과거를 얼마나 기억할지, 새 입력으로 얼마나 갱신할지를 학습. RSSM의 deterministic transition을 구현하는 흔한 선택.
  ```
  z_t  = σ(W_z · [h_{t-1}, x_t])              # update gate
  r_t  = σ(W_r · [h_{t-1}, x_t])              # reset gate
  ĥ_t  = tanh(W · [r_t ⊙ h_{t-1}, x_t])       # candidate
  h_t  = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ĥ_t      # 새 hidden
  ```

&nbsp;

> The posterior hidden state probability distribution is given by $`q(\mathrm{s}_t \mid o_{\le\,t},\, \mathrm{a}_{<\,t}) \sim 𝒩_{\phi}(\mathrm{o}_t, \mathrm{h}_t, \mathrm{a}_t)`$, while the prior hidden state probability distribution, without the input of observed feature $`\mathrm{o}_t`$, is given by $`p(\mathrm{s}_t \mid \mathrm{h}_t, \mathrm{a}_{t-1}) \sim 𝒩_{\theta}(\mathrm{h}_t, \mathrm{a}_{t-1})`$.

> 후행(posterior) 은닉 상태 확률 분포는 $`q(\mathrm{s}_t \mid o_{\le\,t},\, \mathrm{a}_{<\,t}) \sim 𝒩_{\phi}(\mathrm{o}_t, \mathrm{h}_t, \mathrm{a}_t)`$ 이고, 관측 특징 $`\mathrm{o}_t`$ 입력이 없는 사전(prior) 은닉 상태 분포는 $`p(\mathrm{s}_t \mid \mathrm{h}_t, \mathrm{a}_{t-1}) \sim 𝒩_{\theta}(\mathrm{h}_t, \mathrm{a}_{t-1})`$ 로 주어진다.
- **prior vs posterior in RSSM**: posterior는 "관측을 봤을 때" 무엇이 그럴듯한가, prior는 "관측 없이도" 모델이 예측하는 분포. 학습 단계에서 $`\mathrm{KL}(\text{posterior} \,\Vert\, \text{prior})`$ 을 최소화하면 prior가 posterior를 따라가게 되어, 미래에 관측 없이도 prior로 잘 imagine 가능.
  ```
  학습:  posterior(o_t 사용) ──┬──KL──▶ prior(o_t 미사용)
                                │
                                └─ posterior에서 s_t 샘플 → 디코딩 (recon loss)
  추론(imagine): prior에서 ŝ_t 샘플 → 미래 관측 ô_t 생성 (관측 불요)
  ```

&nbsp;

> Here, $`𝒩_{\phi}`$ and $`𝒩_{\theta}`$ are probability models modelled by a MLP.

> 여기서 $`𝒩_{\phi}`$ 와 $`𝒩_{\theta}`$ 는 MLP로 모델링되는 확률 모델이다.

&nbsp;

> Given observations, $`\mathrm{s}_t`$ is sampled from the posterior distribution $`q`$.

> 관측이 주어진 경우 $`\mathrm{s}_t`$ 는 posterior $`q`$ 에서 샘플링된다.

&nbsp;

> In the absence of observations, i.e., during prediction, $`\hat{\mathrm{s}}_t`$ is sampled from the prior distribution $`p`$.

> 관측이 없을 때, 즉 예측 단계에서는 $`\hat{\mathrm{s}}_t`$ 를 prior $`p`$ 에서 샘플링한다.

---

&nbsp;

> We utilize the output tokens $`\mathrm{t}_{\text{new}}`$ from the sensor fusion in the form of a two-dimensional latent state as the encoded observations $`\mathrm{o}_t`$ for the transition model.

> sensor fusion의 출력 토큰 $`\mathrm{t}_{\text{new}}`$ 를 **2D 잠재 상태(two-dimensional latent state)** 형태로 transition model의 인코딩된 관측 $`\mathrm{o}_t`$ 로 사용한다.
- **1D latent vs 2D latent state**: 기존 RSSM은 잠재 상태를 평탄한 벡터 $`(B, S, D)`$ 로 둠. 본 논문은 공간 구조를 보존하기 위해 토큰 차원을 살린 $`(B, S, C, \sum_j T_j)`$ 형태로 확장. paper §IV-B의 headline ("2D latent significantly benefits camera predictions")의 근거 구조.
  ```
  1D state : (B, S, D)              flat vector (D = 256~512)
  2D state : (B, S, C, N_token)    C × token grid (예: 256 × 369 in alpha-2D)
                                    여기서 N_token = Σ_j T_j (모달리티별 토큰 합)
  ```

&nbsp;

> Compared to 1D states, we set the stochastic hidden states $`\mathrm{s}_t`$, and deterministic historical states $`\mathrm{h}_t`$ to shape $`C_h \times (\sum_j T_j)`$, where $`T_j`$ is the number of tokens of each output modality.

> 1D 상태와 비교해, 확률 은닉 상태 $`\mathrm{s}_t`$ 와 결정 과거 상태 $`\mathrm{h}_t`$ 를 $`C_h \times (\sum_j T_j)`$ 형태로 설정한다. 여기서 $`T_j`$ 는 각 출력 모달리티의 토큰 수이다.

&nbsp;

> For $`f_{\theta}`$, we replace all fully connected layers with convolutional layers in order to utilize two-dimensional states.

> 2D 상태를 활용하기 위해 $`f_{\theta}`$ 의 모든 fully-connected 레이어를 convolutional 레이어로 교체한다.
- **FC → Conv 교체 (ConvGRU)**: 평탄 GRU는 FC layer 사용 → 토큰 간 모든 위치를 일률적으로 섞음. ConvGRU는 Conv1d로 인접 토큰 사이의 weight 공유 + 작은 receptive field → 토큰 grid의 공간 구조를 유지하며 빠르고 파라미터 효율적.
  ```
  flat GRU :  h = GRU(s, h)               # FC(W: D × D)
  ConvGRU  :  h = ConvGRU(s, h)           # Conv1d on token dim (N=369)
                  ▲                       # 각 토큰 위치별로 동일 weight 공유
                  N_token 축
  ```

&nbsp;

> The probability models $`𝒩_{\phi}`$ and $`𝒩_{\theta}`$ are modeled by a transformer decoder.

> 확률 모델 $`𝒩_{\phi}`$ 와 $`𝒩_{\theta}`$ 는 Transformer 디코더로 모델링한다.

&nbsp;

> Learnable embeddings, which have the same shape as the stochastic hidden states $`\mathrm{s}_t`$, are used as queries, where $`\mathrm{h}_t`$, $`\mathrm{a}_t`$, ($`\mathrm{o}_t`$) are concatenated as key-value pairs.

> $`\mathrm{s}_t`$ 와 동일한 형태의 학습 가능 임베딩을 query로 사용하고, $`\mathrm{h}_t`$, $`\mathrm{a}_t`$, ($`\mathrm{o}_t`$)를 이어 붙여 key-value 쌍으로 제공한다.

&nbsp;

> Then, we query the state information in these through the attention mechanism to obtain stochastic hidden state tokens.

> Attention 메커니즘을 통해 이들로부터 상태 정보를 질의하여 확률 은닉 상태 토큰을 얻는다.
- **TransformerDecoder as $`𝒩_{\phi}`$ / $`𝒩_{\theta}`$**: prior/posterior 분포를 MLP가 아닌 cross-attention으로 모델링. query = 학습 가능한 토큰 임베딩, key/value = ($`h`$, $`a`$, $`o`$)를 concat. 결과 토큰을 linear로 $`\mu, \sigma`$ 로 보냄. 토큰별로 서로 다른 정보를 attend할 수 있어 표현력 ↑.
  ```
  Q = learnable embedding (N_token, C)        ← 출력 형태와 동일
  KV = concat([h_tokens, embedding_tokens, latent_action])
       │
       ▼
  TransformerDecoderLayer × L (cross-attention)
       │
       ▼
  Linear → (μ, σ) for each token  →  s_t ~ N(μ, σ²)
  ```

---

## D. Multimodal Decoder (L278–L299)

&nbsp;

> We decode into camera and lidar data and introduce an optional head for 3D occupancy.

> 카메라·LiDAR 데이터로 디코드하며, 3D occupancy를 위한 선택적(optional) 헤드를 추가로 둔다.

&nbsp;

> The input is a latent dynamic state $`(\mathrm{s}_t, \mathrm{h}_t)`$ with shape $`C \times \sum_j T_j`$ which is provided by the transition model.

> 입력은 transition model이 제공하는 형태 $`C \times \sum_j T_j`$ 의 잠재 동역학 상태 $`(\mathrm{s}_t, \mathrm{h}_t)`$ 이다.

&nbsp;

> We divide those tokens based on modalities and reshape each modality's tokens $`C \times T_j`$ to fit the output shape.

> 토큰을 모달리티별로 분할한 뒤, 각 모달리티 토큰 $`C \times T_j`$ 를 출력 형태에 맞도록 reshape한다.

&nbsp;

> The occupancy decoder is of shape $`C \times X \times Y \times Z`$.

> Occupancy 디코더의 형태는 $`C \times X \times Y \times Z`$ 이다.

&nbsp;

> For camera and lidar data, first, the input is reshaped to $`C \times H_0 \times W_0`$, where $`H_0`$ and $`W_0`$ are determined by the final output resolution $`H \times W`$.

> 카메라·LiDAR의 경우, 입력을 먼저 $`C \times H_0 \times W_0`$ 로 reshape하며, $`H_0`$, $`W_0`$ 는 최종 출력 해상도 $`H \times W`$ 에 의해 결정된다.

&nbsp;

> Subsequently, we perform upscaling with convolutional networks similar to [80], [81] to produce a feature map of size $`C_n \times H \times W`$.

> 이어서 [80], [81]과 유사한 convolutional 네트워크로 업스케일링해 $`C_n \times H \times W`$ 크기의 feature map을 만든다.

&nbsp;

> For camera and lidar, we utilize two-dimensional convolutions, while we employ three-dimensional convolutions for voxels.

> 카메라·LiDAR에는 2D convolution을, voxel에는 3D convolution을 사용한다.

---

## Fig. 2 캡션 (개요 그림)

**MUVO Overview** — Raw camera images and lidar point clouds are processed and fused. The resulting latent representations are fed into our transition model. Conditioned on actions, future states are predicted. Finally, future states are decoded into 3D occupancy grids, raw point clouds, and raw images.

> MUVO 개요 — 원본 카메라 이미지와 LiDAR 포인트 클라우드를 처리·융합한다. 그 결과의 잠재 표현을 transition model에 입력하고, 행동을 조건으로 미래 상태를 예측한다. 마지막으로 미래 상태를 디코드해 3D occupancy grid, 원본 포인트 클라우드, 원본 이미지를 생성한다.

## Fig. 3 캡션 (Sensor Fusion ablation)

> Sensor Fusion 실험 — $`𝒟_{\text{val}}^{\text{RL}}`$ 은 표현 학습 능력을, $`𝒟_{\text{val}}^{\text{DS}}`$ 은 강건성을 평가한다. Fusion 전략으로 feature averaging(AVG) [69], feature concatenation(FC) [32], transformer 기반(TR) [21]을 비교. LiDAR 인코딩은 PointPillars(PP) [70]와 ResNet 위 range view(RR) [71], 카메라는 BEV 미사용(WOB) vs BEV 매핑(BEV) [73]을 비교한다.

## Fig. 4 캡션 (2D Latent Space)

> 2D Latent Space — 1D 기준선을 여러 2D 잠재 공간들과 비교하며, 동시에 vision transformer 백본 및 perceptual loss(PL) 항의 영향을 함께 조사한다. 백본은 ResNet18(RN)과 MobileVit-V2(VIT)를 비교한다.

---

## 핵심 용어/수치 정리표

| 용어/기호 | 영문 정의 | 한국어 의미 | 등장 위치 |
|---|---|---|---|
| MILE [24] | Reduced-complexity baseline architecture | 본 모델의 기준 단순 구조 | L175 |
| Front camera + top-mounted lidar | Sensor setup | 전방 카메라+상단 LiDAR | L207–L208 |
| 60,000 points | LiDAR point count | 최대 LiDAR 포인트 수 | L209 |
| Cylindrical projection / Range view | 2D lossless projection of point cloud | 손실 없는 원통 투영 표현 | L209–L210 |
| $`600 \times 960`$ | Image input size | 이미지 입력 해상도 | L211–L212 |
| $`ℱ_c`$, $`ℱ_L`$ | Image and LiDAR feature maps | 카메라·LiDAR 특징 맵 | L217–L218 |
| 2D sinusoidal PE | 2D positional embedding | 2D 위치 임베딩 | L229 |
| Sensor embedding $`s`$ | Learnable per-sensor embedding | 학습 가능한 센서 종류 임베딩 | L231 |
| Transformer encoder, $`k`$ layers | Token fusion module | $`k`$ 층 fusion 모듈 | L237–L240 |
| $`\mathrm{h}_t`$, $`\mathrm{s}_t`$ | Deterministic / stochastic state | 결정·확률 잠재 변수 | L247–L248 |
| $`D_a, D_s, D_h`$ | Action/state/hidden dims | 각 벡터 차원 | L250–L251 |
| GRU $`f_{\theta}`$ | Recurrent transition for $`\mathrm{h}_t`$ | $`\mathrm{h}_t`$ 갱신 GRU | L252–L253 |
| Posterior $`q(\mathrm{s}_t \mid o_{\le\,t},\, \mathrm{a}_{<\,t})`$ | Posterior latent dist (with obs) | 관측 조건 후행 분포 | L255 |
| Prior $`p(\mathrm{s}_t \mid \mathrm{h}_t, \mathrm{a}_{t-1})`$ | Prior latent dist (no obs) | 무관측 사전 분포 | L257 |
| 2D latent state $`C_h \times \sum_j T_j`$ | Token-shaped state (paper's headline change) | 토큰 형태 상태 (논문 핵심) | L263–L268 |
| Conv-replacing-FC in $`f_{\theta}`$ | GRU FC → Conv | 2D 상태 위해 FC→Conv | L269–L270 |
| Transformer decoder $`𝒩_{\phi}`$, $`𝒩_{\theta}`$ | Prior/posterior queries via cross-attn | prior/posterior cross-attention 모델 | L271–L277 |
| $`C \times X \times Y \times Z`$ occupancy head | 3D voxel decoder shape | voxel decoder 형태 | L292 |
| 2D conv for cam/lidar, 3D conv for voxel | Decoder ops | 모달리티별 convolution 차원 | L297–L299 |

---

## 알파(alpha26) 코드 관점 메모

- **alpha-1D (`models_muvo.py`)**: paper의 1D baseline에 해당. RSSM은 $`(B, S, D)`$ 단일 벡터 상태. GRUCell + MLP `RepresentationModel`.
- **alpha-2D (`models_muvo_2D.py`)**: paper §III.C의 2D latent state 구조에 대응. `RSSMTD` + `ConvGRUCellGlo`(FC→Conv 교체), `RepresentationModelTD`(Transformer decoder, 학습 가능 query). 토큰 구성 = 이미지(240) + LiDAR(128) + policy(1) = 369. paper의 ($`C \times \sum_j T_j`$) 구조와 동일하나 occupancy/BEV 토큰은 alpha에서는 없음.
- **Observation Encoder**: alpha는 timm ResNet18 백본 + FPN, paper와 동일 계열. 이미지 입력은 alpha $`320 \times 768`$ (paper는 $`600 \times 960`$). LiDAR range-view는 alpha $`32 \times 1024`$.
- **Multimodal Fusion**: alpha는 `SensorFusionTransformer` 3층 (paper와 동등한 self-attention 토큰 융합). 2D sinusoidal PE + type embedding 동일.
- **Multimodal Decoder**: alpha는 카메라·LiDAR만 디코드 (paper의 occupancy 헤드 없음). alpha-2D `TokenConvDecoder2D`는 paper의 "토큰을 모달리티별 분할 후 reshape → ConvT" 구조와 일치.
- **Paper 식 ($`h`$, $`s`$) 차원 비교**: paper는 $`D_h / D_s`$ 명시하지 않으나 §IV에서 1D vs 2D 비교. alpha-1D: $`H_{\text{dim}} = 512`$, $`Z_{\text{dim}} = 256`$; alpha-2D: 토큰당 256 차원.
- **Gap**: paper의 occupancy 디코더($`C \times X \times Y \times Z`$, 3D conv) → alpha에는 미구현. improvement plan에서 P2 후보.
