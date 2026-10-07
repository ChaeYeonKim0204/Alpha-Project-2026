# MUVO Paper §IV Evaluation — 문장별 EN→KR 번역

**원문:** `/home/carol/chaeyeon-kim/muvo/MUVO_A_Multimodal_Generative_World_Model_for_Autonomous_Driving_with_Geometric_Representations.pdf`
**원문 추출본:** `/home/carol/chaeyeon-kim/muvo_paper.txt` (lines 300-588)
**작성일:** 2026-05-19

각 문장 = "**EN 원문**" 다음 줄에 "→ KR 번역". 의역 최소화, 페이퍼 용어 그대로 유지.
맨 아래 §V 비교 표 (Fig 3 / Fig 4 / Fig 5 / Fig 6).

---

## §IV. EVALUATION (도입)

> "For the evaluation, we first present our utilized training setup in Sec. IV-A, followed by the evaluation of sensor fusion strategies in Sec. IV-B."

→ 평가를 위해 먼저 §IV-A에서 사용된 학습 설정을 제시하고, 이어 §IV-B에서 센서 융합 전략의 평가를 다룬다.

> "Fig. 3 shows the impact of different fusion strategies, and Fig. 4 demonstrates the difference of differently sized latent spaces."

→ Fig 3은 서로 다른 융합 전략의 영향을, Fig 4는 서로 다른 크기의 잠재 공간(latent space)의 차이를 보여준다.

> "Finally, we examine the effects of the optional 3D occupancy prediction in Sec. IV-C."

→ 마지막으로 §IV-C에서 선택적(optional) 3D 점유(occupancy) 예측의 효과를 검토한다.

> "Fig. 5 shows the relation between camera-lidar based pre-training and 3D occupancy prediction, and Fig. 6 shows the reverse impact of predicting occupancy on the quality of sensor predictions."

→ Fig 5는 카메라-LiDAR 기반 사전학습(pre-training)과 3D 점유 예측의 관계를, Fig 6은 그 역방향—점유 예측이 센서 예측 품질에 미치는 영향—을 보여준다.

---

## §IV-A Training Setup

### Training Losses

> "For our experiments, we follow a self-supervised training approach and do not require labels at any point."

→ 본 실험은 self-supervised 학습 방식을 따르며 어느 시점에서도 라벨을 요구하지 않는다.

> "For each modality, we downsample multiple times with ratios of 1, 2, and 4."

→ 각 모달리티(modality)에 대해 1, 2, 4의 비율로 여러 차례 다운샘플링한다.

> "With this multi-scale approach, we compute losses at different resolutions."

→ 이러한 multi-scale 접근으로 서로 다른 해상도에서 손실(loss)을 계산한다.

> "For images, we output RGB data that align with the size of the input and utilize the common L1 loss L_img for the minimization of the absolute discrepancies between target and prediction."

→ 이미지의 경우, 입력 크기에 일치하는 RGB 데이터를 출력하고, 타겟과 예측 간 절대 차이를 최소화하기 위해 일반적인 L1 손실 `L_img`를 사용한다.

> "For point clouds, we generate range view images of dimensions H_r × W_r × 4, which can be converted into N × 3 point cloud data."

→ 점군(point cloud)의 경우, `H_r × W_r × 4` 크기의 range view 이미지를 생성하며, 이는 `N × 3` 점군 데이터로 변환 가능하다.

> "The target is the range view image transformed from the ground truth, where an L2 loss L_p,xyz is applied to minimize the Euler distance and an L1 loss L_p,r is based on range r."

→ 타겟은 ground truth로부터 변환된 range view 이미지이며, Euclidean 거리(원문 "Euler distance"는 typo로 추정) 최소화를 위한 L2 손실 `L_p,xyz`와 range `r`에 대한 L1 손실 `L_p,r`이 적용된다.

> "For 3D Occupancy, voxel grids of size 192 × 192 × 64 with 0.5m voxels contain the binary occupancy."

→ 3D 점유의 경우, 0.5m 단위 voxel로 구성된 `192 × 192 × 64` voxel 격자에 이진 점유(binary occupancy) 정보가 들어있다.

> "The target is obtained by voxelizing fused depth maps from depth cameras and point clouds from lidar."

→ 타겟은 depth 카메라에서 얻은 깊이 맵과 LiDAR 점군을 융합·복셀화하여 얻는다.

> "The loss for voxel grids uses a Scene-Class Affinity Loss (SCAL) [42] L_V,scal."

→ Voxel 격자에 대한 손실은 Scene-Class Affinity Loss(SCAL) `L_V,scal`을 사용한다.

> "The total loss is given by L = Σ_i λ_i ( λ_img L_img^i + λ_pcd (L_p,xyz^i + L_p,r^i + L_pcd^i) + λ_V L_V,scal^i )  (Eq. 1)"

→ 전체 손실은 위 식(Eq. 1)으로 주어진다. `λ_i`는 frame-별 가중치, 나머지 λ는 모달리티별 가중치.

### Datasets

> "We collected a training dataset D_train in the CARLA simulation environment [82] using an expert reinforcement learning agent [24], [83] for a more realistic driving style."

→ 더 현실적인 주행 스타일을 위해 expert RL agent를 사용해 CARLA 시뮬레이션 환경에서 학습 데이터셋 `D_train`을 수집했다.

> "Our data collection encompasses four towns (Town01, Town03, Town04, Town06) and four weather conditions (Clear Noon, Wet Noon, Hard Rain Noon, Clear Sunset), gathered at a frequency of 10 FPS."

→ 4개 도시(Town01/03/04/06)와 4개 기상조건(Clear Noon, Wet Noon, Hard Rain Noon, Clear Sunset)을 포함하며, 10 FPS로 수집했다.

> "For each town, we executed 25 runs, each lasting 300 seconds, with randomly selected weather conditions, amounting to 300,000 frames of data."

→ 각 도시당 300초짜리 25회 주행을 무작위 기상에서 실행했으며, 총 300,000 프레임이 모였다.

> "We obtain RBG image I ∈ ℝ^{3×600×960}, depth map I_D ∈ ℝ^{1×600×960}, e.g., derived from stereo cameras, point cloud P ∈ ℝ^{≤60,000×3} obtained from a lidar with 64 vertical channels, route map route ∈ ℝ^{1×64×64} as the planned route in BEV space, speed v ∈ ℝ, and actions a ∈ ℝ^2 in the form of acceleration and steering angle."

→ 프레임당 데이터: RGB 이미지 `3×600×960`, depth 맵 `1×600×960` (스테레오 카메라에서), 점군 `≤60,000×3` (64-채널 LiDAR), BEV 공간상 계획 경로 route `1×64×64`, 속도 `v`, 액션 `a∈ℝ²` (가속도 + 조향각).

> "We adopt the same setup as before for two distinct validation sets."

→ 두 개의 별도 검증 셋(validation set)을 같은 설정으로 구성한다.

> "For each town, we execute five 300-second long driving sessions with the following settings:"

→ 각 도시당 300초짜리 5회 주행 세션을 다음 설정으로 실행한다:

> "D^RL_val: This set uses the same cities and weather conditions as the training set."

→ `D^RL_val`: 학습 셋과 동일한 도시 및 기상조건 사용.

> "However, the driving routes are randomized."

→ 단 주행 경로는 무작위화한다.

> "The goal is to evaluate the effectiveness of our model in representation learning (RL) in familiar environments."

→ 목적은 익숙한 환경에서 representation learning(RL) 효과를 평가하는 것.

> "D^DS_val: We maintain the same cities as in the training set but introduce different weather conditions."

→ `D^DS_val`: 학습과 동일한 도시지만 기상은 다른 조건을 도입한다.

> "The driving routes are also randomized to evaluate the model performance under domain shifts (DS)."

→ 주행 경로 역시 무작위로 하여, domain shift(DS) 상황에서의 모델 성능을 평가한다.

### Training Parameters

> "We sampled data at intervals of 0.2 seconds, creating sequences of length 12 to serve as training inputs."

→ 데이터는 0.2초 간격(=5 Hz)으로 샘플링하여 길이 12의 시퀀스를 학습 입력으로 만든다.

> "All 12 frames were treated as known data."

→ 학습 시 12 프레임 전부를 알려진(known) 데이터로 취급했다.

> "In the experiments containing voxel reconstructions, we reduced the length of sequences to 6 to speed up the training."

→ Voxel 재구성을 포함하는 실험에서는 학습 가속을 위해 시퀀스 길이를 6으로 줄였다.

> "We trained with a batch size of 16 and the AdamW optimizer with a learning rate of 10^{-4} and a weight decay of 0.01."

→ 배치 16, AdamW 옵티마이저, learning rate `1e-4`, weight decay `0.01`로 학습.

> "For validation, we used 6/4 frames as given observations, while 6/2 served as ground truth."

→ 검증 시 관측(observations)으로 6/4 프레임을, ground truth로 6/2 프레임을 사용했다. (해석: non-voxel 실험은 seq=12에서 RF=6 + FH=6; voxel 실험은 seq=6에서 RF=4 + FH=2.)

> "For all experiments, we use a pre-trained ResNet18 [72] as our baseline backbone."

→ 모든 실험에서 사전학습된 ResNet18을 baseline backbone으로 사용했다.

---

## §IV-B Sensor Fusion Strategies

### 도입

> "Several prior multimodal world models rely on naive fusion approaches [26], [32], [84]."

→ 기존 멀티모달 world model 중 일부는 naive 융합 방식에 의존한다.

> "For our experiments, we compare those to a transformer-based architecture."

→ 본 실험은 그것들을 transformer 기반 구조와 비교한다.

> "To evaluate the effect of different sensor fusion strategies, we used metrics based on the evaluated modality:"

→ 서로 다른 융합 전략의 효과를 평가하기 위해, 평가 모달리티별 메트릭을 사용한다:

> "For assessing the quality of image predictions, we use the common Peak Signal-to-Noise Ratio (PSNR) to assess average differences."

→ 이미지 예측 품질 평가에는 일반적인 PSNR을 사용한다.

> "We use the Chamfer Distance to evaluate the accuracy of point cloud predictions."

→ 점군 예측의 정확도는 Chamfer Distance로 평가한다.

> "For the predictions of 3D occupancy grids, we used the metrics Intersection over Union (IoU), Precision, and Recall."

→ 3D 점유 격자 예측에는 IoU, Precision, Recall 메트릭을 사용한다.

> "We differentiate between IoU+ for occupied voxels and IoU− for empty ones."

→ 점유된 voxel은 IoU+, 비어있는 voxel은 IoU− 로 구분한다.

> "We first present the examined decoders and fusion methods."

→ 먼저 검토된 디코더와 융합 방법들을 제시한다.

> "Subsequently, we provide an overview and a comparison of all analyzed combinations, as shown in Figure 3."

→ 이후 Fig 3에 보인 대로 모든 분석된 조합의 개요와 비교를 제공한다.

### Encoders

> "For image features F_c, we compare the standard encoder introduced in Sec. III-A to approaches that map features to BEV space [15], [24], [73], [85]."

→ 이미지 특징 `F_c`에 대해, §III-A의 표준 encoder를 BEV 공간으로 특징을 매핑하는 접근과 비교한다.

> "Here, features are first elevated into a 3D space."

→ 이 접근에선 특징을 먼저 3D 공간으로 끌어올린다.

> "Then, these 3D feature voxels are aggregated into the BEV space, leading to image features F_b ∈ ℝ^{C×H_b×W_b}."

→ 그 후 3D 특징 voxel을 BEV 공간으로 집계하여 이미지 특징 `F_b ∈ ℝ^{C×H_b×W_b}`를 얻는다.

> "We evaluate both lossless and lossy representations for point clouds."

→ 점군에 대해 lossless 및 lossy 표현 둘 다 평가한다.

> "We compare a range view-based representation with PointPillars [70] as an encoder, where point clouds are segmented into discrete pillars along the X and Y axes followed by data processing and feature extraction, resulting in a 2D BEV pseudo-image."

→ Range view 기반 표현을 PointPillars(점군을 X, Y 축 기준 pillar로 분할 후 2D BEV pseudo-image 생성)와 비교한다.

### Latent Space

> "In prior works, the latent space was commonly modeled as one-dimensional vectors [24], [80], which may limit model performance by introducing a representational bottleneck."

→ 선행 연구에서 잠재 공간은 보통 1차원 벡터로 모델링되었는데, 이는 표현적 병목(representational bottleneck)을 유발해 모델 성능을 제한할 수 있다.

> "We perform experiments with both a 1D and a 2D latent space."

→ 본 연구는 1D와 2D 잠재 공간 양쪽으로 실험을 진행한다.

> "In addition, we examine an additional perceptual loss [86] and a vision transformer backbone [87]."

→ 또한 추가로 perceptual loss와 vision transformer backbone을 검토한다.

> "Figure 4 shows four evaluation graphs."

→ Fig 4에 4개의 평가 그래프가 제시된다.

> "It is divided into the prediction performance for camera (a), lidar (b), and 3D occupancy (c,d) on D^DS_val."

→ `D^DS_val` 상의 카메라(a), LiDAR(b), 3D 점유(c, d) 예측 성능으로 나뉜다.

> "The 2D latent state significantly benefits predictions for camera images and spatial voxel occupancies, while lidar predictions do not see any benefit."

→ **2D 잠재 상태는 카메라 이미지와 공간 voxel 점유 예측에 유의하게 유리하지만, LiDAR 예측에는 어떠한 이득도 없다.** (페이퍼의 핵심 finding 1)

> "This might be due to the fact that camera data is much more complex than lidar data."

→ 이는 카메라 데이터가 LiDAR 데이터보다 훨씬 복잡하기 때문일 가능성이 있다.

> "Compared to the baseline 2D model (dark blue), we do not see a strong effect of utilizing a perceptual loss, as it produces visually poorer reconstructions and does not show any significant advantages."

→ Baseline 2D 모델(짙은 파랑) 대비 perceptual loss는 큰 효과가 없으며, 오히려 시각적으로 더 흐릿한 재구성을 만들고 의미있는 이점이 없다.

> "Using the vision transformer as an encoder does provide advantages for the prediction of camera images but shows no effect on other metrics."

→ Vision transformer를 encoder로 쓰면 카메라 이미지 예측에는 이점이 있으나 다른 메트릭에는 영향이 없다.

> "This shows that the 2D latent space itself provides the largest boost in performance, while other changes have little effect."

→ **결국 2D 잠재 공간 그 자체가 성능 향상에 가장 큰 기여를 하고, 다른 변경(perceptual, ViT)은 거의 영향이 없다.** (페이퍼의 핵심 finding 2)

### Fusion Methods

> "We compare a transformer-based sensor fusion approach, as described in Sec. III-B, with naive combinations of encoded 1D features from each sensor modality, as found in the literature."

→ §III-B에서 기술한 transformer 기반 센서 융합 방식을, 기존 문헌의 각 모달리티별 1D 특징 naive 조합과 비교한다.

> "We perform experiments for both averaging features as well as concatenating them, followed by a fully connected layer."

→ 평균(averaging) 방식과 concat+FC 방식 양쪽 실험한다.

> "To generate such latent states, the output tokens are reshaped into their original shape after the encoding, namely F^new_c ∈ ℝ^{C×H_c×W_c} and F^new_L ∈ ℝ^{C×H_L×W_L}."

→ 그러한 잠재 상태 생성을 위해, encoding 후 출력 토큰을 원본 모양 `F^new_c`, `F^new_L`로 reshape한다.

> "Each feature is then downsampled by convolutional layers, followed by pooling layers to get one-dimensional features f_d ∈ ℝ^D, which are subsequently concatenated and then passed through fully connected layers to reduce its dimensionality, producing the vector o_t ∈ ℝ^D."

→ 각 특징은 conv + pooling으로 1D 특징 `f_d ∈ ℝ^D`로 다운샘플 후 concat, FC layer로 차원 축소하여 벡터 `o_t`를 만든다.

> "We evaluate the prediction performance of eight encoder-fusion combinations, as visible in Figure 3."

→ Fig 3과 같이 총 8개의 encoder-fusion 조합의 예측 성능을 평가한다.

> "We follow a naming scheme A-B-C: A represents the method of processing point clouds: PP stands for the use of PointPillars as the encoder; RV indicates the conversion of point clouds into range view."

→ A-B-C 네이밍 스킴: A = 점군 처리 방식 (PP = PointPillars, RV = range view 변환).

> "B denotes the approach of image processing: BEV implies mapping to BEV followed by feature extraction with a backbone; WOB denotes that no BEV mapping is performed."

→ B = 이미지 처리 방식 (BEV = BEV로 매핑 후 backbone feature 추출, WOB = BEV 매핑 없음).

> "C describes the method of sensor fusion: AVG stands for the averaging of 1D features; FC means that concatenation followed by a fully connected layer is performed; TR denotes that the transformer-based multi-head self-attention mechanism was used, as described in Sec. III-B."

→ C = 융합 방식 (AVG = 1D 특징 평균, FC = concat + FC layer, TR = §III-B의 transformer multi-head self-attention).

> "In the following, we first discuss the effects on image predictions, followed by the effects on point cloud predictions."

→ 이하 먼저 이미지 예측에 미치는 영향, 다음으로 점군 예측에 미치는 영향을 논의한다.

### Image Prediction

> "The impact of the different experiments on the quality of camera predictions is shown in Fig. 3 a) and 3 b)."

→ 카메라 예측 품질에 미치는 영향이 Fig 3 a, b에 나타나있다.

> "We observe a drop in performance for all networks in the D^DS_val dataset compared to D^RL_val, but the relative performance of different networks remains consistent across both datasets."

→ 모든 네트워크에서 `D^DS_val`이 `D^RL_val` 대비 성능 하락을 보이지만, 네트워크 간 상대적 순위는 두 데이터셋에서 일관되게 유지된다.

> "Generally, the transformer-based architecture RV-WOB-TR performs on par or better compared to the other combinations, and range view-based lidar encodings show clear advantages over PointPillars."

→ **일반적으로 transformer 기반 RV-WOB-TR이 다른 조합과 동급 또는 더 우수하며, range view 기반 LiDAR encoding이 PointPillars 대비 명확한 이점을 보인다.**

> "Methods with an additional BEV mapping of image features perform worse, and combinations with PointPillars suffer especially."

→ **이미지 특징을 BEV로 추가 매핑하는 방법은 성능이 떨어지고, PointPillars 조합은 특히 더 심하다.**

> "We can see that the effectiveness of introducing a transformer-based architecture depends on the encoder used."

→ Transformer 기반 구조의 효과성은 사용된 encoder에 의존함을 알 수 있다.

> "It outperforms other approaches when combined with a ResNet-18 for feature extraction."

→ ResNet-18을 특징 추출기로 함께 쓰면 다른 접근을 능가한다.

> "In contrast, when combined with PP and BEV, its performance is lower than concatenating (FC) but higher than averaging (AVG)."

→ 반면 PP + BEV와 결합하면 concat(FC)보다는 낮지만 평균(AVG)보다는 높다.

### Point Cloud Prediction

> "The impact of the different experiments on the quality of camera predictions is shown in Fig. 3 c) and 3 d)."

→ (페이퍼 원문 typo로 추정 — 본 subsection은 점군 예측인데 "camera predictions"로 표기됨. 실제 내용은 점군 예측이며 Fig 3 c, d에 나타난다.)

> "Examining the Chamfer Distance plots, where lower values mean better performance, we find no significant performance disparity between both validation datasets."

→ Chamfer Distance 플롯(낮을수록 좋음)에서 두 검증셋 간 큰 성능 차이는 발견되지 않는다.

> "For D^RL_val, the transformer-based architecture RV-WOB-TR performs on par or better compared to the other combinations."

→ `D^RL_val`에서 RV-WOB-TR이 다른 조합과 동급 또는 더 우수하다.

> "However, on D^DS_val, its performance drops."

→ 그러나 `D^DS_val`에서는 성능이 하락한다.

> "As before, range view-based methods demonstrate superiority over PointPillars."

→ 앞서와 같이 range view 기반 방법이 PointPillars보다 우수하다.

> "Utilizing BEV features shows no clear disadvantage for this task."

→ 점군 예측 과제에선 BEV 특징 사용이 뚜렷한 단점을 보이지 않는다.

> "We can see that transformer-based architectures generally outperform other fusion techniques."

→ Transformer 기반 구조가 다른 융합 기법보다 일반적으로 우월하다.

> "We determine a transformer-based architecture with a 2D latent space and lossless range-view representations for point clouds as an optimal fusion strategy, while performance benefits are more pronounced for camera predictions."

→ **결론: 2D 잠재 공간 + 점군의 lossless range-view 표현 + transformer 기반 구조의 조합이 최적 융합 전략이며, 카메라 예측에서 성능 이득이 더 두드러진다.** (페이퍼 §IV-B 최종 verdict)

---

## §IV-C 3D Occupancy Prediction

### 도입

> "Next to analyzing fusion strategies, we are interested in the effects of also predicting more actionable 3D occupancies."

→ 융합 전략 분석에 더해, 더 actionable한 3D 점유 예측의 효과에도 관심이 있다.

> "Our experiments analyze whether an occupancy model can benefit from a pre-trained model which was trained by only predicting camera and lidar data, as shown in Fig. 5."

→ Fig 5에서 보듯, 카메라와 LiDAR만 예측하도록 학습된 사전학습 모델이 점유 모델에 도움을 줄 수 있는지 분석한다.

> "Subsequently, we analyze if occupancy prediction improves the prediction of camera and lidar data, as shown in Fig. 6."

→ 이어 Fig 6에서 보듯, 점유 예측이 역으로 카메라/LiDAR 예측을 개선하는지를 분석한다.

### 3D Occupancy Prediction

> "We perform experiments in three scenarios, as shown in Figure 5."

→ Fig 5와 같이 세 가지 시나리오로 실험한다.

> "As we want to examine the effect of encoded knowledge of predicting camera and lidar data on 3D occupancy, we first train a model as a pre-trained starting point that predicts camera and lidar data alone for 50,000 steps."

→ 카메라/LiDAR 예측을 통해 encoding된 지식이 3D 점유에 미치는 영향을 보기 위해, 먼저 카메라+LiDAR만 예측하는 모델을 50,000 step 사전학습한다.

> "For the first scenario, we employ the pre-trained model but freeze (PTF) all of its weights so that only the weights of the voxel decoder are trained."

→ 첫 번째 시나리오(**PTF**, Pre-Trained Frozen): 사전학습 모델 가중치를 전부 freeze하고 voxel decoder만 학습.

> "This approach allows us to assess the impact of fine-tuning only the voxel-specific aspects of the model while keeping the rest of the network, in particular all encoders, constant to evaluate if any information about a discrete geometry of the world is already encoded based on camera and lidar data."

→ 이 접근은 나머지 네트워크(특히 모든 encoder)를 고정한 채 voxel-특화 부분만 fine-tune해서, 카메라/LiDAR 학습만으로도 세계의 이산적 기하 정보가 이미 encoding되어 있는지 평가하게 해준다.

> "For the second scenario, the pre-trained weights were used as a starting point, but the entire network was open (PTO) for weight updates during training."

→ 두 번째 시나리오(**PTO**, Pre-Trained Open): 사전학습 가중치를 시작점으로 쓰되 학습 중 네트워크 전체를 업데이트 허용.

> "Here, we analyze how the pre-trained weights influence the learning process when the whole network adapts and evolves during training."

→ 여기선 네트워크 전체가 적응·진화하는 과정에서 사전학습 가중치가 학습에 미치는 영향을 분석한다.

> "For the third scenario, no pre-training (NPT) is utilized, and we train the network from scratch."

→ 세 번째 시나리오(**NPT**, No Pre-Training): 사전학습 없이 scratch부터 학습.

> "In Figure 5 we observe that the model trained from scratch (NPT) exhibits a similar performance on both validation datasets across all four metrics, while the other two models using pre-trained weights (PTF and PTO) generally performed better on D^RL_val than on D^DS_val across three metrics, excluding IoU−."

→ Fig 5에서 NPT는 4개 메트릭 모두에서 두 검증셋 간 비슷한 성능을 보이는 반면, PTF·PTO 둘은 IoU−를 제외한 3개 메트릭에서 `D^RL_val`이 `D^DS_val`보다 일반적으로 더 나았다.

> "Interestingly, for IoU− we observe an opposite behavior, where the models perform better on D^DS_val."

→ 흥미롭게도 IoU−에서는 반대 패턴, 즉 `D^DS_val`에서 성능이 더 좋았다.

> "This is attributed to voxel occupancy grid predictions focusing more on occupied grids."

→ 이는 voxel 점유 격자 예측이 점유된 격자에 더 집중하기 때문이다.

> "Since voxel grids are mostly empty, models on D^DS_val tend to predict more noise, leading to lower IoU− scores."

→ Voxel 격자 대부분이 비어있으므로, `D^DS_val`에서 모델이 더 많은 노이즈를 예측해 IoU− 점수가 낮아진다. (역설적으로 이게 더 좋은 점수 산출로 이어진다는 의미는 페이퍼 텍스트만으로는 불완전 — Fig 5의 IoU− 값 방향 정의에 의존.)

> "Comparing PTO to NPT, the PTO model showed advantages early on, supporting the idea that pre-trained weights contribute valuable spatial knowledge."

→ PTO vs NPT 비교: PTO가 초기에는 우위를 보였고, 이는 사전학습 가중치가 가치 있는 공간 지식을 제공한다는 생각을 뒷받침한다.

> "However, in the later stages of training, the NPT model overtook the PTO model in Precision, while the PTO model remained superior for IoU− and Recall."

→ 그러나 학습 후반부에서는 NPT가 Precision에서 PTO를 추월했고, PTO는 IoU−와 Recall에서 우위를 유지했다.

> "This indicates that the non-pre-trained model adopts a more conservative strategy for 3D occupancy prediction."

→ 이는 사전학습 없는 모델이 더 보수적(conservative)인 점유 예측 전략을 채택함을 시사한다.

> "When we examine the PTF scenario, although the model underperformed compared to the other two, its performance improved over time by only training the voxel decoder."

→ PTF 시나리오는 다른 두 시나리오보다 성능이 낮았지만, voxel decoder만 학습해도 시간에 따라 성능이 향상되었다.

> "This improvement underscores that the pre-trained weights already contain some, however limited, spatial information, indicating that the model partially integrates image and point cloud features to form spatial voxel features even when trained only on these two modalities."

→ 이 개선은 사전학습 가중치가 제한적이나마 공간 정보를 이미 담고 있음을 강조하며, 카메라+LiDAR 두 모달리티만 학습해도 모델이 부분적으로 이미지와 점군 특징을 공간 voxel 특징으로 통합함을 시사한다.

> "As learning 3D occupancy is computationally intensive, we conclude that pre-training strategies on only camera and lidar data are generally recommendable, as they both speed up training and show overall superior performance."

→ **결론: 3D 점유 학습은 계산 비용이 크므로, 카메라+LiDAR만으로 사전학습하는 전략이 일반적으로 권장된다. 학습 속도도 빠르고 전반적 성능도 우수하다.** (페이퍼 §IV-C 핵심 결론)

### Sensor Data Predictions

> "We perform experiments to determine whether knowledge encoded through occupancy can be leveraged by lidar and camera predictions, as shown in Figure 6."

→ Fig 6과 같이, 점유 학습을 통해 encoding된 지식이 LiDAR/카메라 예측에 활용될 수 있는지 실험한다.

> "Based on the chamfer Distance for point clouds and the PSNR metric for images, we observe only slightly increased performance gains for both modalities when occupancy prediction is included, with a more pronounced benefit for camera predictions under the D^RL_val setting."

→ 점군 Chamfer Distance와 이미지 PSNR을 보면, 점유 예측을 추가했을 때 두 모달리티 모두 **약간(slightly)** 의 성능 향상만 보이며, `D^RL_val` 설정에서 카메라 예측 쪽 이득이 더 두드러진다.

---

## 비교 표 (페이퍼 §IV의 4개 figure 정성 정리)

페이퍼 본문에 정확한 수치 값은 제시되지 않음(Fig 3-6의 plot으로만 확인 가능). 본 표는 페이퍼 텍스트에 명시된 정성적 verdict만 정리.

### 표 1. Fig 3 — Sensor Fusion Strategies (8개 조합 × 2 metric × 2 val set)

페이퍼는 12개 조합(2 LiDAR × 2 image × 3 fusion) 중 8개를 평가. 네이밍: `<LiDAR>-<Image>-<Fusion>`.

| 조합 | RGB PSNR `D^RL_val` | RGB PSNR `D^DS_val` | LiDAR Chamfer `D^RL_val` | LiDAR Chamfer `D^DS_val` |
|---|---|---|---|---|
| **RV-WOB-TR** (페이퍼의 최적) | par/better | par/better | par/better | drops (열위로 전환) |
| RV-WOB-FC | inferior | inferior | inferior | par |
| RV-WOB-AVG | inferior | inferior | inferior | par |
| RV-BEV-TR | worse | worse | par | par |
| RV-BEV-FC | worse | worse | par | par |
| RV-BEV-AVG | worse | worse | par | par |
| PP-WOB-TR | TR=middle | TR=middle | inferior | inferior |
| PP-BEV-TR | TR < FC, TR > AVG | 동일 패턴 | inferior (PP suffers) | inferior |

**핵심 verdict**:
- "transformer-based architecture **RV-WOB-TR** performs on par or better"
- "range view-based lidar encodings show clear advantages over PointPillars"
- "Methods with an additional BEV mapping of image features perform worse, and combinations with PointPillars suffer especially"
- "It outperforms other approaches when combined with a ResNet-18 for feature extraction"
- "when combined with PP and BEV, its performance is lower than concatenating (FC) but higher than averaging (AVG)"

### 표 2. Fig 4 — Latent Space Variants (4개 변형 × 4 metric on `D^DS_val`)

| 변형 | RGB (Fig 4a) | LiDAR (Fig 4b) | Voxel IoU+ (Fig 4c) | Voxel IoU− (Fig 4d) |
|---|---|---|---|---|
| **1D baseline** | inferior (baseline) | par | inferior | inferior |
| **2D (페이퍼 baseline 2D, dark blue)** | **significant boost** | no benefit (1D와 동급) | **significant boost** | **significant boost** |
| 2D + PerceptualLoss (PL) | "visually poorer" (불리) | par | par | par |
| 2D + ViT (MobileViT-V2) | "advantages for camera images" (약간 ↑) | no effect | no effect | no effect |

**핵심 verdict**:
- "**The 2D latent state significantly benefits predictions for camera images and spatial voxel occupancies, while lidar predictions do not see any benefit**"
- "This might be due to the fact that camera data is much more complex than lidar data"
- "perceptual loss ... produces visually poorer reconstructions and does not show any significant advantages"
- "Using the vision transformer as an encoder does provide advantages for the prediction of camera images but shows no effect on other metrics"
- "**the 2D latent space itself provides the largest boost in performance, while other changes have little effect**"

### 표 3. Fig 5 — Occupancy Pre-training Scenarios (3 시나리오 × 4 metric × 2 val set)

| 시나리오 | IoU+ `D^RL_val` | IoU+ `D^DS_val` | IoU− `D^RL_val` | IoU− `D^DS_val` | Precision | Recall |
|---|---|---|---|---|---|---|
| **PTF** (encoders frozen, decoder만) | underperformed (lowest) | underperformed | underperformed | underperformed | underperformed | underperformed |
| **PTO** (사전학습 → 전체 unfreeze) | superior (RL > DS) | inferior to RL | superior (RL > DS) | **반대: DS > RL** | early: leader; late: NPT 추월 | leader |
| **NPT** (scratch) | 두 val set 비슷 | 두 val set 비슷 | 두 val set 비슷 | 두 val set 비슷 | **late: PTO 추월** | inferior to PTO |

**핵심 verdict**:
- NPT: "exhibits a similar performance on both validation datasets across all four metrics"
- PTF/PTO: "generally performed better on `D^RL_val` than on `D^DS_val` across three metrics, excluding IoU−"
- IoU−: "we observe an opposite behavior, where the models perform better on `D^DS_val`" (이유: voxel mostly empty + DS predicts more noise)
- PTO vs NPT: "the PTO model showed advantages early on ... in the later stages of training, the NPT model overtook the PTO model in Precision, while the PTO model remained superior for IoU− and Recall"
- NPT는 "**more conservative strategy**"
- PTF의 "performance improved over time by only training the voxel decoder" → 사전학습 가중치에 공간 정보가 일부 이미 들어있다는 evidence
- **결론**: "pre-training strategies on only camera and lidar data are generally recommendable" (PTO/PTF > NPT, 학습 속도 + 전반 성능 모두)

### 표 4. Fig 6 — Occupancy의 역방향 영향 (camera/lidar 예측에)

| 시나리오 | RGB PSNR `D^RL_val` | RGB PSNR `D^DS_val` | LiDAR Chamfer `D^RL_val` | LiDAR Chamfer `D^DS_val` |
|---|---|---|---|---|
| with occupancy prediction | slightly higher (**more pronounced**) | slightly higher | slightly higher | slightly higher |
| without occupancy | baseline | baseline | baseline | baseline |

**핵심 verdict**:
- "we observe only **slightly increased performance gains for both modalities** when occupancy prediction is included"
- "with a more pronounced benefit for camera predictions under the `D^RL_val` setting"

---

## 페이퍼 §IV 종합 한 줄

> **2D latent + RV LiDAR + transformer fusion + 카메라/LiDAR로 사전학습 → 점유 학습 추가** 가 페이퍼가 endorse한 recipe. 단 점유의 역기여는 marginal, perceptual loss와 ViT는 부정.

---

## 한계 / 주의

- **수치 값 부재**: 페이퍼는 Fig 3-6의 plot 데이터를 본문 표로 제시하지 않음. 본 표의 모든 cell은 페이퍼 텍스트의 정성적 표현만 인용. 정확한 PSNR/Chamfer/IoU 값은 plot에서 읽거나 release weight로 재현 필요.
- **Fig 3 typo**: 페이퍼 §IV-B "Point Cloud Prediction" 소제목 아래 첫 문장에 "camera predictions"라고 잘못 쓰여 있음. 실제 내용은 점군 예측.
- **IoU− 해석 미묘**: 페이퍼는 "models on `D^DS_val` tend to predict more noise, leading to lower IoU− scores"라고 했는데, 이 문맥에서 "lower IoU− = worse"인지 "lower IoU− = better"인지는 IoU−의 정의 방향에 따라 달라짐. 페이퍼는 메트릭 정의를 명시하지 않음.

---

## 참고 문서

- `muvo_paper_summary.md` — 5개 섹션 통합 요약 (TL;DR 포함)
- `muvo_full_comparison.md` — 4-way 코드 비교 (upstream/2D/alpha)
- `muvo_4way_1D_2D_comparison.md` — alpha 1D/upstream 1D/alpha 2D/upstream 2D 매트릭스
