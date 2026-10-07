# IV. Evaluation — 영-한 대역본

- **원문 위치**: `muvo_paper.txt` L300–L588
- **한 줄 요지 (KR)**: §IV-A 학습 셋업(자기지도, 다중 스케일 L1/L2 + SCAL voxel loss, CARLA 300K frames, seq 12/6); §IV-B 8가지 fusion 조합 비교에서 RV-WOB-TR 및 2D latent이 최적; §IV-C voxel pre-training은 camera+lidar 만으로 미리 학습한 가중치가 occupancy 학습을 가속하며, occupancy 추가 예측은 sensor 예측에 약간의 개선.

---

## 개요 (L300–L310)

&nbsp;

> For the evaluation, we first present our utilized training setup in Sec. IV-A, followed by the evaluation of sensor fusion strategies in Sec. IV-B.

> 평가에서는 먼저 §IV-A에서 사용한 학습 셋업을 제시한 뒤, §IV-B에서 sensor fusion 전략을 평가한다.

&nbsp;

> Fig. 3 shows the impact of different fusion strategies, and Fig. 4 demonstrates the difference of differently sized latent spaces.

> Fig. 3은 여러 fusion 전략의 영향을, Fig. 4는 잠재 공간 크기별 차이를 보여준다.

&nbsp;

> Finally, we examine the effects of the optional 3D occupancy prediction in Sec. IV-C.

> 마지막으로 §IV-C에서 선택적 3D occupancy 예측의 효과를 분석한다.

&nbsp;

> Fig. 5 shows the relation between camera-lidar based pre-training and 3D occupancy prediction, and Fig. 6 shows the reverse impact of predicting occupancy on the quality of sensor predictions.

> Fig. 5는 camera-lidar 기반 pre-training과 3D occupancy 예측의 관계를, Fig. 6은 반대로 occupancy 예측이 sensor 예측 품질에 미치는 영향을 보여준다.

---

## A. Training Setup (L311–L379)

### Training Losses

&nbsp;

> For our experiments, we follow a self-supervised training approach and do not require labels at any point.

> 모든 실험은 자기지도(self-supervised) 학습이며, 어느 단계에서도 라벨이 필요하지 않다.

&nbsp;

> For each modality, we downsample multiple times with ratios of 1, 2, and 4.

> 각 모달리티는 1, 2, 4 비율로 여러 번 다운샘플링한다.

&nbsp;

> With this multi-scale approach, we compute losses at different resolutions.

> 이 multi-scale 접근으로 서로 다른 해상도에서 손실을 계산한다.
- **multi-scale loss**: 디코더 출력을 원본 1×, 절반 1/2×, 1/4× 해상도로 동시에 만들고 각각에 손실 부여. 저해상도는 거시 구조, 고해상도는 세부 디테일을 잡게 해서 안정적 학습 + blur 완화.
  ```
  decoder ── head_1 ─▶ pred_1   (H, W)         ──L1 vs gt_1
           ├─ head_2 ─▶ pred_2   (H/2, W/2)     ──L1 vs gt_2  (×1/2 weight)
           └─ head_4 ─▶ pred_4   (H/4, W/4)     ──L1 vs gt_4  (×1/4 weight)
  ```

&nbsp;

> For images, we output RGB data that align with the size of the input and utilize the common L1 loss $`ℒ^{\text{img}}`$ for the minimization of the absolute discrepancies between target and prediction.

> 이미지의 경우, 입력 크기와 같은 RGB를 출력하고 표준 L1 손실 $`ℒ^{\text{img}}`$ 를 사용해 target과 예측의 절대 차이를 최소화한다.

&nbsp;

> For point clouds, we generate range view images of dimensions $`H_r \times W_r \times 4`$, which can be converted into $`N \times 3`$ point cloud data.

> 포인트 클라우드는 $`H_r \times W_r \times 4`$ 차원의 range view 이미지를 생성하며, 이는 $`N \times 3`$ 형태의 포인트 클라우드로 변환할 수 있다.

&nbsp;

> The target is the range view image transformed from the ground truth, where an L2 loss $`ℒ^{\text{pxyz}}`$ is applied to minimize the Euler distance and an L1 loss $`ℒ^{\text{pr}}`$ is based on range $`r`$.

> target은 ground truth로부터 변환된 range view 이미지이며, 유클리드(Euclidean) 거리를 최소화하는 L2 손실 $`ℒ^{\text{pxyz}}`$ 와 range $`r`$ 에 대한 L1 손실 $`ℒ^{\text{pr}}`$ 를 적용한다.

&nbsp;

> For 3D Occupancy, voxel grids of size $`192 \times 192 \times 64`$ with 0.5m voxels contain the binary occupancy.

> 3D occupancy의 경우, 0.5m 셀의 $`192 \times 192 \times 64`$ voxel grid가 이진(binary) occupancy를 담는다.

&nbsp;

> The target is obtained by voxelizing fused depth maps from depth cameras and point clouds from lidar.

> target은 depth 카메라의 depth map과 LiDAR 포인트 클라우드를 융합해 voxel화하여 얻는다.

&nbsp;

> The loss for voxel grids uses a Scene-Class Affinity Loss (SCAL) [42] $`ℒ^{V,\text{scal}}`$.

> Voxel grid 손실은 Scene-Class Affinity Loss(SCAL) [42] $`ℒ^{V,\text{scal}}`$ 를 사용한다.
- **SCAL (Scene-Class Affinity Loss)**: MonoScene에서 도입. voxel grid의 sparse·imbalanced 특성을 다루기 위해 클래스별 precision/recall/specificity를 미분 가능하게 손실로 사용. cross-entropy만 쓸 때보다 class-wise 균형이 좋아짐.
  ```
  SCAL(class c) = (1 - P_c) + (1 - R_c) + (1 - S_c)
     P_c = TP_c / (TP_c + FP_c)   ← precision (예측 c 중 맞은 비율)
     R_c = TP_c / (TP_c + FN_c)   ← recall    (실제 c 중 맞춘 비율)
     S_c = TN_c / (TN_c + FP_c)   ← specificity (c가 아닌데 c로 안 예측한 비율)
  ```

&nbsp;

> The total loss is given by

$$ℒ = \sum_i \lambda_i \left( \lambda_{\text{img}} ℒ_i^{\text{img}} + \lambda_{\text{pcd}} (ℒ_i^{\text{pxyz}} + ℒ_i^{\text{pr}} + ℒ_i^{\text{pcd}}) + \lambda_V ℒ_i^{V,\text{scal}} \right) \quad \text{(eq. 1)}$$

> 전체 손실은 식 (1)로 주어지며,

$$ℒ = \sum_i \lambda_i \left( \lambda_{\text{img}} ℒ_i^{\text{img}} + \lambda_{\text{pcd}} (ℒ_i^{\text{pxyz}} + ℒ_i^{\text{pr}} + ℒ_i^{\text{pcd}}) + \lambda_V ℒ_i^{V,\text{scal}} \right) \quad \text{(1)}$$

형태이다.

---

### Loss 보충 설명

paper §IV-A 식 (1)의 각 항을 분해해서 정리. alpha 구현과의 매핑은 마지막 표 참조.

#### (1) Multi-scale 인덱스 $`i`$ 와 가중치 $`\lambda_i`$

paper는 디코더 출력을 1, 2, 4 배율로 다운샘플해서 각 해상도마다 loss를 따로 계산한다. 인덱스 $`i \in \{1, 2, 4\}`$ 가 해상도, $`\lambda_i`$ 가 해상도별 가중치.

- 일반적인 선택: $`\lambda_i = 1/i`$ → 저해상도일수록 가중치 작음. alpha 구현도 동일 (`trainer_muvo.py`의 `discount factors [1, 2, 4]`에 `× 1/factor`).
- 의도: 고해상도 디테일을 잡으면서, 저해상도 거시 구조도 동시에 supervise해서 학습 안정화 + blur 완화.

```
prediction at scale i ∈ {1, 2, 4}    target at scale i
        │                                    │
        └─────────── L_i(pred, target) ──────┘
        │
        × λ_i  (예: 1, 1/2, 1/4)
        │
        ▼
        Σ over i  ──▶ 최종 multi-scale loss
```

#### (2) $`ℒ^{\text{img}}`$ — RGB L1 reconstruction

```
L_img = (1 / N_pixel) · Σ_{px} | pred_RGB[px] - target_RGB[px] |
```

- **왜 L1 (MSE 아님)?** L1은 outlier에 강하고 RGB 픽셀 차이의 절댓값을 그대로 학습. MSE는 큰 오차를 제곱해서 over-smooth(전체적으로 blur)되기 쉬움. 영상 재구성에서 L1이 표준.
- alpha 매핑: `trainer_muvo.py:compute_loss`에서 `F.l1_loss(pred, target) × WEIGHT_RGB × 1/factor`. 가중치 `WEIGHT_RGB=1.0`.

#### (3) $`ℒ^{\text{pxyz}}`$ — LiDAR 좌표 L2

```
L_{p,xyz} = (1 / N_valid) · Σ_{px ∈ valid}
              ||pred_xyz[px] - target_xyz[px]||²
```

- **valid mask**: range-view 이미지에서 실제 점이 있는 픽셀만 포함. 빈 픽셀(empty)은 별도 항으로 처리.
- **L2 이유**: 3D 거리(유클리드) 자체를 줄이는 목적. xyz는 연속·정밀 값이라 L2가 자연스러움.
- alpha 매핑: `F.mse_loss(pred[..., :3], target[..., :3])` (유효 점만). 가중치 `WEIGHT_LIDAR_RE` ⚠ **현재 0.0** — improvement_plan P0.

#### (4) $`ℒ^{\text{pr}}`$ — LiDAR range L1

```
L_{p,r} = (1 / N_valid) · Σ_{px ∈ valid}
            | pred_range[px] - target_range[px] |
```

- **range channel** $`r = \sqrt{x^2 + y^2 + z^2}`$ — 깊이 정보만 별도로 supervise. xyz와 중복이지만 paper는 두 항을 모두 사용해 깊이 정확도를 강조.
- **L1 이유**: range는 단일 채널이고 outlier(원거리 점)에 안정적. L1이 잘 작동.
- alpha 매핑: `F.l1_loss(pred_depth, target_depth)` (유효 점만). 가중치 동일.

#### (5) $`ℒ^{\text{pcd}}`$ — 정의 모호, 두 해석 가능

paper §IV-A 본문은 $`ℒ^{p,\text{xyz}}`$ (L2) 와 $`ℒ^{p,r}`$ (L1) 만 정의 후 식 (1) 에 $`ℒ^{\text{pcd}}`$ 가 추가 등장 — **정확한 정의가 paper 에 명시되지 않음**. 두 해석 가능:

**해석 A — Chamfer Distance (PCD 정합 손실)**: 일반 ML 관습. 예측 점 분포 ↔ GT 점 분포 거리.
```
L_chamfer = (1/|P_pred|) Σ_{p∈P_pred} min_{q∈P_gt} ‖p-q‖² 
          + (1/|P_gt|)   Σ_{q∈P_gt}   min_{p∈P_pred} ‖q-p‖²
```
upstream 저자도 이 해석으로 `muvo_2d/muvo/losses.py:354 CDLoss` 클래스 정의.

**해석 B — Empty smooth-L1 (빈 픽셀 hallucinate 방지)**: alpha 의 해석.
```
L_empty = (1/N_invalid) · Σ_{px ∉ valid} smooth_L1( pred_range[px], 0 )
```
의도: 빈 픽셀에서도 "여기는 점 없다(range≈0)" 예측 강제.

**핵심 관찰**: paper 식 (1) $`ℒ^{\text{pcd}}`$ 항은 **upstream / alpha 두 구현 모두에서 학습에 사용되지 않음**:
- upstream: `trainer.py:124 # self.lidar_cd_loss = CDLoss()` **주석처리**. CDMetric 으로 validation/test 평가만 사용.
- alpha: `WEIGHT_LIDAR_EMPTY=0.0` 으로 꺼짐.

→ paper 본문 정의 모호 + 양쪽 구현 모두 실제 학습 미사용 → "paper $`ℒ^{\text{pcd}}`$ 가 정확히 무엇이었나" 는 paper 만으로 결정 불가. upstream 저자가 Chamfer 로 의도했다는 것은 클래스 명에서 추론할 수 있지만 절대적 정답 아님. **alpha 의 empty 해석도 paper 식 (1) 의 한 후보로 유효** — 두 가지 모두 ablation 가치 있음.

| 항목 | 해석 A (Chamfer, upstream) | 해석 B (Empty smooth-L1, alpha) |
|---|---|---|
| 손실 함수 | `CDLoss` (Chamfer) | `F.smooth_l1_loss(pred_depth, zeros)` |
| 마스크 | valid 점만 | invalid 점만 (빈 픽셀) |
| 의도 | 점 분포 매칭 | 빈 픽셀 hallucinate 방지 |
| 코드 | `muvo_2d/losses.py:354` (주석처리, 학습 X) | `trainer_muvo_2D.py:576-578` (weight 0) |
| cfg 키 | `WEIGHT_LIDAR_RE` 묶음 | `WEIGHT_LIDAR_EMPTY` |
| 현재 학습 사용? | ❌ 주석처리 | ❌ weight 0 |

#### (6) $`ℒ^{V,\text{scal}}`$ — SCAL voxel loss (MonoScene)

```
SCAL = Σ_{class c} [ (1 - P_c) + (1 - R_c) + (1 - S_c) ]

  P_c = Σ TP_c / Σ(TP_c + FP_c)   ← precision (class c)
  R_c = Σ TP_c / Σ(TP_c + FN_c)   ← recall    (class c)
  S_c = Σ TN_c / Σ(TN_c + FP_c)   ← specificity (class c)
```

- **왜 SCAL?** voxel grid는 99%가 빈 셀 → 단순 cross-entropy는 다수 클래스(empty)에 편향. SCAL은 클래스마다 precision·recall·specificity를 미분 가능하게 직접 최소화 → class-wise 균형이 자연스럽게 맞춰짐.
- 추가 항 `SemScalLoss + GeoScalLoss`도 upstream에서 함께 사용 (의미/기하 분리). alpha에는 미구현 (occupancy 헤드 없음).

#### (7) RSSM의 KL probabilistic loss (paper 식 (1)에는 없지만 RSSM 내부 항)

paper 본문 §III-C는 prior/posterior 분포를 정의하지만 식 (1)에 KL 항을 명시하지 않음. 실제 RSSM 구현(MILE, MUVO upstream, alpha 모두)은 KL 정규화를 사용:

```
KL_balanced = α · KL( sg(post) ‖ prior ) + (1 - α) · KL( post ‖ sg(prior) )

  sg : stop_gradient
  α  : KL_BALANCING_ALPHA (alpha 0.75)

가능한 옵션: free-bits
  per-dim KL_sum = max( KL_sum, free_bits )
  → KL이 너무 작아지면 (posterior collapse 방지) 더 이상 줄지 않게 clamp
```

- alpha 매핑: `trainer_muvo.py:balanced_kl_loss` + `gaussian_kl_loss(free_bits=...)`. 가중치 `WEIGHT_PROBABILISTIC=1e-2`, `KL_FREE_BITS=1.0`.
- alpha-1D는 free-bits 항상 적용, alpha-2D는 `KL_FREE_BITS_ENABLED` cfg 게이트로 ON/OFF 선택 가능.

#### (8) Future loss factor $`\lambda_{\text{future}}`$ (alpha 전용)

paper 식 (1)은 sequence 전체에 동일 가중치. alpha는 receptive-field(RF) 부분과 future-horizon(FH) 부분을 분리하고 FH에 별도 multiplier 적용:

```
sequence : [ frame_0 ... frame_{RF-1} | frame_{RF} ... frame_{RF+FH-1} ]
                  observation 구간            future prediction 구간
                  posterior 사용              prior(imagine) 사용

L_total = L_obs (위 항들 합) + WEIGHT_FUTURE × L_fut (위 항들 합, future 부분)
```

- alpha 매핑: `trainer_muvo.py`의 `future_*` prefix loss + `× WEIGHT_FUTURE`. 기본 `WEIGHT_FUTURE=1.0` (= future도 동등 가중치).

#### (9) Action loss (alpha-2D 전용, paper에 없음)

alpha-2D는 RSSMTD의 policy token 출력으로 행동을 회귀:

```
L_action = (1 / (B × S)) · Σ | pred_action - target_action |
```

- 가중치 `WEIGHT_ACTION=1.0`. paper 본문에는 없으나 alpha-2D `models_muvo_2D.py:PolicyDecoder` + `trainer_muvo_2D.py` 추가.

#### (10) Loss 항 ↔ 가중치 ↔ alpha 코드 매핑 표

| paper 기호 | 의미 | upstream 가중치 키 | alpha-1D 값 | alpha-2D 값 | 코드 위치 |
|---|---|---|---|---|---|
| $`ℒ^{\text{img}}`$ | RGB L1 multi-scale | `LOSSES.WEIGHT_*_RGB` | `WEIGHT_RGB=1.0` ✓ | `WEIGHT_RGB=1.0` ✓ | `trainer_muvo*.py:compute_loss` |
| $`ℒ^{\text{pxyz}}`$ | LiDAR xyz L2 | `LOSSES.WEIGHT_LIDAR_RE` | **0.0 ⚠** | **0.0 ⚠** | 동 |
| $`ℒ^{\text{pr}}`$ | LiDAR range L1 | `LOSSES.WEIGHT_LIDAR_RE` | **0.0 ⚠** | **0.0 ⚠** | 동 |
| $`ℒ^{\text{pcd}}_{\text{empty}}`$ | empty 픽셀 smooth-L1 | `LOSSES.WEIGHT_LIDAR_EMPTY` | **0.0 ⚠** | **0.0 ⚠** | 동 |
| $`ℒ^{V,\text{scal}}`$ | SCAL voxel | `LOSSES.WEIGHT_VOXEL` | ✗ (헤드 없음) | ✗ (헤드 없음) | – |
| $`ℒ^{\text{prob}}`$ (KL) | balanced KL | `LOSSES.WEIGHT_PROBABILISTIC` | `1e-2` | `1e-2` | `balanced_kl_loss` |
| free-bits | KL clamp | `LOSSES.KL_FREE_BITS` | `1.0` (항상) | `1.0` (cfg 게이트) | `gaussian_kl_loss` |
| $`\lambda_{\text{future}}`$ | future multiplier | `LOSSES.WEIGHT_FUTURE` | `1.0` | `1.0` | `compute_loss` future 분기 |
| SSIM | 보조 RGB | `LOSSES.WEIGHT_SSIM` | `0.0` (꺼짐) | `0.0` (꺼짐) | `SSIMLoss` |
| $`ℒ^{\text{action}}`$ | action L1 (alpha-2D 추가) | `LOSSES.WEIGHT_ACTION` | ✗ | `1.0` | `trainer_muvo_2D.py` |

---

### Datasets

&nbsp;

> We collected a training dataset $`𝒟_{\text{train}}`$ in the CARLA simulation environment [82] using an expert reinforcement learning agent [24], [83] for a more realistic driving style.

> 학습 데이터 $`𝒟_{\text{train}}`$ 은 CARLA 시뮬레이션 [82]에서, 보다 현실적인 주행 스타일을 위해 expert RL 에이전트 [24], [83]로 수집했다.

&nbsp;

> Our data collection encompasses four towns (Town01, Town03, Town04, Town06) and four weather conditions (Clear Noon, Wet Noon, Hard Rain Noon, Clear Sunset), gathered at a frequency of 10 FPS.

> 4개 town(Town01·03·04·06)과 4가지 날씨(Clear Noon, Wet Noon, Hard Rain Noon, Clear Sunset)를 10 FPS로 수집했다.

&nbsp;

> For each town, we executed 25 runs, each lasting 300 seconds, with randomly selected weather conditions, amounting to 300,000 frames of data.

> 각 town당 25회의 주행(각 300초, 랜덤 날씨)을 실행해 총 300,000 프레임 데이터를 수집했다.

&nbsp;

> We obtain RGB image $`ℐ \in ℝ^{3 \times 600 \times 960}`$, depth map $`ℐ_D \in ℝ^{1 \times 600 \times 960}`$, e.g., derived from stereo cameras, point cloud $`𝒫 \in ℝ^{\le 60{,}000 \times 3}`$ obtained from a lidar with 64 vertical channels, route map $`\text{route} \in ℝ^{1 \times 64 \times 64}`$ as the planned route in BEV space, speed $`v \in ℝ`$, and actions $`a \in ℝ^2`$ in the form of acceleration and steering angle.

> 데이터로는 RGB 이미지 $`ℐ \in ℝ^{3 \times 600 \times 960}`$, 스테레오 카메라 등에서 얻는 depth map $`ℐ_D \in ℝ^{1 \times 600 \times 960}`$, 64채널 LiDAR의 포인트 클라우드 $`𝒫 \in ℝ^{\le 60{,}000 \times 3}`$, BEV 공간의 계획 경로 $`\text{route} \in ℝ^{1 \times 64 \times 64}`$, 속도 $`v \in ℝ`$, 그리고 가속·조향각 형태의 행동 $`a \in ℝ^2`$ 를 얻는다.

&nbsp;

> We adopt the same setup as before for two distinct validation sets.

> 동일한 셋업을 두 종류의 검증(validation) 데이터에 적용한다.

&nbsp;

> For each town, we execute five 300-second long driving sessions with the following settings:

> 각 town에서 다음 설정으로 300초 길이 주행을 5회 실행한다.

&nbsp;

> $`𝒟_{\text{val}}^{\text{RL}}`$: This set uses the same cities and weather conditions as the training set. However, the driving routes are randomized.

> $`𝒟_{\text{val}}^{\text{RL}}`$: 학습과 동일한 도시·날씨이지만 주행 경로는 무작위화. 표현 학습(RL) 능력을 평가하는 친숙한 환경.

&nbsp;

> The goal is to evaluate the effectiveness of our model in representation learning (RL) in familiar environments.

> 목적은 익숙한 환경에서의 표현 학습(RL) 효과를 평가하는 것이다.

&nbsp;

> $`𝒟_{\text{val}}^{\text{DS}}`$: We maintain the same cities as in the training set but introduce different weather conditions.

> $`𝒟_{\text{val}}^{\text{DS}}`$: 학습과 동일한 도시지만 날씨를 변경해 도메인 이동(DS) 상황을 도입한다.

&nbsp;

> The driving routes are also randomized to evaluate the model performance under domain shifts (DS).

> 주행 경로도 무작위화하여 도메인 이동(DS) 하에서의 성능을 평가한다.

### Training Parameters

&nbsp;

> We sampled data at intervals of 0.2 seconds, creating sequences of length 12 to serve as training inputs.

> 0.2초 간격으로 데이터를 샘플링하여 길이 12의 시퀀스를 학습 입력으로 만든다.

&nbsp;

> All 12 frames were treated as known data.

> 12프레임 전부를 알려진 데이터로 취급한다.
- **학습 시 imagine 처리 — paper vs alpha 차이**: 이 문장은 학습 단계에서 12 프레임 전체가 observation (posterior 입력) 으로 처리됨을 의미. future prediction (imagine) 은 검증에서만 평가. alpha 는 학습 때부터 imagine 도 supervise 함.
  ```
  paper / upstream 2D:
    학습 forward : 전체 seq (12 or 6) 모두 posterior reconstruction loss
                   imagine_step 은 prior μ/σ 계산 (KL loss 용) 만, decoder X
    검증 forward : observe_and_imagine() — RF posterior + FH prior imagine,
                   FH 구간 imagine 결과로 future prediction metric 측정

  alpha 2D:
    학습 forward : 전체 seq posterior reconstruction + FH 구간 imagine 도
                   별도 호출 (trainer_muvo_2D.py:621-645 _observe_and_imagine
                   → model.imagine(state, future_horizon=fh))
                   → FH 구간 decoder 결과에도 reconstruction loss
                   (future_* prefix × WEIGHT_FUTURE=1.0)
    검증 forward : 동일 _observe_and_imagine 사용 (학습/검증 logic 같음)
  ```
- **시사점**: alpha 는 future prediction 을 학습 단계에서 강하게 supervise → paper 와 학습 셋업 불일치. paper 와 정량 비교를 위해서는 `WEIGHT_FUTURE=0` (옵션 A) 또는 training_step 에서 imagine 부분 제거 (옵션 B) 필요. 자세한 ablation 제안은 `improvement_plan/alpha_{1D,2D}_improvements.md` G13/G15 참조.

&nbsp;

> In the experiments containing voxel reconstructions, we reduced the length of sequences to 6 to speed up the training.

> voxel reconstruction을 포함하는 실험에서는 학습 속도를 위해 시퀀스 길이를 6으로 줄였다.

&nbsp;

> We trained with a batch size of 16 and the AdamW optimizer with a learning rate of $`10^{-4}`$ and a weight decay of $`0.01`$.

> 배치 크기 16, AdamW(LR$`=10^{-4}`$, weight_decay$`=0.01`$)로 학습한다.
- **AdamW vs Adam**: Adam에서 weight decay 처리를 개선한 변형. 기존 Adam은 L2 정규화를 그래디언트에 합쳐 momentum과 섞였지만, AdamW는 weight decay를 따로 분리 적용해 더 안정적·일반화 좋음. Transformer 학습에서 사실상 표준.
  ```
  Adam  : θ_t ← θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λ·θ_{t-1})  ← 결합
  AdamW : θ_t ← θ_{t-1} - lr · m̂_t / (√v̂_t + ε) - lr·λ·θ_{t-1}  ← 분리
                                                      ▲
                                                      weight decay 항을 별도로 적용
  ```

&nbsp;

> For validation, we used 6/4 frames as given observations, while 6/2 served as ground truth.

> 검증에서는 6/4 프레임을 관측으로, 6/2 프레임을 ground truth로 사용한다.

&nbsp;

> For all experiments, we use a pre-trained ResNet18 [72] as our baseline backbone.

> 모든 실험에서 사전 학습된 ResNet18 [72]을 기준 백본으로 사용한다.

---

## B. Sensor Fusion Strategies (L390–L515)

&nbsp;

> Several prior multimodal world models rely on naive fusion approaches [26], [32], [84].

> 기존 멀티모달 world model 다수는 단순(naive) 융합 방식 [26], [32], [84]에 의존한다.

&nbsp;

> For our experiments, we compare those to a transformer-based architecture.

> 우리는 이를 transformer 기반 구조와 비교한다.

&nbsp;

> To evaluate the effect of different sensor fusion strategies, we used metrics based on the evaluated modality.

> sensor fusion 전략별 효과를 평가하기 위해 모달리티별 지표를 사용한다.

&nbsp;

> For assessing the quality of image predictions, we use the common Peak Signal-to-Noise Ratio (PSNR) to assess average differences.

> 이미지 예측 품질은 일반적인 PSNR(Peak Signal-to-Noise Ratio)로 평균 차이를 평가한다.

&nbsp;

> We use the Chamfer Distance to evaluate the accuracy of point cloud predictions.

> 포인트 클라우드 예측 정확도는 Chamfer Distance로 평가한다.

&nbsp;

> For the predictions of 3D occupancy grids, we used the metrics Intersection over Union (IoU), Precision, and Recall.

> 3D occupancy grid 예측에는 IoU, Precision, Recall 지표를 사용한다.
- **PSNR / Chamfer Distance / IoU±**: 모달리티별 표준 지표.
  ```
  PSNR (RGB)        : 20·log10(MAX / √MSE)         ← 높을수록 좋음 (이미지 품질)
  Chamfer (LiDAR)   : Σ_x min_y ||x-y||² + Σ_y min_x ||x-y||²
                                                     ← 낮을수록 좋음 (점 집합 거리)
  IoU+  (occupancy) : |pred=1 ∩ gt=1| / |pred=1 ∪ gt=1|   ← 점유 voxel
  IoU−  (occupancy) : |pred=0 ∩ gt=0| / |pred=0 ∪ gt=0|   ← 빈 voxel
  ```

&nbsp;

> We differentiate between IoU+ for occupied voxels and IoU− for empty ones.

> IoU는 점유된 voxel에 대한 IoU+, 비어있는 voxel에 대한 IoU−로 나누어 표기한다.

### Encoders

&nbsp;

> For image features $`ℱ_c`$, we compare the standard encoder introduced in Sec. III-A to approaches that map features to BEV space [15], [24], [73], [85].

> 이미지 특징 $`ℱ_c`$ 에 대해, §III-A의 표준 인코더를 BEV 공간으로 특징을 매핑하는 [15], [24], [73], [85] 방식과 비교한다.

&nbsp;

> Here, features are first elevated into a 3D space.

> BEV 방식은 특징을 먼저 3D 공간으로 들어 올린다.

&nbsp;

> Then, these 3D feature voxels are aggregated into the BEV space, leading to image features $`ℱ_b \in ℝ^{C \times H_b \times W_b}`$.

> 이후 3D 특징 voxel을 BEV 공간으로 집계해 이미지 특징 $`ℱ_b \in ℝ^{C \times H_b \times W_b}`$ 를 얻는다.

&nbsp;

> We evaluate both lossless and lossy representations for point clouds.

> 포인트 클라우드에 대해서는 손실 없는(lossless) 표현과 손실 있는(lossy) 표현을 모두 평가한다.

&nbsp;

> We compare a range view-based representation with PointPillars [70] as an encoder, where point clouds are segmented into discrete pillars along the X and Y axes followed by data processing and feature extraction, resulting in a 2D BEV pseudo-image.

> range view 기반 표현을 PointPillars [70] 인코더(X·Y 축으로 포인트 클라우드를 이산 pillar로 분할 후 처리하여 2D BEV pseudo-image 생성)와 비교한다.
- **PointPillars vs Range View**: 둘 다 LiDAR 인코딩 방법.
  ```
  PointPillars (lossy):
    3D points ─grid on (X,Y)─▶ pillars (각 셀이 점들 묶음)
              ─PointNet─▶ pillar feature
              ─scatter─▶ 2D BEV pseudo-image (Z 정보 손실)

  Range View (lossless):
    3D points ─spherical (yaw, pitch)─▶ 2D image grid
              ─4 channels (x, y, z, r)─▶ 손실 없는 1:1 매핑
  ```

### Latent Space

&nbsp;

> In prior works, the latent space was commonly modeled as one-dimensional vectors [24], [80], which may limit model performance by introducing a representational bottleneck.

> 기존 연구에서 잠재 공간은 일반적으로 1D 벡터 [24], [80]로 모델링되었으며, 이는 표현 병목(representational bottleneck)을 초래해 성능을 제한할 수 있다.

&nbsp;

> We perform experiments with both a 1D and a 2D latent space.

> 우리는 1D와 2D 잠재 공간 모두로 실험한다.

&nbsp;

> In addition, we examine an additional perceptual loss [86] and a vision transformer backbone [87].

> 더불어 perceptual loss [86] 항과 vision transformer 백본 [87]의 효과도 조사한다.

&nbsp;

> Figure 4 shows four evaluation graphs. It is divided into the prediction performance for camera (a), lidar (b), and 3D occupancy (c,d) on $`𝒟_{\text{val}}^{\text{DS}}`$.

> Fig. 4는 카메라(a), LiDAR(b), 3D occupancy(c, d) 예측 성능을 $`𝒟_{\text{val}}^{\text{DS}}`$ 기준으로 4개의 그래프로 보여준다.

&nbsp;

> The 2D latent state significantly benefits predictions for camera images and spatial voxel occupancies, while lidar predictions do not see any benefit.

> **2D 잠재 상태(2D latent state)는 카메라 이미지와 공간 voxel occupancy 예측에 큰 이득을 주는 반면, LiDAR 예측에는 이득이 없다.**

&nbsp;

> This might be due to the fact that camera data is much more complex than lidar data.

> 이는 카메라 데이터가 LiDAR 데이터보다 훨씬 복잡하기 때문일 가능성이 있다.

&nbsp;

> Compared to the baseline 2D model (dark blue), we do not see a strong effect of utilizing a perceptual loss, as it produces visually poorer reconstructions and does not show any significant advantages.

> 기준 2D 모델(짙은 파랑)과 비교했을 때 perceptual loss는 시각적으로 더 나쁜 reconstruction을 만들고 의미 있는 이점도 없어, 강한 효과가 나타나지 않는다.

&nbsp;

> Using the vision transformer as an encoder does provide advantages for the prediction of camera images but shows no effect on other metrics.

> Vision transformer 인코더는 카메라 이미지 예측에는 이득을 주지만, 다른 지표에는 영향이 없다.

&nbsp;

> This shows that the 2D latent space itself provides the largest boost in performance, while other changes have little effect.

> 결과적으로 **2D 잠재 공간 자체가 가장 큰 성능 향상을 제공**하며, 다른 변경은 거의 효과가 없다.
- **paper의 headline 결과**: 2D latent 추가 = 카메라 PSNR 향상의 가장 큰 단일 요인. perceptual loss / ViT 같은 보조 트릭은 미미. 따라서 alpha-2D의 `RSSMTD + ConvGRU + TransformerDecoder` 토큰 상태 디자인이 paper §IV-B의 핵심 권고를 그대로 반영하는 셋업.
  ```
  ablation 결과 요약 (Fig. 4 기준):
    1D baseline           ─▶ 카메라 PSNR ★
    2D latent             ─▶ 카메라 PSNR ★★★  ← 가장 큰 향상
    2D + Perceptual Loss  ─▶ 효과 없음 (오히려 시각적으로 더 나쁨)
    2D + ViT backbone     ─▶ 카메라에만 약간, 다른 지표 변화 없음
  ```

### Fusion Methods

&nbsp;

> We compare a transformer-based sensor fusion approach, as described in Sec. III-B, with naive combinations of encoded 1D features from each sensor modality, as found in the literature.

> §III-B의 transformer 기반 sensor fusion을, 기존 연구의 각 모달리티 1D 특징의 단순 결합과 비교한다.

&nbsp;

> We perform experiments for both averaging features as well as concatenating them, followed by a fully connected layer.

> 특징을 평균(averaging)하거나, 이어 붙인 뒤 FC 레이어를 거치는(concatenation) 두 가지 방식을 모두 실험한다.

&nbsp;

> To generate such latent states, the output tokens are reshaped into their original shape after the encoding, namely $`ℱ^{\text{new}}_c \in ℝ^{C \times H_c \times W_c}`$ and $`ℱ^{\text{new}}_L \in ℝ^{C \times H_L \times W_L}`$.

> 이러한 잠재 상태를 만들기 위해, 인코딩 후 출력 토큰을 원래 형태인 $`ℱ^{\text{new}}_c \in ℝ^{C \times H_c \times W_c}`$ 와 $`ℱ^{\text{new}}_L \in ℝ^{C \times H_L \times W_L}`$ 로 reshape한다.

&nbsp;

> Each feature is then downsampled by convolutional layers, followed by pooling layers to get one-dimensional features $`f_d \in ℝ^D`$, which are subsequently concatenated and then passed through fully connected layers to reduce its dimensionality, producing the vector $`\mathrm{o}_t \in ℝ^D`$.

> 각 특징을 convolutional layer로 다운샘플링하고 pooling으로 1D 특징 $`f_d \in ℝ^D`$ 를 얻은 뒤, 이어 붙여 FC 레이어를 통해 차원을 줄여 최종 벡터 $`\mathrm{o}_t \in ℝ^D`$ 를 얻는다.

&nbsp;

> We evaluate the prediction performance of eight encoder-fusion combinations, as visible in Figure 3.

> Fig. 3과 같이 총 8가지 encoder-fusion 조합을 평가한다.
- **A-B-C 명명 8조합**: paper의 Fig. 3 비교 매트릭스. 2(PP/RV) × 2(BEV/WOB) × 2~3(AVG/FC/TR)의 조합을 다양하게 시도.
  ```
  A: LiDAR 인코딩  — PP (PointPillars) / RV (Range View)
  B: 이미지 처리   — BEV (BEV mapping) / WOB (Without BEV)
  C: Fusion 방식   — AVG (averaging) / FC (concat+FC) / TR (transformer)

  예시: "RV-WOB-TR" = Range View LiDAR + BEV 미사용 + Transformer fusion (paper 권고 셋업)
  ```

&nbsp;

> We follow a naming scheme A-B-C: A represents the method of processing point clouds: PP stands for the use of PointPillars as the encoder; RV indicates the conversion of point clouds into range view.

> 명명 규칙은 A-B-C: A는 포인트 클라우드 처리 방식 — PP는 PointPillars 인코더, RV는 range view 변환.

&nbsp;

> B denotes the approach of image processing: BEV implies mapping to BEV followed by feature extraction with a backbone; WOB denotes that no BEV mapping is performed.

> B는 이미지 처리 방식 — BEV는 BEV로 매핑 후 backbone 특징 추출, WOB는 BEV 매핑 없음.

&nbsp;

> C describes the method of sensor fusion: AVG stands for the averaging of 1D features; FC means that concatenation followed by a fully connected layer is performed; TR denotes that the transformer-based multi-head self-attention mechanism was used, as described in Sec. III-B.

> C는 sensor fusion 방식 — AVG는 1D 특징 평균, FC는 concat+FC, TR은 §III-B의 transformer 기반 multi-head self-attention.

&nbsp;

> In the following, we first discuss the effects on image predictions, followed by the effects on point cloud predictions.

> 다음에서 먼저 이미지 예측에 대한 영향, 이어서 포인트 클라우드 예측에 대한 영향을 논한다.

### Image Prediction

&nbsp;

> The impact of the different experiments on the quality of camera predictions is shown in Fig. 3 a) and 3 b).

> 실험별 카메라 예측 품질에 대한 영향은 Fig. 3 a), b)에 제시되어 있다.

&nbsp;

> We observe a drop in performance for all networks in the $`𝒟_{\text{val}}^{\text{DS}}`$ dataset compared to $`𝒟_{\text{val}}^{\text{RL}}`$, but the relative performance of different networks remains consistent across both datasets.

> 모든 네트워크에서 $`𝒟_{\text{val}}^{\text{DS}}`$ 성능은 $`𝒟_{\text{val}}^{\text{RL}}`$ 대비 떨어지지만, 네트워크 간 상대적 성능 순서는 두 데이터셋에서 일관된다.

&nbsp;

> Generally, the transformer-based architecture RV-WOB-TR performs on par or better compared to the other combinations, and range view-based lidar encodings show clear advantages over PointPillars.

> 일반적으로 transformer 기반 구조 RV-WOB-TR이 다른 조합과 동등하거나 더 우수하며, range view 기반 LiDAR 인코딩은 PointPillars 대비 명확한 이점을 보인다.

&nbsp;

> Methods with an additional BEV mapping of image features perform worse, and combinations with PointPillars suffer especially.

> 이미지 특징을 BEV로 추가 매핑하는 방법은 더 나쁜 성능을 보이며, 특히 PointPillars와 결합 시 더욱 악화된다.

&nbsp;

> We can see that the effectiveness of introducing a transformer-based architecture depends on the encoder used.

> Transformer 기반 구조의 효과는 함께 사용하는 인코더에 의존한다.

&nbsp;

> It outperforms other approaches when combined with a ResNet-18 for feature extraction.

> ResNet-18 기반 특징 추출과 결합하면 다른 접근을 능가한다.

&nbsp;

> In contrast, when combined with PP and BEV, its performance is lower than concatenating (FC) but higher than averaging (AVG).

> 반면 PP·BEV와 결합하면 성능이 FC보다 낮고 AVG보다 높다.

### Point Cloud Prediction

&nbsp;

> The impact of the different experiments on the quality of camera predictions is shown in Fig. 3 c) and 3 d).

> (포인트 클라우드) 예측에 대한 영향은 Fig. 3 c), d)에 제시된다. *(원문은 "camera"라고 표기되어 있으나 c/d 그래프 위치 및 문맥상 LiDAR 예측 의미)*

&nbsp;

> Examining the Chamfer Distance plots, where lower values mean better performance, we find no significant performance disparity between both validation datasets.

> 낮을수록 좋은 Chamfer Distance를 보면 두 검증 데이터셋 간 의미 있는 차이는 없다.

&nbsp;

> For $`𝒟_{\text{val}}^{\text{RL}}`$, the transformer-based architecture RV-WOB-TR performs on par or better compared to the other combinations.

> $`𝒟_{\text{val}}^{\text{RL}}`$ 에서 RV-WOB-TR은 다른 조합과 동등하거나 더 좋다.

&nbsp;

> However, on $`𝒟_{\text{val}}^{\text{DS}}`$, its performance drops.

> 그러나 $`𝒟_{\text{val}}^{\text{DS}}`$ 에서는 성능이 떨어진다.

&nbsp;

> As before, range view-based methods demonstrate superiority over PointPillars.

> 이미지 예측과 마찬가지로 range view 기반 방식이 PointPillars보다 우월하다.

&nbsp;

> Utilizing BEV features shows no clear disadvantage for this task.

> BEV 특징은 LiDAR 예측 과제에서 뚜렷한 불이익을 보이지 않는다.

&nbsp;

> We can see that transformer-based architectures generally outperform other fusion techniques.

> Transformer 기반 구조가 다른 fusion 기법을 일반적으로 능가한다.

&nbsp;

> We determine a transformer-based architecture with a 2D latent space and lossless range-view representations for point clouds as an optimal fusion strategy, while performance benefits are more pronounced for camera predictions.

> 따라서 **transformer 기반 + 2D 잠재 공간 + 손실 없는 range-view LiDAR**가 최적 fusion 전략이며, 카메라 예측에서 이점이 더 두드러진다.

---

## C. 3D Occupancy Prediction (L516–L587)

&nbsp;

> Next to analyzing fusion strategies, we are interested in the effects of also predicting more actionable 3D occupancies.

> Sensor fusion 분석에 더해, 행동에 유용한 3D occupancy를 함께 예측하는 효과도 살펴본다.

&nbsp;

> Our experiments analyze whether an occupancy model can benefit from a pre-trained model which was trained by only predicting camera and lidar data, as shown in Fig. 5.

> Fig. 5와 같이, 카메라+LiDAR만 예측하도록 사전 학습한 모델이 occupancy 학습에 도움이 되는지 분석한다.

&nbsp;

> Subsequently, we analyze if occupancy prediction improves the prediction of camera and lidar data, as shown in Fig. 6.

> 이어서 Fig. 6에서 occupancy 예측이 카메라·LiDAR 예측을 개선하는지 분석한다.

### 3D Occupancy Prediction

&nbsp;

> We perform experiments in three scenarios, as shown in Figure 5.

> Fig. 5에 표시된 세 가지 시나리오로 실험한다.

&nbsp;

> As we want to examine the effect of encoded knowledge of predicting camera and lidar data on 3D occupancy, we first train a model as a pre-trained starting point that predicts camera and lidar data alone for 50,000 steps.

> 카메라+LiDAR 예측에서 학습된 지식이 occupancy에 미치는 효과를 보기 위해, 먼저 카메라·LiDAR만 50,000 step 동안 학습해 사전 학습 모델을 만든다.

&nbsp;

> For the first scenario, we employ the pre-trained model but freeze (PTF) all of its weights so that only the weights of the voxel decoder are trained.

> 첫 시나리오(PTF): 사전 학습 모델의 모든 가중치를 동결(freeze)하고 voxel 디코더 가중치만 학습한다.

&nbsp;

> This approach allows us to assess the impact of fine-tuning only the voxel-specific aspects of the model while keeping the rest of the network, in particular all encoders, constant to evaluate if any information about a discrete geometry of the world is already encoded based on camera and lidar data.

> 이는 인코더 등 나머지 네트워크는 그대로 둔 채 voxel 관련 부분만 미세 조정하여, 이미 카메라·LiDAR 학습에 의해 이산적(discrete) 세계 기하 정보가 인코딩됐는지 평가할 수 있게 한다.

&nbsp;

> For the second scenario, the pre-trained weights were used as a starting point, but the entire network was open (PTO) for weight updates during training.

> 두 번째 시나리오(PTO): 사전 학습 가중치를 시작점으로 사용하되 전체 네트워크를 갱신(open)한다.

&nbsp;

> Here, we analyze how the pre-trained weights influence the learning process when the whole network adapts and evolves during training.

> 전체 네트워크가 학습 중에 적응·진화할 때 사전 학습 가중치가 학습 과정에 어떤 영향을 주는지를 본다.

&nbsp;

> For the third scenario, no pre-training (NPT) is utilized, and we train the network from scratch.

> 세 번째 시나리오(NPT): 사전 학습 없이 처음부터 학습한다.
- **PTF / PTO / NPT**: 사전 학습 전략 비교. 결과: PTO 초기 우세 → NPT가 후반 Precision 추월 → PTF는 voxel decoder만 학습해도 점진적 개선 (인코더에 공간 정보가 이미 어느 정도 들어 있음을 시사).
  ```
  PTF (Pre-Trained Frozen): cam+lidar 사전학습 모델 동결 + voxel head만 fine-tune
                            → 인코더가 occupancy를 "알고 있는지" 검사용

  PTO (Pre-Trained Open)  : cam+lidar 사전학습 시작점 + 전체 네트워크 갱신
                            → 사전 학습이 학습 가속에 도움이 되는지 검사

  NPT (No Pre-Training)   : 처음부터 cam + lidar + occupancy 학습
                            → 사전 학습 없이 같은 성능에 도달 가능한지 검사
  ```

&nbsp;

> In Figure 5 we observe that the model trained from scratch (NPT) exhibits a similar performance on both validation datasets across all four metrics, while the other two models using pre-trained weights (PTF and PTO) generally performed better on $`𝒟_{\text{val}}^{\text{RL}}`$ than on $`𝒟_{\text{val}}^{\text{DS}}`$ across three metrics, excluding IoU−.

> Fig. 5에서 NPT는 4개 지표 모두 두 검증 데이터셋에서 유사한 성능을 보이며, PTF와 PTO는 IoU−를 제외한 세 지표에서 $`𝒟_{\text{val}}^{\text{RL}}`$ 이 $`𝒟_{\text{val}}^{\text{DS}}`$ 보다 일반적으로 더 좋다.

&nbsp;

> Interestingly, for IoU− we observe an opposite behavior, where the models perform better on $`𝒟_{\text{val}}^{\text{DS}}`$.

> 흥미롭게도 IoU−에서는 반대로 $`𝒟_{\text{val}}^{\text{DS}}`$ 성능이 더 좋다.

&nbsp;

> This is attributed to voxel occupancy grid predictions focusing more on occupied grids.

> 이는 voxel occupancy grid 예측이 점유 grid 쪽에 더 집중하기 때문이다.

&nbsp;

> Since voxel grids are mostly empty, models on $`𝒟_{\text{val}}^{\text{DS}}`$ tend to predict more noise, leading to lower IoU− scores.

> Voxel grid는 대부분 비어 있어 $`𝒟_{\text{val}}^{\text{DS}}`$ 에서는 모델이 더 많은 noise를 예측하기 쉽고, 이로 인해 IoU− 점수가 낮아진다. *(원문 흐름상 "more noise … lower IoU−"는 $`𝒟_{\text{val}}^{\text{RL}}`$ 쪽이 더 낮음을 의미; $`𝒟_{\text{val}}^{\text{DS}}`$ 의 IoU−가 더 높게 나타나는 것과 일치)*

&nbsp;

> Comparing PTO to NPT, the PTO model showed advantages early on, supporting the idea that pre-trained weights contribute valuable spatial knowledge.

> PTO와 NPT를 비교하면 PTO는 초기에 우세하여, 사전 학습 가중치가 유용한 공간 지식을 제공한다는 주장을 뒷받침한다.

&nbsp;

> However, in the later stages of training, the NPT model overtook the PTO model in Precision, while the PTO model remained superior for IoU− and Recall.

> 학습 후반에는 Precision에서 NPT가 PTO를 추월하지만, IoU−와 Recall에서는 여전히 PTO가 우세하다.

&nbsp;

> This indicates that the non-pre-trained model adopts a more conservative strategy for 3D occupancy prediction.

> 이는 사전 학습 없는 모델이 3D occupancy 예측에서 더 보수적인 전략을 채택함을 시사한다.

&nbsp;

> When we examine the PTF scenario, although the model underperformed compared to the other two, its performance improved over time by only training the voxel decoder.

> PTF 시나리오는 두 시나리오보다 성능이 낮지만, voxel 디코더만 학습하는 것으로도 시간이 지남에 따라 성능이 개선된다.

&nbsp;

> This improvement underscores that the pre-trained weights already contain some, however limited, spatial information, indicating that the model partially integrates image and point cloud features to form spatial voxel features even when trained only on these two modalities.

> 이는 사전 학습 가중치가 제한적이지만 공간 정보를 이미 담고 있음을 보여주며, 이미지·LiDAR만 학습해도 모델이 부분적으로 두 모달리티 특징을 결합해 공간 voxel 특징을 형성함을 의미한다.

&nbsp;

> As learning 3D occupancy is computationally intensive, we conclude that pre-training strategies on only camera and lidar data are generally recommendable, as they both speed up training and show overall superior performance.

> 3D occupancy 학습은 계산 비용이 크므로, 일반적으로 카메라+LiDAR만 사용한 사전 학습 전략이 학습 가속과 전반적 성능 우위 측면에서 권장된다.

### Sensor Data Predictions

&nbsp;

> We perform experiments to determine whether knowledge encoded through occupancy can be leveraged by lidar and camera predictions, as shown in Figure 6.

> Fig. 6과 같이, occupancy 학습에서 인코딩된 지식을 LiDAR·카메라 예측이 활용할 수 있는지 실험한다.

&nbsp;

> Based on the chamfer Distance for point clouds and the PSNR metric for images, we observe only slightly increased performance gains for both modalities when occupancy prediction is included, with a more pronounced benefit for camera predictions under the $`𝒟_{\text{val}}^{\text{RL}}`$ setting.

> 포인트 클라우드의 Chamfer Distance와 이미지의 PSNR을 보면, occupancy 예측을 포함했을 때 두 모달리티 모두 **약간의** 성능 향상만 관찰되며, $`𝒟_{\text{val}}^{\text{RL}}`$ 에서 카메라 예측 이점이 보다 뚜렷하다.

---

## Fig. 5 캡션 (Pre-Training)

> Fig. 5 — 카메라·LiDAR 50,000 step 사전 학습이 3D occupancy 예측에 미치는 영향. $`𝒟_{\text{val}}^{\text{RL}}`$ 과 $`𝒟_{\text{val}}^{\text{DS}}`$ 에서 평가. 녹색은 사전 학습 없는 기준선, 보라색은 사전 학습 가중치 동결(PTF), 파랑은 가중치 개방(PTO).

## Fig. 6 캡션 (Occupancy → Sensor)

> Fig. 6 — 3D occupancy 예측이 카메라·LiDAR 예측 품질에 미치는 영향. $`𝒟_{\text{val}}^{\text{RL}}`$ 과 $`𝒟_{\text{val}}^{\text{DS}}`$ 에서 평가.

---

## 핵심 용어/수치 정리표

| 용어/기호 | 영문 정의 | 한국어 의미 | 등장 위치 |
|---|---|---|---|
| Self-supervised | No labels needed | 라벨 불요 자기지도 학습 | L312–L313 |
| Multi-scale (1, 2, 4) | Downsample ratios | 다중 해상도 1/2/4 손실 | L314–L315 |
| $`ℒ^{\text{img}}`$ (L1) | Image RGB loss | 이미지 L1 | L316–L319 |
| $`ℒ^{\text{pxyz}}`$ (L2) | Range-view xyz L2 | 포인트 클라우드 좌표 L2 | L323 |
| $`ℒ^{\text{pr}}`$ (L1) | Range-view range L1 | 거리 채널 L1 | L324 |
| Voxel $`192 \times 192 \times 64`$, 0.5 m | Binary occupancy grid | 이진 occupancy 격자 | L325–L326 |
| SCAL [42] $`ℒ^{V,\text{scal}}`$ | Scene-Class Affinity voxel loss | voxel 손실 | L328–L329 |
| Total loss eq. (1) | $`\sum_i \lambda_i \cdot`$ per-modality weighted sum | 총 손실식 | L330–L341 |
| CARLA 4 towns $`\times`$ 4 weather | Train env | 학습 환경 | L345–L348 |
| 10 FPS, 25 runs $`\times`$ 300 s | Sampling/episode setup | 수집 설정 | L348–L350 |
| 300,000 frames | Dataset size | 데이터 크기 | L350 |
| Image $`600 \times 960`$ | Input image size | 입력 이미지 크기 | L351 |
| LiDAR $`\le 60{,}000`$ points, 64 ch | LiDAR specs | LiDAR 사양 | L353 |
| route $`64 \times 64`$ | Planned route map | BEV 경로 맵 | L354 |
| Action $`a \in ℝ^2`$ | Acceleration + steering | 가속+조향 | L355–L356 |
| $`𝒟_{\text{val}}^{\text{RL}}`$ | Same train env, randomized routes | 표현 학습 검증 | L361–L364 |
| $`𝒟_{\text{val}}^{\text{DS}}`$ | Different weather, same cities | 도메인 이동 검증 | L365–L369 |
| Sample 0.2 s, seq 12 | Sampling interval, seq len | 샘플링 간격·길이 | L370–L372 |
| Seq 6 (voxel exp) | Reduced seq | voxel 실험 단축 | L373–L374 |
| Batch 16, AdamW, LR $`=10^{-4}`$, WD $`=0.01`$ | Training hyperparams | 학습 하이퍼파라미터 | L375–L376 |
| 6/4 obs, 6/2 GT | Validation split | 검증 관측/예측 비율 | L377–L378 |
| ResNet18 [72] | Baseline backbone | 기준 백본 | L378–L379 |
| PSNR | Image metric | 이미지 지표 | L396–L397 |
| Chamfer Distance | PCD metric | 포인트 클라우드 지표 | L398 |
| IoU+ / IoU− | Occupied / empty voxel IoU | 점유/비점유 voxel IoU | L401–L402 |
| Encoder PP vs RV | LiDAR encoding (PointPillars vs range view) | LiDAR 인코딩 비교 | L412–L416 |
| Encoder BEV vs WOB | Image to BEV vs without BEV | 이미지 BEV 사용/미사용 | L406–L411 |
| Fusion AVG/FC/TR | Averaging/concat-FC/Transformer | 융합 방식 3종 | L463–L466 |
| 8 combos | Total ablation count | 총 조합 수 | L454 |
| Naming A-B-C | Lidar-Image-Fusion | 명명 규칙 | L455–L466 |
| **RV-WOB-TR** | Optimal combo | **최적 조합** | L487, L512–L515 |
| 1D vs 2D latent | Latent shape comparison | 잠재 공간 차원 비교 | L417–L437 |
| Perceptual Loss (PL) [86] | Optional perceptual term | 선택적 perceptual 손실 | L422 |
| Vision Transformer (VIT) [87] | Optional backbone | 선택적 ViT 백본 | L422 |
| **2D latent biggest boost** | Headline §IV-B verdict | **2D 잠재 공간이 핵심** | L436–L438 |
| PTF/PTO/NPT | Pre-trained-frozen / open / no pre-train | 세 가지 occupancy 시나리오 | L530–L544 |
| 50,000 steps pre-train | Pre-training duration | 사전 학습 step 수 | L530 |
| Occupancy → sensor (Fig 6) | Reverse direction effect | occupancy가 sensor 예측에 미치는 효과 | L579–L587 |
| Slight occupancy benefit | Camera/LiDAR slight gain | 약간의 sensor 예측 이득 | L583–L587 |

---

## 알파(alpha26) 코드 관점 메모

- **§IV-A 손실식 ↔ alpha**:
  - alpha의 RGB L1 multi-scale (factors 1, 2, 4): paper와 일치 ✓ (`trainer_muvo.py` discount factor [1, 2, 4]).
  - alpha의 LiDAR L2 xyz + L1 range: paper와 일치 ✓ (단 `WEIGHT_LIDAR_RE=0.0`로 꺼져 있음 — improvement plan P0).
  - alpha의 LiDAR empty L1(=smooth L1 to 0): paper에는 명시 없음, alpha 자체 추가.
  - alpha에는 SCAL voxel loss 미존재 (alpha occupancy 헤드 자체 없음).
  - alpha 추가 항: KL (probabilistic) 및 SSIM(현재 weight 0).
- **§IV-A 데이터/시퀀스**:
  - paper: 0.2s 샘플링, seq 12 (또는 seq 6 for voxel). 
  - alpha: `SAMPLE_EVERY_N=2`, `FRAME_STEP=5`, `CARLA_HZ=20` → 0.5s 간격(~2 Hz)으로 paper 대비 더 듬성. alpha seq = RF(4) + FH(4 또는 2) = 8 또는 6. paper의 seq 6 voxel 실험과 일치하는 경우는 alpha-2D (FH=2).
- **§IV-B 8 combos vs alpha**:
  - alpha 두 갈래 모두 **RV-WOB**: ✓ (range view + no BEV). 
  - 융합: alpha-1D는 transformer 3층 + post-flat MLP → RV-WOB-TR'(약식) 변형.
  - alpha-2D는 transformer 3층 fusion + RSSMTD decoder → paper의 RV-WOB-TR + 2D latent 조합과 가장 가까움.
  - paper의 PP/BEV/AVG/FC 비교군은 alpha에는 없음.
- **§IV-B latent space**:
  - alpha-1D: 1D latent (RSSM, $`(B, S, 512) + (B, S, 256)`$).
  - alpha-2D: 2D latent (RSSMTD, $`(B, S, 256, 369)`$) — paper의 headline 결론(2D latent 최대 효과)을 반영하는 디자인.
  - alpha에는 PL/VIT 옵션 없음 (paper에서도 큰 효과 없음으로 결론).
- **§IV-C voxel pre-training/sensor 영향**:
  - alpha는 occupancy 헤드가 없어 PTF/PTO/NPT 실험 자체 불가.
  - 만약 alpha occupancy 헤드를 추가한다면 paper 권고(camera+lidar pre-train → voxel fine-tune)가 직접 적용 가능 → improvement plan에서 P2.
- **§IV-B validation**:
  - alpha 검증은 단일 dataloader (val_rl/val_ds 분리는 일부 있으나 weather/route shift 적용 아님).
  - paper의 $`𝒟_{\text{val}}^{\text{RL}}`$ vs $`𝒟_{\text{val}}^{\text{DS}}`$ 비교는 alpha에 부재.
