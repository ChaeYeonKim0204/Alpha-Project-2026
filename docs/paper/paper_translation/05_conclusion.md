# V. Conclusion — 영-한 대역본

- **원문 위치**: `muvo_paper.txt` L593–L619
- **한 줄 요지 (KR)**: BEV는 높이 정보 부재로 인한 병목이 맞고, **손실 없는 LiDAR 표현 + 표준 transformer fusion + 2D 잠재 공간**이 카메라+LiDAR 융합에 유효함을 실험적으로 확인. 3D occupancy 학습은 카메라+LiDAR 사전 학습으로부터 이득을 받으며, 반대로 occupancy 예측은 sensor 예측에 **약간의** 개선만 제공.

---

## 본문 (문장 단위 영-한)

&nbsp;

> We have presented an extensive set of experiments on sensor fusion strategies for predictive world models in autonomous driving.

> 본 논문에서는 자율주행을 위한 예측형(predictive) world model의 sensor fusion 전략에 대해 광범위한 실험을 제시하였다.

&nbsp;

> In addition, we have examined the effects of additionally predicting 3D occupancy.

> 또한 3D occupancy를 추가로 예측했을 때의 효과도 함께 분석하였다.

&nbsp;

> This way, the effects of using typical sensor setups combined with learning actionable occupancy data can be better understood.

> 이를 통해 전형적인 센서 구성과 행동에 활용 가능한(actionable) occupancy 데이터 학습을 결합했을 때의 효과를 더 잘 이해할 수 있게 되었다.

---

&nbsp;

> We were particularly interested in the question of whether the introduction of a BEV feature representation, as typical in the literature [15], [16], might act as a bottleneck, as it misses height information.

> 우리는 특히 기존 연구 [15], [16]에서 일반적인 BEV 특징 표현이 높이(height) 정보 부재로 인해 병목으로 작용할 수 있는지에 관심을 두었다.

&nbsp;

> Our experiments demonstrate that lossless lidar representations with a standard transformer-based fusion and an increased 2D latent space are indeed beneficial in the case of camera-lidar fusion.

> 우리의 실험은 **손실 없는 LiDAR 표현 + 표준 transformer 기반 fusion + 확장된 2D 잠재 공간** 조합이 카메라-LiDAR 융합에 실제로 유리함을 입증한다.
- **세 가지 권고 셋업 (요약)**: paper의 conclusion에서 한 번에 묶어 제시.
  ```
  ① Lossless LiDAR representation : Range View (1:1 매핑) > PointPillars (lossy)
  ② Transformer fusion             : multi-head self-attention > AVG / FC
  ③ 2D latent state                : token grid 형태 > 1D flat vector
  ```
- **alpha 매핑**: alpha-2D는 ①②③ 모두 반영(`models_muvo_2D.py:RSSMTD + SensorFusionTransformer + range-view $32 \times 1024$`). alpha-1D는 ①②만 반영 (③은 1D baseline).

---

&nbsp;

> When analyzing the effects of introducing actionable but computationally intensive spatial 3D occupancy predictions, our experiments show that occupancy predictions benefit from more efficient pre-training with only camera and lidar predictions.

> 행동에는 유용하지만 계산 비용이 큰 3D occupancy 예측을 도입했을 때의 효과를 분석한 결과, occupancy 예측은 카메라+LiDAR 예측만으로 효율적으로 사전 학습한 가중치로부터 이득을 얻는 것으로 나타났다.
- **권장 학습 흐름 (alpha에 적용한다면)**:
  ```
  Stage 1: cam + lidar reconstruction 만 50,000 step (cheap)
              │
              ▼ 인코더 + RSSM이 공간 정보를 어느 정도 학습
              │
  Stage 2: voxel decoder 헤드 추가 + 전체 네트워크 fine-tune (PTO 시나리오)
              │
              ▼ occupancy 예측 학습 가속 + 최종 정확도 향상
  ```

&nbsp;

> In addition, we observed that occupancy prediction leads to minor improvements for both camera and lidar predictions.

> 또한 occupancy 예측이 카메라·LiDAR 예측 양쪽에 **약간의** 개선을 가져옴을 관찰하였다.

---

&nbsp;

> For future work, we would be interested in performing large-scale experiments with real-world data as our approach does not require labels, leading to a better understanding of the implications of utilizing simulated data only.

> 향후 연구로는 본 접근이 라벨을 요구하지 않는 점을 활용해 실제(real-world) 데이터로 대규모 실험을 수행해, 시뮬레이션 데이터만 사용했을 때의 함의를 더 잘 이해하는 방향에 관심이 있다.

&nbsp;

> This might include a more complex sensor setup, raising the need to examine computational efficiency more thoroughly.

> 더 복잡한 센서 구성도 포함될 수 있어, 계산 효율성을 보다 면밀히 검토할 필요가 생긴다.

---

## 핵심 용어/수치 정리표

| 용어/기호 | 영문 정의 | 한국어 의미 | 등장 위치 |
|---|---|---|---|
| Predictive world model | Model that predicts future obs given actions | 행동 조건 미래 관측 예측 모델 | L594–L595 |
| BEV bottleneck (height info missing) | Height-information loss in BEV | 높이 정보 부재로 인한 BEV 병목 | L601–L603 |
| Lossless LiDAR representation | Range view (vs PointPillars/voxel) | 손실 없는 LiDAR 표현 (range view) | L604 |
| Standard transformer fusion | Multi-head self-attention fusion | 표준 transformer 융합 | L604–L605 |
| **Increased 2D latent space** | Token-shaped latent $`\sum_j T_j \times C`$ | **확장된 2D 잠재 공간** | L605 |
| **Occupancy benefits from pre-training** | Pre-train cam+lidar → fine-tune voxel | **사전 학습→voxel** 유리 | L609–L611 |
| Minor occupancy → sensor benefit | Slight cam/lidar prediction gain | sensor 예측에 약간의 이득 | L611–L613 |
| Future: real-world large-scale | Label-free advantage exploited at scale | 실제 데이터 대규모 실험 | L614–L617 |

---

## 알파(alpha26) 코드 관점 메모

- alpha-2D 디자인의 정당성: paper conclusion이 명시한 "transformer fusion + 2D latent" 권고와 alpha-2D의 `RSSMTD + TransformerDecoder` 구조가 정확히 일치 — alpha-2D는 paper의 권고를 그대로 구현한 셋업이라 볼 수 있다.
- alpha-1D의 의의: paper §IV-B가 "2D latent significantly benefits camera, not lidar"라고 결론지었으므로, alpha-1D는 카메라 예측 측면에서 paper 기준 비교적 약한 baseline. 단 LiDAR-only 평가에서는 1D-2D 차이가 크지 않다는 paper 결론이 alpha의 single-window overfit 결과(1D > 2D)와 부분적으로 호환.
- alpha에 occupancy 부재 → paper conclusion의 "minor improvement" 효과를 검증할 수 없음. 향후 alpha에 voxel head 추가 시 paper의 pre-train 전략(50,000 step cam+lidar 후 voxel head)을 그대로 적용 가능.
- Future work 항(real-world 데이터)은 alpha 범위 밖. 그러나 alpha의 라벨 없는 학습 흐름(Arrow dataset → RGB+LiDAR raw recon)은 paper의 "label-free → real-world 확장" 비전과 동일 선상.
