# MUVO Paper — 통합 요약 (6개 섹션 종합)

원문: `/home/carol/chaeyeon-kim/muvo_paper.txt` (862 lines).
세부 문장 대역본은 `00_abstract.md` ~ `05_conclusion.md` 참조.

---

## 1. 한 줄 요약 (KR)

MUVO는 카메라+LiDAR(range view) 멀티모달 입력을, **BEV 매핑 없이** transformer로 융합하고 **2D 토큰 형태 잠재 공간**의 RSSM으로 미래를 예측한 뒤, RGB·LiDAR·선택적 3D occupancy로 디코드하는 자기지도 world model이다. 핵심 실험 결론은 (1) **RV-WOB-TR + 2D latent**가 카메라 예측 품질에서 가장 큰 이득을 주고 (2) **occupancy 예측은 카메라+LiDAR 사전 학습으로부터 가속**되며 (3) 반대로 occupancy를 추가해도 sensor 예측은 약간만 개선된다는 것.

---

## 2. 섹션별 핵심 요지

| 섹션 | 핵심 요지 (KR) |
|---|---|
| **Abstract** | 자율주행 world model에서 카메라/LiDAR 멀티모달 + 3D occupancy 결합 효과를 처음으로 체계적으로 분석. BEV 의존 없는 멀티모달 셋업으로 fusion 전략과 occupancy 예측의 효과를 실험. |
| **I. Introduction** | 기존 driving world model은 카메라 단일 모달이 다수, 멀티모달은 두 편(BEVWorld/HoloDrive)뿐이며 모두 BEV 병목 보유. MUVO는 BEV 미사용 멀티모달을 단순 구조로 광범위 평가. 기여 2가지: (a) BEV-free 멀티모달 + geometric voxel world model, (b) sensor fusion 전략의 광범위한 평가. |
| **II. Related Work** | World model 계열(라벨/자기지도/시퀀스 모델), LiDAR/occupancy 단일 모달, BEV 기반 멀티모달(BEVWorld·HoloDrive), Scene Completion(MonoScene 등), Forecasting(action-free) 등 4갈래 정리. MUVO 차별점: 라벨 없음 + BEV 없음 + 멀티모달 + action-conditioned. |
| **III. World Model** | MILE 기반 단순 구조. (A) Observation Encoder: cam $`600 \times 960`$ + LiDAR cylindrical projection($`\le 60\text{K}`$ pts → range view) → ResNet18 + multi-layer feature. (B) Multimodal Fusion: 2D sinusoidal PE + 학습 가능 sensor embedding, $`k`$-layer transformer encoder. (C) Transition: 1D 대신 **$`C \times \sum_j T_j`$ 형태 2D 잠재 상태**, FC→Conv 교체 GRU($`f_{\theta}`$), prior/posterior는 transformer decoder + 학습 가능 query. (D) Decoder: 모달리티별 토큰 분할 → reshape → ConvT(이미지/LiDAR는 2D, occupancy는 3D). |
| **IV. Evaluation** | (A) 자기지도 multi-scale (1/2/4) L1+L2 + SCAL; CARLA 300K frame, 0.2s 간격 seq 12 (voxel 시 seq 6); batch 16, AdamW LR$`=10^{-4}`$. 두 검증셋 ($`𝒟_{\text{val}}^{\text{RL}}`$/$`𝒟_{\text{val}}^{\text{DS}}`$). (B) 8조합(A-B-C) 비교: **RV-WOB-TR + 2D latent 최적**, BEV+PP는 더 나쁨. PL/VIT는 영향 적음. (C) PTF/PTO/NPT 세 시나리오: PTO가 초기 우세, NPT가 후반 Precision 추월. occupancy → sensor 예측 개선은 $`𝒟_{\text{val}}^{\text{RL}}`$ 에서 카메라에 약간 두드러짐. |
| **V. Conclusion** | BEV 병목 입증. RV + transformer + 2D latent가 fusion 최적. occupancy는 cam+lidar 사전 학습으로부터 효율 이득. occupancy 예측이 sensor 예측에 약간의 이득. Future: 실제 데이터 대규모 실험, 더 복잡한 sensor 셋업, 계산 효율 분석. |

---

## 3. 통합 정량 결과 표

### 3.1 §IV-A 학습/검증 셋업

| 항목 | 값 |
|---|---|
| Towns | Town01, Town03, Town04, Town06 |
| Weather | Clear Noon, Wet Noon, Hard Rain Noon, Clear Sunset |
| FPS | 10 |
| Runs per town | $`25 \times 300\text{s}`$ |
| Total frames | 300,000 |
| Image input | $`3 \times 600 \times 960`$ |
| Depth (stereo) | $`1 \times 600 \times 960`$ |
| LiDAR | $`\le 60{,}000`$ points, 64 vertical ch → range view |
| Route map (BEV) | $`1 \times 64 \times 64`$ |
| Speed $`v`$ | scalar |
| Actions $`a`$ | 2 (accel, steering) |
| Sampling interval | 0.2 s |
| Sequence length | 12 (default) / 6 (voxel) |
| Validation split | obs 6 / GT 6 (default), obs 4 / GT 2 (voxel) |
| Batch | 16 |
| Optimizer | AdamW |
| LR / WD | $`10^{-4}`$ / $`0.01`$ |
| Backbone | ResNet18 pretrained |
| Voxel grid | $`192 \times 192 \times 64`$, 0.5 m |
| Validation sets | $`𝒟_{\text{val}}^{\text{RL}}`$ (same env, random route), $`𝒟_{\text{val}}^{\text{DS}}`$ (different weather) |

### 3.2 §IV-A 손실 항

| 손실 | 종류 | 적용 대상 |
|---|---|---|
| $`ℒ^{\text{img}}`$ | L1 multi-scale (1/2/4) | RGB 예측 |
| $`ℒ^{\text{pxyz}}`$ | L2 multi-scale | LiDAR range view xyz 채널 |
| $`ℒ^{\text{pr}}`$ | L1 multi-scale | LiDAR range view $`r`$ 채널 |
| $`ℒ^{V,\text{scal}}`$ | SCAL [42] | 3D voxel occupancy |
| 총 식 (1) | $`\sum_i \lambda_i \left( \lambda_{\text{img}} ℒ_i^{\text{img}} + \lambda_{\text{pcd}} (ℒ_i^{\text{pxyz}} + ℒ_i^{\text{pr}} + ℒ_i^{\text{pcd}}) + \lambda_V ℒ_i^{V,\text{scal}} \right)`$ | 가중 합 |

### 3.3 §IV-B Sensor Fusion 8조합 (A-B-C 명명)

| 조합 | LiDAR | Image | Fusion | 카메라 PSNR | LiDAR Chamfer |
|---|---|---|---|---|---|
| RV-WOB-TR | Range View | no BEV | Transformer | **최고/동급 최고** | $`𝒟_{\text{val}}^{\text{RL}}`$ 최고, $`𝒟_{\text{val}}^{\text{DS}}`$ 에서 떨어짐 |
| RV-WOB-FC | Range View | no BEV | FC | 차상위 | – |
| RV-WOB-AVG | Range View | no BEV | Average | 하위 | – |
| RV-BEV-TR | Range View | BEV | Transformer | RV-WOB-TR보다 낮음 | – |
| PP-WOB-TR | PointPillars | no BEV | Transformer | 더 낮음 | 더 낮음 |
| PP-WOB-FC | PointPillars | no BEV | FC | PP-WOB-TR보다 좀 더 높음(이미지) | – |
| PP-BEV-TR | PointPillars | BEV | Transformer | 가장 낮은 군 | – |
| PP-BEV-FC | PointPillars | BEV | FC | TR보다 약간 높음 (이미지) | – |

> 결론: RV는 PP보다, WOB는 BEV보다, TR은 AVG보다 일반적으로 좋음. **RV-WOB-TR + 2D latent**이 최적.

### 3.4 §IV-B 잠재 공간 비교 (Fig. 4)

| 변형 | 카메라 | LiDAR | Voxel | 메모 |
|---|---|---|---|---|
| 1D baseline | 낮음 | 거의 동일 | 낮음 | 기준선 |
| **2D latent (dark blue)** | **큰 이득** | 차이 없음 | **큰 이득** | 핵심 결론 |
| 2D + perceptual loss (PL) [86] | 거의 동일 또는 시각적으로 더 나쁨 | – | – | 효과 없음 |
| 2D + ViT (MobileVit-V2) [87] | 약간 이득 | – | – | 카메라에만 약간 효과 |

### 3.5 §IV-C Occupancy Pre-training (Fig. 5)

| 시나리오 | 약자 | 가중치 처리 | 결과 |
|---|---|---|---|
| Pre-Trained Frozen | PTF | 사전 학습 모델 동결 + voxel 디코더만 학습 | 처음에는 약하나 시간 지나며 개선; 인코더에 일부 spatial 정보 잠재 |
| Pre-Trained Open | PTO | 사전 학습 모델 시작점 + 전체 갱신 | 초기 우세; 후반 Precision은 NPT에 추월, IoU−/Recall 우세 유지 |
| No Pre-Training | NPT | 처음부터 학습 | 두 검증셋 성능 유사; 후반 Precision 최고 |

→ 결론: cam+lidar 50K-step 사전 학습 후 occupancy 학습이 권장.

### 3.6 §IV-C Sensor 예측에 대한 Occupancy 효과 (Fig. 6)

| 모달리티 | 지표 | with occupancy | 비교 |
|---|---|---|---|
| 카메라 | PSNR | 약간 상승 ($`𝒟_{\text{val}}^{\text{RL}}`$ 에서 보다 뚜렷) | 작은 이득 |
| LiDAR | Chamfer | 약간 상승 | 작은 이득 |

---

## 4. Paper가 alpha 구현에 시사하는 점 (improvement_plan 사전 요약)

> 자세한 변경 제안은 `docs/improvement_plan/alpha_{1D,2D}_improvements.md` 참고.

1. **2D latent state**: paper §IV-B의 headline (카메라 예측 향상 최대). alpha-2D(`RSSMTD`)가 이미 반영 → 검증/튜닝이 우선. alpha-1D는 검증/baseline 용도로 유지.
2. **Lossless LiDAR (Range View)**: alpha 두 갈래 모두 이미 range view 사용 ✓.
3. **No BEV**: alpha 두 갈래 모두 BEV 없음 ✓.
4. **Transformer fusion**: alpha 두 갈래 모두 ✓ (3-layer encoder).
5. **Multi-scale L1+L2 손실**: alpha-1D/2D 모두 RGB L1 multi-scale 적용 ✓. 단 **`WEIGHT_LIDAR_RE=0.0`로 LiDAR 손실 꺼져 있음** → P0 이슈. paper 식 (1)의 $`ℒ^{\text{pxyz}} + ℒ^{\text{pr}}`$ 을 켜야 함.
6. **Self-supervised**: alpha 두 갈래 모두 자기지도 ✓ (라벨 없음).
7. **3D occupancy head + SCAL loss**: alpha 둘 다 부재 → P2 후보. paper의 PTO 전략(cam+lidar pre-train → voxel head 추가 + 전체 갱신)이 그대로 적용 가능.
8. **Validation split ($`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$)**: alpha 검증은 single set. paper처럼 weather/route shift validation을 도입하면 generalization 평가 강화 → P1.
9. **Sequence length 12 (또는 6 for voxel)**: alpha-2D는 seq 6 (RF=4, FH=2) — paper voxel exp과 일치 ✓. alpha-1D는 seq 8 — paper baseline seq 12와 차이.
10. **Image input $`600 \times 960`$**: alpha는 $`320 \times 768`$ 로 축소. 데이터 해상도가 다른 만큼 paper 절대 PSNR 수치와 비교 불가.
11. **Action loss**: paper에는 명시 없음(observation 예측 모델). alpha-2D는 추가 도입(WEIGHT_ACTION=1.0); paper 모델과는 diverge하나 alpha 디자인의 의도된 차이.

---

## 5. Paper-Code 트레이서빌리티 매트릭스 (요약)

| Paper 요소 | alpha-1D 위치 | alpha-2D 위치 | upstream 위치 |
|---|---|---|---|
| §III.A Observation Encoder | `models_muvo.py:FPNDecoder + SensorFeatureConv` | `models_muvo_2D.py:FPNDecoder` | `muvo/muvo/models/mile.py + common.py` |
| §III.B Transformer Fusion | `models_muvo.py:SensorFusionTransformer` 3L | `models_muvo_2D.py:SensorFusionTransformer` 3L | `muvo/muvo/models/mile.py` + transformer block |
| §III.C Transition (1D RSSM) | `models_muvo.py:RSSM` $`(B, S, D)`$ | — | `muvo/muvo/models/transition.py:RSSM` |
| §III.C Transition (2D RSSM) | — | `models_muvo_2D.py:RSSMTD + ConvGRUCellGlo + RepresentationModelTD` | `muvo_2d/muvo/models/transition_td.py:RSSMTD` |
| §III.D Decoder (cam/LiDAR) | `models_muvo.py:SensorDecoder` | `models_muvo_2D.py:TokenConvDecoder2D` | `muvo/muvo/models/common.py:ConvDecoder`, `muvo_2d/muvo/models/decoder.py:ConvDecoder2D` |
| §III.D Decoder (occupancy) | 부재 | 부재 | `muvo/muvo/models/common.py:VoxelDecoder1` |
| §IV-A 손실식 (1) | `trainer_muvo.py:compute_loss` (LiDAR 항 꺼짐) | `trainer_muvo_2D.py:compute_loss` (LiDAR 항 꺼짐) | `muvo/muvo/losses.py + trainer.py` |
| §IV-A 데이터 파이프라인 | `data_muvo.py:MUVODataset` | `data_muvo_2D.py:MUVODataset + PixelAugmentation` | `muvo/muvo/data/dataset.py` |
| §IV-B 8 fusion combos | 단일 조합(RV-WOB-TR-like) | 단일 조합(RV-WOB-TR + 2D latent) | upstream 두 갈래 모두 단일 조합 |
| §IV-B 1D vs 2D latent | alpha-1D 자체 | alpha-2D 자체 | `muvo` vs `muvo_2d` |
| §IV-C occupancy PTF/PTO/NPT | 불가 | 불가 | upstream 1D만 voxel decoder 보유 |
| §IV-A validation split | single (`val_rl`/`val_ds` 분리되나 동일 분포) | 동일 | upstream은 $`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$ 분리 |
