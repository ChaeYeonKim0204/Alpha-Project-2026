# 0. Abstract — 영-한 대역본

- **원문 위치**: `muvo_paper.txt` L7–L21
- **한 줄 요지 (KR)**: 자율주행 world model 분야에서 카메라/LiDAR 멀티모달 융합 전략과 3D occupancy 예측이 sensor data 예측 품질에 어떤 영향을 주는지 실험적으로 분석한 연구.

---

## 본문 (문장 단위 영-한)

&nbsp;

> World models for autonomous driving have the potential to dramatically improve the reasoning capabilities of today's systems.

> 자율주행을 위한 world model은 현재 시스템의 추론(reasoning) 능력을 극적으로 향상시킬 잠재력을 가진다.
- **world model**: 관측을 잠재 상태로 인코딩하고, 행동을 조건으로 미래 관측을 시뮬레이션하는 생성 모델. 자율주행에서는 "차가 이렇게 움직이면 다음 프레임 카메라/LiDAR는 어떻게 보일까"를 예측.
  ```
  obs_t ──encoder──▶ z_t  ──+ action_t──▶ z_{t+1} ──decoder──▶ obs_{t+1}
  ```

&nbsp;

> However, most works focus on camera data, with only a few that leverage lidar data or combine both to better represent autonomous vehicle sensor setups.

> 그러나 대부분의 연구는 카메라 데이터에 집중되어 있고, LiDAR 데이터를 활용하거나 두 센서를 결합해 실제 자율주행 차량의 센서 구성을 더 잘 반영하는 연구는 극히 일부에 불과하다.

&nbsp;

> In addition, raw sensor predictions are less actionable than 3D occupancy predictions, but there are no works examining the effects of combining both multimodal sensor data and 3D occupancy prediction.

> 또한 원시(raw) 센서 예측은 3D occupancy 예측에 비해 후속 행동(actionable) 수립에 덜 유용하지만, 멀티모달 센서 데이터와 3D occupancy 예측을 함께 결합했을 때의 효과를 분석한 연구는 존재하지 않는다.
- **raw sensor prediction vs 3D occupancy**: 원시 센서 예측은 픽셀/포인트 그대로 복원하지만 충돌·경로 판단에는 후처리가 필요. 3D occupancy는 격자 형태로 어디가 점유돼 있는지를 직접 알려줘 planning에 바로 사용 가능.
  ```
  raw RGB    : (3, H, W)  픽셀 RGB값          ─ 시각적·풍부, 의미는 미분리
  raw LiDAR  : (4, H, W)  range-view xyz+r    ─ 거리 정보, 객체 분리는 미해결
  3D occupancy: (X, Y, Z) {0=empty, 1=occupied} ─ 점유 격자, planning에 직결
  ```

&nbsp;

> In this work, we perform a set of experiments with a MUltimodal World Model with Geometric VOxel representations (MUVO) to evaluate different sensor fusion strategies to better understand the effects on sensor data prediction.

> 본 연구에서는 기하학적 VOxel 표현을 갖춘 MUltimodal World Model(MUVO)을 이용해 다양한 센서 융합(sensor fusion) 전략을 평가하는 일련의 실험을 수행하고, 이를 통해 sensor data 예측 품질에 미치는 영향을 분석한다.
- **sensor fusion**: 여러 센서(카메라+LiDAR 등)의 특징을 하나의 잠재 표현으로 결합하는 과정. 본 논문은 융합 방식(평균/concat/transformer)과 인코더 종류(BEV/no-BEV, PointPillars/Range-View)를 8가지 조합으로 비교.
  ```
  cam_feat   ─┐
              ├─[fusion module]─▶ fused token (latent)
  lidar_feat ─┘
  ```

&nbsp;

> We also analyze potential weaknesses of current sensor fusion approaches and examine the benefits of additionally predicting 3D occupancy.

> 동시에 현재의 sensor fusion 접근 방식이 갖는 잠재적 약점을 분석하고, 3D occupancy를 추가로 예측했을 때의 이점도 함께 검토한다.

---

## 핵심 용어/수치 정리표

| 용어/기호 | 영문 정의 | 한국어 의미 | 등장 위치 (line) |
|---|---|---|---|
| World model | Generative model that predicts future observations conditioned on actions | 행동 조건부 미래 관측을 생성하는 모델 | L7 |
| Sensor fusion | Combining multiple sensor modalities (cam + LiDAR) | 다중 센서(카메라+LiDAR) 융합 | L17 |
| MUVO | MUltimodal World Model with Geometric VOxel representations | 본 논문 제안 모델 약자 | L16–17 |
| Raw sensor prediction | Direct prediction of RGB image / LiDAR point cloud | 카메라 영상·LiDAR 원본 예측 | L12 |
| 3D occupancy | Voxelized geometric scene occupancy | 3D 격자 점유 예측 | L12–14 |
| Geometric VOxel | Volumetric voxel-based 3D geometry representation | 기하학적 voxel 기반 3D 표현 | L16 |
| Multimodal | Multiple sensor modalities (image + LiDAR) | 카메라+LiDAR 등 다중 모달 | L11, L14 |
| Reasoning capability | Capacity for future scenario inference | 미래 시나리오 추론 능력 | L8 |
| Actionable | Useful as direct input for downstream planning | 후속 의사결정에 바로 활용 가능한 | L12 |

---

## 알파(alpha26) 코드 관점 메모

- alpha26 (1D / 2D 둘 다) 은 **occupancy 헤드 없음** — paper Abstract의 "3D occupancy 추가 예측" 효과는 alpha 구현에서는 검증되지 않음.
- alpha 구현은 카메라+LiDAR (range-view) 멀티모달 raw prediction까지만 다룸 → Abstract 첫 두 문장의 흐름만 부분 일치, 마지막 두 문장(occupancy)은 미반영.
