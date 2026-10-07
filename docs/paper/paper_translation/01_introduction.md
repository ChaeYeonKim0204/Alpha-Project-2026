# I. Introduction — 영-한 대역본

- **원문 위치**: `muvo_paper.txt` L22–L63
- **한 줄 요지 (KR)**: 카메라/LiDAR 멀티모달 world model이 BEV 병목 없이도 동작할 수 있는지, 그리고 3D occupancy 예측을 추가했을 때의 효과를 광범위한 sensor fusion 실험으로 분석하는 것이 본 논문의 목적.

---

## 본문 (문장 단위 영-한)

&nbsp;

> In machine learning, recent world models like Cosmos [1] or Genie [2], [3] have demonstrated the capability to take sequences of high-resolution input images, conditioned on instructions or actions, and generate future images, illustrating possible future scenarios.

> 기계학습 분야에서 Cosmos [1], Genie [2], [3] 같은 최신 world model들은, 명령어(instruction) 또는 행동(action)을 조건으로 받아 고해상도 입력 이미지 시퀀스로부터 가능한 미래 시나리오를 보여주는 미래 영상을 생성할 수 있음을 입증해 왔다.
- **Cosmos / Genie**: 영상 생성형 world model 흐름. Cosmos(NVIDIA)는 시뮬레이션 환경 + 물리 일관성, Genie(DeepMind)는 비디오에서 행동을 추출/조건화하는 action-conditioned 영상 생성. 모두 대규모 모델·데이터.
  ```
  [image sequence] + [action/instruction]
        │
        ▼
  large pretrained video model
        │
        ▼
  future image frames (high-res, temporally coherent)
  ```

&nbsp;

> In autonomous driving, the majority of such world models focus on camera-based inputs [4]–[11] and some that work in lidar-space [12]–[14].

> 자율주행 영역에서는 이러한 world model들의 대부분이 카메라 기반 입력 [4]–[11]에 집중하고 있으며, 일부 연구가 LiDAR-공간에서 동작 [12]–[14]한다.

&nbsp;

> These works neglect typical sensor setups of autonomous vehicles.

> 그러나 이러한 연구들은 실제 자율주행차의 전형적인 센서 구성을 충분히 반영하지 못한다.

&nbsp;

> Only two recent works leverage both camera and lidar data [15], [16].

> 카메라와 LiDAR 데이터를 동시에 활용하는 최근 연구는 단 두 편 [15], [16]뿐이다.

&nbsp;

> However, they rely on Bird's-Eye-View (BEV) features as part of their sensor fusion strategy, which is an acknowledged bottleneck due to missing height information [15].

> 그러나 이들은 sensor fusion 전략의 일부로 Bird's-Eye-View(BEV) 특징에 의존하는데, BEV는 높이(height) 정보가 누락되어 있어 알려진 병목 지점으로 지적된다 [15].
- **BEV (Bird's-Eye-View) bottleneck**: 3D 공간을 위에서 본 평면 격자로 압축하는 표현. (X, Y) 평면만 남기고 Z(높이)는 합쳐버려 신호등·표지판·다리·고가차도 같은 위·아래 정보가 손실됨.
  ```
  3D world (X, Y, Z) ──collapse Z──▶ BEV grid (X, Y, C)
     ▲                                  ▲
     │  높이 정보 유지                  │  높이 정보 손실 ← 병목
  ```
- **3D occupancy**: 3D 격자 (X, Y, Z) 각 셀의 점유 여부 {0/1}을 직접 예측하는 표현. BEV와 달리 높이 정보 보존. 본 논문이 occupancy 헤드를 추가하는 동기.

&nbsp;

> Finally, world models that predict future 3D occupancies have been proposed, which are highly actionable but rely on visual inputs only [17] or operate in the occupancy space alone [18].

> 한편 미래의 3D occupancy를 예측하는 world model들도 제안되었으며, 후속 행동 계획에 매우 유용(actionable)하지만 이들은 시각 입력(visual)만 사용 [17]하거나 occupancy 공간 안에서만 동작 [18]하는 한계를 가진다.

---

&nbsp;

> In this work, we choose a simple model architecture to perform an extensive set of experiments with varying sensor fusion strategies that do not rely on BEV features and compare them against multiple BEV-based baselines.

> 본 연구에서는 BEV 특징에 의존하지 않는 다양한 sensor fusion 전략을 광범위하게 실험하기 위해 단순한 모델 구조를 선택하고, 이를 여러 BEV 기반 기준선(baseline)들과 비교한다.

&nbsp;

> This way we want to shine some light on how sensor fusion strategies impact the prediction quality of world models.

> 이를 통해 sensor fusion 전략이 world model의 예측 품질에 어떤 영향을 미치는지를 조명하고자 한다.

&nbsp;

> While the benefit of leveraging both camera and lidar data is well-established [19]–[21], previous works did not evaluate the impact on future predictions.

> 카메라와 LiDAR 데이터를 함께 활용하는 것의 이점은 이미 잘 알려져 있지만 [19]–[21], 이전 연구들은 그 효과가 미래 예측(future prediction)에 미치는 영향을 따로 평가하지 않았다.

&nbsp;

> Our experiment setup takes camera images and lidar point clouds as inputs to predict future observations conditioned on actions.

> 우리 실험 설정은 카메라 이미지와 LiDAR 포인트 클라우드를 입력으로 받아, 행동(action)을 조건으로 미래 관측을 예측하도록 구성한다.

&nbsp;

> In addition, we examine the effect of an additional decoder component for the prediction of 3D occupancies.

> 추가로, 3D occupancy 예측을 위한 별도 디코더 컴포넌트를 두었을 때의 효과도 함께 조사한다.
- **decoder per modality**: 같은 잠재 상태 (h, s)로부터 모달리티별 별도 디코더(이미지 ConvT, LiDAR ConvT, voxel 3D ConvT)를 둠. 멀티태스크 학습이지만 인코더-잠재공간은 공유.
  ```
                 ┌─▶ camera decoder ──▶ predicted RGB
  (h, s) ──┬─────┼─▶ lidar decoder  ──▶ predicted range-view
            │     └─▶ voxel decoder  ──▶ predicted occupancy (선택)
  ```

&nbsp;

> Our contributions are:

> 본 연구의 기여는 다음과 같다:

&nbsp;

> • A multimodal world model with geometric representations that does not rely on BEV features as a bottleneck

> • BEV 특징을 병목으로 사용하지 않는, 기하학적(geometric) 표현 기반의 멀티모달 world model.

&nbsp;

> • Extensive evaluation of sensor fusion strategies based on the prediction quality of the world model

> • world model의 예측 품질을 기준으로 한 sensor fusion 전략의 광범위한 평가.

&nbsp;

> The code and model weights are available on GitHub.

> 코드와 모델 가중치는 GitHub에 공개되어 있다.

---

## Fig. 1 캡션

&nbsp;

> Qualitative output of a sensor fusion experiment with occupancy prediction activated. The predictions shown for camera and lidar sensors and 3D occupancy are based on past camera and lidar inputs.

> occupancy 예측을 활성화한 sensor fusion 실험의 정성적(qualitative) 결과. 카메라 및 LiDAR 센서와 3D occupancy에 대해 보이는 예측은 모두 과거의 카메라+LiDAR 입력으로부터 생성된 것이다.

---

## 핵심 용어/수치 정리표

| 용어/기호 | 영문 정의 | 한국어 의미 | 등장 위치 (line) |
|---|---|---|---|
| Cosmos [1] | Recent video-generation world model | 최근의 영상 생성 world model | L23 |
| Genie [2], [3] | Action-conditioned video world model | 행동 조건부 영상 world model | L23 |
| Camera-based inputs [4]–[11] | World models using only RGB | RGB만 쓰는 기존 driving world model 군 | L28 |
| Lidar-space [12]–[14] | World models operating only in LiDAR | LiDAR만 쓰는 세 연구 | L29 |
| BEV (Bird's-Eye-View) | 2D top-down feature representation | 평면 투영 특징 | L32–33 |
| BEV bottleneck | Loss of height information when collapsing 3D → BEV | BEV로 압축할 때 높이 정보 손실 | L34 |
| 3D occupancy world model [17], [18] | World models predicting voxel occupancy futures | 미래 occupancy 예측 모델 | L35–38 |
| Action-conditioned | Future predicted given action input | 행동 조건부 예측 | L48 |
| Geometric representations | Voxel-based 3D geometry features | voxel 기반 3D 표현 | L59 |
| Contribution 1 | No-BEV multimodal world model | BEV 없는 멀티모달 world model | L59–60 |
| Contribution 2 | Extensive sensor fusion evaluation | 광범위한 sensor fusion 평가 | L61–62 |

---

## 알파(alpha26) 코드 관점 메모

- alpha26은 **BEV/voxel/route 모두 미사용** → paper의 "no-BEV" 디자인 의도와 일치하나, paper가 정의하는 BEV-기반 baseline(비교군)은 alpha에 없음.
- alpha-1D/2D 모두 **action 조건부**이지만, alpha-2D만 action loss를 명시적으로 사용 (port_plan.md 참조).
- 3D occupancy 디코더가 alpha에 없음 → 향후 improvement plan에서 P2로 고려 가능 (paper는 occupancy를 sensor 예측 보조용으로 사용).
