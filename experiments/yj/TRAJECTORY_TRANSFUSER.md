# MUVO 2D + TransFuser Trajectory Head — 변경 문서

작성일: 2026-05-25
대상 코드: `alpha26/archive/yj/{config,models,trainer}_muvo_2D.py`
참고 논문: **TransFuser — Imitation with Transformer-Based Sensor Fusion for Autonomous Driving**
(Prakash et al., CVPR 2021 / Chitta et al., TPAMI 2022)

---

## 0. 한 줄 요약

기존 MUVO 디코더(RGB / LiDAR 재구성 + RSSM transition + action head)를 **하나도 제거하지 않고**,
TransFuser의 **GRU autoregressive waypoint decoder** 와 **waypoint→이미지 투영**만
순수 추가(additive)로 붙였다. → "미래 trajectory를 예측해 front-camera 이미지에 그려넣기".

---

## 1. "디코더를 살리면서 trajectory만 추가" — 가능한가?

**가능하다.** 아래가 핵심 근거다.

- MUVO의 출력 latent state(`RSSMTD` posterior의 `hidden_state`, `sample`, shape `(B,S,C,N)`)는
  이미 image·lidar·policy 디코더가 공유해서 읽는다. trajectory head도 **같은 state를 읽기만** 하면
  되므로, 기존 디코더의 입력/출력/가중치에 아무 영향이 없다.
- TransFuser가 trajectory를 만드는 방식 = "융합 feature 1개 → 작은 GRU로 미래 waypoint 회귀"라서,
  추가되는 파라미터·연산이 작고 기존 그래프와 분리된다.

그래서 변경은 전부 **추가**다. 제거·치환은 없다.

### 단, 데이터에 한 가지 제약이 있다 (반드시 인지)

현재 Arrow 데이터셋 컬럼은 다음뿐이다:

```
run_id, frame, timestamp, image_front, lidar, steer, throttle, speed_kmh
```

→ **ego pose / location / GPS / yaw 컬럼이 없다.** TransFuser 원본은 기록된 ego 위치를 그대로
waypoint **정답(GT)** 으로 쓰지만, 본 데이터엔 위치 정답이 없다. 따라서:

> **waypoint GT를 `speed_kmh` + `steer`로 kinematic bicycle 모델을 적분해 *근사 생성*한다.**
> 이 GT는 근사치다 (§5의 한계 참조). 정확한 GT가 필요하면 CARLA 재수집 시
> ego `Transform`(location+rotation)을 저장하는 것이 정공법이다.

이 근사 GT 생성은 **데이터로더를 건드리지 않고**(반환 dict 그대로 유지) 트레이너에서 계산한다.
즉 `data_muvo_2D.py`는 **무수정**이다.

---

## 2. TransFuser에서 가져온 것 / 바꾼 것

| 요소 | TransFuser 원본 | 본 통합(yj) |
|------|-----------------|-------------|
| 입력 feature | RGB+LiDAR(BEV) fusion → 512-d 벡터 | MUVO RSSM posterior state를 token축 mean-pool 후 `cat(h,s)` → **512-d** (= 2×256) |
| waypoint decoder | 단층 **GRU** + Linear, **차분(differential)** waypoint를 autoregressive 예측 | **동일** (`GRUWaypointDecoder`, `nn.GRUCell` + Linear, 누적 차분) |
| goal/target point | GPS 목표점을 GRU 입력에 concat | **생략** (데이터에 goal/route 신호 없음). 필요 시 `input_size=4`로 확장 가능 |
| waypoint 손실 | waypoint **L1** | **동일** (masked L1) |
| 제어 변환 | waypoint → **PID** → throttle/steer | **미구현** (이미 별도 `PolicyDecoder` action head가 있음. 추론 제어는 향후) |
| GT 출처 | 기록된 ego 위치 | **kinematic 근사** (speed+steer 적분) — 데이터 제약 |

---

## 3. 파일별 변경 (전부 추가)

### 3.1 `config_muvo_2D.py`

- `cfg.MODEL.WAYPOINT` 네임스페이스 신규:
  ```python
  WAYPOINT = SimpleNamespace(
      ENABLED=True,
      N_WAYPOINTS=2,         # 예측 future waypoint 수 (default = FUTURE_HORIZON)
      GRU_HIDDEN=64,         # TransFuser waypoint GRU hidden size
      DT=0.5,                # 샘플 간 간격 [s] = 1/EFFECTIVE_HZ
      WHEELBASE=2.85,        # [m] 차량 wheelbase 근사 (kinematic GT용)
      MAX_STEER_RAD=1.221,   # [rad] ≈ 70° 최대 조향각 근사 (kinematic GT용)
      Y_RIGHT_POSITIVE=True, # ego +y가 오른쪽(CARLA)인지
  )
  ```
- `cfg.DATA.CAMERA` 네임스페이스 신규 (waypoint→이미지 투영용 calibration):
  ```python
  CAMERA = SimpleNamespace(FOV_DEG=90.0, HEIGHT_M=1.6, X_OFFSET_M=1.2, PITCH_DEG=0.0)
  ```
  > ⚠️ 실제 CARLA front-camera rig 값에 맞춰야 투영이 정확하다. 기본값은 일반적 가정치.
- `cfg.LOSSES.WEIGHT_WAYPOINT = 1.0` 추가.

### 3.2 `models_muvo_2D.py`

- **신규 클래스 `GRUWaypointDecoder`** (PolicyDecoder 다음):
  scene feature → `feat_proj`로 GRU hidden 초기화 → `n_waypoints` 스텝 동안
  `GRUCell(input=현재 xy) → Linear → Δxy` 를 누적해 차분 waypoint 예측.
  반환 `(M, n_waypoints, 2)`. (TransFuser waypoint 디코더와 동일한 구조)
- **`Model.__init__`**: `self.waypoint_decoder` 추가 (`cfg.MODEL.WAYPOINT.ENABLED` 게이트).
  `in_channels = 2*EMBEDDING_DIM = 512`. 비활성 시 `None`.
- **`Model._decode_state`**: action_pred 산출 직후, `hidden_state`/`sample`를 token축으로
  mean-pool하고 `cat` → `(B*S, 512)` feature → `waypoint_decoder` → `out["waypoints_pred"]`
  `(B, S, n_wp, 2)`. **기존 image/lidar/policy 디코더 코드는 무수정.**

> `_decode_state`는 관측 경로(`forward`, posterior, S=RF+FH 또는 RF)와 미래 경로(`imagine`, prior, FH)
> 모두에서 호출되므로, 두 경우 모두 `waypoints_pred`가 자동으로 생성된다.

### 3.3 `trainer_muvo_2D.py`

- **`WorldModelTrainer.__init__`**: `weight_waypoint`, `n_waypoints`, `wp_dt`,
  `wp_wheelbase`, `wp_max_steer` 로드.
- **신규 메서드 `compute_gt_waypoints(action, speed)`**: kinematic bicycle 적분으로
  미래 waypoint GT 근사. 반환 `gt (B,S,n_wp,2)` + `mask (B,S)`.
  프레임 `t`의 GT는 `t`의 ego frame 기준 `t+1..t+n_wp` 위치(`+x` 전방, `+y` 우측).
  미래 프레임이 부족한 끝쪽 프레임(`t+n_wp > S-1`)은 `mask=0`.
- **`compute_loss`**: action loss 다음에 **waypoint masked L1** 항 추가
  (`weight_waypoint>0` 이고 `waypoints_pred`/`action`/`speed`가 있을 때만).
- **`compute_eval_metrics`**: trajectory 지표 **ADE/FDE**(평균/최종 displacement error, meter) 추가.
- **신규 함수 `_waypoints_to_pixels(...)`**: ego-frame waypoint를 핀홀 카메라 모델 +
  지면(z=0) 가정으로 이미지 픽셀에 투영 (`cfg.DATA.CAMERA` 사용).
- **신규 함수 `save_trajectory_overlay_figure(pl_module, dataset, run_cfg, run_name, ...)`**:
  마지막 관측 프레임의 front-camera 이미지에 **GT(초록)·예측(빨강) trajectory를 오버레이** 저장.
  → `results/figures/<run_name>_trajectory_s<idx>.png`.

> 기존 손실(`rgb_*`, `lidar_*`, `probabilistic`, `action`)과 figure 함수는 무수정.

---

## 4. 데이터 흐름 / 텐서 shape

```
batch.image / lidar / speed
   → encode_fuse_sequence → z (B,S,256,368)
   → RSSMTD → posterior {hidden_state, sample} (B,S,256,369)
        ├─ [기존] image_decoder  → rgb_{1,2,4}
        ├─ [기존] lidar_decoder  → lidar_reconstruction_{1,2,4}
        ├─ [기존] policy_decoder → action_pred (B,S,2)
        └─ [신규] mean-pool tokens → cat(h,s) (B*S,512)
                  → GRUWaypointDecoder → waypoints_pred (B,S,n_wp,2)

GT(트레이너): action(B,S,2)+speed(B,S) → compute_gt_waypoints → gt(B,S,n_wp,2), mask(B,S)
손실: masked L1( waypoints_pred , gt )           # 가중치 WEIGHT_WAYPOINT
지표: ADE/FDE [m]
시각화: waypoints → _waypoints_to_pixels → front-camera 이미지 오버레이
```

상수(현 cfg): `RF=4, FH=2, S=6`, `EMBEDDING_DIM=256` → feature 512,
`N_WAYPOINTS=2`, `DT=0.5s`. `S=6, n_wp=2`이면 유효 GT 프레임은 `t=0..3` (프레임당 4개).

---

## 5. 한계 / 다음 단계

1. **waypoint GT가 근사치다.** pose 컬럼이 없어 `speed+steer` kinematic 적분으로 만든다.
   `WHEELBASE`/`MAX_STEER_RAD`는 차량 추정치라 곡선·저속에서 오차가 있다.
   → **정공법**: CARLA 재수집 시 ego `Transform` 저장 후 GT를 그 위치로 대체
   (`compute_gt_waypoints`만 교체하면 나머지 파이프라인은 그대로).
2. **카메라 calibration 의존.** `cfg.DATA.CAMERA`(FOV/높이/오프셋/pitch)가 실제 rig와
   다르면 투영이 어긋난다. `muvo/run/data_collect` 카메라 설정값으로 맞출 것.
3. **goal/route 미사용.** TransFuser는 목표점을 GRU에 넣지만 여기선 생략.
   route/goal 신호가 생기면 `GRUWaypointDecoder` input을 4-d로 확장.
4. **FH=2로 짧다(예측 1s).** 더 긴 horizon은 `FUTURE_HORIZON`↑ + window index 재생성 필요.
   `N_WAYPOINTS`는 FH와 독립적으로 늘릴 수 있으나 GT 유효 프레임이 줄어든다.
5. **PID 제어 미구현.** waypoint → 제어 변환은 추론 배포 시 추가(이미 별도 action head 존재).

---

## 6. 사용법

학습/평가는 기존 entrypoint 그대로 — `cfg.MODEL.WAYPOINT.ENABLED=True`(기본)면 자동으로
waypoint head·손실·지표가 활성화된다. 끄려면 `ENABLED=False` 또는 `WEIGHT_WAYPOINT=0.0`.

trajectory 오버레이 이미지 저장(예):

```python
from trainer_muvo_2D import save_trajectory_overlay_figure
save_trajectory_overlay_figure(pl_module, dataset, cfg, run_name="my_run", sample_idx=0)
```

로깅되는 신규 지표: `*/waypoint`(loss), `*/waypoint_ade`, `*/waypoint_fde`.
