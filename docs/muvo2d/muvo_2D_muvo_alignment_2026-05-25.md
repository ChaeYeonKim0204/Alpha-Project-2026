# MUVO 2D — MUVO 재현 정렬 + LiDAR 데이터 분석 (2026-05-25)

커밋 `aa7fa16`(코드 리뷰 수정) **이후**의 변경. 목표: alpha 2D-token RSSM 포트를
upstream MUVO와 **메커니즘적으로 정합**(파라미터 값·해상도·BEV/voxel 제거 등 의도적
적응은 제외)시키는 것.

대상 파일: `scripts/model_variants/{config,data,models,trainer}_muvo_2D.py`

---

## 1. MUVO 정렬 변경

### 1.1 Decoder를 sample-only로 (`models_muvo_2D.py`)
- 전: `TokenConvDecoder2D`가 `cat(hidden_state, sample)`(in=2C)를 디코드.
- 후: posterior/prior의 **sample만** 디코드(in=C). upstream `ConvDecoder2D`(posterior.sample만 사용)와 일치.
- `_decode_state(sample)`, `image/lidar_decoder(in_channels=embedding_n_channels)`.

### 1.2 KL = MUVO `ProbabilisticLoss` (`trainer_muvo_2D.py`, `config`)
- **t=0 anchor 추가**: t≥1은 `KL(posterior‖prior)`, t=0은 `KL(posterior_t0 ‖ N(0,1))`.
  state는 token-shape `(B,S,C,N)` — latent 채널 C(dim=-2) 합산, 나머지 평균.
- `WEIGHT_PROBABILISTIC` 1e-2 → **1e-3**.
- `KL_FREE_BITS_ENABLED` True → **False** (upstream은 free-bits 없음).
- 참고: upstream `first_kl`은 t=0 mu와 t=1 sigma를 섞는 off-by-one 인덱싱 quirk가 있으나,
  여기서는 t=0의 mu·sigma로 정확히 계산(의도 동일, 버그 미복제).

### 1.3 RGB/LiDAR loss = MUVO `SpatialRegressionLoss` (`trainer_muvo_2D.py`, `config`)
- per-element 오차를 **채널(dim=-3) 합산 후 전 픽셀 평균**. xyz=L2, depth·RGB=L1.
- LiDAR **valid-mask 제거**(전 픽셀 supervise) + 별도 **empty-depth loss 제거**.
- 정리: `WEIGHT_LIDAR_EMPTY`(config), `self.weight_lidar_empty`, 미사용 `_masked_loss` 제거.
- 신규 헬퍼 `_spatial_regression_loss(pred, target, norm)`.
- 영향: 같은 weight에서도 RGB·xyz loss가 ~채널수배로 커짐 = upstream과 동일 스케일.

### 1.4 ~~LIDAR_SCALE 40 → 50~~ → **40 유지 (revert)** (`config`)
- ⚠️ **정정(2026-06-01)**: 이 40→50 변경은 *착오*였고 40으로 되돌림.
- 변경 당시 근거였던 "upstream `LIDAR_RE.SCALE=50` 복사"는 **상수만 복사**한 것. 정규화의 *특성*을
  복사하려면 센서 사거리 차이를 반영해야 함:
  - MUVO 센서 `range=100m`, `SCALE=50` → 정규화 depth ∈ **[0, 2.0]** (100/50).
  - alpha 센서 `range=80m`(데이터 제공자 명시, 데이터도 80.00m에 하드캡) → 같은 [0,2] 특성 재현 = `80/2 = **40**`.
- `/50`은 alpha를 [0,1.6]로 **under-fill** → MUVO 정규화 특성과 불일치 + `WEIGHT_LIDAR_RE`(MUVO의 [0,2] 데이터 기준 튜닝) 전이도 깨짐.
- §2의 "/50이 [0,2]에 들어옴" 판정은 *오버플로우*만 봤지 MUVO가 100m라 자기 데이터를 [0,2]까지 채운다는 점을 놓쳐 /40과 구별 못 함.

### 1.5 empty cell depth 0 → −1 (`data_muvo_2D.py`)
- `point_cloud_to_range_view`: 빈 셀 depth를 **−1 sentinel**로(`rv[3].fill(-1.0)`), xyz는 0 유지,
  유효 점이 덮어씀. **정정(2026-06-01)**: "동일"의 기준을 *정규화 분포 일치*로 통일 →
  empty depth를 scale 무관하게 MUVO와 동일한 **−0.02**(`MUVO_EMPTY_DEPTH_NORM = -1/50`)로 둔다.
  `__getitem__`에서 `/scale` 후 `rv[3]<0` 셀을 −0.02로 set (raw −1 fill은 empty marker 용도로만 유지). xyz empty=0은 이미 일치.
- `range_view_valid_mask`(`>0`)는 빈 셀(−0.02)을 그대로 제외 → Chamfer/지표 영향 없음.
- 근거: scale(40)을 "MUVO 정규화 특성 일치"로 정했으면 empty도 같은 기준이어야 일관됨. 효과 크기는 작지만(학습되는 상수) MUVO 입력 분포 충실 재현이 목적.

### 1.6 scheduled sampling 활성화 (`config`)
- `TRANSITION.USE_DROPOUT` True, `DROPOUT_PROBABILITY` 0.15.
- 동작: nn.Dropout이 아니라, 학습 중 매 timestep(t>0) **15% 확률로 posterior 대신
  prior sample을 forward에 흘림**(`RSSMTD.forward`의 `use_prior` 분기) → 미래 rollout 강건화.
  upstream MUVO와 동일.

---

## 2. LiDAR 데이터 분석 (`processed/train_run_002.arrow`, 10프레임 ~5만 점)

LiDAR 투영을 MUVO와 맞출 필요가 있는지 판단하기 위해 실제 데이터를 분석.

- **배열 형태**: `(N, 4)` = x, y, z, +1채널(intensity 추정). alpha는 `[:, :3]`만 사용.
- **좌표계 = 센서 원점 기준**: z 중앙값 −2.40m(점 다수가 z∈[−3,−2] = 지면) → 원점이 지면
  위 2.4m = LiDAR 센서 위치. x·y는 0 중심 대칭. → alpha의 `r=√(x²+y²+z²)`가 곧 센서거리 depth이며,
  MUVO의 `points − lidar_position` 재센터링은 **불필요**(데이터가 이미 센서 원점).
- **ego(자기 차체) 포인트 = 이미 제거됨**: 최소 r=2.92m, r<2.0m = 0점, 차 footprint 박스
  (|x|<2.6, |y|<1.1) = 0점. → `point_cloud_to_range_view`에 ego 마스킹 추가 **불필요**.
- **LIDAR_SCALE 적합성**: r max=80m(센서 하드캡), p99=75m.
  - `/40` → max **2.00**, p99 1.85 — MUVO(range=100m, /50 → max 2.0)와 동일한 [0,2] 풀-레인지 특성. ✅ 채택.
  - `/50` → max 1.60, p99 1.50 — clipping은 없으나 [0,1.6]로 under-fill, MUVO 특성 불일치. (당초 이 줄은 /40·/50을 구별 못 했음 → §1.4 정정 참고)

→ 결론: LiDAR 쪽에서 실제로 손볼 건 **empty-cell fill(0→−1)** 하나였고(§1.5 적용),
좌표계·ego는 데이터가 이미 충족.

---

## 3. 확인했으나 변경하지 않은 것

- **training 목적함수**: alpha는 observe-only(전 시퀀스 posterior 재구성 + KL). MUVO도
  동일(`shared_step` train 분기 = reconstruction only; imagine은 eval/시각화/deployment 전용).
  → 이미 일치, 변경 불필요.
- **action 채널 순서**: 둘 다 throttle-first(`[throttle, steer]` vs `[throttle_brake, steering]`).
  alpha 데이터에 brake 컬럼이 없어 throttle_brake가 raw throttle로 collapse되는 것만 의도적 적응.
- **eval 전용 차이(보류 권장, alpha 쪽이 더 표준적)**:
  - imagine future-action 1칸 시프트(`action[rf-1+t]` vs `action[rf+t]`) — eval 지표에만 영향.
  - eval 시 BN 모드(MUVO는 평가 때 train-mode BN; alpha는 기본 eval/running-stats).
- **튜닝 대상·의도적 적응(범위 외)**: transformer 용량(3층4헤드), dim(256), 해상도(320×768),
  토큰수(368), BEV/voxel/3D·segmentation 제거, Arrow 로딩, CLI.

---

## 검증

각 변경마다 `kcy-alpha`(torch 2.0.0)에서:
- 4개 파일 `py_compile` 통과, 제거 심볼 잔존 참조 0건.
- RSSMTD/Model forward·imagine shape 확인(sample-only decoder, n_out_tokens 등).
- KL t=0 anchor 단위 테스트(post=prior=N(0,1)→0; post_t0 shift→>0).
- empty-fill 테스트(빈 셀 −1, 유효 양수, valid_mask 정상).
- scheduled sampling 전파 + train-mode forward 확인.
- 매 단계 미니 스모크 학습(overfit 8 + held-out val, 1 epoch) end-to-end 정상(에러 없음).
