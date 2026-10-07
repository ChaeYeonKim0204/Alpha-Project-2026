# Trainer11 Meeting Notes

## Hugging Face CARLA Autopilot Multimodal Dataset 시간 해상도 확인

### 배경

- 사용 중인 데이터셋: `immanuelpeter/carla-autopilot-multimodal-dataset`
- 로컬 Arrow 위치: `/home/carol/chaeyeon-kim/processed/*.arrow`
- 목적: world model 학습에 사용할 수 있을 만큼 temporal resolution이 충분한지 확인
- 처음 우려:
  - dataset card에는 LiDAR가 20 Hz라고 되어 있음
  - 하지만 Arrow의 `timestamp` 간격을 직접 확인해보니 row 간 간격이 약 1.2~2.8초로 나옴
  - 이 값만 보면 저장된 sample frequency가 약 0.4~0.8 Hz로 보여 world model 학습에 부적절해 보였음

### 로컬 Arrow 확인 결과

전체 `/home/carol/chaeyeon-kim/processed/*.arrow` 파일에 대해 `run_id`, `frame`, `timestamp`를 확인했다.

- 모든 run에서 `frame_diff = 5`
- `timestamp` 기준 row 간격은 run마다 크게 다름
  - 일부 train/test run: 약 1.25초
  - 중간 그룹: 약 1.6초
  - 느린 그룹: 약 2.0~2.3초
- 기존 방식으로 `timestamp`를 시간축으로 보면:
  - row frequency: 약 0.4~0.8 Hz
  - 20 Hz 또는 20 Hz에서 5 frame마다 저장한 4 Hz와 맞지 않음

생성한 확인 파일:

- `/home/carol/chaeyeon-kim/processed/all_runs_timestamp_interval_summary.csv`
- `/home/carol/chaeyeon-kim/processed/all_runs_timestamp_intervals.csv`
- `/home/carol/chaeyeon-kim/alpha26/inspect_test_run_001_frame.ipynb`

### 원인 확인

Hugging Face dataset card 자체에는 다음 내용만 명시되어 있다.

- LiDAR: 32-channel ray-cast, 20 Hz, 80 m range
- `timestamp`: Relative timestamp, seconds
- collection process: synchronous sensors, saves every N-th frame
- 단, `N`, `fixed_delta_seconds`, 실제 generation script는 dataset card에 없음

이후 작성자 GitHub repo로 보이는 `immanuel-peter/self-driving-model`에서 collection script를 확인했다.

주요 코드:

```python
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
```

즉 CARLA simulation은 20 Hz로 설정되어 있다.

```python
SAVE_EVERY_N = 5
...
if tick % save_every_n == 0:
    frame_id = tick
```

즉 저장은 5 tick마다 한 번이다. 따라서 simulation 기준 저장 간격은:

```text
0.05 sec/tick * 5 ticks = 0.25 sec
```

즉 실제 sample frequency는:

```text
4 Hz
```

중요한 문제는 timestamp 저장 방식이다.

```python
"timestamp": time.time() - start_time
```

이 값은 CARLA simulation time이 아니라 Python wall-clock elapsed time이다. 따라서 로컬 Arrow에서 보인 `timestamp diff = 1.2~2.8초`는 simulation time 간격이 아니라, 데이터 수집 루프가 실제 컴퓨터에서 돌아간 시간 간격으로 해석하는 것이 맞다.

### 최종 해석

Arrow의 시간 관련 컬럼은 다음처럼 봐야 한다.

- `frame`: CARLA simulation tick id
- `timestamp`: wall-clock collection elapsed time
- simulation time: `frame * 0.05`
- consecutive sample dt: `frame.diff() * 0.05`

현재 Arrow에서는 모든 run에서 `frame_diff = 5`였으므로:

```text
simulation dt = 5 * 0.05 = 0.25 sec
effective sample rate = 4 Hz
```

따라서 이 데이터는 처음 우려했던 0.4~0.8 Hz 데이터가 아니라, **simulation 기준 약 4 Hz 데이터**로 보는 것이 타당하다.

### World Model 관점

- 4 Hz는 dense control-level world model에는 다소 낮지만, 0.5 Hz 수준보다는 훨씬 낫다.
- 1 step이 0.25초이므로 coarse-to-mid temporal dynamics 학습은 가능성이 있다.
- action-conditioned prediction을 할 경우 `timestamp` 컬럼을 그대로 쓰면 안 된다.
- 학습에서는 다음 중 하나를 사용해야 한다.
  - `dt = 0.25`로 고정
  - 또는 `dt = frame.diff() * 0.05`로 계산
- sequence 구성 시 row가 연속되어 있고 `frame_diff = 5`인지 확인하는 guard를 넣는 것이 안전하다.

### 주의점

- Hugging Face dataset card의 `timestamp (s)` 설명은 오해의 소지가 있다.
- 실제 코드 기준으로는 simulation seconds가 아니라 wall-clock seconds이다.
- 데이터셋 card에는 generation script가 직접 포함되어 있지 않아, GitHub repo의 script가 실제 HF dataset 생성에 사용되었다는 것은 강한 정황이지만 완전한 보증은 아니다.
- 다만 Arrow의 `frame_diff = 5` 패턴과 script의 `SAVE_EVERY_N = 5`가 정확히 일치하므로 같은 생성 로직일 가능성이 높다.

### 다음 액션

1. Trainer dataset에서 `timestamp` 대신 `frame` 기반 simulation time을 사용하도록 정리
2. sequence sampling 시 `run_id` 동일성뿐 아니라 `frame_diff == 5` 연속성도 확인
3. 모델/노트북 설명에 effective Hz를 `4 Hz`로 명시
4. 필요하면 `timestamp` 컬럼은 wall-clock collection diagnostics로만 취급

### 참고 링크

- Hugging Face dataset: https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset
- Dataset files: https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset/tree/main
- Generation script candidate: https://github.com/immanuel-peter/self-driving-model/blob/main/scripts/collect_autopilot_data.py

## 2026-05-07 교수님 피드백 반영 실험 설정

기준 모델: `/home/carol/chaeyeon-kim/alpha26/trainer11.ipynb`

우선 모델이 풀기 쉬운 문제부터 시작하기 위해 다음 ablation을 추가했다.

- reconstruction target 축소: RGB target `608x800` -> `216x288`
- LiDAR range view 축소: `64x512` -> `32x256`
- temporal sampling 축소: 저장 기준 4 Hz row에서 `SAMPLE_EVERY_N=2`로 약 2 Hz 사용
- latent 크기는 유지: `h_t=128`, `z_t=64`
- 학습 길이 축소: `STEPS=6000`
- pkl manifest 추가: `/home/carol/chaeyeon-kim/processed/arrow_manifest.pkl`
- 실행 스크립트 추가: `/home/carol/chaeyeon-kim/alpha26/scripts/trainer11_lowres_2hz.py`
- 실험 비교 로그 템플릿: `/home/carol/chaeyeon-kim/alpha26/results/experiment_log.csv`

### LiDAR range-view projection 정리

Immanuel Peter dataset의 수집 script는 CARLA `sensor.lidar.ray_cast` raw point cloud를 저장한다.

- 수집 설정: `channels=32`, `range=80`, `rotation_frequency=20`, `points_per_second=200000`
- 저장 형태: `Nx4 = x, y, z, intensity`
- `upper_fov`, `lower_fov`는 override하지 않았으므로 CARLA 기본값 `[-30, 10]`을 사용한 것으로 보는 것이 타당하다.
- MUVO도 `POINTS.FOV = [-30, 10]`을 사용한다.

따라서 `trainer11`의 range-view 변환에서 pitch를 `[-90, 90]`처럼 정규화하던 방식을 고치고, CARLA/MUVO 공통 vertical FOV `[-30, 10]` 기준으로 row를 계산하도록 수정했다.

현재 `LIDAR_RANGE_VIEW_SIZE=(32, 256)`의 의미:

- `H=32`: 수집 LiDAR의 vertical channel/beam 수
- `W=256`: 360도 azimuth를 rasterize하는 horizontal bin 수이며, raw point cloud에 원래 있는 고정 width가 아니라 local ablation용 선택값
- 더 조밀한 projection을 보고 싶으면 다음 실험에서 `W=320` 또는 `W=512`를 비교

검증된 첫 sample shape:

```text
image      (8, 3, 300, 400)
image_raw  (8, 3, 216, 288)
lidar      (8, 4, 32, 256)
action     (8, 2)
```

다음 실험 순서 제안:

1. `trainer11_lowres_2hz`를 local/server에서 짧게 돌려 TensorBoard loss와 validation metric 확인
2. learning rate 후보 `3e-5`, `1e-4`, `3e-4` 비교
3. reconstruction이 안정되면 `h_t/z_t` 또는 target resolution을 한 단계씩 증가
4. 각 변경은 `experiment_log.csv`에 한 줄씩 남겨 경향성 비교
