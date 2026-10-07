# Trainer11 서버 DDP 멈춤 현상 설명

## 한 줄 요약

이번 문제는 데이터나 모델이 깨진 것이 아니라, **4 GPU 분산 학습(DDP)을 켤 때 필요한 옵션이 빠져서 학습 시작 직전에 멈춘 현상**입니다.

## 용어 먼저: DDP와 NCCL이 뭔가

이번 문제를 이해하려면 두 단어만 알면 됩니다.

### DDP

DDP는 `Distributed Data Parallel`의 약자입니다. 쉽게 말하면 **GPU 여러 개를 한 팀처럼 묶어서 같은 모델을 나눠 학습시키는 PyTorch 방식**입니다.

4 GPU DDP에서는 대략 이런 일이 일어납니다.

```text
GPU 0: batch 일부 계산
GPU 1: batch 일부 계산
GPU 2: batch 일부 계산
GPU 3: batch 일부 계산

각자 계산한 gradient를 서로 맞춘 뒤
같은 모델 상태로 다음 step 진행
```

즉 DDP의 핵심은 **각 GPU가 따로 계산하되, 매 step마다 결과를 서로 맞추는 것**입니다.

### NCCL

NCCL은 `NVIDIA Collective Communications Library`의 약자입니다. 이름은 어렵지만 역할은 단순합니다.

**GPU들이 서로 계산 결과를 주고받게 해주는 NVIDIA 통신 도구**입니다.

우리가 코드에서 NCCL을 직접 호출하는 것은 아닙니다. PyTorch Lightning에서 4 GPU DDP를 켜면, 내부적으로 GPU끼리 gradient를 맞추기 위해 NCCL을 사용합니다.

비유하면:

```text
DDP = GPU 4명을 한 팀으로 묶어 일시키는 학습 방식
NCCL = 그 4명이 매 step마다 결과를 공유할 때 쓰는 통신 장치
```

그래서 DDP 문제가 생기면 NCCL 관련 설정이 같이 등장할 수 있습니다. 단, 이번에 로그로 확실히 확인된 에러는 NCCL 에러가 아니라 **DDP unused parameter 에러**입니다. NCCL 옵션은 4 GPU 실행이 멈추는 상황을 피하려고 붙인 안정화 옵션입니다.

## 비전공자용 핵심 설명

아주 쉽게 말하면, **GPU 4개가 같이 학습하려고 서로 "계산 끝났어? 이제 gradient 맞추자" 하고 기다리는 단계에서 꼬인 상황**입니다.

문제는 크게 두 단계로 나눠서 봐야 합니다.

첫 번째, 그리고 **확실히 에러로 확인된 원인**은 DDP가 우리 모델 구조를 기본 설정으로는 못 받아들인 것입니다.

우리 모델은 매 step마다 모든 parameter가 loss 계산에 쓰이지 않을 수 있습니다. 그런데 기본 DDP는 "모든 parameter가 매번 쓰여야 한다"고 기대합니다. 그래서 어떤 parameter가 이번 step에서 조용히 있으면 DDP가 에러를 냅니다.

이 문제를 해결하는 옵션이 아래입니다.

```bash
--strategy ddp_find_unused_parameters_true
```

이 옵션은 DDP에게 "이번 step에서 안 쓰인 parameter가 있어도 이상한 상황이 아니니 찾아서 정상 처리해라"라고 알려주는 역할을 합니다.

두 번째는 **에러 원인이라기보다 4 GPU 실행을 안정적으로 만들기 위해 붙인 GPU 통신 안전 옵션**입니다.

GPU 4개가 같이 학습하려면 서로 계속 정보를 주고받아야 합니다. 이 통신을 담당하는 도구가 NCCL입니다. 처음 실행에서는 4 GPU 프로세스가 모두 등록된 뒤 실제 학습 progress가 나오지 않았습니다. 정확한 NCCL 내부 원인을 확정한 것은 아니지만, 아래 환경 변수를 붙인 조합에서 4 GPU debug run이 정상 진행됐기 때문에 이 서버에서는 안전 옵션으로 같이 사용합니다.

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1
```

즉 이번 현상은 아래처럼 이해하면 됩니다.

```text
데이터가 느린 것 아님
6000 Ada가 느린 것 아님
코드가 1 GPU에서도 안 되는 것 아님
4 GPU DDP 설정이 우리 모델/서버 조합에 맞지 않았던 것
```

비유하면, GPU 4명이 팀플을 하는데 기본 규칙은 "모든 사람이 매번 발언해야 함"이었습니다. 그런데 우리 모델은 어떤 step에서는 일부 파트가 조용히 있어도 되는 구조입니다. DDP가 그걸 "누가 빠졌는데?"라고 오해해서 멈춘 것이고, `ddp_find_unused_parameters_true`는 "조용한 파트가 있어도 정상으로 처리해"라고 알려주는 옵션입니다.

서버에서 안정적으로 돌린 명령은 아래 조합입니다.

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 \
python scripts/train_full_trainer11_server.py \
  --steps 150000 \
  --devices 4 \
  --strategy ddp_find_unused_parameters_true \
  --batch-size 2 \
  --num-workers 2 \
  --window-index /home/user/chaeyeon-kim/processed/arrow_window_index.pkl \
  --run-name trainer11_server_full_h256_z128_lidar_re10_empty005
```

## 무슨 일이 있었나

처음에는 아래처럼 실행했습니다.

```bash
python scripts/train_full_trainer11_server.py \
  --steps 150000 \
  --devices 4 \
  --batch-size 2 \
  --window-index /home/user/chaeyeon-kim/processed/arrow_window_index.pkl \
  --run-name trainer11_server_full_h256_z128_lidar_re10_empty005
```

터미널에는 데이터셋 준비와 DDP 초기화까지는 정상처럼 보이는 로그가 나왔습니다.

```text
Loaded window index cache: /home/user/chaeyeon-kim/processed/arrow_window_index.pkl
Window index source: cached_files=22 scanned_files=0
Built train_ds: 6680 windows
Built validation: val_rl=264 windows, val_ds=400 windows
All distributed processes registered. Starting with 4 processes
Missing logger folder: /home/user/chaeyeon-kim/alpha26/logs/...
```

하지만 그 뒤에 몇 시간 동안 정상적인 학습 progress bar, TensorBoard log, checkpoint가 생기지 않았습니다.

## 왜 헷갈렸나

`nvidia-smi`에서는 GPU 사용률이 100%처럼 보여서 "학습이 도는 중인가?"처럼 보였습니다.

하지만 각 Python 프로세스가 잡고 있던 GPU 메모리는 대략 560-604 MB 정도였습니다.

```text
GPU0 python 약 596 MB
GPU1 python 약 604 MB
GPU2 python 약 604 MB
GPU3 python 약 564 MB
```

이 정도 메모리는 실제 모델 학습이라기보다 **CUDA/DDP 초기화만 된 상태**에 가깝습니다. 실제 forward/backward 학습이 돌면 보통 이보다 훨씬 많은 VRAM을 잡습니다.

즉 당시 상태는 아래에 가까웠습니다.

```text
GPU 사용률은 높아 보임
하지만 GPU 메모리는 초기화 수준
학습 step 증가 없음
TensorBoard event 파일 없음
checkpoint 없음
```

그래서 "서버가 느린 것"이 아니라 **분산 학습 초기화 이후 동기화 지점에서 막힌 것**으로 판단했습니다.

## 확인 과정

### 1. 1 GPU 단독 실행은 바로 성공

아래 명령은 바로 돌았습니다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_full_trainer11_server.py \
  --steps 20 \
  --devices 1 \
  --batch-size 2 \
  --num-workers 0 \
  --val-check-interval 10 \
  --limit-val-batches 1 \
  --save-top-k 1 \
  --window-index /home/user/chaeyeon-kim/processed/arrow_window_index.pkl \
  --run-name debug_single_gpu_workers0 \
  --no-reconstruction
```

이걸로 확인된 점:

- 모델 코드 자체는 실행 가능
- Arrow 데이터 파일은 정상
- `arrow_window_index.pkl` cache도 정상
- 기본 training step도 정상

즉 문제는 데이터나 모델 전체가 아니라 **4 GPU DDP 설정** 쪽으로 좁혀졌습니다.

### 2. 4 GPU + 기본 DDP는 실패

`--strategy ddp` 또는 기본 DDP 설정에서는 아래 RuntimeError가 발생했습니다.

```text
RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step.
If this is intentional, you must enable the detection of unused parameters in DDP,
either by setting strategy='ddp_find_unused_parameters_true'
or DDPStrategy(find_unused_parameters=True).
```

이건 warning이 아니라 **실제 에러**입니다.

### 3. 4 GPU + unused parameter 탐지 옵션은 성공

아래 debug run은 성공했습니다.

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 \
python scripts/train_full_trainer11_server.py \
  --steps 50 \
  --devices 4 \
  --strategy ddp_find_unused_parameters_true \
  --batch-size 2 \
  --num-workers 2 \
  --val-check-interval 25 \
  --limit-val-batches 1 \
  --save-top-k 1 \
  --window-index /home/user/chaeyeon-kim/processed/arrow_window_index.pkl \
  --run-name debug_ddp_unused_ncclfix_workers2 \
  --no-reconstruction
```

따라서 full run도 같은 핵심 옵션을 사용해야 합니다.

## 원인 1: DDP unused parameter 문제

DDP는 `Distributed Data Parallel`의 약자입니다.

쉽게 말하면:

- GPU 4개가 각각 같은 모델 복사본을 들고 학습합니다.
- 각 GPU가 자기 batch를 계산합니다.
- 매 step마다 gradient를 서로 맞춥니다.

기본 DDP는 모든 trainable parameter가 매 step의 loss 계산에 참여한다고 가정합니다.

그런데 우리 모델은 구조상 어떤 parameter가 특정 step의 loss에 직접 연결되지 않을 수 있습니다. 예를 들어 조건부 경로, decoder 경로, 미래 예측 경로, loss 구성 방식 때문에 DDP 입장에서는 "이번 step에서 gradient가 안 생긴 parameter"가 보일 수 있습니다.

기본 DDP는 이것을 비정상으로 보고 에러를 냅니다.

해결책은 아래 옵션입니다.

```bash
--strategy ddp_find_unused_parameters_true
```

이 옵션은 DDP에게 이렇게 알려줍니다.

```text
이번 step에서 사용되지 않은 parameter가 있을 수 있으니,
그걸 찾아서 에러로 죽이지 말고 정상적으로 처리해라.
```

## 추가 안정화: GPU 통신을 맡는 NCCL 안전 옵션

NCCL은 GPU끼리 통신할 때 쓰는 NVIDIA 라이브러리입니다. 쉽게 말하면 **GPU들이 서로 "내 계산 결과는 이렇다"라고 주고받게 해주는 통신 담당 도구**입니다.

DDP에서 GPU 4개가 학습하려면 각 GPU가 계산한 gradient를 매 step마다 맞춰야 합니다. 이때 PyTorch가 내부적으로 NCCL을 사용합니다.

이번에 NCCL 자체가 에러 메시지로 직접 드러난 것은 아닙니다. 직접 드러난 에러는 위의 unused parameter RuntimeError입니다.

다만 처음 4 GPU 실행에서는 모든 rank가 등록된 뒤에도 실제 학습 진행이 보이지 않았고, GPU 메모리도 초기화 수준에 머물렀습니다. 이 상황은 GPU끼리 서로 맞춰야 하는 단계에서 대기 상태가 생겼을 때도 나타날 수 있습니다. 그래서 서버의 GPU 통신 경로에서 대기 상태가 생겼을 가능성을 피하기 위해 아래 안전 옵션을 같이 붙였습니다.

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1
```

각 옵션의 의미를 풀어 쓰면:

- `NCCL_P2P_DISABLE=1`: GPU끼리 바로 직접 연결해서 주고받는 빠른 길을 쓰지 않게 합니다.
- `NCCL_IB_DISABLE=1`: 서버에 있을 수 있는 고속 네트워크/장비 경로인 InfiniBand/RDMA를 쓰지 않게 합니다.
- `NCCL_ASYNC_ERROR_HANDLING=1`: 통신 중 문제가 생겼을 때 에러를 더 잘 감지하게 합니다.

위 두 disable 옵션은 쉽게 말해 **가장 빠른 길 대신 더 단순하고 보수적인 길로 GPU끼리 통신하게 하는 설정**입니다. 속도를 약간 희생할 수 있습니다. 대신 서버 하드웨어 배치나 드라이버 조합 때문에 생길 수 있는 DDP 초기 멈춤을 피하는 데 도움이 될 수 있습니다.

정리하면, 문서에서 말하는 NCCL 옵션은 "NCCL이 범인이라고 확정됐다"는 뜻이 아닙니다. **4 GPU들이 서로 계산 결과를 주고받는 단계에서 멈추지 않도록 붙인 보수적인 통신 설정**에 가깝습니다.

## 원인이 아니었던 것들

### Arrow window cache 문제는 아니었음

로그에 아래가 찍혔습니다.

```text
cached_files=22 scanned_files=0
```

즉 전체 Arrow 파일을 다시 스캔하느라 느린 상황이 아니었습니다.

### `Missing logger folder`는 핵심 에러가 아님

아래 메시지는 처음 log 폴더가 아직 없을 때 Lightning/TensorBoard가 찍는 메시지입니다.

```text
Missing logger folder: /home/user/chaeyeon-kim/alpha26/logs/...
```

이 문장 자체가 실패 원인은 아닙니다.

진짜 문제는 기본 DDP에서 발생한 unused parameter RuntimeError였습니다.

### `sync_dist=True` 경고는 fatal error가 아님

아래와 같은 경고가 많이 나왔습니다.

```text
It is recommended to use self.log(..., sync_dist=True)
```

이건 분산 validation metric을 GPU 여러 장에서 평균낼 때 `sync_dist=True`를 쓰는 것이 더 정확하다는 권장 경고입니다.

학습을 죽이는 에러는 아닙니다.

조금 더 풀어 쓰면, 4 GPU DDP에서는 GPU 0, 1, 2, 3이 각각 자기 batch의 loss/metric을 계산합니다. 그런데 `self.log(..., sync_dist=True)`가 없으면 Lightning 입장에서는 이 값이 4 GPU 전체 평균인지, rank 0 하나의 값인지 애매합니다.

예를 들어 아래 같은 warning이 반복됩니다.

```text
It is recommended to use self.log('train_future_rgb_2', ..., sync_dist=True)
It is recommended to use self.log('train_future_lidar_depth_4', ..., sync_dist=True)
```

의미는 아래에 가깝습니다.

```text
학습 중단 원인 아님
backward/gradient 문제 아님
로그/validation metric/checkpoint monitor 정확도에는 영향 가능
```

따라서 현재 run을 이 warning 때문에 끊을 필요는 없습니다. 다만 다음 코드 정리 때는 `self.log(...)`에 `sync_dist=True`와 `batch_size=...`를 넣는 것이 좋습니다.

## Window index는 무엇이고, 속도를 느리게 하나

`arrow_window_index.pkl`은 새로 쪼갠 학습 데이터 파일이 아닙니다.

정확히는 각 Arrow 파일에서 **8-frame window가 시작될 수 있는 row 번호 목록**입니다.

비유하면:

```text
Arrow 파일 = 원본 책
arrow_window_index.pkl = 어느 페이지에서 읽기 시작하면 되는지 적어둔 책갈피 목록
```

window index를 안 쓰면 실행할 때마다 모든 Arrow 파일을 훑으면서 아래를 다시 확인해야 합니다.

```text
이 row부터 8개 frame을 뽑아도 되는가?
run_id가 중간에 바뀌지 않는가?
frame 번호가 끊기지 않는가?
```

window index를 쓰면 이 계산을 미리 저장해둔 목록으로 대체합니다.

이번 run에서는 아래처럼 나왔습니다.

```text
Window index source: cached_files=22 scanned_files=0
Built train_ds: 6680 windows
```

즉 window index cache는 정상적으로 사용됐고, 전체 Arrow 파일을 다시 스캔하지 않았습니다.

중요한 점은, window index가 학습 중 무거운 전처리를 미리 끝내주는 것은 아니라는 점입니다. 현재 Dataset은 학습 중에도 매 batch마다 아래 작업을 합니다.

```text
image bytes -> PIL image decode -> image transform
LiDAR point cloud -> range view 변환
```

따라서 window index는 주로 **학습 시작 전 준비 시간**을 줄입니다. step 하나하나의 GPU 학습 속도를 크게 느리게 만드는 요소는 아닙니다.

## Global batch size와 epoch당 step 수

현재 full server run에서 `--batch-size 2 --devices 4`를 사용합니다.

이 스크립트에서 `--batch-size`는 전체 batch size가 아니라 **GPU 한 장당 batch size**입니다.

따라서:

```text
per_gpu_batch_size = 2
GPU 개수 = 4
global_batch_size = 2 * 4 = 8
```

한 optimizer step에서 실제로 처리되는 window 수는 아래와 같습니다.

```text
GPU 0: 2 windows
GPU 1: 2 windows
GPU 2: 2 windows
GPU 3: 2 windows
총합: 8 windows
```

이번 run 로그에 따르면 train dataset은 6680 windows입니다.

```text
Built train_ds: 6680 windows
```

그래서 epoch당 step 수는:

```text
6680 windows / global_batch_size 8 = 835 steps/epoch
```

터미널에 보이는 아래 숫자가 이 값입니다.

```text
Epoch 11: 56% | 468/835 ...
```

이전 local `trainer11_reviewed_v4_alldata_40epoch`도 전체 데이터를 썼습니다. 다만 local 1 GPU에서는 global batch size가 2였기 때문에 checkpoint 이름 기준으로 대략 3500 steps/epoch가 나왔습니다.

```text
local 1 GPU:
  global_batch_size = 2
  약 7000 windows / 2 ~= 3500 steps/epoch

server 4 GPU:
  global_batch_size = 8
  6680 windows / 8 = 835 steps/epoch
```

따라서 epoch당 step 수 차이는 hidden state 크기 때문이 아닙니다. **global batch size가 달라졌기 때문**입니다.

## 서버 run이 더 오래 걸려 보일 수 있는 이유

4 GPU를 쓰면 epoch 한 바퀴의 step 수는 줄어듭니다. 하지만 전체 run 시간이 반드시 4배 빨라지는 것은 아닙니다.

이번 server run에서는 아래 요인들이 겹칩니다.

```text
h=128, z=64 -> h=256, z=128로 모델이 커짐
DDP는 매 step마다 GPU 4개의 gradient를 서로 맞춰야 함
per-GPU batch size가 2라서 GPU 한 장당 일이 작고 통신 비중이 커질 수 있음
ddp_find_unused_parameters_true가 매 step unused parameter를 추적함
NCCL 안전 옵션이 빠른 통신 경로 일부를 끄므로 보수적으로 동작함
학습 중 image decode와 LiDAR range view 변환이 계속 수행됨
validation 때 LiDAR chamfer/cdist metric이 무거움
```

또 하나 중요한 점은 `--steps 150000`입니다.

이 값은 epoch 수가 아니라 optimizer step 수입니다. 즉 4 GPU로 epoch당 step 수가 줄어도 종료 조건은 여전히 150000 step입니다.

그리고 global batch size가 커졌으므로, 같은 150000 step이라도 처리하는 총 window 수는 더 많습니다.

```text
local 1 GPU, global batch 2:
  150000 steps * 2 = 300000 windows 처리

server 4 GPU, global batch 8:
  150000 steps * 8 = 1200000 windows 처리
```

그래서 서버 run이 더 오래 걸려 보이는 느낌은 이상하지 않습니다. 서버가 느려진 것이 아니라, **한 step에서 더 많은 데이터를 보고, 모델도 커졌고, DDP 통신/안전 옵션 오버헤드도 붙은 상태**입니다.

현재 속도가 예를 들어 `3.39 it/s`라면 순수 train step 기준 예상 시간은 대략:

```text
150000 / 3.39 ~= 44248 sec ~= 12.3 hours
```

여기에 validation/checkpoint 시간이 추가됩니다.

## 앞으로 full run 실행 방법

현재 추천 명령:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 \
python scripts/train_full_trainer11_server.py \
  --steps 150000 \
  --devices 4 \
  --strategy ddp_find_unused_parameters_true \
  --batch-size 2 \
  --num-workers 2 \
  --window-index /home/user/chaeyeon-kim/processed/arrow_window_index.pkl \
  --run-name trainer11_server_full_h256_z128_lidar_re10_empty005
```

학습이 제대로 시작됐는지 확인:

```bash
ls -lh /home/user/chaeyeon-kim/alpha26/logs/trainer11_server_full_h256_z128_lidar_re10_empty005
nvidia-smi
```

좋은 신호:

```text
Epoch 0: ... train_loss_step=...
logs/<run-name>/version_0/events.out.tfevents...
logs/<run-name>/checkpoints/...
GPU memory가 초기화 수준보다 훨씬 큼
```

나쁜 신호:

```text
All distributed processes registered 이후 오래 멈춤
progress bar 없음
TensorBoard event 파일 없음
checkpoint 없음
각 Python 프로세스 GPU memory가 500-700 MB 정도에 머묾
```

## 멈춘 run 종료 방법

먼저 프로세스를 확인합니다.

```bash
pgrep -af 'train_full_trainer11_server.py'
```

full run만 죽이려면:

```bash
pkill -9 -f 'train_full_trainer11_server.py.*trainer11_server_full_h256_z128_lidar_re10_empty005'
```

debug run을 죽이려면 run name을 바꿔서 사용합니다.

```bash
pkill -9 -f 'train_full_trainer11_server.py.*debug_ddp_unused_ncclfix_workers2'
```

서버를 혼자 쓰고 있고 이 스크립트만 돌고 있다면 넓게 죽일 수 있습니다.

```bash
pkill -9 -f 'train_full_trainer11_server.py'
```

종료 확인:

```bash
pgrep -af 'train_full_trainer11_server.py'
nvidia-smi
```

## 저장 위치

TensorBoard log:

```text
/home/user/chaeyeon-kim/alpha26/logs/<run-name>/version_0/
```

Checkpoint:

```text
/home/user/chaeyeon-kim/alpha26/logs/<run-name>/checkpoints/
```

학습 완료 후 reconstruction figure:

```text
/home/user/chaeyeon-kim/alpha26/results/figures/<run-name>_reconstruction_s0.png
```

단, debug 명령처럼 `--no-reconstruction`을 붙이면 figure는 저장하지 않습니다.

Experiment CSV:

```text
/home/user/chaeyeon-kim/alpha26/results/experiment_log.csv
```

## 팀 공유용 결론

이번 현상은 "서버가 느려서 학습이 오래 걸린 것"이 아닙니다.

1 GPU에서는 바로 학습이 됐고, 4 GPU에서만 문제가 났습니다. 확실히 확인된 에러 원인은 4 GPU DDP가 우리 모델의 unused parameter를 기본 설정으로 처리하지 못한 것입니다. 여기에 더해, 서버에서 4 GPU 실행이 초기 대기 상태에 빠지는 것을 피하려고 NCCL 안전 옵션을 같이 사용했습니다.

따라서 앞으로 이 서버에서 해당 trainer를 4 GPU로 돌릴 때는 아래 두 가지를 반드시 붙입니다.

```text
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1
--strategy ddp_find_unused_parameters_true
```
