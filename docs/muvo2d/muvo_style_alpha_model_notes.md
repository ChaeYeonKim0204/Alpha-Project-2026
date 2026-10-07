# MUVO-style Alpha Model Notes

작성일: 2026-05-14

> **NOTE (2026-05-15)**: 본 문서는 2026-05-14 작성 후, 2026-05-15에 본문의 dimension
> 수치만 현재 상태로 업데이트됨. 이후 적용된 코드 변경:
>
> - commit `7099f87`: `IMAGE_INPUT_SIZE` (320,800) → **(320,768)**,
>   `RGB_RECON_SIZE` (216,288) → **(320,768)**, SensorDecoder → upstream `ConvDecoder` 패턴
> - commit `7bf7df9`: `SensorFeatureConv` plain conv → **BasicBlock × 2** (ResNet residual)
> - commit `0be4871`: RSSM에 `action_in_gru` flag 도입 (default True = alpha 디자인,
>   False = upstream muvo 동작)
> - commit `08336bc`: `trainer_muvo.py`에 SSIM loss 정식 통합 (`cfg.LOSSES.WEIGHT_SSIM`)
>
> 본문 dim 수치는 commit `7099f87` 기준 갱신되었음. SensorDecoder/SensorFeatureConv/RSSM
> 변경의 자세한 영향은 본문에 미반영 — 정확한 현재 구조는 `scripts/model_variants/models_muvo.py`,
> upstream과의 default config 비교는 `docs/muvo_default_config_audit.md` 참조.

이 문서는 `muvo/` 원본 구조와 `alpha26/` 현재 구조를 비교하면서, `alpha26/scripts/model_variants/models_muvo.py`와 `config_muvo.py`에 반영한 MUVO-style 변경 사항을 정리한다.

## 목표

`alpha26`의 world model encoder/fusion 경로를 MUVO 원본에 더 가깝게 바꾸되, alpha 데이터셋의 실제 센서 조건에 맞춰 가볍게 조정한다.

현재 목표 구조:

```text
RGB / LiDAR input
-> ResNet18 multi-scale features
-> MUVO-style DecoderDS / FPN
-> sensor fusion transformer
-> per-sensor feature conv
-> image/lidar/speed concat
-> features_combine
-> RSSM
```

채택한 alpha 설정:

```text
image input: 320 x 768
lidar range-view: 32 x 1024
FPN / transformer channels: 256
transformer: d_model=256, nhead=8, layers=3
embedding_dim: 256
RSSM hidden/state: 512 / 256
```

## MUVO 원본 구조 요약

MUVO transformer 경로는 대략 다음 흐름이다.

```text
ResNet18 multi-scale features
-> DecoderDS output: TRANSFORMER.CHANNELS
-> sensor fusion transformer
-> per-sensor feature conv
-> image/lidar/route/speed concat
-> features_combine
-> RSSM
```

주요 config:

```text
MODEL.TRANSFORMER.CHANNELS = 384  # muvo.yml에서 override
MODEL.EMBEDDING_DIM = 512
MODEL.TRANSITION.HIDDEN_STATE_DIM = 1024
MODEL.TRANSITION.STATE_DIM = 512
POINTS.CHANNELS = 64
POINTS.HORIZON_RESOLUTION = 1024
POINTS.FOV = [-30, 10]
```

MUVO는 transformer channel과 RSSM embedding dimension을 분리한다.

```text
transformer d_model: 384
post-transformer embedding: 512
```

`mile.py`에서는 transformer 이후 `image_feature_conv`와 `lidar_feature_conv`를 통해 sensor별 feature를 embedding 차원으로 압축한다.

## Image 처리 비교

### MUVO

MUVO 기본 image config:

```text
원본 image size: 600 x 960
crop: [64, 138, 896, 458]
crop 결과: 320 x 832
```

즉 MUVO는 resize가 아니라 crop으로 `320 x 832`를 만든다.

ResNet18 `out_indices=[2, 3, 4]` 기준:

```text
input: 320 x 832
xs[0]: 40 x 104
xs[1]: 20 x 52
xs[2]: 10 x 26
```

MUVO-style `DecoderDS` 최종 image feature:

```text
10 x 26 = 260 image tokens
```

### Alpha

기존 alpha는 `transforms.Resize(tuple(size))`로 image를 바로 resize했다.

기존 설정:

```text
IMAGE_INPUT_SIZE = (300, 400)
```

MUVO-style alpha에서는 다음 방향으로 결정했다.

```text
원본 image: 600 x 800
목표 input: 320 x 768
권장 방식: resize가 아니라 crop
```

`320 x 768`을 쓰면:

```text
input: 320 x 768
xs[0]: 40 x 96
xs[1]: 20 x 48
xs[2]: 10 x 24
```

최종 image tokens:

```text
10 x 24 = 240
```

이는 MUVO 원본의 `260` tokens와 거의 같은 스케일이다.

## LiDAR 처리 비교

### LiDAR 기본 개념

LiDAR raw point cloud는 점들의 목록이다.

```text
point = x, y, z, intensity 또는 semantic attributes
```

`channel`은 수직 방향 레이저 줄 수로 보면 된다.

```text
32-channel LiDAR = 세로 방향 레이저 줄 32개
64-channel LiDAR = 세로 방향 레이저 줄 64개
```

Range-view는 raw point cloud를 2D 이미지처럼 펼친 표현이다.

```text
range-view xyzd:
channel 0 = x
channel 1 = y
channel 2 = z
channel 3 = depth/range
```

### MUVO LiDAR

MUVO repo의 data collection config 기준 semantic LiDAR는 64-channel이다.

```yaml
lidar_points_semantic:
  module: lidar.ray_cast_semantic
  location: [1.0, 0.0, 2.0]
  lidar_options:
    channels: 64
    range: 100
    rotation_frequency: 10
    points_per_second: 600000
    upper_fov: 10.0
    lower_fov: -30.0
```

학습 파이프라인에서도:

```text
POINTS.CHANNELS = 64
POINTS.HORIZON_RESOLUTION = 1024
POINTS.FOV = [-30, 10]
```

즉 MUVO range-view는:

```text
4 x 64 x 1024
```

ResNet18 / DecoderDS 이후:

```text
64 x 1024 -> 2 x 32 = 64 lidar tokens
```

### Alpha / Immanuel Dataset LiDAR

Alpha에서 쓰는 Immanuel Peter dataset은 CARLA `sensor.lidar.ray_cast` raw point cloud를 저장한다.

수집 script 기준:

```python
lidar_bp.set_attribute('range', '80')
lidar_bp.set_attribute('rotation_frequency', '20')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('points_per_second', '200000')
```

저장 형태:

```text
Nx4 = x, y, z, intensity
```

`upper_fov`, `lower_fov`는 override하지 않았으므로 CARLA 기본값을 따른다고 보는 것이 타당하다.

```text
upper_fov = 10.0
lower_fov = -30.0
```

따라서 alpha도:

```text
LIDAR_FOV_DEGREES = (-30.0, 10.0)
```

를 유지한다.

중요한 차이:

```text
MUVO raw: xyz + semantic label 계열
Alpha raw: xyz + intensity
```

Alpha는 semantic LiDAR label이 없으므로 MUVO의 LiDAR semantic segmentation supervision과는 다르다.

### Alpha LiDAR range-view 결정

Alpha raw LiDAR는 실제 32-channel이므로 height를 64로 늘리는 것은 실제 수직 해상도를 왜곡할 수 있다.

따라서 최종 결정:

```text
LIDAR_RANGE_VIEW_SIZE = (32, 1024)
```

이유:

```text
H=32: 실제 32-channel sensor와 일치
W=1024: MUVO와 같은 horizontal resolution 사용
```

ResNet18 / FPN 이후:

```text
32 x 1024 -> 1 x 32 = 32 lidar tokens
```

MUVO와 비교:

```text
MUVO:  64 x 1024 -> 2 x 32 = 64 lidar tokens
Alpha: 32 x 1024 -> 1 x 32 = 32 lidar tokens
```

Alpha의 vertical channel 수가 MUVO의 절반이므로, 최종 LiDAR token 수도 절반인 것이 자연스럽다.

## Token 수 비교

현재 alpha MUVO-style 설정:

```text
image input: 320 x 768
lidar input: 32 x 1024
```

FPN 출력:

```text
image feature: 256 x 10 x 24
lidar feature: 256 x 1 x 32
```

Transformer token 수:

```text
image tokens = 10 * 24 = 240
lidar tokens = 1 * 32 = 32
total = 272 tokens per frame
```

MUVO 원본과 비교:

```text
MUVO image: 320 x 832 -> 10 x 26 = 260
MUVO lidar: 64 x 1024 -> 2 x 32 = 64
MUVO total: 324 tokens

Alpha image: 320 x 768 -> 10 x 24 = 240
Alpha lidar: 32 x 1024 -> 1 x 32 = 32
Alpha total: 272 tokens
```

## 원본 MUVO vs 현재 Alpha `_muvo` 최종 비교

현재 `alpha26/scripts/model_variants/*_muvo.py` 구조는 MUVO full model을 그대로 복제한 것이 아니라, MUVO의 핵심 sensor-token fusion 구조를 alpha dataset 조건에 맞춰 축소 이식한 형태다.

| 구간 | 원본 MUVO | 현재 Alpha `_muvo` |
|---|---|---|
| RGB input | `600 x 960`에서 crop `[64, 138, 896, 458]` -> `320 x 832` | `600 x 800`에서 center crop -> `320 x 768` |
| LiDAR raw | CARLA semantic/ray-cast 계열, config 기준 `64-channel`, `100m`, `1024 horizontal` | Immanuel dataset raw ray-cast, `32-channel`, `80m`, `1024 horizontal` |
| image encoder | ResNet18 `features_only`, `out_indices=[2, 3, 4]` | 동일하게 ResNet18 |
| lidar encoder | ResNet18 `in_chans=4` | 동일하게 ResNet18 `in_chans=4` |
| FPN/feature decoder | `DecoderDS` | MUVO `DecoderDS` 방식의 `FPNDecoder` |
| fusion transformer | image tokens + lidar tokens concat | 동일한 방식 |
| transformer channels | `384` in `muvo.yml` | `256` |
| transformer layers | `6` | `3` |
| embedding dim | `512` | `256` |
| RSSM | hidden `1024`, state `512` | hidden `512`, state `256` |
| output decoder | RGB, LiDAR, BEV/voxel/segmentation/policy 등 | RGB reconstruction, LiDAR reconstruction 우선 |
| route/map input | 사용 | 미사용 |
| speed input | 사용, normalization `5.0` | 사용, normalization `50.0` |

현재 Alpha `_muvo`가 MUVO스럽게 맞춘 핵심 부분:

```text
RGB와 LiDAR를 각각 ResNet18로 인코딩
-> 둘 다 multi-scale feature를 DecoderDS/FPN 스타일로 합침
-> image token과 lidar token에 position embedding + sensor type embedding 추가
-> 두 sensor token을 transformer에 같이 넣어 attention 수행
-> transformer output을 다시 image/lidar feature map으로 분리
-> sensor별 feature conv + adaptive pooling으로 vector화
-> image vector + lidar vector + speed vector concat
-> features_combine으로 RSSM embedding 생성
```

의도적으로 다르게 둔 부분:

```text
MUVO보다 작은 transformer/RSSM 사용:
  MUVO: 384 channels, 6 layers, embedding 512, RSSM 1024/512
  Alpha: 256 channels, 3 layers, embedding 256, RSSM 512/256

Alpha dataset의 실제 LiDAR 조건 반영:
  MUVO: 64 x 1024 range-view
  Alpha: 32 x 1024 range-view

현재는 route map, BEV, voxel, semantic segmentation, policy head 제외
```

결론:

```text
현재 구조 = MUVO full model이 아니라
Alpha dataset에 맞춘 MUVO-style compact world model
```

## 현재 수정한 파일

### `alpha26/scripts/model_variants/config_muvo.py`

현재 MUVO-style alpha config:

```python
DATA.IMAGE_INPUT_SIZE = (320, 768)
DATA.RGB_RECON_SIZE = (320, 768)
DATA.LIDAR_RANGE_VIEW_SIZE = (32, 1024)
DATA.LIDAR_FOV_DEGREES = (-30.0, 10.0)

MODEL.EMBEDDING_DIM = 256
MODEL.FUSION.TRANSFORMER_CHANNELS = 256
MODEL.FUSION.TRANSFORMER_LAYERS = 3
MODEL.FUSION.TRANSFORMER_HEADS = 8

MODEL.TRANSITION.HIDDEN_STATE_DIM = 512
MODEL.TRANSITION.STATE_DIM = 256
MODEL.TRANSITION.ACTION_LATENT_DIM = 64
```

추가로 학습 entrypoint가 기대하는 값을 명시했다.

```python
EPOCHS = 1
STEPS = 400
LOGGING.RUN_NAME = "muvo_style_alpha_fpn256_lidar32"
LOGGING.BASE_FILE = "train_muvo.py"
```

### `alpha26/scripts/model_variants/models_muvo.py`

#### Config import

`models_muvo.py`는 기존 `config.py`가 아니라 `config_muvo.py`를 import하도록 변경했다.

```python
from config_muvo import cfg
```

#### FPNDecoder

기존 alpha FPN은 top-down 방식이었다.

```text
xs[2] -> upsample -> xs[1] -> upsample -> xs[0]
```

MUVO-style FPN은 downsample 방식이다.

```text
xs[0] -> downsample -> xs[1] -> downsample -> xs[2]
```

현재 구현은 다음 흐름이다.

```python
x = self.conv1(xs[0])
for i, conv in enumerate(self.skip_convs):
    skip = xs[i + 1]
    if x.shape[-2:] != skip.shape[-2:]:
        x = F.adaptive_max_pool2d(x, output_size=skip.shape[-2:])
    x = conv(skip) + x
```

`adaptive_max_pool2d`를 쓴 이유는 입력 크기가 항상 32로 완벽하게 나누어떨어지지 않을 수 있기 때문이다.

#### Transformer channel / embedding dimension 분리

현재는 둘 다 256이지만, MUVO 원본처럼 나중에 분리할 수 있도록 다음 두 개념을 구분했다.

```text
transformer_channels: FPN output / transformer d_model
embedding_n_channels: RSSM에 들어가는 최종 embedding dim
```

현재 설정:

```text
transformer_channels = 256
embedding_n_channels = 256
```

#### SensorFeatureConv 추가

기존:

```python
AdaptiveAvgPool2d((1, 1))
Flatten(start_dim=1)
```

변경:

```text
Conv-BN-ReLU
Conv-BN-ReLU
AdaptiveAvgPool2d
Flatten
```

즉 transformer output을 바로 평균내지 않고, sensor별 conv block으로 한 번 더 가공한 뒤 vector로 압축한다.

현재 흐름:

```text
cam_out:   (B*S, 256, 10, 25) -> SensorFeatureConv -> (B*S, 256)
lidar_out: (B*S, 256, 1, 32)  -> SensorFeatureConv -> (B*S, 256)
```

#### Adaptive token pooling 제거

기존 alpha 코드에는 FPN 이후 다음 처리가 있었다.

```python
cam_feat = F.adaptive_avg_pool2d(cam_feat, image_token_hw)
lidar_feat = F.adaptive_avg_pool2d(lidar_feat, lidar_token_hw)
```

MUVO-style FPN을 쓰면 FPN 출력 해상도 자체가 transformer token grid가 되므로 이 pooling을 제거했다.

현재:

```text
FPN output 그대로 transformer 입력
```

### `alpha26/scripts/model_variants/data_muvo.py`

`data_muvo.py`는 MUVO-style input shape에 맞춰 별도 data path로 만든다.

Image transform:

```text
기존: Resize((H, W))
현재: width가 맞는 상태에서 height를 center crop
```

현재 alpha image는 `600 x 800`이므로:

```text
600 x 800 -> center crop -> 320 x 768
```

LiDAR projection:

```text
raw point cloud Nx4
-> range-view xyzd
-> 4 x 32 x 1024
-> LIDAR_SCALE=40.0으로 나눠 모델 입력
```

`40.0`을 쓰는 이유:

```text
Alpha LiDAR range = 80m
MUVO는 100m LiDAR에 SCALE=50을 사용
따라서 Alpha는 80 / 2 = 40이 자연스러운 대응
```

Intensity는 현재 range-view 입력에서 사용하지 않는다. Alpha raw의 네 번째 값은 intensity지만, 모델 입력은 MUVO처럼 `xyzd`를 사용한다.

### `alpha26/scripts/model_variants/trainer_muvo.py`

`trainer_muvo.py`는 `models_muvo.Model`을 사용하도록 연결했다.

```python
from models_muvo import Model
```

`embedding_n_channels`는 더 이상 `128`로 고정하지 않는다.

```python
if embedding_n_channels is None:
    embedding_n_channels = getattr(self.cfg.MODEL, "EMBEDDING_DIM", 256)
```

LiDAR 입력은 scale된 값으로 학습하지만, eval metric과 visualization은 다시 meter 단위로 환산한다.

```text
loss: scaled LiDAR 값 기준
metric / visualization: LIDAR_SCALE을 곱해 meter 기준
```

### `alpha26/scripts/model_variants/train_muvo.py`

학습 entrypoint는 MUVO-style 파일들을 import하도록 정리했다.

```python
from config_muvo import cfg, PROJECT_ROOT
from data_muvo import ...
from trainer_muvo import ...
```

모델 생성도 `embedding_n_channels=128`을 넘기지 않고 config 기본값을 따른다.

```python
model = WorldModelTrainer(cfg=run_cfg, lr=run_cfg.OPTIMIZER.LR)
```

## 검증한 shape

수정 후 단위 shape 검증:

```text
cfg image/lidar: (320, 768), (32, 1024)

image FPN:
  (2, 256, 10, 24)
  tokens = 240

lidar FPN:
  (2, 256, 1, 32)
  tokens = 32

fusion output:
  image: (2, 256, 10, 25)
  lidar: (2, 256, 1, 32)

compress output:
  image: (2, 256)
  lidar: (2, 256)
```

`py_compile`도 통과했다.

추가 forward smoke:

```text
posterior rgb_1:   (1, 4, 3, 320, 768)
posterior lidar_1: (1, 4, 4, 32, 1024)
future rgb_1:      (1, 4, 3, 320, 768)
future lidar_1:    (1, 4, 4, 32, 1024)
```

이 smoke에서는 pretrained weight download를 피하기 위해 `timm.create_model(pretrained=True)`만 임시로 `pretrained=False`로 monkeypatch했다. 구조와 tensor shape 검증 목적이다.

## 아직 남은 작업

1. 실제 dataset sample 재확인

`LIDAR_RANGE_VIEW_SIZE=(32, 1024)`로 바꿨으므로 실제 Arrow sample에서 다음을 확인해야 한다.

```text
batch["lidar"].shape == (S, 4, 32, 1024)
빈 픽셀 비율
range/depth 값 범위
projection 방향이 뒤집히지 않았는지
```

현재 한 샘플 기준 shape와 값 범위는 확인했다.

```text
image:     (4, 3, 320, 768)
image_raw: (4, 3, 320, 768)
lidar:     (4, 4, 32, 1024)
```

2026-05-14 실제 `train_run_002.arrow` sanity check:

```text
dataset len: 599 windows
seq_len: 4
sample_every_n: 2
frame_step: 5

checked indices: 0, 299, 598

image:     (4, 3, 320, 768)
image_raw: (4, 3, 320, 768)
lidar:     (4, 4, 32, 1024)
action:    (4, 2)
speed:     (4,)
```

LiDAR 값 범위:

```text
scaled depth max: 약 2.0
meter depth max: 약 79.95 ~ 79.99m
valid pixel ratio: 약 0.136 ~ 0.140
non-empty rows: 31 / 32
```

Raw LiDAR 첫 frame:

```text
raw lidar shape: (4974, 4)
raw range min/max/mean: 3.567 / 79.972 / 18.468m
raw intensity min/max/mean: 0.726 / 0.986 / 0.931
projection occupied pixels: 4433 / 32768
```

이 결과상 `LIDAR_RANGE_VIEW_SIZE=(32, 1024)`와 `LIDAR_SCALE=40.0`은 현재 dataset에 맞는다.

2. Tiny / short overfit

```bash
python scripts/model_variants/train_muvo.py --epochs 1 --batch-size 1 --limit-train-batches 1 --limit-val-batches 1
```

또는 더 작은 one-window overfit 설정으로 loss가 내려가는지 확인한다.

3. Speed normalization 확인

```text
MUVO:  SPEED.NORMALISATION = 5.0
Alpha: SPEED_NORMALISATION = 50.0
```

Alpha dataset의 speed 단위가 m/s인지, km/h인지, 이미 normalize된 값인지 확인해야 한다.

`data_muvo.py`는 `speed_kmh` 컬럼을 그대로 사용한다.

```python
speeds.append(row.column("speed_kmh")[0].as_py())
```

실제 sample speed:

```text
idx 0:   [10.805, 9.516, 11.496, 0.102] km/h
idx 299: [0.0, 0.0, 0.0, 0.0] km/h
idx 598: [15.269, 21.934, 29.528, 29.704] km/h
```

따라서 현재 `SPEED_NORMALISATION=50.0`은 km/h 입력 기준으로 자연스럽다.

4. LiDAR loss weight 결정

현재 config:

```text
WEIGHT_LIDAR_RE = 0.0
WEIGHT_LIDAR_EMPTY = 0.0
```

따라서 현재 초기 학습은 사실상 RGB/RSSM 중심이다. LiDAR reconstruction까지 학습하려면 이후 작은 weight부터 켜야 한다.

## 현재 설계의 핵심 결론

Alpha는 MUVO와 큰 방식은 같게 간다.

```text
raw point cloud -> range-view xyzd -> ResNet18 -> DecoderDS/FPN -> transformer fusion
```

하지만 센서가 다르므로 해상도는 다르게 간다.

```text
MUVO LiDAR: 64-channel -> H=64
Alpha LiDAR: 32-channel -> H=32
```

따라서 현재 alpha 설정은:

```text
image: 320 x 768 -> 240 tokens
lidar: 32 x 1024 -> 32 tokens
total: 272 tokens
```

이 설정은 MUVO 원본의 token scale을 크게 벗어나지 않으면서, alpha dataset의 실제 32-channel LiDAR 조건을 반영한다.

## Transformer Fusion의 역할

이 구조에서 transformer는 camera feature와 LiDAR feature가 서로 정보를 주고받게 하는 sensor fusion module이다.

현재 alpha MUVO-style 설정에서는 한 프레임마다 다음 token이 만들어진다.

```text
image feature: 256 x 10 x 24 -> 240 tokens
lidar feature: 256 x 1 x 32  -> 32 tokens
total: 272 tokens
```

코드 흐름:

```text
cam_feat:   (B*S, 256, 10, 25)
lidar_feat: (B*S, 256, 1, 32)

cam_tokens:   (250, B*S, 256)
lidar_tokens: (32,  B*S, 256)
fused:        (282, B*S, 256)
```

Transformer는 token 수를 바꾸지 않는다.

```text
input:  (282, B*S, 256)
output: (282, B*S, 256)
```

하지만 attention을 통해 각 token이 다른 token을 참고할 수 있다.

```text
camera token -> lidar token 참고 가능
lidar token  -> camera token 참고 가능
camera token끼리도 참고 가능
lidar token끼리도 참고 가능
```

예를 들어 camera feature가 "앞쪽에 차량처럼 보이는 패턴"을 갖고 있고, LiDAR feature가 "그 방향 18m 근처에 실제 점들이 있음"을 갖고 있다면, transformer는 두 정보를 같은 fused representation 안에서 결합할 수 있다.

Transformer output은 다시 sensor별로 쪼갠다.

```text
out[:250]  -> image tokens -> (B*S, 256, 10, 25)
out[250:]  -> lidar tokens -> (B*S, 256, 1, 32)
```

즉 transformer 이후에도 image와 lidar의 spatial grid는 유지된다.

## B, S의 의미

입력 tensor의 앞 두 축:

```text
B = batch size
S = sequence length / time steps
```

예를 들어 `B=2`, `S=4`이면 한 batch에 2개의 주행 window가 있고, 각 window는 4프레임짜리라는 뜻이다.

입력:

```text
image: (B, S, 3, H, W)
lidar: (B, S, 4, H, W)
```

ResNet/FPN은 frame 단위 image-like tensor를 처리하므로, encoder에 넣기 전에 `B`와 `S`를 합친다.

```text
(B, S, C, H, W)
-> (B*S, C, H, W)
```

이후 sensor fusion과 feature compression까지 `(B*S, ...)`로 처리하고, 최종 embedding을 다시 sequence 형태로 복원한다.

```text
(B*S, 256)
-> (B, S, 256)
```

RSSM은 이 `(B, S, embedding_dim)` 시퀀스를 시간 순서대로 처리한다.

## Sensor Feature Compression의 의미

Transformer output은 아직 공간 feature map이다.

```text
image: (B*S, 256, 10, 25)
lidar: (B*S, 256, 1, 32)
```

이는 "하나의 숫자"가 아니라, 각 위치마다 256차원 feature vector가 있다는 뜻이다.

```text
image: 10 x 24 = 240개 위치
각 위치마다 256차원 vector
```

RSSM은 프레임당 하나의 observation vector를 받으므로, 공간 feature map을 sensor별로 하나의 vector로 요약해야 한다.

```text
image: 250개의 256차원 vector -> 1개의 256차원 vector
lidar: 32개의 256차원 vector  -> 1개의 256차원 vector
```

현재 `SensorFeatureConv`는 다음 흐름이다.

```text
Conv-BN-ReLU
Conv-BN-ReLU
AdaptiveAvgPool2d((1, 1))
Flatten
```

의미:

```text
학습 가능한 conv가 spatial feature를 먼저 정리한다.
그 다음 전체 spatial grid를 평균내어 sensor별 vector로 만든다.
```

결과:

```text
image_feature: (B*S, 256)
lidar_feature: (B*S, 256)
speed_feature: (B*S, 16)
```

이후 concat:

```text
256 + 256 + 16 = 528
```

`features_combine`은 이 528차원 vector를 RSSM 입력 크기인 256차원으로 다시 만든다.

```text
(B*S, 528) -> Linear -> (B*S, 256)
```

이 256차원 vector가 한 프레임의 fused observation embedding이다.

```text
(B*S, 256) -> (B, S, 256) -> RSSM
```

## MUVO의 Image ResNet과 LiDAR ResNet 차이

MUVO는 image와 LiDAR에 같은 ResNet18 구조를 쓰지만, 같은 network instance를 공유하지 않는다.

```python
self.encoder = timm.create_model(
    cfg.MODEL.ENCODER.NAME,
    pretrained=True,
    features_only=True,
    out_indices=[2, 3, 4],
)

self.range_view_encoder = timm.create_model(
    cfg.MODEL.LIDAR.ENCODER,
    pretrained=True,
    features_only=True,
    out_indices=[2, 3, 4],
    in_chans=4,
)
```

의미:

```text
image encoder: ResNet18, RGB 3채널 입력
lidar encoder: ResNet18, xyzd 4채널 입력
```

둘은 구조는 같지만 서로 다른 모델 객체이며, 학습되면서 서로 다른 weights를 갖는다.

```text
image ResNet18:
  RGB edge, texture, road marking 등을 학습

lidar ResNet18:
  range-view xyzd pattern, distance structure, empty-space pattern 등을 학습
```

Alpha MUVO-style도 같은 방식을 따른다.

```python
self.image_encoder = timm.create_model(
    "resnet18",
    pretrained=True,
    features_only=True,
    out_indices=[2, 3, 4],
)

self.lidar_encoder = timm.create_model(
    "resnet18",
    pretrained=True,
    features_only=True,
    out_indices=[2, 3, 4],
    in_chans=4,
)
```

즉 같은 설계도의 ResNet18을 두 번 만든다.

```text
하나는 RGB용
하나는 LiDAR range-view용
```

## Pretrained와 Freeze

`pretrained=True`는 pretrained weight로 초기화한다는 뜻이지, 자동으로 freeze한다는 뜻은 아니다.

```text
pretrained=True:
  ImageNet pretrained weights에서 시작

requires_grad=True:
  학습 중 gradient를 받고 update됨
```

Freeze하려면 명시적으로 다음 코드가 있어야 한다.

```python
for p in self.image_encoder.parameters():
    p.requires_grad = False
```

MUVO 원본에는 encoder freeze 코드가 없다. 따라서 MUVO도 image/lidar ResNet18을 pretrained weight로 시작하지만 학습 중 update한다.

Alpha MUVO-style도 현재 freeze하지 않는다.

```text
image_encoder weights update됨
lidar_encoder weights update됨
```

LiDAR encoder는 `in_chans=4`이므로 ImageNet ResNet18의 3채널 첫 conv weight를 timm이 4채널에 맞게 adapt해서 초기화한다. 이후 LiDAR 데이터에 맞게 학습된다.

## LiDAR와 ReLU에 대한 논의

LiDAR raw channel에는 부호가 물리적 의미를 가진 값이 있다.

```text
x: 앞/뒤 방향, 음수 가능
y: 좌/우 방향, 음수 가능
z: 위/아래 방향, 음수 가능
range: 양수
```

따라서 raw 입력 초반에서 음수 좌표를 무작정 잘라내는 것은 좋지 않다.

다만 현재 ReLU가 적용되는 위치는 raw LiDAR 값 자체가 아니라, conv를 지난 learned feature이다. ResNet18 내부에서도 이미 여러 ReLU가 존재한다.

MUVO 원본도 LiDAR용 ResNet18, DecoderDS, feature conv에서 ReLU 계열 activation을 사용하며, LiDAR만 별도로 activation을 바꾸지는 않는다.

따라서 "MUVO 원본을 따른다"는 관점에서는 ReLU 사용이 자연스럽다. 다만 alpha에서 LiDAR 부호 보존을 더 보수적으로 보고 싶다면, 새로 추가한 `SensorFeatureConv` 정도는 `ReLU` 대신 `LeakyReLU(0.1)`로 바꾸는 개선을 실험할 수 있다.

현재 상태:

```text
MUVO 원본에 더 가깝게 ReLU 유지
추후 ablation으로 LeakyReLU 검토 가능
```
