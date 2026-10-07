# muvo 원본 vs alpha 현재 — Decoder/Head 구조 비교 (비전공자용)

작성일: 2026-05-15

`_muvo_comparison.md` Step 2 (A)/(B) 진입 전 정지(整地). 흐릿한 RGB recon의
구체적 원인 후보를 코드 흐름과 ASCII 다이어그램으로 설명한다.

## 0. 전체 그림: Decoder가 하는 일

World model은 4프레임을 보고 머릿속에 작은 "장면 요약"(RSSM hidden state +
stochastic state, 합쳐서 약 256~384차원 vector)을 만든다. Decoder는 그 작은
vector를 받아 다시 **RGB 이미지**(216×288 픽셀)나 **LiDAR range-view**로
복원하는 모듈이다.

```text
RSSM 출력 (B, S, 256~384차원 vector)
                    ↓ Decoder
RGB 이미지 (B, S, 3, 216, 288)
또는 LiDAR  (B, S, 4, 32,  1024)
```

이게 잘 안 되면 "보기엔 비슷한데 디테일이 뭉개진" 흐릿한 그림이 나온다.

## 1. 작은 vector를 큰 그림으로 펼치는 방법

핵심 도구: `nn.ConvTranspose2d` (이름이 길지만 그냥 **학습 가능한 업샘플**이라고
생각하면 된다).

```text
일반 Conv:      큰 그림  → 작은 feature  (요약)
ConvTranspose:  작은 feature → 큰 그림    (펼침)
```

비유로 비교:

```text
bilinear interpolate:
  복사기로 이미지를 2배로 확대 → 픽셀 사이를 평균으로 채움
  → 학습 안 함, 항상 부드럽게 흐려짐

ConvTranspose2d:
  학습된 화가가 작은 스케치를 보고 큰 캔버스에 다시 그림
  → 학습됨, 디테일을 살려서 채울 수 있음
```

bilinear는 빠르고 안전하지만 본질적으로 흐릿하다. ConvTranspose는 학습되지만,
**kernel size와 stride가 안 맞으면 격자 무늬(checkerboard) 아티팩트**가 생긴다.
이건 잘 알려진 현상이다 (https://distill.pub/2016/deconv-checkerboard/).

### Checkerboard가 왜 생기나

`ConvTranspose2d(kernel_size=4, stride=2, padding=1)`는 출력 크기가 정확히 2배가
되지만, 가중치 겹침이 위치마다 불균등하다. 일부 출력 픽셀은 2개의 kernel weight,
일부는 4개의 kernel weight가 겹쳐서 만들어진다 → 학습 가능하지만 시작점부터
편향이 생기고, 출력에 격자 패턴이 자주 보인다.

회피 방법:

```text
방법 1: kernel_size를 stride의 정수배로 (예: k=4, s=2 또는 k=6, s=2)
방법 2: 홀수 kernel + output_padding (예: k=5, s=2, p=2, output_padding=1)
방법 3: bilinear upsample + 그 위에 stride=1 Conv ("resize-conv", 안 쓰는 곳도 많음)
```

upstream muvo는 방법 1과 2를 혼합해서 쓴다. alpha 현재 코드는 `k=4`도 쓰지만
`s=2`와 함께 쓸 때 `p=1`을 쓰므로 방법 1을 따른다. **이론적으로 alpha의 k=4/s=2/p=1도
checkerboard를 피할 수 있는 조합이지만**, upstream의 다양한 kernel 조합(k=5+output_padding,
k=6)이 경험적으로 더 자연스러운 출력을 만든다는 게 일반적인 관찰이다.

## 2. upstream muvo `ConvDecoder` 단계별 흐름

위치: `muvo/muvo/models/common.py:549-632`. RGB 출력 기준, target=320×832 가정.

```text
입력: (B, latent_n_channels)  ← RSSM에서 나온 vector

linear:
  Linear(latent_n_channels → 512)
  Unflatten → (B, 512, 1, 1)

pre_transpose_conv: 5단계 업샘플
  ConvTranspose(k=(5,13))            → (B, 512,   5,  13)   ← base 크기 직접 키움
  ConvTranspose(k=5, s=2, p=2, op=1) → (B, 512,  10,  26)   ← 2배
  ConvTranspose(k=5, s=2, p=2, op=1) → (B, 512,  20,  52)   ← 2배
  ConvTranspose(k=6, s=2, p=2)       → (B, 512,  40, 104)   ← 2배

trans_conv1:
  ConvTranspose(k=6, s=2, p=2)       → (B, 256,  80, 208)   ← 2배, 채널 절반
head_4:
  Conv2d(256 → 3, k=1)               → (B,   3,  80, 208)   ← rgb_4 (1/4 해상도)

trans_conv2:
  ConvTranspose(k=6, s=2, p=2)       → (B, 128, 160, 416)   ← 2배, 채널 절반
head_2:
  Conv2d(128 → 3, k=1)               → (B,   3, 160, 416)   ← rgb_2 (1/2 해상도)

trans_conv3:
  ConvTranspose(k=6, s=2, p=2)       → (B,  64, 320, 832)   ← 2배, 채널 절반
head_1:
  Conv2d(64 → 3, k=1)                → (B,   3, 320, 832)   ← rgb_1 (full 해상도)

출력: {'rgb_4': ..., 'rgb_2': ..., 'rgb_1': ...}
```

핵심 관찰:

- `head_4`, `head_2`, `head_1` **각각 자기 전용 학습 가능 업샘플(trans_conv*)을
  거친 후**에 1×1 conv head를 통과한다. 즉 full resolution은 학습 가능한
  방식으로 정확히 만들어진다.
- head는 정말 1×1 conv만 — 보간이나 resize 없음.
- 채널 수 progression: 512 → 256 → 128 → 64. 해상도가 커질수록 채널 수가 줄어
  메모리/연산이 폭주하지 않게 균형 잡힘.

## 3. alpha `SensorDecoder` 현재 흐름

> **NOTE (2026-05-15 정정)**: 본 §3은 작성 당시(2026-05-14) 기준 — `models.py`에서
> 가져온 미마이그레이션 상태의 SensorDecoder (target=216×288, bilinear interpolate
> 포함) 분석이다. **commit `7099f87`에서 upstream ConvDecoder 패턴으로 마이그레이션
> 완료**되었고 target도 (320, 768)로 변경됨. 아래 분석은 §6.1의 "흐릿함 원인" 추적
> 맥락을 이해하기 위한 historical snapshot으로 유지. 현재 디코더 구조는 upstream의
> ConvDecoder와 사실상 동일 (자세히는 `models_muvo.py`의 `ConvDecoder` 직접 참조).

위치: `alpha26/scripts/model_variants/models_muvo.py:304-365` (작성 당시 라인 번호).
target=216×288 가정.

```text
입력: (B, S, hidden+sample = 256+128 = 384)  ← RSSM hidden + stochastic 합침

cat + reshape → (B*S, 384)

linear:
  Linear(384 → 512)
  Unflatten → (B*S, 512, 1, 1)

base_conv: 4단계 업샘플 (base 크기 = ceil(216/32), ceil(288/32) = (7, 9))
  ConvTranspose(k=(7,9))             → (B*S, 512,  7,  9)
  ConvTranspose(k=4, s=2, p=1)       → (B*S, 512, 14, 18)
  ConvTranspose(k=4, s=2, p=1)       → (B*S, 512, 28, 36)
  ConvTranspose(k=4, s=2, p=1)       → (B*S, 512, 56, 72)

up1:
  ConvTranspose(k=4, s=2, p=1)       → (B*S, 256, 112, 144)   ← 2배, 채널 절반
head_4 (output_size=(108, 144)):
  Conv2d(256 → 3, k=1)               → (B*S,   3, 112, 144)
  F.interpolate(bilinear)            → (B*S,   3, 108, 144)   ← rgb_4 (실제로는 1/2 해상도)

up2:
  ConvTranspose(k=4, s=2, p=1)       → (B*S, 128, 224, 288)   ← 2배, 채널 절반
head_2 (output_size=(216, 288)):
  Conv2d(128 → 3, k=1)               → (B*S,   3, 224, 288)
  F.interpolate(bilinear)            → (B*S,   3, 216, 288)   ← rgb_2 (target 해상도)

refine (no upsample):
  Conv2d(128 → 128, k=3, p=1)        → (B*S, 128, 224, 288)
head_1 (output_size=(216, 288)):
  Conv2d(128 → 3, k=1)               → (B*S,   3, 224, 288)
  F.interpolate(bilinear)            → (B*S,   3, 216, 288)   ← rgb_1 (target 해상도)

reshape back → (B, S, 3, H, W)
```

핵심 관찰 (upstream과 다른 점):

1. **`head_1`을 위한 별도 학습 가능 업샘플이 없다.**
   - upstream은 `trans_conv3 (k=6, s=2)`로 한 번 더 학습 가능하게 키워서
     full resolution을 만든다.
   - alpha는 `head_2`와 `head_1`이 **같은 224×288 feature map을 공유**하고,
     1×1 conv weight만 다르다. 즉 full resolution을 만드는 추가 학습이 없다.

2. **`refine`은 학습 가능하지만 업샘플은 아니다.**
   - 3×3 conv 한 번. 공간 해상도 변화 없음.
   - upstream의 `trans_conv3`(s=2 업샘플)를 대체하지 않는다.

3. **모든 head 출력에 bilinear interpolate가 강제됨.**
   - decoder의 ConvTranspose 결과 해상도(7→14→28→56→112→224)가 target(216×288)과
     **정확히 안 맞기 때문**.
   - 112 → 108은 ~3.6%, 224 → 216은 ~3.6%, 224 → 216은 ~3.6% 다운샘플.
   - bilinear는 학습 안 되는 평균 보간이라 흐릿함을 한 번 더 추가한다.

4. **head_4와 head_1이 실제로는 "downsample 4"가 아니다.**
   - 이름은 `rgb_4` (downsample 4), `rgb_2` (downsample 2), `rgb_1` (full)이지만
     실제 출력 해상도는 (108, 144), (216, 288), (216, 288) — head_2와 head_1이
     같은 해상도!
   - trainer는 target을 pred shape에 맞춰 resize하므로 loss는 계산되지만,
     multi-scale supervision 구조가 의도와 다르게 동작 중일 수 있다.

5. ConvTranspose kernel
   - upstream: `k=5, s=2, p=2, output_padding=1` + `k=6, s=2, p=2` 혼합
   - alpha: `k=4, s=2, p=1` 단일
   - alpha 조합도 이론적으로 checkerboard 회피 가능하지만, upstream 패턴이
     경험적으로 더 부드러운 출력을 만든다는 보고가 많다.

## 4. 비유로 차이 정리

```text
upstream ConvDecoder:
  화가가 캔버스를 점점 큰 사이즈로 옮겨가며 그린다.
  마지막 캔버스(full resolution)에 도착해서 색만 RGB로 정리하고 끝.
  → 모든 디테일이 학습된 붓질로 채워짐.

alpha SensorDecoder (현재):
  화가가 1/2 사이즈 캔버스까지만 그린다.
  그 결과를 복사기로 1.04배 줄여서 출력 사이즈에 맞춤.
  rgb_2와 rgb_1은 같은 그림에 다른 사인을 한 셈.
  → 사이즈 보정과 full resolution 학습이 사라짐.
```

## 5. 흐릿함과의 직접 연관

`_muvo_comparison.md` §7에서 RGB blurriness 원인 후보 순위:

1. **SSIM/perceptual loss 부재** — `trainer_ssim.py`로 검증 중 (별도 작업).
2. **Decoder 단순화로 인한 checkerboard 가능성** — alpha k=4/s=2/p=1.
3. **`SensorHead`의 bilinear interpolate** — 학습 불가 보간으로 디테일 평균화.

이 문서 기준 직접 타깃은 **2 + 3**. 특히 **rgb_1을 위한 trans_conv3 추가**가
핵심이다. 그렇게 하면:

- bilinear interpolate 불필요 (decoder가 정확한 해상도 생성)
- rgb_1과 rgb_2가 진짜로 다른 해상도/다른 trans_conv 결과가 됨
- head는 upstream처럼 1×1 conv만 → 보간 노이즈 0

## 6. 마이그레이션 시 알아야 할 디자인 결정

### 6.0 upstream muvo가 target size 320×832를 고른 이유

upstream은 RGB target을 임의로 정하지 않았다. **decoder 수학이 깨끗하게
떨어지도록** CARLA raw 캡처를 명시적으로 crop했다.

```text
muvo/muvo/config.py:
  _C.IMAGE.SIZE = (600, 960)
  _C.IMAGE.CROP = [64, 138, 896, 458]
    crop width  = 896 - 64  = 832
    crop height = 458 - 138 = 320
  → crop 결과 320 × 832 (이게 encoder INPUT 이자 decoder OUTPUT target)
```

수학적 의도:

```text
320 = 2^6 × 5      ← constant_size 높이 = 5
832 = 2^6 × 13     ← constant_size 너비 = 13

ConvDecoder 6단계 stride-2 업샘플 + 시작 constant_size 곱하기:
  (1, 1) → (5, 13)              ConvTranspose(k=constant_size)
        → (10, 26)               2x
        → (20, 52)               2x
        → (40, 104)              2x    ← pre_transpose_conv 끝 (8x)
        → (80, 208)              2x    ← trans_conv1 → head_4
        → (160, 416)             2x    ← trans_conv2 → head_2
        → (320, 832)             2x    ← trans_conv3 → head_1

총 6번의 2x upsample × constant_size = 64 × (5, 13) = (320, 832) ✓
모든 단계 정수, bilinear interpolate 0번.
```

즉 upstream은 **input == output**을 강제하고, 그 둘을 동시에 `2^N × 작은 정수`
형태가 되는 사이즈로 골랐다.

### 6.1 alpha의 현재 target 216×288은 어디서 왔나

```text
alpha CARLA raw: 600 × 800 (Immanuel dataset)
encoder input (non-_muvo): (300, 400)  ← transforms.Resize, 32 배수 아님
encoder input (_muvo):     (320, 800)  ← center crop, 32 배수
decoder target RGB_RECON_SIZE: (216, 288)  ← encoder와 독립 선택
```

216의 인수분해: `216 = 2^3 × 27`. 27은 작은 소수배가 아니라 upstream의
`2^6 × 작은 정수` 패턴을 따라갈 수 없다.

현재 alpha decoder는 5번 doubling만 사용한다:

```text
constant_size = ceil((216, 288) / 32) = (7, 9)
(1,1) → (7,9) → (14,18) → (28,36) → (56,72) → (112,144) → (224,288)
       base                                        up1          up2
총 5 doublings × 32 × constant_size = (224, 288)
```

(224, 288)이지 (216, 288)이 아니라서, head의 `F.interpolate(bilinear, size=(216,288))`이
강제 발동 — **이게 흐릿함 직접 원인.**

`216`이라는 숫자가 어디서 왔는지 코드만으로는 명확치 않다. 아마도 "encoder 입력보다
작은 reconstruction target으로 메모리/속도 절약" 의도 (216×288 ≈ 0.6 megapixel vs
320×800 ≈ 2.6 megapixel). 다만 그 트레이드오프가 흐릿한 출력 비용으로 돌아오고 있다.

### 6.2 32-multiple 그리고 upstream 패턴까지 만족하는 후보

`alpha` target을 바꾼다면 candidates:

| target (H, W) | constant for 6-doubling | constant for 5-doubling | 비고 |
|---|---|---|---|
| 192 × 256 | (3, 4) | (6, 8) | 현재보다 작음 |
| 224 × 288 | (3.5, 4.5) ✗ | (7, 9) | 5-doubling만 가능 |
| 256 × 384 | (4, 6) | (8, 12) | 현재보다 큼 |
| **320 × 800** | (5, 12.5) ✗ | **(10, 25)** | 작성 당시 IMAGE_INPUT_SIZE로 골라뒀던 사이즈 |
| **320 × 768** | **(5, 12)** ✓ | **(10, 24)** | **commit `7099f87`에서 실제 채택. 6/5-doubling 둘 다 깨끗** |
| 320 × 832 | (5, 13) | (10, 26) | upstream과 동일 |

### 6.3 추천: target = IMAGE_INPUT_SIZE = (320, 800) + 5-doubling

> **실제 채택 (commit `7099f87`)**: `(320, 768)` (§6.2 표의 추가된 행 참조).
> 작성 당시 추천은 `(320, 800)` (5-doubling만 가능)이었으나, 구현 시점에 한 번 더
> 검토하면서 `(320, 768)`로 변경 — `768 = 64 × 12 = 32 × 24`로 **6-doubling과
> 5-doubling 둘 다 깨끗하게 떨어짐** + width 메모리 약간 절약. input==output 철학은
> 동일. 아래 shape 계산식의 `(320, 800)` / `(10, 25)` 부분만 `(320, 768)` / `(10, 24)`로
> 대체하면 현재 구조와 동일.

`config_muvo.py`가 이미 `IMAGE_INPUT_SIZE=(320, 800)`을 골라놓은 상태이므로,
RGB_RECON_SIZE도 같은 값으로 두면 upstream의 **input == output** 철학이
그대로 따라온다. 5-doubling 패턴으로 깨끗하게 떨어진다.

```text
(1, 1)
  ConvTranspose(k=(10, 25))         → (B, 512,  10,  25)   ← constant_size
  ConvTranspose(k=5, s=2, p=2, op=1) → (B, 512,  20,  50)   ← pre_transpose × 2
  ConvTranspose(k=6, s=2, p=2)       → (B, 512,  40, 100)
  ─────────────────────────────────────────────────────────
  trans_conv1: k=6, s=2, p=2         → (B, 256,  80, 200)   ← rgb_4 (1/4)
  trans_conv2: k=6, s=2, p=2         → (B, 128, 160, 400)   ← rgb_2 (1/2)
  trans_conv3: k=6, s=2, p=2         → (B,  64, 320, 800)   ← rgb_1 (full)

각 head: Conv2d(채널 → 3, k=1) 만. bilinear interpolate 없음.
```

upstream과의 차이:

| | upstream | alpha 제안 |
|---|---|---|
| 총 doublings | 6 (×64) | 5 (×32) |
| constant_size | (5, 13) | (10, 25) |
| pre_transpose doublings | 3 | 2 |
| trans_conv doublings (= heads) | 3 | 3 (동일) |

upstream의 `pre_transpose_conv` 첫 stage를 빼고 그 자리를 더 큰 `constant_size`로
채운 형태. trans_conv 3단계 + 각 head 전용 학습 가능 업샘플이라는 핵심 설계는 동일.

#### LiDAR도 같은 패턴으로 자연 적합

`LIDAR_RANGE_VIEW_SIZE=(32, 1024)`도 5-doubling clean:

```text
constant_size = (32/32, 1024/32) = (1, 32)
head_4: (8, 256) — 1/4
head_2: (16, 512) — 1/2
head_1: (32, 1024) — full
```

즉 같은 `SensorDecoder` 한 곳을 고치면 RGB와 LiDAR 모두 cover.

### 6.4 (216, 288) 유지하면서 흐릿함 줄이기 (대안)

비표준 decoder 설계. 마지막 한 단계의 kernel/stride/padding을 (216, 288) 정확히
떨어지도록 직접 계산해야 함. 가능하지만 upstream 패턴에서 멀어지고 잠재적 버그
risk가 큼. 위 6.3 추천 대비 명확한 이점 없음.

### 6.5 target_size가 32로 안 나누어떨어질 때 (일반론)

```text
upstream: constant_size 하드코딩 (5, 13) — RGB 160×416 전제
  → 5 × 2 × 2 × 2 × 2 × 2 = 160 ✓
  → 13 × 2 × ... = 416 ✓
  깔끔하게 떨어짐.

alpha: target=(216, 288), 32로 안 떨어짐.
  → constant_size=(7, 9), 5단계 업샘플하면 (224, 288)
  → 224 ≠ 216, bilinear가 강제됨.
```

해결 방법 후보:

```text
(a) target_size를 32 배수로 조정 (예: 192×288 또는 224×288).
    데이터 파이프라인 영향: rgb_label_1 크기 변경.

(b) base_conv 단계 하나를 줄이고 마지막에 다른 kernel로 정확히 맞춤.
    예: 4단계 → 3단계 + 마지막 trans_conv를 k=(특정,특정)로.

(c) constant_size를 다르게 (예: (7,9) 대신 직접 (216/32, 288/32) 계산식 변경).
    가능한 결과 크기를 다양화.
```

upstream 패턴(`ConvDecoder`)을 가장 가깝게 따르려면 (a)가 깔끔하지만 data 변경이
다른 곳에 영향을 줄 수 있다. (b)/(c)는 코드 변경만으로 가능하지만 신중한 shape
계산이 필요하다.

### 6.6 채널 수 progression

```text
upstream: 512 → 256 → 128 → 64
alpha:    512 → 256 → 128 → 128 (refine, 같은 채널)
```

upstream을 따라가면 마지막 trans_conv3에서 64 채널까지 줄인다. 1×1 conv head가
64 채널만 보고 RGB를 생성. 이게 정보 압축을 한 번 더 강제해서 좋은 representation을
만들도록 한다는 게 통상적 직관.

## 7. (D) RSSM pre_gru 의 latent_action concat

다른 후보: alpha `RSSM`의 `pre_gru`에 `latent_action`을 concat하는 부분이
upstream에 없다. `_muvo_comparison.md` §4에서 "의도된 변경인지 확인 필요"로
flag된 부분. 이건 decoder/head와 직교한 변경이라 우선순위는 낮다.

다음 작업 진행 결정 시 참고:

- **(A+B) decoder/head 동시 변경**이 흐릿함을 가장 직접 공격한다. SSIM 결과와
  독립적으로 의미 있는 변경.
- **(C) SensorFeatureConv → BasicBlock**은 encoder쪽이라 decoder 흐릿함과는
  직접 관련이 약하다.
- **(D) pre_gru 검토**는 별도 의사결정 (수정 vs 문서화).
