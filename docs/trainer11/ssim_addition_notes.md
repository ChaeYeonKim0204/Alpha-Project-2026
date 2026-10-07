# SSIM Loss Addition Notes

작성일: 2026-05-15

흐릿한 RGB reconstruction 문제 진단(`scripts/model_variants/_muvo_comparison.md` §7)의 1순위 후보인
**SSIM/perceptual loss 부재**를 가장 가성비 높은 방법으로 검증하기 위해, non-_muvo
`trainer.py` baseline 위에 SSIM term만 얹은 sub-trainer를 추가했다.

## 동기

`alpha26` 기존 RGB recon loss는 L1만 사용한다 (`scripts/model_variants/trainer.py` `compute_loss`의
`rgb_label_1` 블록). L1만 쓰면 픽셀 평균을 맞추는 방향이 안전 minimum이라
모델이 "평균색으로 칠하는" 흐릿한 출력을 선호하게 된다.

upstream `muvo/muvo/trainer.py:97-98, 312-318`은 RGB recon에 SSIM을 함께 쓴다:

```python
self.ssim_loss = SSIMLoss(channel=3)
...
ssim_loss = 1 - self.ssim_loss(prediction=output[f'rgb_{factor}'],
                               target=batch[f'rgb_label_{factor}'])
ssim_weight = 0.6
losses[f'ssim_{factor}'] = rgb_weight * discount * ssim_loss * ssim_weight
```

`alpha26` 코드 전체에는 SSIM이 한 줄도 없다 (`grep -ri ssim alpha26/` 확인 완료).
이게 흐릿함의 단일 가장 가능성 높은 원인이라는 게 `_muvo_comparison.md` §7의 결론.

`_muvo` 분기는 muvo-style 전체 fusion/encoder 구조를 옮기는 큰 작업이다.
SSIM 추가는 그것과 직교한 작은 변경으로 먼저 효과를 가늠해보는 게 합리적이다.

## 설계 결정

> **NOTE (2026-05-15 정정)**: 아래 §설계 결정 + §추가/수정한 파일은 초기 SSIM 시도
> 시점 (non-_muvo `trainer.py` 위에 `trainer_ssim.py` + `train_ssim.py` subclass +
> monkey-patch 패턴) 의 기록이다. **commit `08336bc`에서 `trainer_muvo.py`에 SSIM
> 직접 통합 (`cfg.LOSSES.WEIGHT_SSIM` 토글)** 으로 마이그레이션 완료 — `trainer_ssim.py`/
> `train_ssim.py`는 non-_muvo baseline 비교용으로 잔존하지만 새 실험은 모두
> `train_muvo.py --weight-ssim ...` 형태로 진행한다. 자세한 통합은 이 문서 아래쪽
> §`_muvo` decoder × SSIM × dim 2×2 매트릭스 + §정정 참조.

### Subclass + monkey-patch

`trainer.py`(720줄)를 통째로 복제해서 SSIM만 추가하는 건 유지보수 부담이 크다.
대신:

```text
scripts/model_variants/trainer_ssim.py
  - SSIMLoss (upstream losses.py에서 인라인)
  - class SSIMWorldModelTrainer(WorldModelTrainer):
      __init__ → super 호출 후 self.ssim_loss / self.weight_ssim 추가
      compute_loss → super 호출 후 rgb_{1,2,4}별 SSIM term 추가

scripts/model_variants/train_ssim.py
  import train
  from trainer_ssim import SSIMWorldModelTrainer
  train.WorldModelTrainer = SSIMWorldModelTrainer  # monkey-patch
  if __name__ == "__main__":
      train.main()
```

baseline `trainer.py`/`train.py`는 한 줄도 건드리지 않는다.

### SSIM weight

upstream pattern을 alpha LOSSES 키에 맞춰 재구성:

```python
losses[f"{prefix}ssim_{factor}"] = (
    self.weight_rgb * discount * self.weight_ssim * (1 - ssim_value)
)
```

- `self.weight_rgb`: `cfg.LOSSES.WEIGHT_RGB` (현재 1.0)
- `discount`: `1 / factor` (1, 1/2, 1/4)
- `self.weight_ssim`: `cfg.LOSSES.WEIGHT_SSIM`을 `getattr` fallback `0.6`으로 읽음 → `config.py` 수정 없음
- `ssim_value`: SSIMLoss forward 결과 (1에 가까울수록 유사)

`prefix=""`와 `prefix="future_"` 양쪽 모두에 진입한다 (alpha trainer는
posterior recon과 future imagination 양쪽에 같은 `compute_loss`를 호출).

### Pixel clamp

SSIM은 동적 범위 `L=1` 가정으로 만들어졌다. AMP/FP16 환경에서 decoder 출력이 [0,1]을
약간 넘는 경우가 있어, SSIM 입력 직전에 `clamp(0.0, 1.0)`을 명시한다.

```python
pred_clamped   = pred.clamp(0.0, 1.0)
target_clamped = target.clamp(0.0, 1.0)
ssim_term      = 1.0 - self.ssim_loss(pred_clamped, target_clamped)
```

target은 이미 `image_raw`에서 [0,1] 범위이므로 사실상 prediction쪽 가드.

## 추가/수정한 파일

### `scripts/model_variants/trainer_ssim.py` (신규)

- `SSIMLoss` 클래스: `muvo/muvo/losses.py:292-348`을 그대로 인라인. (B, S, C, H, W) 입력 지원.
- `SSIMWorldModelTrainer(WorldModelTrainer)`: __init__ + compute_loss 오버라이드.

총 약 90줄.

### `scripts/model_variants/train_ssim.py` (신규)

- `train` 모듈을 import한 뒤 `train.WorldModelTrainer`를 SSIM subclass로 monkey-patch.
- `train.main()` 호출.

총 약 20줄. train.py 본문(337줄)을 복제하지 않는다.

## 검증

### Smoke test (20 epoch, one-window-overfit)

```bash
python train_ssim.py --one-window-overfit --epochs 20 \
  --run-name ssim_smoke_20ep --num-workers 0 \
  --no-reconstruction --no-metric-export
```

PyTorch Lightning model summary에 새 모듈이 등록됨:

```text
| Name      | Type     | Params
0 | model     | Model    | 76.9 M
1 | ssim_loss | SSIMLoss | 0
```

TensorBoard scalar에서 SSIM 항목 출력 확인:

```text
train_ssim_1        : 0.5844 → 0.5791 (n=20)
train_ssim_2        : 0.2600 → 0.2596
train_ssim_4        : 0.1410 → 0.1401
train_future_ssim_1 : 0.5970 → 0.5943
train_future_ssim_2 : 0.2796 → 0.2796
train_future_ssim_4 : 0.1459 → 0.1455
train_loss_epoch    : 3.5849 → 3.5669
```

20 epoch은 OneCycleLR 워밍업이 끝나기도 전이라 절대값 변화 자체는 미미하지만:

- SSIM 항목이 `prefix=""`와 `prefix="future_"` 양쪽 모두 진입함.
- 총 loss에서 SSIM 합 ≈ 1.0 / 3.58 ≈ **28%** 비중 (L1 + future L1 ≈ 44%, KL ≈ 0.3%).
- gradient flow 정상, 모델 학습은 천천히 진행.

### 1000-epoch full overfit — 결과: **현재 decoder에서는 SSIM이 더 악화**

동일 설정 (1-window, h=256/z=128, batch=1, OneCycleLR, lr=1e-4, RGB-only)에서
SSIM term만 추가한 결과는 baseline 대비 **눈에 띄게 나빠짐**.

#### Reconstruction figure 비교

```text
baseline:
  results/figures/one_window_overfit_train002_h256_z128_rgbonly_1000epoch_reconstruction_s0.png
  → Posterior RGB가 Observed와 거의 동일. 트럭/도로/원근감 모두 살아있음. clean.

SSIM 추가:
  results/figures/ssim_one_window_overfit_train002_h256_z128_1000epoch_reconstruction_s0.png
  → Posterior RGB에 **체커보드 아티팩트 명확히 보임**. blocky grid pattern.
  → future imagination도 같은 패턴.
```

#### Loss curve

```text
baseline: total loss 1.6 → 0.2 부드럽게 단조 감소. KL ≈ 0.02에서 안정.

SSIM:     total loss 3.5 시작 → step 250 근처에서 ~15로 폭발 → 진동 후
          6-8에서 settle. baseline 0.2에 근처도 못 옴. KL도 같은 시점에 spike(0.4).
          → 학습 자체가 불안정.
```

#### 원인 분석

**(1) SSIM weight 과도** — upstream과 alpha의 weight 합성이 다름:

```text
upstream:  rgb_weight(0.1) × ssim_weight(0.6) = effective 0.06
alpha현재:  WEIGHT_RGB(1.0) × ssim_weight(0.6) = effective 0.6  ← upstream의 10×
```

`SSIMWorldModelTrainer.compute_loss`는 `self.weight_rgb * discount * self.weight_ssim`
조합인데, 알파의 `WEIGHT_RGB=1.0`이 upstream의 `0.1`과 다르다. SSIM term이
L1 term과 동등 스케일에서 일종의 "structure만 맞추고 픽셀은 무시" 압력으로 작용.

**(2) 현재 alpha `SensorDecoder` (k=4/s=2/p=1)는 checkerboard 내재 편향** —
L1만 쓰면 모델은 smooth 도피로 격자 안 보이는 평균값 출력. SSIM은 그 도피를
막아 decoder grid 편향을 노출시킴.

#### 결론

이 한 번의 실험으로 SSIM 자체를 폐기하긴 이름. 합리적 후속 가설:

```text
decoder가 checkerboard-free일 때만 SSIM이 도움이 된다.
```

직접 후속 실험 후보:

1. **새 `_muvo` decoder (upstream `ConvDecoder` 패턴, k=5/k=6+output_padding)
   + SSIM 적정 weight (0.06 정도)** 로 재시도. decoder가 격자 안 만들고,
   SSIM이 high-freq을 강제로 요구해도 capacity 있게.
2. (1)도 실패면 SSIM 폐기 후 다른 우선순위(`_muvo_comparison.md` §7의 KL weight,
   latent dim 확장 등)로.

### 1000-epoch full overfit (weight 0.06, upstream effective와 매칭)

`cfg.LOSSES.WEIGHT_SSIM=0.06` 으로 낮춰 재시도 (`SSIMWorldModelTrainer` getattr
fallback도 0.06으로 변경).

#### Reconstruction figure 비교

```text
results/figures/ssim06_one_window_overfit_train002_h256_z128_1000epoch_reconstruction_s0.png
  → 체커보드는 사라짐 (weight 0.6의 격자 패턴 없음).
  → 그러나 baseline 대비 **더 흐릿함**. 트럭 윤곽 거의 안 보임, 디테일 평균화.
  → Prior future RGB도 비슷한 blur.
```

#### Loss curve

```text
SSIM 0.06: total loss 1.7 → ~0.55에서 oscillating. baseline 0.2 대비 ~3×.
           KL 0.045까지 spike (baseline 0.02). 학습 noisy하고 덜 안정적.
```

#### 해석

- weight 0.6 → 0.06으로 **체커보드 문제는 사라졌으나** L1-only baseline 대비
  recon이 더 **smooth/blurry**해짐. SSIM이 local luminance·contrast·structure
  매칭을 시도하면서, "exact pixel match" 대신 "통계적 평균에 가까운 부드러운"
  솔루션을 선호하게 되는 부작용.
- 즉 **SSIM 단독으로는 알파의 현재 decoder 위에서 이득 없음**. 어떤 weight에서도
  baseline보다 좋지 않음 (0.6 = 체커보드, 0.06 = 평균 흐림).

#### 종합 결론

```text
원래 동기: full-dataset 학습에서 RGB recon이 흐릿함.
1-window overfit 환경에서 baseline이 이미 sharp.
→ 이 실험 setting은 "흐릿함 → SSIM 효과" 검증에는 부적합.

SSIM은 현재 alpha SensorDecoder (k=4, ConvTranspose, bilinear interp 강제)와
조합했을 때 시각적으로 더 나빠짐. 두 가설:

(a) SSIM과 decoder 구조가 함께 작동해야 효과 (decoder 흐릿함 직접 타깃).
(b) SSIM은 1-window overfit이 아닌 large-scale 학습에서만 효과 (regularization 측면).

(a) 검증을 위한 다음 단계: 새 _muvo decoder (upstream ConvDecoder, k=5/6)
+ SSIM 0.06 조합 테스트. trainer_ssim.py가 non-_muvo 전용이라 _muvo용 entry
별도 필요.
```

## `_muvo` decoder × SSIM × dim 2×2 매트릭스 (1-window overfit)

이전 §의 옵션 B 채택: `trainer_muvo.py`에 `SSIMLoss`를 인라인하고 `cfg.LOSSES.WEIGHT_SSIM`
으로 토글, `train_muvo.py`에 `--h-dim / --z-dim / --weight-ssim` CLI flag 추가
(`build_callbacks`의 change_summary도 cfg 값을 그대로 읽도록 동기화).

기존 `muvo_one_window_overfit_1000` baseline은 그 사이의 `_muvo` 변경 3건
(`7099f87` decoder→upstream ConvDecoder, `7bf7df9` SensorFeatureConv→BasicBlock×2,
`0be4871` RSSM `action_in_gru` flag) 이전이라 stale → 재실행 포함 4개 fresh run.

### Setup

| run | dim | WEIGHT_SSIM | run_name |
|---|---|---|---|
| #1 | h256/z128 | 0.0  | `muvo_one_window_overfit_h256_z128_1000epoch` |
| #2 | h256/z128 | 0.06 | `muvo_ssim06_one_window_overfit_h256_z128_1000epoch` |
| #3 | h512/z256 | 0.0  | `muvo_one_window_overfit_h512_z256_1000epoch_v2` |
| #4 | h512/z256 | 0.06 | `muvo_ssim06_one_window_overfit_h512_z256_1000epoch` |

공통: 1-window overfit, 1000 epoch, `_muvo` 현재 코드 (ConvDecoder + BasicBlock×2 +
`action_in_gru=True`), `WEIGHT_LIDAR_RE=0.0` (LiDAR supervision 꺼짐, recon depth row가
전부 검정으로 나오는 이유).

### Reconstruction figure 비교

```text
results/figures/muvo{_ssim06}_one_window_overfit_{h256_z128,h512_z256{_v2}}_*_reconstruction_s0.png
  → 4장 모두 Posterior RGB가 트럭 윤곽 흐릿. Prior future도 비슷한 average smoothing.
  → 육안으로 SSIM on/off, dim 256/512 간 sharpness 차이 거의 식별 불가.
  → LiDAR row는 4개 다 검정 (WEIGHT_LIDAR_RE=0).
```

### Loss curve

| run | total loss 종점 | KL 안착 |
|---|---|---|
| #1 (h256, L1) | ~0.2 단조 감소, 깨끗 | step ~300부터 ~0.01 안정 |
| #2 (h256, SSIM 0.06) | ~0.55에서 plateau, 큰 oscillation | step 600까지 oscillation 지속 |
| #3 (h512, L1) | ~0.2 단조 감소, 깨끗 | step ~300부터 ~0.01 안정 |
| #4 (h512, SSIM 0.06) | ~0.55에서 plateau, 큰 oscillation | step 600까지 oscillation 지속 |

(* SSIM 추가시 total loss 절대값은 `1-SSIM` 노이즈 floor 때문에 0으로 안 내려감 →
L1-only와 직접 비교 불가. 비교 포인트는 **shape**.)

### 해석

1. **SSIM은 sharpness 개선 없음 + 학습 불안정 추가**: recon figure에 시각적 차이가
   없고, KL이 안착하는 시점이 L1-only보다 ~2× 늦음.
2. **Dim (h256 vs h512) 효과 거의 없음**: L1-only끼리도 SSIM끼리도 convergence가
   사실상 동일. 1-window 메모라이즈는 h256 capacity로도 충분.
3. **`_muvo` decoder 교체 자체도 1-window setting에선 의미 없음**: alpha decoder 시절
   baseline(`one_window_overfit_train002_h256_z128_rgbonly_1000epoch`)과 비교해도
   recon 품질이 시각적으로 동일.

### 종합 결론 — 1-window overfit testbed 자체의 한계

이전 §의 가설 "1-window overfit setting 자체가 baseline에서 이미 sharp"가
**완전히 검증됨**. 4-run 매트릭스로 통제해도:

- decoder 구조 (alpha → _muvo ConvDecoder)도,
- decoder capacity (h256 → h512)도,
- SSIM regularization 추가도,

전부 **시각적으로 무차이**. 한 개 window를 1000 epoch 외우는 setting에서는
모든 변형이 같은 수렴점에 도달.

**원본 동기 (full-dataset 학습에서의 blurry RGB)** 와 1-window overfit은 다른 문제이고,
**1-window는 SSIM 효과를 잴 수 없는 testbed**임이 확인됨. 다음 실험은 반드시
multi-window (적어도 partial dataset) 환경에서 진행해야 함.

## 다음 단계

1. ~~`_muvo` decoder + SSIM 0.06 조합 실험 (1-window)~~ → 완료 (위 매트릭스).
2. **Partial dataset (`train_run_002` 하나 전체 또는 train 전체의 N%) + 짧은 epoch
   으로 SSIM 효과 재평가**. overfit 회피 후 SSIM regularization 가치 측정.
3. 만약 partial-dataset에서도 SSIM이 sharpness 개선 없으면 SSIM 폐기 →
   다른 후보 (KL weight, decoder dilation, perceptual loss 등) 로 이동.
4. (병행 검토) `WEIGHT_LIDAR_RE=0.0` 으로 LiDAR가 학습되지 않고 있음. _muvo branch
   동기와 별개로, 모델이 RGB + LiDAR fusion 전제 위에 만들어진 만큼 LiDAR loss
   재활성화 여부를 결정해야 함.

## 정정 (2026-05-15, 4-run 매트릭스 commit 직후)

위 분석들의 한 전제 — **"upstream muvo는 SSIM을 사용한다"** — 가 사실 틀렸음을 확인.
`docs/muvo_default_config_audit.md` 참조.

### 정정 1: upstream muvo default config는 SSIM이 OFF

`muvo/configs/muvo.yml`의 `LOSSES.SSIM: False`이고 `muvo/trainer.py:97`이
`if cfg.LOSSES.SSIM:` 으로 gated → upstream default 학습은 SSIM 없이 RGB는
L1만 사용. 즉 원래 SSIM 실험의 동기였던 **"muvo는 SSIM 덕분에 안 흐릿함"은
잘못된 전제**였음.

### 정정 2: upstream의 sharp recon 비결은 SSIM이 아니라 auxiliary loss head

upstream default ON: `RGB_L1 + LIDAR_RE + VOXEL_SEG` (+ KL + Action). ROUTE는
입력만 (loss 아님). BEV/Depth/Sem 다 default OFF.

`LIDAR_RE`와 `VOXEL_SEG`가 RGB encoder/backbone의 ResNet18 + ConvDecoder에
gradient를 추가로 흘려서 representation을 단단히 만듦. 알파 `_muvo`는
- VOXEL/BEV → 의도적 제거 (_muvo branch 정책)
- LIDAR_RE → **`WEIGHT_LIDAR_RE=0.0` 으로 OFF**

→ RGB recon gradient만 받음 → blur 도피 동력 충분.

### 결과적으로 SSIM 실험의 위치 재평가

위 매트릭스 4개 run에서 SSIM이 blur 만든 4가지 구조적 이유 (§ 2×2 매트릭스)
는 여전히 유효한 SSIM 자체의 특성 설명. 다만 **"왜 그럼 알파에 SSIM 추가했어도
upstream만큼 안 sharp한가?"** 의 답은 단순함: upstream 자체가 SSIM으로 sharp
한 게 아니었기 때문.

→ 다음 단계 우선순위 변경. SSIM 추가 검증은 **후순위로**, **`WEIGHT_LIDAR_RE`
복원 (0.0 → 0.1)** 이 가장 cheap한 첫 시도. 자세한 우선순위는 `muvo_default_config_audit.md`
참조.
