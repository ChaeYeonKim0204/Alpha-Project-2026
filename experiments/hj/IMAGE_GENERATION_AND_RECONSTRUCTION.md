# 이미지 생성 & Reconstruction 파이프라인 (hj / MUVO 2D-token + latent diffusion)

이 문서는 `hj/` 모델이 **(1) 관측 프레임을 어떻게 reconstruction 하는지**와
**(2) 미래 프레임 이미지를 어떻게 *생성*(generate) 하는지**를 코드 단위로 설명한다.
두 경로는 **디코더는 공유하지만 latent를 만드는 방식이 완전히 다르다** — 이 구분이 핵심이다.

관련 파일:
- `models_muvo_2D.py` — 인코더 / 디코더 / latent diffusion (`Model`, `TokenConvDecoder2D`, `LatentFutureDiffusion`)
- `trainer_muvo_2D.py` — 손실, `_observe_and_imagine`, figure 저장 함수
- `data_muvo_2D.py` — range view 변환, batch 구성
- 설계 배경: `DIFFUSION_FUTURE_DESIGN.md`

---

## 0. 한눈에 보기

```
                         ┌─────────────── 공유 디코더 ───────────────┐
 image (B,S,3,H,W) ─┐                                                  │
 lidar  (B,S,4,H,W) ─┼─► encode_fuse_sequence ─► latent z (B,S,C,368) ─┤
 speed              ─┘        (ResNet18+FPN +                          │
                              SensorFusionTransformer)                 │
                                                                       │
   [Reconstruction 경로] ──────────────────────────────────────────────┤
     z (raw)  ──► _decode_state(z) ──► rgb_{1,2,4} / lidar_reconstruction_{1,2,4}
                                       = "obs t=*" 칼럼 (autoencoder)   │
                                                                       │
   [미래 생성 경로] ────────────────────────────────────────────────────┤
     z_norm = latent_normalize(z)                                      │
     z_past = z_norm[:, :RF]                                           │
     noise ~ N(0,I) ──► diffusion.sample(z_past, action, speed) ─DDIM─► z_fut(norm)
     z_fut_raw = latent_denormalize(z_fut)                             │
     z_fut_raw ──► _decode_state(z_fut_raw) ──► rgb_* / lidar_*        │
                                       = "fut t=*" 칼럼 (diffusion)    ┘
```

- **Reconstruction = autoencoder**: 실제 프레임을 인코딩→디코딩. 입력이 있으니 항상 그럴듯하게 나온다.
- **미래 생성 = diffusion 샘플링**: **노이즈에서 시작**해 과거+행동 조건으로 latent를 만들고 디코딩.
  diffusion이 학습이 안 됐으면 샘플된 latent ≈ 노이즈 → 디코딩 결과도 노이즈.

---

## 1. 입력 batch

`data_muvo_2D.py`의 dataset이 윈도우 1개당 다음을 만든다 (S = RF+FH = 4+4 = 8 프레임):

| 키 | shape | 의미 |
|----|-------|------|
| `image`     | (S, 3, H, W) | 인코더 입력용 **정규화된** RGB |
| `image_raw` | (S, 3, 320, 768) | reconstruction **타깃**(raw RGB). figure의 GT도 이것 |
| `lidar`     | (S, 4, 32, 1024) | range view `[x, y, z, r]` (= `range_view_label_1`) |
| `speed`     | (S,) 또는 (S,1) | ego 속도 |
| `action`    | (S, 2) | `[throttle_brake, steering]` |

`WorldModelTrainer.prepare_custom_batch` (`trainer_muvo_2D.py:513`)가
`rgb_label_1 ← image_raw`, `range_view_label_1 ← lidar`로 reconstruction 타깃을 채운다.

### range view 채널 (`data_muvo_2D.py:126` `point_cloud_to_range_view`)
LiDAR 포인트클라우드를 구면 투영해 `(4, H, W)` 텐서로 만든다:
- 채널 0,1,2 = `x, y, z`
- 채널 3 (마지막) = `r` (range/거리). **`r > 0` = 유효 픽셀(실제 반사점)**, `r == 0` = 빈 픽셀.

---

## 2. 인코딩: 프레임 → latent 토큰

`Model.encode_fuse_sequence` (`models_muvo_2D.py:715`):

1. **이미지 인코더**: `resnet18`(pretrained, `out_indices=[2,3,4]`) → `FPNDecoder` → `(B*S, C, 10, 24)`
2. **LiDAR 인코더**: `resnet18`(`in_chans=4`, `out_indices=LIDAR_OUT_INDICES`) → `FPNDecoder` → `(B*S, C, 2, 64)`
3. **속도 인코더**: `SpeedEncoder`로 `(B*S, C)` 임베딩
4. **융합**: `SensorFusionTransformer`가 카메라 토큰 + LiDAR 토큰(+속도)을 self-attention으로 융합

출력: **token-shape latent** `z` = `(B, S, C, N)`
- `N = n_image_tokens + n_lidar_tokens = (10×24) + (2×64) = 240 + 128 = 368`
- `C = cfg.MODEL.EMBEDDING_DIM`

이 `z`가 두 경로(reconstruction / 미래 생성)의 공통 출발점이다.

---

## 3. Reconstruction 경로 (관측 프레임, "obs t=*")

### 3.1 디코더 `TokenConvDecoder2D` (`models_muvo_2D.py:465`)

token-shape latent → 픽셀. **ConvTranspose 사다리 + 멀티스케일 head** 구조:

```
state_slice (B,S,C,h,w)
   └ reshape (B*S, C, h, w)
   └ pre_transpose_conv : Conv3x3 + (ConvTranspose2d ×n_basic_conv)   # 큰 배율용 추가 doubling
   └ trans_conv1 (stride2) ─► head_4  →  {sensor}_4   (1/4 해상도)
   └ trans_conv2 (stride2) ─► head_2  →  {sensor}_2   (1/2 해상도)
   └ trans_conv3 (stride2) ─► head_1  →  {sensor}_1   (full 해상도)
```

- `head_*`는 1×1 Conv (`_TokenHead`, `:450`)로 출력 채널을 맞춘다.
- 해상도 사다리는 `base_hw`에서 `target_hw`까지 **2의 거듭제곱 배율**이어야 함 (`:482` 검증).
  - **카메라**: base `(10,24)` → 5× doubling → `(320,768)`, 출력 **3채널 RGB**
  - **LiDAR**: base `(2,64)` → 4× doubling → `(32,1024)`, 출력 **4채널** `[x,y,z,r]`
- `_{1,2,4}` 세 스케일을 모두 내보내 손실에 deep-supervision으로 쓴다 (factor별 `discount=1/factor` 가중).

### 3.2 `_decode_state` (`models_muvo_2D.py:667`)

```python
img, lid = self._slice_token_state(state)        # 368 토큰을 240(img)/128(lid)로 분리, 2D로 reshape
out.update(self.image_decoder(img))              # rgb_1, rgb_2, rgb_4
out.update(self.lidar_decoder(lid))              # lidar_reconstruction_1/2/4
pooled = state.mean(dim=-1)                       # 토큰 평균 → action_pred
out["action_pred"] = self.policy_decoder(pooled)
```

### 3.3 forward에서의 reconstruction (`models_muvo_2D.py:762`)

```python
z = self.encode_fuse_sequence(image, lidar, speed=speed)   # (B,S,C,N) raw
recon = self._decode_state(z)                              # ★ 전체 6(=8)프레임 직접 디코딩
```

즉 reconstruction은 **`decode(encode(실제 프레임))`** 인 순수 autoencoder다.
입력 프레임이 그대로 들어가므로 디코더만 학습되면 (blur는 있어도) 항상 그럴듯하게 복원된다.
figure의 **"obs t=0..3" 칼럼**이 이 경로(`posterior_output`)다.

### 3.4 reconstruction 손실 (`trainer_muvo_2D.py:701` `compute_loss`)

- **RGB**: `rgb_{1,2,4}` 각각 L1 (+ 선택적 SSIM), `weight_rgb * discount`
- **LiDAR xyz**: `pred[:, :, :3]` vs target, **유효 마스크**(`r>0`)로 마스킹한 MSE
- **LiDAR depth**: range 채널 L1 (유효 픽셀), `lidar_empty_depth`는 빈 픽셀을 0으로 (smooth L1)
- **action**: `action_pred` L1

> 마스킹은 `_masked_loss`(`:695`)가 `mask.expand_as(pred)`로 채널 broadcast 처리.
> 단일채널 유효마스크 `(B,S,1,H,W)`가 xyz 3채널로 자동 확장된다.

---

## 4. 미래 생성 경로 (예측 프레임, "fut t=*") — latent diffusion

여기가 reconstruction과 **완전히 다른** 부분이다. 미래는 입력 프레임이 없으므로
**노이즈에서 diffusion으로 latent를 생성**한 뒤 같은 디코더로 픽셀화한다.

### 4.1 latent 정규화 (`models_muvo_2D.py:681`)

diffusion은 **정규화 공간**에서 동작하고, 디코더는 **raw 공간**을 쓴다.

- `latent_normalize(z)`: 채널별 EMA mean/std(`momentum=0.99`)로 `(z-μ)/σ`
- `latent_denormalize(z)`: 디코딩 직전 `z*σ+μ`로 되돌림
- 모드: `"ema"`(기본) / `"fixed"` / `"none"`

### 4.2 학습 타깃 (`models_muvo_2D.py:368` `training_targets`) — **샘플링 없음**

```python
z_norm   = latent_normalize(z)
z_past   = z_norm[:, :RF]                 # 과거 4프레임 (조건)
z_future = z_norm[:, RF:RF+FH]            # 미래 4프레임 (★ diffusion 타깃)
diff = diffusion.training_targets(z_past, z_future.detach(), action_future, speed_obs)
```

`training_targets` 내부 (v-parameterization, cosine schedule):
```python
tau   = randint(0, T)                     # 노이즈 레벨 (T=1000)
eps   = randn_like(z_future)
z_tau = alpha*z_future + sigma*eps        # forward diffusion (노이즈 주입)
v_target = alpha*eps - sigma*z_future     # v-예측 타깃
v_pred   = denoise(z_tau, tau, z_past, action_future, speed_obs)   # DiT
```
손실은 `trainer`가 `MSE(v_pred, v_target)` (`trainer_muvo_2D.py:711`).
**학습 스텝에서는 DDIM 샘플링을 하지 않는다** (MSE만 → 빠름).

> `z_future.detach()`: diffusion 그래디언트가 인코더로 흐르지 않게 막음.
> = 인코더는 reconstruction 손실로만 학습, diffusion은 그 latent를 **타깃**으로만 추격.
> 인코더가 같이 학습되므로 타깃이 움직인다 (joint+detach 모드, `DIFFUSION_FUTURE_DESIGN.md:272` 참고).

### 4.3 조건부 DiT denoiser (`models_muvo_2D.py:342` `denoise`, `:219` `DiTBlock`)

`z_tau`(노이즈 낀 미래 토큰)를 입력, **memory cross-attention**으로 조건을 받는다:
```
memory = [ z_past 토큰 | speed 토큰 | action 토큰(FH개) ]
```
- 각 토큰에 2D sine pos-emb + image/lidar type-emb + temporal-emb 추가
- `DiTBlock`: AdaLN-Zero self-attn (timestep `t_emb`로 modulate) → memory cross-attn → AdaLN-Zero MLP
- `final_linear`/`adaLN` **zero-init** → 학습 초기 출력 ≈ 0 (표준 DiT)
- **action**만 CFG 대상: `p_uncond` 확률로 `null_action`으로 드롭 → 추론 시 guidance scale `w`로 행동 효과 강도 조절

### 4.4 추론 샘플링 (`models_muvo_2D.py:399` `sample`) — **DDIM + CFG**

```python
z = noise ~ N(0, I)                                  # (B*K, FH, C, N)  ★ 노이즈에서 시작
steps = linspace(T-1, 0, ddim_steps)                 # 기본 50 step
for tau in steps:
    pred       = _guided_pred(z, tau, z_past, action, speed, w)   # CFG
    x0, eps    = _pred_to_x0_eps(pred, z, alpha, sigma)           # v → (x0, eps) 복원
    z          = alpha_next*x0 + sigma_next*eps                   # DDIM(eta=0) 한 스텝
# 마지막엔 z = x0
```
v-param 복원식 (`:386`):
- `x0  = alpha*z_tau - sigma*v`
- `eps = sigma*z_tau + alpha*v`
(둘 다 `alpha²+sigma²=1`에서 유도되는 정확한 식)

### 4.5 latent → 픽셀 (`models_muvo_2D.py:802` `generate_futures`, `:813` `decode_future_per_frame`)

```python
z_past, speed_emb = self._encode_past(batch)               # 과거만 인코딩 → 정규화
z_fut = diffusion.sample(z_past, action_future, speed_emb, ddim_steps=50)
z_fut = latent_denormalize(z_fut)                          # 정규화→raw
out   = self._decode_state(z_fut)                          # ★ reconstruction과 동일 디코더
```

**핵심**: 미래 프레임은 `decode( denorm( DDIM_sample(noise | 과거,행동) ) )`.
diffusion이 학습 안 됐으면 `v_pred ≈ 0` → 샘플된 latent ≈ 입력 노이즈 → **디코딩 결과 = 노이즈**.
(현재 overfit 디버깅에서 future가 노이즈로 나온 직접 원인.)

### 4.6 반사실(counterfactual) (`models_muvo_2D.py:426`, `:828`)

같은 `z_past`/속도/**노이즈 seed**를 고정하고 **action만** 바꿔(직진/좌/우/급제동) 미래를 생성.
노이즈가 공유되므로 결과 차이 = 순수 행동 효과. → "어떤 행동을 하면 어떤 미래가 되는가".

---

## 5. Figure가 만들어지는 방식 (`trainer_muvo_2D.py`)

학습이 끝나면(`save_*` 함수) `_observe_and_imagine`(`:795`)이 두 출력을 만든다:
- `posterior_output` = `_decode_state(encode(전체 프레임))`을 관측 프레임(0..RF-1)으로 slice → **reconstruction**
- `future_output` = `generate_futures(...)`의 0번째 샘플 → **diffusion 미래**

| 함수 | 파일 | 내용 |
|------|------|------|
| `save_reconstruction_figure` | `:97` | 칼럼 = obs(GT vs **posterior recon**) RF개 + fut(GT vs **diffusion pred**) FH개. 행 = RGB GT/Pred + depth GT/Pred (+ raw depth) |
| `save_future_frames_per_frame` | `:223` | 미래 FH프레임을 **각각 독립 디코딩**(`decode_future_per_frame`), 시점별 PNG |
| `save_counterfactual_figure` | `:309` | 같은 과거, 행동만 바꾼 미래 그리드 (`build_probe_actions`, `:293`) |

depth 시각화는 `cmap="magma"`, `r` 채널(인덱스 3)만 표시, `vmax=2.0*LIDAR_SCALE`.

---

## 6. 왜 "obs는 그럴듯한데 fut는 노이즈"인가 (요약)

| 구분 | latent 출처 | 학습 난이도 | 결과 |
|------|------------|------------|------|
| **obs reconstruction** | `encode(실제 프레임)` (결정적) | 쉬움 — autoencoder가 ~100스텝에 overfit | 그럴듯(약간 blur) |
| **fut 생성** | `DDIM_sample(noise\|조건)` (확률적) | 어려움 — 매 스텝 random τ·ε, 타깃도 움직임 | 학습 부족 시 노이즈 |

→ reconstruction이 잘 나오는 것과 미래 예측 품질은 **별개**다.
미래는 diffusion(`train/diffusion` 손실)이 충분히 수렴해야 비로소 의미 있는 프레임이 나온다.
자세한 디버깅 진단은 본 디렉토리의 overfit 실험 결과 참고.
