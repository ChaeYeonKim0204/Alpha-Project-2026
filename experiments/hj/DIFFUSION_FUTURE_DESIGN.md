# MUVO 2D — "어떤 행동을 하면 어떤 미래가 되는가" 설계서

작성일: 2026-05-27 (재작성: action → future 인과를 문서 전체 축으로)
대상 코드: `alpha26/archive/hj/{config,data,models,trainer,train}_muvo_2D.py`

---

## 0. 이 문서가 답하려는 단 하나의 질문

> **"지금 이 상황에서 내가 행동 a를 하면, 미래가 어떻게 되는가?"**
> 그리고 **"행동을 a 대신 a'로 바꿨다면, 미래는 어떻게 달라졌을까?"**

자율주행 world model의 본질적 가치는 "미래를 그럴듯하게 그린다"가 아니라
**행동(action)과 미래(future) 사이의 인과를 모델 안에서 묻고 답할 수 있다**는 데 있다.

이 설계는 그 질문에 답하기 위해, 기존 RSSMTD(recurrent state-space transition)를
**제거**하고, 인코더가 만든 **latent 토큰** 위에서 동작하는
**행동 조건부(action-conditional) diffusion model**로 교체한다.

핵심은 세 가지다.

1. **미래는 행동의 조건부 분포다.** 모델이 학습하는 것은
   `p(z_future | z_past, speed, action)` — "과거 + 행동이 주어졌을 때의 미래".
   행동을 바꾸면 조건이 바뀌고, 따라서 생성되는 미래가 바뀐다.
2. **행동의 순수 효과를 분리한다.** classifier-free guidance(CFG)로
   "행동을 무시한 미래"와 "행동을 반영한 미래"를 둘 다 만들 수 있어,
   그 차이가 곧 **"이 행동이 미래에 미친 영향"** 이다.
3. **그 영향의 강도를 손잡이로 노출한다.** guidance scale `w`가
   "미래가 행동을 얼마나 따를지"를 연속 hyperparameter로 만든다 →
   `w`-sweep으로 **action sensitivity(행동 민감도)** 를 정량 측정.

> RSSM은 행동을 GRU에 deterministic하게 먹여 **한 개의 미래**만 내놓고,
> "행동을 다르게 했다면?"을 물을 손잡이가 없다. 바로 이 한계가 교체의 동기다.

### 확정된 방향
- 결과물: **설계서 우선**(본 문서) → 검토 후 코드 구현
- 시간 모델링: **전체 horizon 동시 생성** (FH 프레임을 한 번에 denoise)
- 생성 공간: **latent 토큰 공간** (인코더 임베딩 위, 기존 디코더 재사용)
- 조건 구조: **action을 별도 CFG 조건으로 분리**(드롭 가능), `z_past`/speed는 항상-on 컨텍스트

---

## 1. 큰 그림 — "행동 → 미래"가 파이프라인 어디에서 일어나는가

```
[변경 없음]  Dataset (B, S=RF+FH=8, ...)  image(320×768) / lidar(4×32×1024) / action(2) / speed
[변경 없음]  Encoder: ResNet18+FPN → SensorFusionTransformer
                 → z = (B, S, C=256, N=368)      # image 240 + lidar 128 토큰
                 → z_past (과거: "지금까지의 상황"),  z_future (미래: 맞혀야 할 정답)
──────────────────────────── 교체 경계 ────────────────────────────
[제거]       RSSMTD  (prior/posterior, ConvGRUCellGlo, RepresentationModelTD)
             └ 행동을 deterministic하게 받아 미래 1개만 냄 → "행동을 바꿨다면?"에 답 못 함
[신규]       LatentFutureDiffusion  (action-conditional DiT denoiser + CFG)
             └ p(z_future | z_past, speed, action) 를 학습 → 행동을 조건으로 미래를 생성
             └ CFG로 "행동의 순수 효과"를 분리하고 강도(w)를 조절
────────────────────────────────────────────────────────────────
[수정]       TokenConvDecoder2D: 입력 채널 2C → C (cat(h,s) 제거, state 1개)
[수정]       PolicyDecoder: 2C → C
[변경 없음]  RGB/LiDAR 업샘플 ladder, head_4/2/1   # 미래 latent를 다시 픽셀/포인트로
```

**데이터 흐름으로 본 "행동 → 미래":**
```
   과거 관측 ──encoder──▶ z_past ─┐
   speed ──────────────────────┐ ├──(항상-on 컨텍스트)
                                │ │
   행동 a ──MLP──▶ action token ┘ │   ← 여기를 바꾸면(직진/좌회전/급제동)
                                  ▼
              LatentFutureDiffusion (DDIM + CFG, w)
                                  │
                                  ▼
              z_future^(a) ──decoder──▶ 미래 RGB / LiDAR
   (같은 과거에서 행동만 바꾸면 미래가 갈라진다 = counterfactual)
```

상수 (현 cfg 기준):
- `C = EMBEDDING_DIM = 256`
- `IMAGE_TOKEN_HW=(10,24)` → 240, `LIDAR_TOKEN_HW=(2,64)` → 128, **N = 368**
- `RECEPTIVE_FIELD(RF)=4`, `FUTURE_HORIZON(FH)=4`, `S=8`
- 토큰 텐서: `z_full=(B,8,256,368)`, `z_past=(B,4,256,368)`, `z_future=(B,4,256,368)`

> 참고: 현재 `trainer._observe_and_imagine`(trainer_muvo_2D.py:619)에 다중 샘플 인프라가
> 반쯤 들어가 있으나 `cfg.PREDICTION` 부재로 1로만 동작. 더 본질적으로 RSSM은 action을
> deterministic하게 받아(`models_muvo_2D.py:446 imagine_step`, `:792 imagine`)
> "행동을 바꿨다면 미래가 어떻게 달라지는가"를 표현할 수단 자체가 없다 → 교체 동기.

---

## 2. 핵심 모듈 `LatentFutureDiffusion` — 행동을 조건으로 미래를 만든다

### 2.1 역할 (한 줄: "과거 + 행동 → 미래"를 생성 모델로 학습)
- **학습**: 미래 latent `z_future`를 조건 하에 denoise하도록 학습.
  조건은 **항상-on 컨텍스트**(`z_past` + speed)와 **CFG 대상 행동**(`a_future`)으로 분리.
  학습 중 행동을 확률 `p_uncond`로 null 토큰으로 드롭(§2.5)해 "행동을 무시한 미래" 분기도 동시 학습.
- **추론**: 조건 + 행동 + guidance scale `w`로 noise→latent 역확산.
  같은 `z_past`에서 **행동만** 바꿔(a₁/a₂/a₃) 미래가 어떻게 달라지는지(counterfactual)를 생성하고,
  `w`로 "미래가 행동을 얼마나 따를지"를 조절.

### 2.2 Diffusion 정식화 (1차 권장: v-prediction + cosine schedule)
- 학습 스텝 τ ~ Uniform{1..T}, `T = NUM_TRAIN_STEPS = 1000`
- cosine schedule로 `ᾱ_τ` 계산
- forward: `z_τ = sqrt(ᾱ_τ)·z0 + sqrt(1-ᾱ_τ)·ε`,  `ε ~ N(0, I)`
- 타깃: `v = sqrt(ᾱ_τ)·ε − sqrt(1-ᾱ_τ)·z0`
- 손실: `L_diffusion = MSE( D_θ(z_τ, τ, c, a),  v )`  (token·channel·batch 평균)
- 샘플러: 추론은 **DDIM** `NUM_SAMPLING_STEPS = 50`
- cfg로 `PARAMETERIZATION ∈ {"v","eps"}` 전환 가능 (eps도 동일 골격)

> **Latent 정규화(중요):** 인코더 임베딩 z는 단위분산이 아니므로 그대로 diffusion에
> 넣으면 학습이 불안정. running mean/std(EMA) 또는 고정 `LATENT_SCALE`로
> `z ← (z − μ)/σ` 정규화 후 diffusion, 디코딩 직전 역정규화. 1차 구현은 첫 N step
> 동안 batch 통계로 μ,σ를 EMA 추정하는 방식을 권장.

### 2.3 Denoiser 아키텍처 (action-conditional DiT)
미래 토큰을 `(B, FH·N, C) = (B, 1472, 256)` 시퀀스로 펼쳐 처리.

입력 임베딩(미래 토큰):
- spatial pos embedding: 기존 `PositionEmbeddingSine`을 image/lidar 영역별로 재사용
- temporal embedding: 학습형 per-frame `(FH, C)`
- modality type embedding: image=0 / lidar=1 (기존 `type_embedding` 컨벤션 차용)

조건 컨텍스트 — **"항상 주는 맥락" vs "바꿔가며 묻는 행동"으로 분리**:
- (항상-on, 미래의 *기준*) 과거 토큰 `z_past` → `(B, RF·N, C) = (B, 1472, 256)` + pos/temporal/type embedding
- (항상-on) **관측 speed**(마지막 RF 프레임 또는 RF 시퀀스의 speed) → 기존 `SpeedEncoder`(models_muvo_2D.py:176)
  → `(B, C)` modulation token. **⚠️ 미래 speed를 쓰면 안 됨** — 미래 speed는 추론 시 알 수 없고(예측 대상),
  GT 미래 speed를 넣으면 counterfactual과 모순(급제동→속도↓여야 하는데 GT speed가 행동과 싸움)이며 정보 누설.
- (**CFG 대상, "어떤 행동"의 자리**) 미래 action `a_future` → MLP → `(B, FH, C)` action token
  - 학습 시 확률 `p_uncond`로 **학습형 null-action 임베딩 `a_∅`** 으로 치환(§2.5)
- 위를 concat → cross-attention memory `(B, RF·N + FH + 1, C)`

> **⚠️ action 시간 정렬(off-by-one):** 데이터/기존 RSSM 컨벤션은 `action[t-1]`이 프레임 t로의 전이를 일으킨다
> (models:402-404, trainer:631-632의 `future_actions = cat([last_obs_action, a_fh[:,:-1]])`).
> diffusion의 `action_future`도 **같은 정렬**을 따라야 한다. 어긋나면 행동↔미래 인과 신호가 한 칸 밀려
> action sensitivity가 무너지므로, 정렬을 코드에 명시하고 단위 테스트로 확인.

> action을 cross-attn memory의 **별도 토큰**으로 두는 이유: 추론 시 그 토큰만 `a_∅`로
> 바꿔 한 번 더 forward하면 "행동을 무시한 미래"가 되어, 동일 `z_past`/speed 위에서
> **순수 행동 효과(행동 반영 − 행동 무시)** 를 분리할 수 있다(§2.6 CFG).
> 즉 "어떤 행동을 하면 어떤 미래"의 *행동* 부분만 외과적으로 조작 가능한 구조.

DiT 블록 (× `DEPTH`, 권장 6):
```
t_emb = MLP(SinusoidalTimestep(τ))                 # (B, C)
(shift1,scale1,gate1, shift2,scale2,gate2) = AdaLN(t_emb)   # 6×C

x = x + gate1 · SelfAttn( modulate(LN(x), shift1, scale1) )
x = x + CrossAttn( LN(x), memory )                 # 조건(과거·speed·행동) 주입
x = x + gate2 · MLP( modulate(LN(x), shift2, scale2) )
```
- `HEADS=8`, `DIM=C=256`, `MLP_RATIO=4`, AdaLN-Zero 초기화(gate≈0)
- 최종 `Linear(C, C)` → `(B, FH·N, C)` → reshape `(B, FH, C, N)` = v(또는 ε) 예측

> 간단 폴백: AdaLN 없이 `nn.TransformerDecoder`(self+cross attn)를 쓰고 t_emb를 모든
> future 토큰에 additive로 더하는 방식도 동작. 단 noise level 표현력은 DiT가 우위.

### 2.4 API (의사 시그니처)
```python
class LatentFutureDiffusion(nn.Module):
    def __init__(self, channels, n_tokens, future_horizon,
                 image_token_hw, lidar_token_hw,
                 depth=6, heads=8, mlp_ratio=4,
                 num_train_steps=1000, parameterization="v", schedule="cosine",
                 p_uncond=0.1): ...     # ← 행동 조건 드롭 확률 (CFG 학습용)

    # 학습: 행동 일부를 null로 드롭하여 v_pred / v_target 반환 (손실은 trainer가 계산)
    def training_targets(self, z_past, z_future, action_future, speed):
        # returns dict(v_pred=(B,FH,C,N), v_target=(B,FH,C,N), z0_hat=(B,FH,C,N))
        ...

    # 추론: 과거 + 행동 + guidance_scale 로 미래 latent 생성 (DDIM + CFG)
    @torch.no_grad()
    def sample(self, z_past, action_future, speed,
               guidance_scale=1.0, num_samples=1, ddim_steps=50):
        # guidance_scale w: 0=행동 무시, 1=표준 조건, >1=행동 효과 증폭
        # returns (B, K, FH, C, N)
        ...

    # counterfactual: 같은 z_past/speed에서 "행동 리스트"별로 미래 생성 (핵심)
    @torch.no_grad()
    def counterfactual(self, z_past, action_list, speed,
                       guidance_scale=1.0, ddim_steps=50, seed=None):
        # action_list: [a1(직진), a2(좌회전), a3(급제동), ...]  각 (B,FH,2)
        # 같은 noise seed 공유 → 미래 차이가 순수 "행동 효과"로 해석됨
        # returns (B, A, FH, C, N)   # A = len(action_list)
        ...
```
`z0_hat`(예측 x0)은 옵션인 future-decode 손실(§4)과 시각화에 사용.

### 2.5 CFG 학습 — "행동을 무시하는 법"도 같이 배운다 (action 조건 드롭아웃)
`training_targets()` 내부에서 **배치 샘플별로 확률 `p_uncond`(권장 0.1)** 로 action 토큰을
학습형 null 임베딩 `a_∅`로 치환한다.

```
mask ~ Bernoulli(p_uncond)            # (B,) per-sample
a_emb = where(mask, a_∅, MLP(a_future))   # 드롭된 샘플은 "행동 무시" 분기로 학습
```
- 효과: 같은 denoiser가 **행동 반영 `D(·|z_past,speed,a)`** 와
  **행동 무시 `D(·|z_past,speed,∅)`** 를 동시에 학습 → 추론 시 CFG로 둘을 결합 가능.
- `z_past`/speed는 **드롭하지 않음** — "어떤 행동을 하면 어떤 미래"에서 *미래의 기준이 되는 과거*는
  항상 고정해야 하므로 **행동만 드롭**하는 것이 핵심 설계 차이.
  (표준 text-to-image CFG는 전체 조건을 드롭하지만, 여기서는 "행동의 효과"를 분리하는 게 목적.)
- `p_uncond`는 cfg `DIFFUSION.P_UNCOND`로 노출.

### 2.6 CFG 추론 — "행동이 미래에 미친 영향"을 분리하고 그 강도를 조절
v(또는 ε) 예측에 선형 guidance를 적용 (각 DDIM 스텝마다 2회 forward):
```
v_cond   = D(z_τ, τ, z_past, speed, a)        # 행동 반영
v_uncond = D(z_τ, τ, z_past, speed, a_∅)      # 행동 무시
v_w      = v_uncond + w · (v_cond − v_uncond)   # (v_cond − v_uncond) = 행동의 순수 효과
```
- **`w = 0`**: 행동 무시 → 환경 prior만 따르는 "행동 무관" 미래.
- **`w = 1`**: 표준 조건부 생성(데이터 분포가 말하는 행동 효과).
- **`w > 1`**: 행동 효과 증폭 → "직진/좌회전/급제동"의 미래가 더 또렷하게 분리.
- 비용: 스텝당 forward 2배. batch로 cond/uncond를 쌓아 한 번에 처리(`(2·B·K,...)`)하여 상쇄.

> **이게 RSSM이 못 하는 지점.** RSSM은 행동을 GRU에 deterministic하게 먹여 한 점을 내므로
> "행동을 얼마나 따를지"라는 손잡이가 없다. CFG의 `w`는 그 손잡이를 연속 hyperparameter로
> 제공하고, `w`-sweep으로 **action sensitivity 곡선**(§5)을 그릴 수 있다.

---

## 3. `Model` 변경 (`models_muvo_2D.py`)

### 3.1 제거 / 추가
- 제거: `RSSMTD`(:316), `RepresentationModelTD`(:250), `ConvGRUCellGlo`(:219),
  `self.rssm`, `imagine`(:792)/`imagine_step`(:446)
- 추가: `self.diffusion = LatentFutureDiffusion(...)` (null-action 임베딩 포함)

### 3.2 디코더 리팩터 (state 1개) — **세 군데 모두 손대야 함**
RSSM이 사라지면 (a) 디코더 입력이 `cat(h, s)=2C`에서 encoder latent `z=C` 하나로 줄고,
(b) 인코더 z에는 **policy 토큰이 없다**(아래 ⚠️). 다음을 **모두** 바꿔야 동작한다:

- `TokenConvDecoder2D.__init__(in_channels=2*C→C)`(:513) **그리고** `forward`(:575):
  현재 `forward(hidden_state_slice, sample_slice)`가 `torch.cat([h,s], dim=2)→2C`를 함 →
  **단일 입력 `forward(state_slice)`** 로 바꾸고 cat 제거. `__init__` 채널만 바꾸면 안 됨.
- `_decode_state`(:714): 현재 `(hidden_state, sample)` 둘을 받아 각각 slice 후 디코더에 전달 →
  **`_decode_state(state)`** 하나만 받도록. policy cat(:725-728 `cat([pol_h,pol_s])→2C`)도 제거.
- **⚠️ policy 토큰 부재(핵심):** 인코더 `encode_fuse_sequence`(:734)는 image 240 + lidar 128 = **368 토큰만**
  내놓고 policy(369번째) 토큰이 없다(그건 RSSM의 `RepresentationModelTD`가 만들던 것).
  현재 `_slice_token_state`(:696)는 인덱스 `N_img+N_lid`(=368)에서 policy 토큰을 꺼내는데,
  z가 368 토큰(인덱스 0..367)이면 **인덱스 초과 에러**. 따라서:
  - `n_total_tokens`(:692)에서 `policy_tokens` 가산 제거 → N=368.
  - `_slice_token_state`는 image/lidar 두 블록만 반환(policy 분기 삭제).
  - `PolicyDecoder(in_channels=C)`(:195); action_pred는 policy 토큰이 없으므로
    **프레임별 mean-pooled token `(B,S,C)`** 에서 산출.

### 3.3 forward (학습)
```python
def forward(self, batch):
    z = self.encode_fuse_sequence(image, lidar, speed)   # (B, S=8, C, N=368)  ← :734
    z = self.latent_normalize(z)                         # §2.2
    z_past, z_future = z[:, :RF], z[:, RF:RF+FH]          # 과거 / 정답 미래

    # ★ recon은 z_past가 아니라 z 전체(6프레임)를 디코딩한다 (아래 두 이유)
    recon = self._decode_state(z)                        # rgb_{1,2,4}, lidar_*, action_pred

    # diffusion 타깃: "과거+행동→미래"를 맞히도록. action은 내부에서 p_uncond로 드롭.
    # speed는 관측(과거) speed만 (§2.3 ⚠️). 미래 speed 넣지 말 것.
    diff = self.diffusion.training_targets(
        z_past, z_future.detach(), action_future, speed_obs)

    out = {**recon, "v_pred": diff["v_pred"], "v_target": diff["v_target"],
           "z0_hat": diff["z0_hat"]}
    return out, {"z": z}
```
- **왜 z_past가 아니라 z 전체를 recon하나 (중요):**
  1. **디코더가 미래 latent도 디코딩할 줄 알아야 한다.** diffusion이 생성하는 건 *미래* latent인데,
     recon을 과거 프레임으로만 학습하면 디코더가 생성된 미래 latent를 픽셀로 못 푼다.
  2. **shape 정합.** `compute_loss`(trainer:590)는 `output["rgb_1"]`을 `batch["rgb_label_1"]`(6프레임)과
     L1 비교. z_past(4프레임)만 디코딩하면 S=4 vs 6 **shape mismatch 에러**.
- `z_future.detach()`: diffusion 타깃이 인코더로 흐르는 gradient를 끊어 타깃을
  안정화(권장). 인코더는 관측 recon 손실로 학습됨.
- **대안(더 안정적):** 2단계 학습 — ①AE(encoder+decoder) 사전학습 후 encoder freeze,
  ②diffusion만 학습. cfg `FREEZE_ENCODER_FOR_DIFFUSION`로 토글. 1차는 joint+detach 권장.

### 3.4 generate_futures / counterfactual_futures (추론·검증)
```python
def generate_futures(self, batch, num_samples=K, guidance_scale=1.0, ddim_steps=50):
    # 같은 행동(GT)에서 noise만 달리한 미래 K개 (환경 응답 다양성, 부차)
    z = self.latent_normalize(self.encode_fuse_sequence(...))
    z_past = z[:, :RF]
    z_fut_k = self.diffusion.sample(z_past, action_future, speed_obs,
                                    guidance_scale=guidance_scale,
                                    num_samples=K, ddim_steps=ddim_steps)  # (B,K,FH,C,N)
    return [self._decode_state(self.latent_denormalize(z_fut_k[:, k])) for k in range(K)]

def counterfactual_futures(self, batch, action_list, guidance_scale=1.0, ddim_steps=50):
    # ★핵심★ 같은 과거에서 "행동만" a1/a2/a3로 바꿔 미래를 디코딩
    z = self.latent_normalize(self.encode_fuse_sequence(...))
    z_past = z[:, :RF]
    z_fut_a = self.diffusion.counterfactual(z_past, action_list, speed_obs,
                                            guidance_scale=guidance_scale,
                                            ddim_steps=ddim_steps)  # (B,A,FH,C,N)
    return [self._decode_state(self.latent_denormalize(z_fut_a[:, a]))
            for a in range(len(action_list))]   # list of A dict(rgb_*, lidar_*)
```
- `counterfactual_futures`는 **동일 noise seed**를 행동 간 공유하여, 결과 차이가
  noise가 아니라 **순수 행동 효과**로 귀속되게 한다 → "어떤 행동을 하면 어떤 미래"의 직접 답.

---

## 4. 손실 (`trainer_muvo_2D.py`)

`compute_loss`에서 **KL(probabilistic) 항을 제거**하고 diffusion 항으로 대체:

| 항목 | 정의 | 가중치(cfg) | 비고 |
|------|------|-------------|------|
| `diffusion` | `MSE(v_pred, v_target)` (action drop 포함) | `WEIGHT_DIFFUSION` | KL 대체 (핵심: 행동→미래 학습) |
| `rgb_{1,2,4}` | 관측 프레임 L1 | `WEIGHT_RGB` | 기존 유지 |
| `lidar_xyz/depth/empty_{1,2,4}` | 관측 프레임 | `WEIGHT_LIDAR_*` | 기존 유지 |
| `action` | L1 | `WEIGHT_ACTION` | 관측 프레임 (BC) |
| `future_decode`(옵션) | `z0_hat` 디코딩 후 미래 GT와 L1 | `WEIGHT_FUTURE_DECODE` | latent↔pixel 결합 (Phase 2) |

- diffusion 손실은 model이 반환한 `v_pred/v_target`로 trainer가 계산(기존 스타일 유지).
  action 드롭은 diffusion 모듈 내부에서 일어나므로 손실 식 자체는 동일.
- **구현 메모:** 현재 `compute_loss`(trainer:546)엔 diffusion 분기가 없다. `if "v_pred" in output:`
  분기를 추가해야 함. KL 분기(trainer:549)는 `"prior" in output` 가드라서, diffusion forward가
  prior/posterior를 안 내놓으면 **자동으로 스킵**된다(별도 삭제 불필요). 학습 경로는 `_observe_full`
  (trainer:614)→`compute_loss(batch, output, include_kl=True)` 그대로 쓰되, diffusion MSE가 추가됨.
- `balanced_kl_loss` / `gaussian_kl_loss` / `KL_*` cfg는 deprecated (키는 호환 위해 잔존 가능, 가중치 0).
- `future_decode`는 1차에서 off, 학습이 도는 것 확인 후 낮은 가중치로 켜서 미래 화질을 픽셀공간에 묶음.

---

## 5. 검증/지표 — "행동을 바꾸면 미래가 (올바르게) 바뀌는가"

> **⚠️ `_observe_and_imagine`(trainer:619)는 내부 루프만이 아니라 통째로 재작성해야 한다.**
> 현재는 ① `model.forward(batch_rf)`로 **RF 프레임만** 넣고(새 forward는 RF+FH 전체를 받음),
> ② `state_dict["posterior"]["hidden_state"]`로 imagine을 시작한다(diffusion엔 posterior 없음).
> 새 버전: full batch로 forward(관측 recon) → `z_past` 추출 → `generate_futures`/
> `counterfactual_futures`로 미래 생성. **반환 5-튜플 계약**
> `(losses, output, state_dict, losses_imagine, output_imagine)`와
> figure(trainer:114)가 읽는 키(`output["rgb_1"]`, `output_imagine["rgb_1"/"lidar_reconstruction_1"]`)는
> 유지해야 기존 시각화/검증 흐름이 안 깨진다. counterfactual 그리드(§5.4)는 그 위에 추가.

루프 교체 방향: (1) `generate_futures(K, w)` — 같은 행동에서 noise 다양성(부차),
(2) `counterfactual_futures(action_list, w)` — **핵심: 행동→미래 인과 검증**.

### 5.1 Probe action set — "어떤 행동"의 후보들
검증 시 GT action 외에 **고정된 probe action 세트**를 주입:
- `a₁ = 직진` (steer≈0, throttle 유지)
- `a₂ = 좌회전` (steer<0)  / `a₂' = 우회전` (steer>0)
- `a₃ = 급제동` (throttle=0, brake↑)

(스케일은 데이터 action 분포의 분위수로 잡아 OOD 회피.)

### 5.2 핵심 지표 — action sensitivity (행동이 미래를 얼마나 바꾸나)
- **action sensitivity** `S = mean_pairwise_dist({future(aᵢ)})`
  - 같은 `z_past`·같은 noise seed에서 **행동만** 바꾼 미래 간 거리(latent L2 / pixel L1 / LPIPS).
  - 높을수록 모델이 행동에 민감 = "행동→미래" 인과를 강하게 학습.
  - 로깅: `cf_sensitivity_latent`, `cf_sensitivity_rgb`, `cf_sensitivity_lidar`.
- **guidance sweep** `w ∈ {0, 1, 2, 4}`: 각 `w`에서 `S(w)`를 측정 → **action sensitivity 곡선**.
  - `w=0`에서 S≈0(행동 무관)부터 `w↑`에 따라 S 증가하는지 → CFG가 손잡이로 작동하는지 검증.
- **방향 정합성(directional check)** — "올바른" 미래인가:
  - a₂(좌회전) 미래에서 차선/heading이 실제로 좌측으로,
  - a₃(급제동) 미래에서 ego 속도/전방차 거리 변화가 제동 방향인지 (정성 + 가능하면 정량 proxy).
  - 단순히 "달라지는가"를 넘어 **"행동에 맞는 방향으로 달라지는가"** 까지 본다.
- **현실성(realism) trade-off**: `w↑` 시 행동 분리는 커지나 화질이 망가질 수 있음.
  - `future_rgb_psnr` / `future_lidar_chamfer`를 `w`별로 함께 로깅 → sensitivity vs realism 곡선.

### 5.3 보조 지표 (부차) — 같은 행동 안에서의 미래 다양성
- 같은 행동·다른 seed로 K개 생성 시 pairwise 거리 = **환경 응답 다양성**(다른 차/보행자/신호).
  - `future_rgb_diversity`, `future_lidar_diversity` (참고용, headline 아님).
- best-of-K 재구성(GT action에 대해): `future_rgb_psnr_bestK` 등 (mode-covering 참고).

cfg: `PREDICTION.GUIDANCE_SCALES=[0,1,2,4]`, `PREDICTION.N_SAMPLES=K`(부차),
`PREDICTION.CF_ACTIONS`(probe action 정의). 학습 step은 diffusion MSE만(샘플링 불필요 → 빠름).

### 5.4 시각화 (`save_reconstruction_figure`)
- **counterfactual 그리드**: 행=GT/관측 recon, 그 아래 **행동별(a₁ 직진 / a₂ 좌회전 / a₃ 급제동)**
  미래를 RGB·depth로. 동일 과거에서 **행동만 바꿨을 때 미래가 갈라지는지** 한눈에.
  → 문서 제목 그대로 "어떤 행동을 하면 어떤 미래가 되는가"를 그림 한 장으로 보여줌.
- **guidance 그리드**(옵션): 한 행동을 고정하고 `w=0/1/2/4`로 나열 → CFG 강도 효과 정성 확인.

---

## 6. cfg 변경 (`config_muvo_2D.py`)

```python
MODEL.DIFFUSION = SimpleNamespace(
    ENABLED=True,
    NUM_TRAIN_STEPS=1000,
    NUM_SAMPLING_STEPS=50,        # DDIM
    SCHEDULE="cosine",
    PARAMETERIZATION="v",         # or "eps"
    DEPTH=6, HEADS=8, MLP_RATIO=4,
    P_UNCOND=0.1,                 # CFG 학습: 행동 조건 드롭 확률
    SELF_COND=False,              # 옵션
    LATENT_NORM="ema",            # {"ema","fixed","none"}
    LATENT_SCALE=1.0,             # LATENT_NORM=="fixed"일 때
    FREEZE_ENCODER_FOR_DIFFUSION=False,  # True면 2단계 학습
)
PREDICTION = SimpleNamespace(
    N_SAMPLES=4,                  # noise 다양성 검증 K (부차)
    GUIDANCE_SCALES=[0.0, 1.0, 2.0, 4.0],   # CFG sweep (핵심)
    CF_ACTIONS="default",         # probe action 세트 정의 (직진/좌/우/급제동)
)

LOSSES.WEIGHT_DIFFUSION = 1.0
LOSSES.WEIGHT_FUTURE_DECODE = 0.0           # Phase 2에서 ↑
# TRANSITION / KL_* : deprecated (유지하되 미사용)
```

> **⚠️ 현재 `config_muvo_2D.py`는 이전 "multi-future" 방향이라 CFG 키가 빠져 있다.**
> 이미 들어간 것: `MODEL.DIFFUSION`(ENABLED/STEPS/SCHEDULE/PARAM/DEPTH/HEADS/LATENT_*),
> `LOSSES.WEIGHT_DIFFUSION/WEIGHT_FUTURE_DECODE`, `PREDICTION.{N_SAMPLES, DDIM_STEPS}`.
> **추가 필요(counterfactual/CFG 필수):** `MODEL.DIFFUSION.P_UNCOND`(§2.5),
> `PREDICTION.GUIDANCE_SCALES`, `PREDICTION.CF_ACTIONS`(§5). 토큰 HW(`IMAGE_TOKEN_HW`/
> `LIDAR_TOKEN_HW`)는 여전히 `MODEL.RSSM_2D`에 있으므로 새 Model도 거기서 읽으면 됨(유지).

`train_muvo_2D.py`: `--h-dim==--z-dim` 강제(RSSMTD 제약) 로직 제거 가능,
대신 `--diffusion-depth`, `--ddim-steps`, `--p-uncond`, `--guidance-scale`,
`--n-samples`, `--weight-diffusion` 플래그 추가.

---

## 7. 데이터 (`data_muvo_2D.py`)
**변경 없음.** 반환 dict(image, image_raw, lidar, action, speed)와 window 구성(seq_len=RF+FH) 그대로.
diffusion은 인코더 출력만 소비하므로 데이터 계층 영향 없음.
probe action(직진/좌/우/급제동)은 **검증 시 코드에서 합성**하므로 데이터 라벨 추가 불필요.

---

## 8. 구현 순서 (Phase)

1. **cfg 스캐폴딩** — `MODEL.DIFFUSION`(P_UNCOND 포함), `PREDICTION`(GUIDANCE_SCALES, CF_ACTIONS),
   `LOSSES.WEIGHT_DIFFUSION` 추가.
2. **Denoiser + LatentFutureDiffusion** — DiT 블록, cosine schedule, v-pred, null-action 임베딩,
   `training_targets()`(action 드롭) + DDIM `sample()`(CFG) + `counterfactual()`.
   단위 테스트: 랜덤 텐서 in/out shape, 한 step 학습 시 MSE 감소, `w=0/1/2`에서 출력이 달라지는지.
3. **Model 통합** — RSSM 제거, 디코더 in_channels=C 리팩터, `forward`/`generate_futures`/
   `counterfactual_futures`, latent 정규화. 단위 테스트: forward dict 키/shape.
4. **Trainer 손실 교체** — KL→diffusion, 관측 recon 유지, `_observe_full`(trainer:614) 갱신.
5. **Counterfactual 검증** — `counterfactual_futures(action_list, w)`, action sensitivity·
   guidance sweep·realism 지표, figure(행동 그리드 + guidance 그리드).
6. **Tiny-overfit sanity** — 윈도우 몇 개로 diffusion MSE↓ + 미래 디코딩이 그럴듯한지,
   **행동을 바꿨을 때(특히 w>1) 미래가 실제로 갈라지는지** 확인 (핵심 sanity).

각 Phase는 기존 RSSM 파일을 보존하고 새 변형 파일(`*_diff.py`)로 두거나, hj 내에서 직접
교체하는 방식 중 선택 (구현 착수 시 확정).

---

## 9. 리스크 / 열린 질문

- **Latent 정규화**가 학습 안정성의 핵심. EMA 통계 추정 워밍업 필요.
- **이동 타깃 문제**: encoder가 같이 학습되면 diffusion 타깃 z_future가 흔들림.
  1차 = stop-grad joint, 불안정 시 = 2단계(encoder freeze).
- **"행동→미래" 인과가 데이터에 약하면 sensitivity가 안 오름**: 데이터가 대부분 직진 주행이면
  행동↔미래 상관이 약해 `w↑` 해도 미래가 안 갈라질 수 있음. → 회전·제동이 풍부한 윈도우
  비중 점검, 필요 시 검증 셋 큐레이션.
- **OOD action 위험**: probe action(급제동·급회전)이 데이터 action 분포 밖이면 미래가 깨짐.
  → probe는 분위수 기반으로, `w`도 과도하게(예: 8↑) 올리지 않기.
- **FH=4** → 미래 4프레임을 한 번에 denoise. 행동 효과가 더 긴 horizon에 드러남.
  더 늘리려면(예: 6) `cfg.FUTURE_HORIZON`만 바꾸면 됨 — 이 파이프라인은
  `MultiArrowStreamDataset(seq_len=RF+FH)`로 윈도우를 매 실행 동적 생성하므로
  `build_arrow_window_index` 캐시 재생성은 불필요(seq_len이 길수록 윈도우 수는 줄어듦).
- **평가 baseline**: 동일 RF/FH/인코더에서 RSSM(action deterministic, 한 미래) vs
  diffusion+CFG(action sweep)의 **action sensitivity 비교**가 ablation의 핵심.
  RSSM은 `w`-손잡이가 없으므로 "행동→미래 인과를 정량 측정 가능한가" 자체가 차별점.

---

## 10. RSSM ↔ Action-Conditional Diffusion 핵심 대응표

| 측면 | RSSMTD (현재) | Action-Conditional Diffusion + CFG (제안) |
|------|---------------|--------------------------|
| "행동→미래" 질문 | **불가** (한 미래만, 행동 효과 분리 손잡이 없음) | **가능** (`w`-sweep으로 행동 민감도 정량화) |
| action 주입 | GRU에 deterministic 입력, 한 결과 | cross-attn 조건 + **CFG로 강도 조절(`w`)** |
| 미래 생성 | prior GRU 1-step Gaussian, 자기회귀 | 조건부 DiT, 전체 horizon 동시 denoise |
| 다양성 원천 | 한 스텝 가우시안 분산 (약함) | (부차) noise seed별 역확산 |
| state | h(hidden)+s(stochastic), 369 토큰 | encoder latent z, 368 토큰 |
| 확률 손실 | KL(prior‖posterior) | diffusion MSE(v) + action 드롭아웃 |
| 핵심 검증 | `n_samples` 평균 (미완) | **counterfactual 행동 그리드 + sensitivity 곡선** |
| 디코더 입력 | cat(h,s)=2C | z=C |

---

## 11. 개념 보충 Q&A

### 11.1 이 방식은 강화학습인가? reward가 데이터셋에 없는데 괜찮나
**아니다 — 자기지도(self-supervised) + 모방학습(imitation)이며 reward가 필요 없다.**
학습 신호는 두 종류이고 둘 다 reward와 무관:
1. **미래 예측 = 자기지도**: "정답 라벨"이 곧 미래 관측 프레임 자체. 데이터에 미래
   RGB/LiDAR가 있으므로 그걸 맞추면 됨. 손실 = RGB L1 / LiDAR chamfer·depth /
   diffusion MSE — 전부 "예측이 실제 미래 관측과 가까운가"이지 "좋은 행동인가"가 아님.
2. **행동 예측 = 모방학습(BC)**: `trainer_muvo_2D.py`의
   `F.l1_loss(action_pred, batch["action"])`는 기록된 실제 throttle/steer를 따라하는 것.
   reward 최대화(RL)가 아님.

즉 데이터셋에 `(관측, 행동)`만 있으면 충분. "어떤 행동을 하면 어떤 미래"를 묻는 counterfactual
생성은 RL이 아니라 **조건부 생성 모델의 추론(intervention)** 일 뿐이다.

> **RL이 필요해지는 경우(향후 확장):** world model 위에서 정책을 *최적화*하려면
> (Dreamer/MILE식 상상 rollout 정책학습) reward가 필요 → 데이터에 reward 컬럼 추가,
> 또는 reward model 별도 학습, 또는 규칙 기반 reward(충돌=−, 경로추종=+) 정의.
> 현재 목표(action-conditional counterfactual 생성)는 거기까지 불필요.

### 11.2 모델은 z_future를 직접 맞히나? (v/ε 예측)
**직접 출력하지 않는다.** denoiser 입력 = 노이즈 섞인 미래 latent `z_τ` + τ + 조건(`z_past`,
speed, action), 출력 = `v_pred`(또는 `ε_pred`). 단, **v와 z_future는 동치** — 대수적으로 복원 가능.

α=√ᾱ, σ=√(1−ᾱ), `z_τ = α·z0 + σ·ε`, `v = α·ε − σ·z0` 일 때 (α²+σ²=1):
```
z0_hat = α·z_τ − σ·v_pred      # ← 복원된 z_future (디코딩/future-decode 손실/시각화)
ε_hat  = σ·z_τ + α·v_pred
```
그래서 `training_targets()`가 `v_pred, v_target, z0_hat`을 함께 반환(§2.4).

**왜 z0 직접 회귀가 아니라 v/ε인가:**
1. 고노이즈(τ↑)에서 z_τ≈순수노이즈 → z0 직접 복원은 거의 불가, 타깃 스케일 폭발.
   ε는 모든 τ에서 단위분산~1로 정규화돼 학습 균일.
2. v는 고노이즈에선 x0-pred처럼, 저노이즈에선 ε-pred처럼 → timestep 전체 손실 균형.
3. ε·v 타깃은 분산이 τ에 무관하게 일정 → 별도 재가중 없이 MSE면 됨.

`cfg PARAMETERIZATION="eps"`로 전환 시 복원식만 `z0_hat = (z_τ − σ·ε_pred)/α`로 달라짐.
**CFG는 v/ε 어느 쪽이든 동일하게 `pred_w = pred_uncond + w·(pred_cond − pred_uncond)`** 로 적용.

### 11.3 "어떤 행동을 하면 어떤 미래"가 구체적으로 어떻게 생성되나
**핵심: 조건 중 `z_past`·speed·noise seed는 고정, 행동만 바꾼다.** diffusion은 "과거+행동→미래"를
조건부 분포 `p(z_future | z_past, speed, action)`로 학습하므로, 행동을 a₁→a₂→a₃로 바꾸면
조건이 달라져 미래 분포가 이동한다. 같은 noise seed를 공유하면 그 이동분이 **순수 행동 효과**.

```
공통 고정: z_past, speed, noise seed z_T~N(0,I)        ← A개 행동에 공통
   action a1(직진)     action a2(좌회전)    action a3(급제동)   ← 여기만 다름
        │ DDIM 50스텝 + CFG(w):  v_w = v_∅ + w·(v_a − v_∅)
   z0^(a1)            z0^(a2)              z0^(a3)              ← 행동별 미래 latent
        │ 디코더(공유)
   미래(직진 유지)    미래(좌측 차선/heading)  미래(감속·전방차 접근)
```
- **`w`(guidance scale)** 가 "행동→미래"의 영향 강도를 조절: `w=0`이면 행동 무시(분기 사라짐),
  `w↑`이면 분기가 또렷. 즉 **행동 민감도가 hyperparameter로 노출**된다.
- 구현은 배치로: 행동 A개 + cond/uncond 2분기를 쌓아 `(2·B·A,...)` 한 번에 역확산 →
  `(B,A,FH,C,N)` reshape (§3.4 `counterfactual_futures`).

**RSSM과의 차이(핵심 동기):** RSSM은 행동을 deterministic하게 받아 한 점만 내고,
"얼마나 따를지" 손잡이가 없어 행동 민감도를 측정할 수 없다. diffusion+CFG는
`w`-sweep으로 sensitivity 곡선을 그려 **"행동→미래" 인과를 정량화**한다.

### 11.4 행동을 고정하면 다양성은 어디서 오나 (그리고 그게 핵심이 아닌 이유)
같은 행동에서도 환경 응답(다른 차·보행자·신호)은 여러 갈래라 noise seed별로 미래가 갈린다
— 이게 §5.3의 "noise 다양성"이다. 하지만 **이건 부차적**이다.

**이번 설계의 headline은 "다양성"이 아니라 "어떤 행동을 하면 어떤 미래가 되는가"** 다:
- 묻는 질문이 "미래가 얼마나 다양한가?"가 아니라 **"내가 좌회전했다면 / 급제동했다면
  미래가 어떻게 달라졌을까?"** 이기 때문.
- 따라서 1차 평가의 중심은 noise diversity가 아니라 **action sensitivity(§5.2)** 이고,
  CFG `w`가 그 강도를 조절하는 손잡이다.

**행동 자체까지 생성·계획하고 싶다면(향후 planning 확장):**
- action을 조건이 아니라 **생성 대상에 포함**(z_future와 함께 a_future도 denoise)
  → 정책 분포까지 샘플링. "여러 미래 + 여러 행동" 동시 생성.
- 여기에 reward/cost(§11.1 확장)를 붙이면 **sampling-based planning**(후보 rollout 생성 →
  비용 최소 행동 선택, MPC/CEM류)으로 확장. 이때부터 reward 필요.
- 1차 설계 범위 밖. 본 설계는 **행동을 입력 조건으로 고정한 "행동→미래" 생성**까지.

---

## 12. 구현 정합성 체크리스트 (코드 대조 결과, 2026-05-27)

설계서를 실제 `models/trainer/config_muvo_2D.py`와 대조해 찾은, **그대로 구현하면 막히는 지점**들.
착수 전 이 표를 먼저 처리한다.

| # | 문제 | 위치 | 처리 |
|---|------|------|------|
| 1 | 인코더 z는 **368 토큰**(image240+lidar128), policy 토큰 없음. `_slice_token_state`가 인덱스 368에서 policy를 꺼내려다 **인덱스 초과**. | `_slice_token_state`(models:696), `n_total_tokens`(:692), `encode_fuse_sequence`(:734) | policy 분기 삭제, N=368, action_pred는 mean-pooled token에서 (§3.2) |
| 2 | 디코더가 입력 2개(`h`,`s`)를 cat→2C. `__init__`뿐 아니라 **`forward`도** 바꿔야. | `TokenConvDecoder2D.forward`(:575), `_decode_state`(:714), policy cat(:725-728) | 단일 입력 `forward(state_slice)`, `_decode_state(state)` (§3.2) |
| 3 | recon을 z_past(4프레임)만 하면 ① 디코더가 **미래 latent 디코딩 미학습**, ② `rgb_label_1`(6프레임)과 **shape mismatch**. | `forward`(§3.3), `compute_loss`(trainer:590) | recon은 **z 전체(6프레임)** 디코딩 (§3.3) |
| 4 | speed를 **미래 speed**로 조건하면 추론 시 알 수 없고 counterfactual과 모순(누설). | §2.3 / §3.3 / §3.4 | **관측(과거) speed**로만 조건 (§2.3 ⚠️) |
| 5 | `_observe_and_imagine`는 `forward(batch_rf)`(RF만)+`posterior` state로 동작 → 새 forward/diffusion과 비호환. **메서드 통째 재작성**. | `_observe_and_imagine`(trainer:619), figure(trainer:114) | full-batch forward→z_past→generate/counterfactual. **5-튜플·키 계약 유지** (§5 ⚠️) |
| 6 | action 시간 정렬(off-by-one) 안 맞으면 sensitivity 붕괴. | §2.3, models:402-404, trainer:631-632 | 데이터 컨벤션과 동일 정렬 + 단위 테스트 (§2.3 ⚠️) |
| 7 | 현재 config는 "multi-future"용 → `P_UNCOND`/`GUIDANCE_SCALES`/`CF_ACTIONS` 없음. | `config_muvo_2D.py` DIFFUSION/PREDICTION | Phase 1에서 키 추가 (§6 ⚠️) |
| 8 | `compute_loss`에 diffusion 분기 없음. KL 분기는 prior 부재로 자동 스킵됨. | `compute_loss`(trainer:546-552) | `if "v_pred" in output:` 분기 추가 (§4 구현 메모) |
| 9 | latent EMA 정규화 μ/σ: train에서만 갱신·eval 고정·DDP 동기 필요. 정규화 이중 적용 금지. | `latent_normalize`/`latent_denormalize`(신규) | register_buffer + `if self.training` 갱신, DDP all-reduce 또는 rank0 (§2.2) |

**판정:** 알고리즘 설계(diffusion+CFG, v-pred, counterfactual)는 일관적이고 구현 가능하다.
다만 위 1–5는 *코드 시그니처/형상* 수준의 정합성 문제라 손대지 않으면 **첫 forward에서 에러**가 난다.
체크리스트 처리 후 §8 Phase 순서대로 가면 된다. (RSSM 파일은 보존하고 `*_diff.py`로 분기 권장.)
