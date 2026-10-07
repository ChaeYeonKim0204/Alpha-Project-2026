# 코드 개발 계획서: Action-conditioned Latent Diffusion Future Prediction (Diffuvo_WM)

> 기존 RSSM 기반 deterministic future 예측을 **latent diffusion**으로 교체하여,
> 같은 과거·action 조건에서도 여러 가능한 미래 분포를 생성한다.
> (대화로 확정한 설계 결정을 코드 작업 전 단계로 정리한 문서)

---

## 1. 목적

기존 world model은 `z_future = f(z_past, speed, action)` 형태로 **하나의 deterministic 미래**만
출력한다. 본 개발은 이를

```
p_θ(z_future | z_past, speed, action)
```

조건부 분포 학습으로 바꿔, **(1) action별 counterfactual 미래**와 **(2) 동일 action 내 미래
불확실성**을 함께 생성·분석할 수 있게 한다.

---

## 2. 확정된 설계 결정 (요약)

| 항목 | 결정 |
|------|------|
| Diffusion latent | **융합 encoder 토큰** `z=(B,S,C=256,N=368)`. RSSM 제거 |
| 관측(past) 복원 | **Encoder→Decoder AE** (별도 transition 없음) |
| AE recon 타깃 | **RF+FH 전체 프레임** RGB |
| LiDAR | **encoder fusion 입력으로만 유지**, 출력은 RGB만 (LiDAR decoder/loss 제거) |
| Denoiser 백본 | **Transformer / DiT** |
| 예측 타깃·스케줄 | **v-prediction + cosine schedule** (학습), DDIM 샘플링 |
| 시간 범위 | **FH 프레임 joint** diffusion (동일 timestep 공유) |
| 조건화 | past latent → **cross-attention**, action·speed·timestep → **AdaLN** |
| CFG | **action classifier-free guidance** (학습 중 action drop) |
| 학습 방식 | **2-stage** (① AE 학습 → ② AE freeze 후 diffusion 학습) |
| 손실 | ① AE: RGB recon L1 / ② diffusion: latent v-MSE |
| 학습 길이 | **epoch 기준** (`--epochs`) |
| 추론 스텝 | `--ddim-steps` CLI, **매 run 사용자에게 확인** |
| 코드 구조 | **신규 diffusion 모듈 분리** |
| 실험 산출물 | `Diffuvo_WM/figures/<run_name>/` **run별 폴더**로 분리, 안에 실험별 PNG (§6.0) |
| 평가 샘플 | **val held-out 3개** |

> 토큰 수: image `(10,24)`=240, lidar `(2,64)`=128 → 합 **N=368**, 채널 **C=256**.

---

## 3. 아키텍처 변경 (현재 → 신규)

### 3.0 src/ 파일 구성 (각 .py 역할 + 구현 상태)

| 파일 | 역할 | 상태 |
|------|------|------|
| `config_Diffuvo.py` | `cfg` SimpleNamespace 트리. data 크기·모델 dim·loss 가중치·스케줄 + **`DIFFUSION` namespace 추가 완료**(§8-1). **(KL·LiDAR-loss 항목 정리는 §8-2/§8-4 때)** | DIFFUSION 추가됨 |
| `data_Diffuvo.py` | Arrow → RF+FH 윈도우 streaming dataset/dataloader. RGB+LiDAR+speed+action 텐서 제공 | 재사용 |
| `models_Diffuvo.py` | `Model`: `encode_fuse_sequence`(→`(B,S,256,368)`), `decode_rgb`(**추가 완료**: image 토큰→`{rgb_1,2,4}`), 이미지/LiDAR encoder·FPN·fusion transformer·`TokenConvDecoder2D`. **(RSSMTD·lidar_decoder·imagine 제거는 §8-4와 동시에)** | decode_rgb 추가됨 |
| `diffusion_Diffuvo.py` | **신규.** latent future diffusion 백본: `CosineNoiseSchedule`(v-pred), `DiTBlock`(self+cross attn, adaLN-Zero), `LatentFutureDiffusion`(`forward`→v_pred / `loss`→v-MSE / `sample`→DDIM+CFG / `from_cfg`) | **구현 완료** |
| `trainer_Diffuvo.py` | **`AEModule`(stage1, RGB L1)** + **`DiffusionModule`(stage2, v-MSE, AE freeze)** 추가 완료. figure 저장(`save_reconstruction_figure`/`save_ae_reconstruction_figure`/`save_loss_metric_figure`) + `ExperimentCSVLogger`. legacy `WorldModelTrainer` 유지. | 2-stage 추가됨 |
| `train_Diffuvo.py` | CLI 학습 엔트리. **`--stage 1\|2` 분리 실행**(+`--ae-checkpoint`), `resolve_figure_dir`(→`figures/<run_name>/`), DDP/precision, scheduler, resume. | 구현 완료 |
| `predict_Diffuvo.py` | **신규.** checkpoint 로드(`--checkpoint` prefix 라우팅 / `--ae-checkpoint`+`--diffusion-checkpoint`) → DDIM 샘플링 → 실험 1/2/3 figure(§5·§6·§7). `resolve_figure_dir` 재사용 → `figures/<run_name>/`. CLI: `--ddim-steps`(필수)·`--cfg-scale`·`--n-seeds`/`--seeds`·`--allow-untrained` | **구현 완료** |

> 의존: `train_Diffuvo` → `trainer_Diffuvo` → (`models_Diffuvo`, `diffusion_Diffuvo`) → `config_Diffuvo`.
> `data_Diffuvo`는 학습/추론 양쪽에서 dataloader 제공.

### 3.1 유지하는 부분 (그대로 재사용)
- `FPNDecoder`, `SensorFusionTransformer`, `SpeedEncoder`
- `Model.encode_fuse_sequence(image, lidar, speed)` → `(B,S,256,368)` 융합 토큰
- `TokenConvDecoder2D` (image branch만; RGB 복원)

### 3.2 제거하는 부분
- `ConvGRUCellGlo`, `RepresentationModelTD`, `RSSMTD` (전체 transition)
- `Model.lidar_decoder` 및 LiDAR reconstruction 손실/메트릭 (Chamfer 등)
- KL 관련: `gaussian_kl_loss`, `balanced_kl_loss`, `WEIGHT_PROBABILISTIC`, free-bits
- `Model.imagine` (RSSM rollout) → diffusion sampling으로 대체

### 3.3 신규 모듈 — `LatentFutureDiffusion` ✅ 구현 완료
파일 `diffusion_Diffuvo.py`. 구성:

- **`CosineNoiseSchedule`**: cosine ᾱ 스케줄. `q_sample`(z_t, v_target 생성)·
  `predict_start_and_noise`(v_pred→z₀_hat, ε_hat) 헬퍼. (`v = √ᾱ·ε − √(1−ᾱ)·z₀`)
- **`DiTBlock`**: self-attention(future 토큰) + cross-attention(past 토큰)
  + **adaLN-Zero**(timestep+action+speed → scale/shift/gate ×3, gate=0 init) + MLP.
- **`LatentFutureDiffusion(nn.Module)`**:
  - `forward(z_future_noisy, t, z_past, action, speed, drop_action=None)` → `v_pred` (학습용)
  - `loss(z_future_clean, z_past, action, speed, p_drop, generator)` → v-MSE (CFG action drop 포함)
  - `sample(z_past, action, speed, n_steps, cfg_scale, generator)` → `z_future_hat`
    (DDIM eta=0 루프, action CFG 적용; seed는 `generator`로 제어)
  - `from_cfg(cfg)` 클래스메서드 — cfg.MODEL/RF/FH + cfg.DIFFUSION(없으면 default)에서 차원·하이퍼 로드.
  - timestep embedding(sinusoidal) + action/speed embedding(MLP) 합산 conditioning,
    CFG용 학습 가능 null-action 임베딩 보유. FH 프레임 joint, past는 cross-attn memory.
  - latent `(B,S,C=256,N=368)`를 `(B,frames·N,C)`로 펴서 처리, C↔D(hidden) projection.

> smoke test 통과: v_pred 모양 `(B,FH,C,N)` 일치 / v-pred 역변환 오차 0 / loss backward /
> 같은 seed→동일·다른 seed→다른 미래(다양성) / `from_cfg` 차원(N=368,C=256) 확인.

### 3.4 `Model` 재구성 (`models_Diffuvo.py`)
```text
encode_fuse_sequence(...)        # 유지: (B,S,256,368)
decode_rgb(z)                    # ✅ 추가 완료: image 토큰 슬라이스(240) → RGB {rgb_1,2,4}
                                 #    (_slice_token_state + image_decoder 재사용, lidar_decoder 미호출)
# RSSM / lidar_decoder / imagine 삭제  ← §8-4(trainer 재작성)와 동시에 제거 예정
```
> **latent 구성 주의**: diffusion은 전체 368 토큰을 denoise한다(LiDAR 토큰 포함, 사용자 선택).
> AE recon은 image 토큰(240)→RGB만 감독한다. LiDAR 토큰은 frozen encoder의 결정적 출력이라
> stage 2에서 안정적 타깃이 된다. 만약 LiDAR 토큰 drift가 문제되면 **fallback: latent를 image
> 토큰(240)만으로 한정**(LiDAR는 fusion 입력으로만)으로 축소한다. → §9 참조.

---

## 4. 학습 절차 (2-stage)

> **실행 방식: `--stage` 기반 분리 실행** (단일 실행 내 순차 아님).
> Stage 1을 끝내 AE checkpoint를 만든 뒤, Stage 2를 **별도 실행**으로 그 checkpoint를 freeze 로드한다.
> ```bash
> # Stage 1 (AE)
> python train_Diffuvo.py --stage 1 --epochs <N> --run-name <run>
> # Stage 2 (diffusion) — stage1 checkpoint를 freeze 로드
> python train_Diffuvo.py --stage 2 --ae-checkpoint logs/<run>/version_*/checkpoints/last.ckpt \
>     --epochs <M> --run-name <run>
> ```
> `--stage` 미지정 시 legacy RSSM `WorldModelTrainer`로 동작(점진적 전환).

### Stage 1 — AutoEncoder  (`AEModule`)
1. RF+FH 윈도우 → `encode_fuse_sequence` → 토큰 `z` → `decode_rgb(z)` → RGB
2. 손실: **RGB recon L1** (multi-scale `rgb_{1,2,4}`, 기존 `_spatial_regression_loss` 재사용),
   RF+FH **전체 프레임** 감독
3. 결과: encoder+decoder weight 저장 (latent space 고정용)

### Stage 2 — Diffusion (AE **완전 freeze**)  (`DiffusionModule`)
1. frozen encoder로 `z_past=(B,RF,256,368)`, `z_future_clean=(B,FH,256,368)` 추출
2. `t ~ U(1,T)`, `ε ~ N(0,I)` → cosine schedule로 `z_future_noisy` 생성, `v_target` 계산
3. `LatentFutureDiffusion.forward` → `v_pred`
   - action은 확률 `p_drop`(예: 0.1)로 null-action 치환 (CFG 학습)
4. 손실: **latent v-MSE** `‖v_pred − v_target‖²`
5. 학습 길이는 epoch 기준. OneCycleLR(또는 cosine) 유지.

> **학습 시 "미래"는 GT future 프레임을 encoder에 통과시킨 `z_future_clean`이 정답(teacher)이다.**

---

## 5. 추론 절차 (future 생성)

1. 과거 프레임만 encode → `z_past`
2. `z_T ~ N(0,I)` `(B,FH,256,368)` (seed로 제어)
3. DDIM 루프 `t=T→0` (`--ddim-steps`): `z_past` + 선택 `action` + `speed` 조건,
   CFG `v = v_uncond + s·(v_cond − v_uncond)` → `z_future_hat`
4. `decode_rgb(z_future_hat)` → 미래 RGB 프레임
5. **다양성**: seed 변경 → 다른 미래 / **intervention**: action 변경 → counterfactual 미래

---

## 6. 실험 & 시각화 설계

### 6.0 출력 디렉토리 구조 (모든 결과 PNG 공통)

한 번의 실험에서 결과 PNG가 여러 장 나오므로(학습 recon·loss·주기적 recon + 추론 실험 1/2/3),
**run마다 `figures/<run_name>/` 하위 폴더 하나로 묶는다.** run끼리 파일이 섞이지 않고,
폴더가 이미 run을 식별하므로 **파일명엔 run_name prefix를 붙이지 않는다**.

- base figures 경로 결정 순서(둘 다 `train`·`predict` 공통):
  `--figure-dir` 인자 → 환경변수 `ALPHA26_OUTPUT_ROOT/figures` → `./results/figures` (fallback).
  그 뒤에 `<run_name>/` 를 붙여 최종 경로를 만든다.
- 입력 샘플: **val held-out 3개** (없으면 폴더 자동 생성).

```
figures/
  <run_name>/                # 예: exp001_diffuvo
    # ── 학습 산출 (train_Diffuvo.py) ──
    reconstruction_s0.png    # 최종 checkpoint recon (held-out 샘플)
    loss_metrics.png         # TensorBoard 스칼라 → loss/metric 패널
    e0100.png                # --recon-every-n-epochs 주기적 recon (epoch 100)
    # ── 추론 산출 (predict_Diffuvo.py) ── ※ val 샘플마다 _s<idx> 접미사
    intervention_s0.png      # 실험1: action별 counterfactual 미래
    seed_diversity_s0.png    # 실험2: seed별 미래 다양성
    mean_var_s0.png          # 실험3: action별 Mean / Variance / Error
```

> **구현 상태**:
> - 학습 측 폴더 분리는 **구현 완료** — `train_Diffuvo.resolve_figure_dir(args, run_cfg)`가
>   `figures/<run_name>/` 를 반환하고, `save_reconstruction_figure` / `save_loss_metric_figure` /
>   `PeriodicReconstructionCallback` 모두 그 폴더에 prefix 없는 파일명으로 저장한다.
> - `predict_Diffuvo.py` 도 **구현 완료** — 동일한 `resolve_figure_dir`를 재사용해
>   같은 `figures/<run_name>/` 에 `intervention_s<idx>` / `seed_diversity_s<idx>` / `mean_var_s<idx>` 저장.

### Counterfactual action 정의 — 고정 프리셋 `(throttle, steer)`
| 라벨 | throttle | steer |
|------|----------|-------|
| 직진(straight) | 현재 유지 | 0.0 |
| 좌회전(left) | 현재 유지 | −0.5 |
| 우회전(right) | 현재 유지 | +0.5 |
| 브레이크(brake) | 0.0 | 0.0 |
> 값은 cfg에 상수로 두고 조정 가능. (action_dim=2 = [throttle, steer])

### 실험 1 — Action intervention (PNG #1)
- 고정: past, speed, **seed**. 변경: action(프리셋). action별 future 비교.
- **no-action baseline 행 포함**: action 프리셋 행들 위에, action을 주지 않은
  순수 미래 예측(`diffusion.sample(z_past, None, speed, ...)`, null-action 임베딩)을
  한 행 그린다. action 조건 행들과의 대조로 "action이 미래를 실제로 바꾸는지" 판정.
  null-action은 cond==uncond라 CFG가 무효이므로 `cfg_scale=1.0`(off)로 호출한다.
  title에 no-action / straight 각각의 `latentMSE(sample,teacher)`·`PSNR(sample/GT)` 표기.

### 실험 2 — Seed diversity (PNG #2)
- 고정: past, speed, action. 변경: seed. seed별 future sample 비교.

### 실험 3 — Mean / Variance (PNG #3)
- action별 다수 seed → 픽셀별 **Mean**(RGB), **Variance**(magma heatmap),
  **Error=|mean−GT|**(magma heatmap).

### 시각화 row 구성 (계획서 §5 기준)
```
[Observed GT]    obs0..obs_{RF-1}
[Observed Recon] AE recon (frozen decoder)
[Future GT]      gt0..gt_{FH-1}
# 실험1(intervention)
[no-action]      action 미입력 자유 미래 (null-action baseline)
[act=straight/left/right/brake]  action별 counterfactual 미래
# 실험2/3
[Seed1..K]       각 seed future
[Mean]           seed 평균 (RGB)
[Variance]       seed 분산 (heatmap)
[Error]          |mean - future GT| (heatmap)
```

---

## 7. CLI 인터페이스 (run마다 사용자 확인 노브)

학습(`train_Diffuvo.py`):

| flag | 의미 | 비고 |
|------|------|------|
| `--epochs` | 학습 길이(AE/diffusion 각 stage) | epoch→step 환산을 답변 시 함께 안내 |

추론(`predict_Diffuvo.py`, 구현 완료):

| flag | 의미 | 비고 |
|------|------|------|
| `--ddim-steps` | 추론 denoise 스텝 수 | **required=True** — default 강제 안 함, 매 run 명시 |
| `--cfg-scale` | action guidance 강도 | default 1.0(off). 실험1 효과 조절 |
| `--n-seeds` / `--seeds` | diffusion 샘플 seed 개수/값 | 실험2·3 (`--seeds`가 `--n-seeds` 우선) |
| `--checkpoint` / `--ae-checkpoint` / `--diffusion-checkpoint` | 가중치 소스 | 통합 prefix 라우팅 또는 stage별 |
| `--sample-indices` | 시각화할 val 샘플(기본 `0,1,2`) | held-out 3개 |
| `--exp3-action` | 실험3 mean/var 대상 action | 기본 `straight`, `all`이면 전체 프리셋 |
| `--allow-untrained` | checkpoint 없이 random weight 실행 | 파이프라인 점검용 |

> (이 선호는 메모리 `diffuvo-run-config-prefs`에 기록됨)

---

## 8. 단계별 작업 체크리스트 (구현 순서)

1. [~] `config_Diffuvo.py`: `DIFFUSION` namespace 추가 — ✅ 완료
       (T·SCHEDULE_S·HIDDEN_DIM·DEPTH·N_HEADS·MLP_RATIO·DROPOUT·P_DROP·CFG_SCALE·ACTION_PRESETS,
       `from_cfg`와 키 1:1). **RSSM/KL/LiDAR-loss 항목 정리는 §8-2/§8-4 리팩토링 때 함께(아직 사용 중이라 보류).**
2. [~] `models_Diffuvo.py`: `decode_rgb` 추가 — ✅ 완료(image 토큰만→`{rgb_1,2,4}`, lidar_decoder 미호출).
       `encode_fuse_sequence` 유지. **RSSMTD·lidar_decoder·imagine 제거는 §8-4와 동시에**
       (trainer가 아직 호출 중 — 먼저 지우면 import/학습 체인 깨짐).
3. [x] `diffusion_Diffuvo.py`(신규): `CosineNoiseSchedule`, `DiTBlock`,
       `LatentFutureDiffusion` (forward / loss / sample+CFG / from_cfg) — ✅ 완료, smoke test 통과
4. [x] `trainer_Diffuvo.py`: **`AEModule`(RGB L1) + `DiffusionModule`(latent v-MSE, AE freeze)** 신규 추가 — ✅ 완료.
       AE freeze는 `requires_grad_(False)`+`eval()`, `train()` override로 학습 중에도 AE는 eval 유지(검증 통과).
       공용 헬퍼 `multiscale_rgb_l1`/`_adamw_onecycle`/`save_ae_reconstruction_figure`.
       **(legacy `WorldModelTrainer`·KL/Chamfer 코드는 유지 — RSSM/lidar/KL 제거는 후속 cleanup. §8-2 참조)**
5. [x] `train_Diffuvo.py`: **`--stage 1|2` 분리 실행** + `--ae-checkpoint`(stage2 필수) 추가 — ✅ 완료.
       stage별 모듈 라우팅, stage1=AE recon figure / stage2=정성 figure는 predict가 담당.
6. [x] `predict_Diffuvo.py`(신규): checkpoint 로드(통합/stage별 prefix 라우팅), 실험 1/2/3 figure 생성 →
       `resolve_figure_dir` 재사용해 `figures/<run_name>/` 에 `intervention_s<idx>` / `seed_diversity_s<idx>` /
       `mean_var_s<idx>` 저장 — ✅ 완료, 합성 샘플 end-to-end 통과.
       (RGB 디코드는 `Model.decode_rgb` 생기기 전까지 `_slice_token_state`+`image_decoder` 폴백.)
7. [ ] import 경로 확인(이미 `_Diffuvo`로 정리 완료)

---

## 9. 검증 방법

- **AE sanity**: Stage 1 후 val recon PSNR이 합리적인지(예: >20dB), recon figure 육안 확인
- **Diffusion sanity**: 작은 overfit subset에서 v-MSE 감소 + 샘플이 GT future와 유사해지는지
- **실험 1 검증**: action을 좌/우로 바꿨을 때 미래 장면이 실제로 좌/우로 달라지는지
- **실험 2 검증**: 같은 action·다른 seed에서 미래가 다양하되 그럴듯한지(variance>0, 붕괴 아님)
- **숫자 메트릭**: future RGB PSNR(mean sample vs GT), seed 간 variance 평균

---

## 10. 미해결 / 주의 사항

- **LiDAR 토큰 감독 부재**: diffusion latent에 LiDAR 토큰(128)이 포함되나 AE는 RGB만 복원.
  frozen encoder의 결정적 출력이라 타깃은 안정적이지만, 표현이 비효율적일 수 있음 →
  필요 시 **latent를 image 토큰(240)만으로 축소**하는 fallback 준비.
- **2-stage 핸드오프**: Stage 1 checkpoint 경로를 Stage 2에서 로드(encoder/decoder),
  freeze 확실히(`requires_grad=False` + `eval()`).
- **CFG null-action**: 학습/추론에서 null-action 임베딩 동일하게 사용.
- **seq_len 일치**: `RECEPTIVE_FIELD=4`, `FUTURE_HORIZON`(현재 2) → window index와 일관.
- 기존 `experiment_log.csv`/ClearML 항목 중 LiDAR·KL 관련 컬럼은 의미 없어짐 → 정리 대상.
