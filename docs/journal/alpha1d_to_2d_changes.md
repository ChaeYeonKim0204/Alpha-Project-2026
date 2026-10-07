# alpha-1D → alpha-2D 변경사항 (5/14 ~ 현재, 2026-05-23 확인 기준)

> **검증 상태 (sub-agent 재검산 반영)**:
> - 모든 commit 시각은 git timestamp (KST) 그대로 사용 (이전 초안의 추정치 `~HH:MM` 전부 제거)
> - 숫자 claim 중 param 수/loss/KL nats/실험 row 수 등은 commit message 및 실험 기록 기반 claim으로 분류하고, diff만으로 확인 가능한 claim과 구분함
> - working tree 변경은 현재 `scripts/model_variants` 기준 7 path(소스 `.py` 4개 + `.pyc` 3개)이며, 아래 미커밋 섹션에 소스 변경 4개를 기준으로 정리함
> - commit history 기준으로 5/14에는 `scripts/model_variants/data.py`, `scripts/model_variants/train.py` 변경이 있었고, 5/16, 5/17, 5/21, 5/22, 5/23에는 committed `scripts/model_variants/*.py` 변경 없음
> - 단, 현재 미커밋 working tree의 `_muvo_2D` 소스 4개(`config_muvo_2D.py`, `models_muvo_2D.py`, `train_muvo_2D.py`, `trainer_muvo_2D.py`)는 파일 수정시각 기준 5/20에 작성된 코드 변경으로 보임

---

## 5/14 (목) 17:53:39 — 원본 alpha one-window overfit / resume 정비 (커밋 `87bd950`)

### 원본 alpha train/data 실험 편의 기능 정비
- `data.py`의 `MUVODataset.__init__`에서 임시로 걸려 있던 "8프레임만 받기" 제한을 제거하고, `self.n = self.table.num_rows`로 전체 Arrow table 길이를 사용하도록 복원함.
- 이 시점의 원본 alpha image transform은 encoder input과 reconstruction target 모두 `transforms.Resize` 기반이었고, crop 기반 transform은 아직 도입되지 않은 상태. 단, 원본 alpha 설정은 `IMAGE_INPUT_SIZE=(300, 400)`, `RGB_RECON_SIZE=(216, 288)`였으므로 두 transform 모두 resize였지만 해상도는 서로 달랐음.
- `train.py`에 `--one-window-overfit` CLI flag를 추가함. 이 옵션을 켜면 `USE_ALL_TRAIN_RUNS=False`, `overfit_samples=1`, `batch_size=1`로 강제하여 정확히 한 window만 학습/시각화하도록 구성함.
- `train.py`에 `--resume-from-checkpoint` CLI flag와 `maybe_extend_onecycle_resume_checkpoint()`를 추가함. OneCycleLR checkpoint를 더 긴 epoch로 이어 학습할 때 scheduler `total_steps` 및 `_schedule_phases`를 새 총 step 수에 맞춰 patch한 임시 checkpoint를 생성하도록 함.
- `trainer.fit()` 호출에 `ckpt_path=resume_ckpt_path`를 전달하도록 수정하여 validation 사용 여부와 무관하게 resume 경로가 적용되도록 함.

---

## 5/14 (목) 밤 ~ 5/15 (금) 03:13 — `_muvo` 1D 브랜치 신규 도입 + 새벽 후속 정비

실제 작업 흐름상 5/14 밤 작업으로 정리함. 단, git commit timestamp는 KST 기준 5/15 01:58:07~03:12:56에 남아 있음.

### 01:58:07 — `_muvo` 스캐폴딩 신규 분기 (커밋 `a89c9de`)

**MUVO 스타일 변종 분기 생성**
- 기존 자체 제작 모델은 image transform이 `transforms.Resize`로 종횡비를 왜곡하는 형태였고, decoder/head 구조도 upstream MUVO와 차이가 큼을 확인함.
- 기존 alpha baseline을 보존한 채 MUVO 스타일을 시험할 수 있도록 `_muvo` 접미사 5개 파일을 신규 생성함:
  - `config_muvo.py`: image (320, 800), LiDAR range-view (32, 1024), embedding=256, RSSM hidden=512/state=256
  - `data_muvo.py`: `VerticalCropToSize` (height만 crop), `LIDAR_SCALE=40.0`, Arrow 데이터셋 파이프라인
  - `models_muvo.py`: `FPNDecoder`(bottom-up), `SensorFeatureConv`(plain Conv-BN-ReLU×2 + pool), `RSSM`, `SensorDecoder`/`SensorHead`, `Model`(BEV/voxel 제거)
  - `train_muvo.py`/`trainer_muvo.py`: 기존 alpha train/trainer.py를 기반으로 `_muvo` import로 교체하고, MUVO-style logging/visualization에 맞춰 일부 동작도 함께 조정
- encoder feature aggregation 쪽 `FPNDecoder`도 MUVO-style로 수정함. 기존 alpha `FPNDecoder`는 가장 작은 feature map `xs[2]`에서 시작해 `F.interpolate`로 위로 올리는 top-down 구조라 upstream 1D MUVO의 `Decoder`(TRANSFORMER.LARGE=True branch)에 가까웠음.
- upstream `muvo.yml`의 기본값은 `TRANSFORMER.LARGE=False`이고, 이 경우 `mile.py`는 `DecoderDS`를 사용함. 이에 맞춰 `_muvo`의 `FPNDecoder`를 `xs[0]`에서 시작해 `adaptive_max_pool2d`로 downsample하며 skip을 더하는 DecoderDS 계열의 bottom-up 구조로 변경함.
- 기존 alpha는 `cfg.MODEL.FUSION.IMAGE_TOKEN_DOWNSAMPLE=(20, 20)`, `LIDAR_TOKEN_DOWNSAMPLE=(4, 16)`을 사용하여 FPN 이후 `F.adaptive_avg_pool2d`로 명시적 token grid를 강제했음.
- `_muvo`에서는 MUVO-style FPN output 해상도 자체를 transformer token grid로 사용하도록 이 명시적 pooling 단계를 제거하고, `config_muvo.py`의 fusion cfg도 `TRANSFORMER_CHANNELS/LAYERS/HEADS/DROPOUT` 중심으로 단순화함.
- `train_muvo.py`는 `config_muvo`/`data_muvo`/`trainer_muvo` import로 전환하고, `change_summary`를 MUVO-style alpha 설명으로 교체함. 또한 `WorldModelTrainer(..., embedding_n_channels=128)` 명시를 제거하여 trainer/model 쪽 cfg embedding default를 사용하도록 변경함.
- `config_muvo.py`는 처음부터 RGB reconstruction 중심 baseline으로 들어와 `WEIGHT_LIDAR_RE=0.0`, `WEIGHT_LIDAR_EMPTY=0.0`, `WEIGHT_RGB=1.0`으로 설정됨.
- 5/14 원본 `train.py`에 추가된 `--one-window-overfit`, `--resume-from-checkpoint`, `maybe_extend_onecycle_resume_checkpoint()`, `trainer.fit(..., ckpt_path=resume_ckpt_path)` 흐름도 `train_muvo.py`에 함께 상속됨.
- `trainer_muvo.py`는 reconstruction figure를 full-resolution head(`rgb_1`/`lidar_reconstruction_1`) 기준으로 표시하고, `LIDAR_SCALE`을 반영해 depth visualization 및 LiDAR metric(chamfer/euclidean/range_mae)을 meter scale로 환산하도록 수정함. `embedding_n_channels` default도 고정 128에서 cfg `MODEL.EMBEDDING_DIM` 기반으로 변경하고, CSV `base_file` fallback을 `train_muvo.py`로 바꿈.
- 즉 5/14 원본 alpha에는 crop이 없었고, 5/15 `_muvo` 분기 생성 시점에 encoder 입력에는 `VerticalCropToSize(320, 800)`가 새로 들어갔으나 reconstruction target은 여전히 `transforms.Resize(216, 288)`를 사용 — transform 종류와 output 해상도 양쪽에서 encoder input과 reconstruction target이 불일치.

### 01:58:19 — 원본 alpha 코드에 1-window overfit baseline 기록 (커밋 `0ad74bf`, 12초 후)

**SSIM 도입 전 control 상태 baseline 확보**
- `a89c9de`(`_muvo` 스캐폴딩) 직후 12초 만에 추가 커밋. `_muvo` 파일은 손대지 않고 **원본 alpha 파일** (`config.py`, `trainer.py`, `models.py`) 만 수정함.
- LiDAR 영향 배제를 위해 원본 `config.py`의 `WEIGHT_LIDAR_RE=1.0 → 0.0`, `WEIGHT_LIDAR_EMPTY=0.05 → 0.0`으로 RGB-only 모드 활성화.
- 원본 `trainer.py` 시각화에서 stride-2 head(`rgb_2`/`lidar_re_2`) 대신 full-resolution head(`rgb_1`/`lidar_re_1`)를 표시하도록 수정.
- 원본 `models.py`의 `SensorDecoder` 레이어 정의 순서를 forward 흐름과 일치하도록 정리 (functional 변경 없음).
- experiment_log.csv에 h512/z256, h256/z128 RGB-only 1000ep overfit + `muvo_one_window_overfit_1000` 합쳐 5개 row 기록. results/figures에 PNG 7개 추가.

### 02:08:05 — SSIM-augmented trainer 분기 (커밋 `07b5aa2`)

**흐릿함 가설 검증용 별도 trainer 추가**
- alpha의 흐릿한 RGB가 SSIM 부재 때문일 수 있다는 가설을 `_muvo` trainer를 건드리지 않은 상태로 검증할 필요가 있음을 판단함.
- 신규 2개 파일 추가:
  - `train_ssim.py`: `train.py`를 import한 뒤 `train.WorldModelTrainer`를 `SSIMWorldModelTrainer`로 monkey-patch하는 방식으로 본문 복제 없이 구성함.
  - `trainer_ssim.py`: `SSIMLoss` 클래스를 upstream `muvo/losses.py:292-348`에서 inline 이식함. Gaussian window conv2d 기반.
- `SSIMWorldModelTrainer`는 `WorldModelTrainer`를 상속하며 `compute_loss`에서 multi-scale `rgb_{1,2,4}` 각각에 SSIM term을 합산: `(1 - ssim) × weight_rgb × (1/factor) × weight_ssim`. SSIM weight는 cfg.LOSSES.WEIGHT_SSIM (getattr fallback=0.6).

### 02:41:16 — upstream `ConvDecoder` 마이그레이션 (커밋 `7099f87`, resize→crop 완전 전환)

**Image transform 통일 (resize → crop 완전 전환)**
- `_muvo` pipeline에서 encoder input은 `VerticalCropToSize(320, 800)`, reconstruction target은 `transforms.Resize(216, 288)`를 쓰는 불일치 상태를 확인함. upstream MUVO는 encoder input과 reconstruction target이 동일 spatial 변환을 거치는 것이 원칙임을 확인함.
- raw transform의 `transforms.Resize`를 제거하고 `CenterCropToSize`로 통일함. 이로써 `transforms.Resize`가 `_muvo` 파이프라인에서 완전히 제거됨.
- `VerticalCropToSize` → `CenterCropToSize`로 변경하여 height뿐 아니라 width도 crop하도록 수정함 (좌우 16px씩 center crop).

**Image input size 정렬 (320×800 → 320×768)**
- `IMAGE_INPUT_SIZE`를 (320, 800) → (320, 768)로 변경함. 800은 `2^6=64`로 떨어지지 않아 decoder 6-doubling이 불가능하고 5-doubling만 가능했지만, `768 = 2^6 × 12`이라 6-doubling이 깔끔하게 떨어짐.
- `RGB_RECON_SIZE`를 (216, 288) → (320, 768)로 변경하여 **encoder input == decoder output** 원칙을 적용함. decoder 출력 이후 `F.interpolate(bilinear)` 보정 단계 제거 가능.

**Decoder 구조 재작성 (upstream `ConvDecoder` 패턴 이식)**
- 기존 `SensorDecoder`의 `head_1`/`head_2` feature 공유 + bilinear interpolate `(216, 288)` 보정 구조를 폐기.
- upstream `ConvDecoder`(`common.py:549-632`) 패턴으로 재작성: `pre_transpose_conv` + `trans_conv1/2/3` + 별도 multi-scale `head_4/2/1`. `_decoder_base_size`/`_auto_n_pre_doublings`로 target에 맞는 doubling 수 자동 결정 (RGB 6-doubling=n=3, LiDAR 5-doubling=n=2). channel `decoder_ch→/2→/4→/8`. kernel 초기 `k=5/s=2/p=2/output_padding=1`, 마지막 `k=6/s=2/p=2` (checkerboard 회피).
- `SensorHead`도 1×1 Conv만 남기고 `F.interpolate` 분기 제거.
- 검증: RGB `rgb_4(80,192)/rgb_2(160,384)/rgb_1(320,768)`, LiDAR `lidar_re_4(8,256)/lidar_re_2(16,512)/lidar_re_1(32,1024)` 모두 정상 출력. 모델 파라미터 77M → 112M.

### 03:05:25 — SSIM 가중치 정정 (커밋 `646775c`)

**SSIM weight를 upstream effective와 일치**
- 기존 `WEIGHT_SSIM` default 0.6은 upstream effective(`rgb_weight=0.1 × ssim_weight=0.6 = 0.06`)의 10배에 해당하여 over-weighting 됨을 확인함.
- 1000ep overfit에서 step 250 근처에서 total loss 폭발 + checkerboard artifact 노출 관찰.
- 원본 `config.py`의 `LOSSES`에 `WEIGHT_SSIM=0.06` 키 신규 추가, `trainer_ssim.py`의 `SSIMWorldModelTrainer.__init__` getattr fallback default를 0.6 → 0.06으로 수정.

### 03:05:39 — `SensorFeatureConv`에 residual 도입 (커밋 `7bf7df9`, 646775c의 14초 후)

**Encoder feature compression에 BasicBlock×2 적용**
- `_muvo`의 `SensorFeatureConv`가 plain `Conv-BN-ReLU × 2 + AdaptiveAvgPool + Flatten` 구조여서 gradient flow와 학습 안정성에 한계가 있음을 확인함.
- upstream `muvo/layers.py:BasicBlock`을 alpha 내부에 inline 이식 (Conv-BN-ReLU×2 + 1×1 shortcut).
- `SensorFeatureConv`의 conv block 부분을 `BasicBlock × 2`로 교체 (pooling/flatten 유지). 7099f87 이후 image width가 768로 정렬된 상태이므로 feature spatial은 `(10, 24)`이고, shape `(B, 8, 256, 10, 24) → (B, 8, 256)` 정상.
- 모델 파라미터 112M → 114M.

### 03:12:56 — RSSM `action_in_gru` 토글 도입 (커밋 `0be4871`)

**RSSM 디자인 차이를 명시적 옵션으로 노출**
- `_muvo`의 RSSM이 표준 Dreamer V1/V2/V3 패턴(action을 pre_gru에 concat)을 따르고 있어, upstream MUVO의 outlier 디자인(action은 prior μ/σ 생성에만 사용)과 차이남을 확인함.
- 단순화 누락이 아니라 의도된 개선으로 판단해서 default는 alpha 디자인을 유지하되, 향후 upstream 비교용 ablation을 위해 toggle 옵션을 추가함.
- `RSSM.__init__`에 `action_in_gru` 파라미터 추가 (default True = alpha 디자인):
  - True: `pre_gru_net([sample_t, latent_action_t])`, action이 GRU hidden state 진화에도 영향 (Dreamer)
  - False: `pre_gru_net(sample_t)`, action은 GRU 이후 prior 분포에만 영향 (upstream)
- `config_muvo.py`에 `MODEL.TRANSITION.ACTION_IN_GRU=True` cfg key 신규.

→ 새벽 marathon session 종료 (01:58–03:13, 약 1시간 15분).

## 5/15 (금) 오후 — `_muvo` 1D 후속 정비

이날 오후 commit은 13:51:59~20:10:57 사이 3건임.

### 13:51:59 — SSIM을 `trainer_muvo`로 정식 통합 + 2×2 ablation (커밋 `08336bc`)

**SSIM term을 `_muvo` 본 trainer로 통합**
- `trainer_ssim.py`로 검증한 SSIM term을 본 trainer로 통합할 필요가 있음을 확인함.
- `trainer_muvo.py`에 `SSIMLoss` 클래스를 inline 이식하고 `compute_loss`의 RGB 블록에 SSIM term 추가 (weight_ssim > 0일 때만 활성화).
- `config_muvo.py`에 `LOSSES.WEIGHT_SSIM=0.0`(default off) 추가.
- `train_muvo.py`에 `--h-dim`, `--z-dim`, `--weight-ssim` CLI flag 추가.
- 같은 commit에서 `train_muvo.py`의 `change_summary`도 `RSSM h{...}/z{...}`, `WEIGHT_SSIM={...}`를 cfg에서 읽어 기록하도록 1차 동적화함. 이후 `515a213`에서 image/LiDAR/embedding/fusion 값까지 추가로 동적화됨.

**1-window 2×2 ablation 실험 (dim × SSIM)**
- 1000ep overfit으로 4개 변형 학습:
  - h256/z128 SSIM=0.0: loss ~0.2 단조 수렴, KL ~step 300에 안정
  - h256/z128 SSIM=0.06: loss ~0.55 진동, KL ~step 600에 안정
  - h512/z256 SSIM=0.0: loss ~0.2 단조, KL ~step 300에 안정
  - h512/z256 SSIM=0.06: loss ~0.55 진동, KL ~step 600에 안정
- 결론: 4 변형 모두 동일 수렴점에 도달 → **1-window overfit 환경이 SSIM/dim 효과 검증 testbed로 부적절** 함을 확인함. SSIM은 instability만 추가, dim 차이 의미 없음 (h256 capacity가 1-window에는 이미 충분). 멀티-window 환경에서 재평가 필요.

### 14:28:17 — `change_summary` 동적화 (커밋 `515a213`)

**experiment_log.csv 갱신 자동화**
- `train_muvo.py`의 `change_summary` 문자열에 hardcoded "320x800" 등 stale 값이 남아 `7099f87`의 변경 후 갱신이 누락됐음을 확인함.
- `change_summary`를 cfg에서 동적으로 읽도록 수정: `IMAGE_INPUT_SIZE`/`LIDAR_RANGE_VIEW_SIZE`/`LIDAR_SCALE`/`EMBEDDING_DIM`/`FUSION.TRANSFORMER_LAYERS` 매번 cfg에서 로드.

### 20:10:57 — CSV `series` 컬럼 신설 (커밋 `33d3faa`)

**다중 trainer 실험의 분류 인프라**
- trainer가 5개(`trainer.py`, `trainer_muvo.py`, `trainer_ssim.py` 등)로 늘었으나 공통 `experiment_log.csv`에서 실험을 분류할 수단이 없어 36 row가 평면화된 상태임을 확인함.
- 19번째 컬럼 `series` 신설 + 36 row 전체를 9 카테고리로 분류.
- `scripts/model_variants/*.py` 범위에서는 `trainer.py`/`trainer_muvo.py` 2개 trainer의 `FIELDNAMES`에 `series` 추가, row dict에 `getattr(cfg.LOGGING, "SERIES", "")` 적용. commit message의 "5 trainer"는 `scripts/trainer11_*` 등 `scripts/model_variants` 밖 파일까지 포함한 범위로 해석해야 함.
- `config.py`/`config_muvo.py`에 `LOGGING.SERIES=""` 추가, `train_muvo.py`에 `--series` CLI flag.

→ 5/14 밤~5/15 prep 단계 (새벽 7건 + 오후 3건)에서 정해진 (320, 768) image / (32, 1024) LiDAR / `CenterCropToSize` / multi-scale head / `BasicBlock SensorFeatureConv` / `action_in_gru` toggle / `WEIGHT_SSIM` cfg / CSV `series` 인프라가 5/18 alpha-2D 도입 시 모두 그대로 상속됨.

---

## 5/18 (월) 19:35:14 — alpha-2D 전체 신규 구현 (커밋 `ca9f350`, 5개 파일 2,581줄)

### MUVO 기반 2D latent 구조 분석 및 코드 구현
- 논문의 핵심 발견 중 하나는, 2D latent space가 카메라 이미지 예측 및 공간 voxel occupancy 예측에 큰 이득을 준다는 것
- MUVO github는 1D latent 방식으로 구현되어 있었기에, 이를 기반으로 구현한 our 모델 또한 1D latent space 방식이었음
- README의 하이퍼링크를 타고 들어가, 2D latent state 모델을 구현해 놓은 repo 발견 (이하 MUVO_2D)
- MUVO_2D 코드와 현재 모델 구조를 비교하며 학습 구조 차이를 분석함.
- 기존 `_muvo` 5개 파일은 baseline 보존을 위해 그대로 두고, `_muvo_2D` 접미사 5개 신규 파일로 분기함.
- 2D latent 구조를 자체 제작 모델에 반영하기 위해 MUVO_2D의 Convolution 기반 GRU 모델인 `ConvGRUCellGlo` 구조를 분석하고 구현함.
- MUVO와 최대한 유사한 학습 환경 및 성능 도출을 위해 MUVO 논문 evaluation의 training setup 부분을 확인함.
  - 논문은 0.2초 간격으로 샘플링하여 길이 12의 시퀀스를 학습 입력으로 사용함을 확인함.
  - voxel reconstruction을 포함하는 실험에서는 학습 속도를 위해 시퀀스 길이를 6으로 축소함을 확인함.
  - 자체 제작 모델은 upstream MUVO `muvo.yml`의 `RECEPTIVE_FIELD=4, FUTURE_HORIZON=2`(seq_len=6)와 일치시켜 설정함.

### 자체 제작 모델의 Action 및 Speed 반영 코드 수정
- RSSMTD(RSSM Transformer Decoder)의 구조를 분석한 결과, action이 GRU에 직접적으로 입력되고 있지 않은 것과 달리, our 모델에서는 action이 GRU에 직접적으로 입력되고 있음을 확인함.
- MUVO_2D 코드 구현과 동일하게 prior/posterior의 mu, sigma를 만들 때만 action을 사용하도록 코드를 구현함. posterior 쪽은 `observe_step`에서 `posterior_action_module(action_t)`를 token으로 추가하고, prior 쪽은 `imagine_step`에서 `prior_action_module(action_t)`를 token으로 추가함.
- 기존 자체 제작 1D 계열에서는 `SensorFusionTransformer.forward`가 image/LiDAR feature만 받고, fusion 이후 `cam_emb`/`lidar_emb`/16-d speed embedding을 `features_combine`에서 concat하는 구조였음을 확인함.
- MUVO_2D의 fusion 구조에 맞춰 type embedding을 image/LiDAR/policy/action/speed의 5 slot으로 확장하고, `SensorFusionTransformer` forward에 `speed_embed` 인자를 추가하여 fusion 내부에서 speed token을 다룰 수 있도록 수정함. ca9f350 시점 구현은 speed token을 cam/lidar token에 broadcast-add하는 방식이며, 별도 token concat 정렬은 working tree에서 추가 진행.

### 자체 제작 모델의 Encoder 코드 수정
- 기존 자체 제작 모델의 encoder는 fusion 이후 `SensorFeatureConv`(BasicBlock×2 + AdaptiveAvgPool + Flatten)와 `features_combine`(Linear)을 통해 image/LiDAR feature를 프레임당 1D vector로 압축하고 있음을 확인함.
- 해당 압축 단계에서 spatial 정보가 사라져 2D latent 구조와 양립할 수 없음을 확인하고, `SensorFeatureConv`와 `features_combine`을 제거하여 token-shape `(B, S, C, N_tokens)`를 RSSMTD까지 그대로 전달하도록 수정함.
- 이에 따라 `encode_fuse_sequence`의 반환 shape을 기존 `(B, S, C)`에서 `(B, S, C, N_tokens)`로 변경함.
- MUVO_2D의 LiDAR encoder가 ResNet18 `out_indices=[1, 2, 3]`(stride-16 종료)로 설정되어 있음을 확인함. 기존 자체 제작 모델은 image/LiDAR 모두 `[2, 3, 4]`(stride-32 종료)였음.
- MUVO_2D와 동일하게 LiDAR encoder만 `out_indices=[1, 2, 3]`으로 변경하여 stride-16 종료로 수정함. LiDAR token spatial이 `(1, 32)=32` → `(2, 64)=128`로 확장됨. fusion encoder 입력 token은 image 240 + LiDAR 128 = 368개이며, `RepresentationModelTD` query/output 쪽에서는 policy 1개를 더해 image 240 + LiDAR 128 + policy 1 = 369개 구조가 됨.
- 기존 자체 제작 모델의 SpeedEncoder는 학습 코드 내부에 inline `nn.Sequential`로 정의되어 있고 output channel이 16으로 transformer embedding channel(256)과 일치하지 않아 fusion token으로 사용할 수 없었음을 확인함.
- MUVO_2D와 동일하게 `SpeedEncoder`를 독립 클래스로 분리하고 output channel을 256으로 정렬하여 fusion 단계에서 image/LiDAR token과 동일한 차원으로 활용 가능하도록 구현함.

### RepresentationModel 및 Decoder 코드 수정
- MUVO 논문은 prior/posterior mu, sigma 생성 모듈을 transformer decoder 기반 구조로 사용하는 것을 확인함.
- 이를 반영하여 기존 mu, sigma 생성 모듈인 RepresentationModel class(`Linear → LeakyReLU → Linear` MLP)를 transformer decoder 구조 `RepresentationModelTD`로 수정함.
- 각 modality(image/LiDAR/policy)별로 학습 가능한 query embedding을 생성하고, query마다 type embedding을 추가한 뒤 concatenate하여 transformer decoder의 query 입력으로 사용함. key/value는 fusion된 observation token임.
- MUVO 논문의 2D latent 구조를 유지하기 위해 decoder 입력 및 출력을 `(B, S, C, H, W)` token shape으로 유지하도록 `TokenConvDecoder2D`를 구현함. Linear+Unflatten 단계 없이 token state를 직접 ConvTranspose ladder에 투입.
- 현재 모델의 task는 RGB 및 LiDAR observation을 기반으로 latent state를 학습하고 future observation을 예측하는 것이므로 action을 별도의 output으로 예측하는 head는 불필요하다고 판단함. 다만 ca9f350 단계에서는 upstream `PolicyDecoder`를 그대로 포함하고 `WEIGHT_ACTION=1.0`로 활성화된 상태로 도입함 — 추후 별도 작업으로 제거 예정.

### 자체 제작 모델의 차원 및 hyperparameter 정렬
- RSSMTD가 학습 안정성을 위해 `embedding_dim == hidden_state_dim == state_dim` 제약을 요구함을 확인함.
- 기존 `HIDDEN_STATE_DIM=512, STATE_DIM=256` 설정을 모두 256으로 정렬. 학습 entrypoint에는 `--h-dim`/`--z-dim` override를 추가하고, 두 값을 모두 지정했는데 서로 다를 때만 warning 후 z-dim으로 강제하는 conflict resolution을 넣음. 단 `--h-dim`만 단독 지정하면 `STATE_DIM`은 그대로 남을 수 있어 완전한 보호 로직은 아님.
- 기존 `SensorFusionTransformer`가 `nhead=8`(d_model=256 → 32 dim/head)이었으나 upstream 컨벤션이 64 dim/head임을 확인하고 `TRANSFORMER_HEADS=4`로 변경함.

### 자체 제작 모델의 손실 함수 수정
- 기존 자체 제작 모델의 LiDAR loss는 전체 point cloud에 대한 global Chamfer만 사용함을 확인함.
- 자율주행에서는 자기 차량 근방의 정확도가 더 중요함을 고려하여 `[x: -20~20, y: -20~20, z: -2~6]` 박스 내 point만 masking하여 Chamfer를 계산하는 near-field Chamfer metric을 추가함. cfg key `NEARFIELD_PC_RANGE`로 box 범위 조정 가능하도록 구현함.
- 기존 KL free-bits가 항상 적용되어 있어 upstream과의 controlled comparison이 불가능함을 확인하고 `KL_FREE_BITS_ENABLED` cfg flag를 추가함. 처음 도입 시 default는 OFF — 같은 날 21:34에 True로 복원, 아래 참조.

### 자체 제작 모델의 데이터 파이프라인 및 학습 인프라
- `data_muvo_2D.py`를 `data_muvo.py` 기반으로 신규 생성하고, import를 `config_muvo_2D`로 전환함.
- `data_muvo_2D.py`에는 upstream `muvo_2d`의 preprocess 구현을 참고한 로컬 `PixelAugmentation`을 추가함. Blur / Sharpen / ColorJitter를 `cfg.DATA.AUGMENTATION.ENABLED` 단일 flag로 gate하고, encoder input transform에만 적용하며 reconstruction target(`image_raw`)에는 augmentation과 normalisation을 적용하지 않도록 구성함.
- `MUVODataset` / `StreamWindowDataset` / `MultiArrowStreamDataset` 구조와 dataset 반환 key/shape는 기존 `_muvo` 계열과 동일하게 유지함. 반환 dict도 `image`, `image_raw`, `lidar`, `action`, `speed`, `run_id`, `start_row` 구조를 유지함.
- 데이터 샘플링이 `SAMPLE_EVERY_N`을 직접 지정해야 했던 구조를, CARLA 20Hz → Arrow 저장 `FRAME_STEP=5` → 학습 시 `EFFECTIVE_HZ` 명시 → `SAMPLE_EVERY_N` 자동 도출 방식으로 변경함. CLI `--effective-hz N` override 가능.
- Augmentation 설정이 학습 코드에 hardcode되어 있어 toggle 불가능함을 확인하고 `cfg.DATA.AUGMENTATION` namespace 추가 + CLI `--no-augmentation` 제공. blur/sharpen/colorjitter 항목을 cfg로 관리.
- 기존 TensorBoard만 사용하여 hyperparameter/git commit/artifact 통합 관리가 불가능함을 확인하고 ClearML `Task.init`을 학습 entry point에 추가함. cfg에 `CML_ENABLED`/`CML_PROJECT`/`CML_TASK`/`CML_TYPE`/`CML_TAGS` 추가, CLI `--no-clearml`/`--cml-project`/`--cml-task` 제공.
- `trainer_muvo_2D.py`의 `test_step`에 ClearML prediction artifact dump를 추가함. `Task.current_task()`가 존재하고 batch_idx < 4일 때 `rgb_pred`/`lidar_pred`/`action_pred`를 `task.upload_artifact()`로 업로드함.
- reconstruction figure 기본 동작도 2D용으로 확장함. `LOGGING.RECON_FIG_ALL_FRAMES=True`, `LOGGING.RECON_FIG_BOTH_VIEWS=True` cfg를 추가하고, `save_reconstruction_figure`가 6 frame 전체 column 및 LiDAR scaled/unscaled 두 view를 표시하도록 구현함.

---

## 5/18 (월) 20:12:44 — Periodic reconstruction figure callback (커밋 `431d588`)

### 자체 제작 2D 모델의 학습 도중 시각화 자동화
- overfit 검증 시 loss curve만으로는 reconstruction 품질 변화 추세를 즉시 파악하기 어려움을 확인함.
- 매 N epoch마다 한 sample을 `model.eval()`로 decode하여 figure를 PNG로 dump하는 `PeriodicReconstructionCallback` 추가. 저장 직후 train mode 복원하여 학습 흐름이 끊기지 않도록 함.
- CLI `--recon-every-n-epochs N`, `--recon-sample-idx K`. 실제 저장 파일명은 `save_reconstruction_figure`의 suffix까지 포함되어 `<run_name>_e0001_reconstruction_s0.png` 형태가 됨.

---

## 5/18 (월) 20:55:54 — RGB plateau 진단 및 decoder capacity 보강 (커밋 `813882c`)

### 자체 제작 2D 모델의 RGB reconstruction plateau fix
- `431d588`의 callback으로 1000ep overfit 결과를 추적한 결과, alpha-2D의 RGB reconstruction loss가 epoch 500 이후 약 0.105에서 plateau에 빠지는 현상을 확인함. 동일 setup의 alpha-1D baseline은 약 0.01까지 떨어졌음.
- 진단 결과 KL collapse는 아니며 (KL ~4.5 nats 정상), `TokenConvDecoder2D`의 capacity가 upstream `ConvDecoder` 대비 절반이라 RGB 정보를 충분히 표현하지 못함을 확인함.
- `cfg.MODEL.RSSM_2D.DECODER_CHANNELS=512` cfg key를 신규 추가하고, decoder 첫 stage가 채널을 즉시 256으로 축소하지 않고 512로 보존하도록 수정함. trans_conv 채널 `512→256→128→64`로 정렬, head input channel을 2배로 확대.
- embedding_dim=256은 그대로 유지하여 RSSM 메모리가 4배 증가하지 않도록 함.

---

## 5/18 (월) 21:34:10 — KL free-bits 정책 복원 (커밋 `8458af7`)

### KL_FREE_BITS_ENABLED default 복원
- `ca9f350` 도입 시점에 "upstream과 일치" 명목으로 default OFF로 설정했으나, alpha-1D `_muvo` baseline이 항상 ON으로 학습되어 controlled comparison이 불가능함을 확인함.
- alpha-2D vs alpha-1D controlled comparison을 위해 default를 True로 복원함.
- upstream MUVO는 free-bits 없이 학습되지만(KL weight 1e-3 + 큰 모델), alpha는 KL weight 1e-2 + free-bits 1.0 조합으로 posterior collapse를 막아왔으므로 free-bits도 동일하게 켜는 것이 일관적임.

---

## 5/19 (화) 00:09:36 — 1D baseline에 동일 figure callback 포팅 (커밋 `ce77b48`)

### 1D/2D apples-to-apples 비교 인프라
- `431d588`에서 alpha-2D에 추가한 `PeriodicReconstructionCallback` 패턴을 alpha-1D `train_muvo.py`/`trainer_muvo.py`에도 이식하여 1D/2D 비교가 가능하도록 수정함.
- 1D는 4-column 2-row 단순 figure, 2D는 6-frame full + scaled/unscaled view로 출력 형태는 약간 다름.

---

## 5/19~5/20 — Working tree 정비 (미커밋, 소스 최종 편집 5/20 23:05)

### 학습 단계 imagine supervise 제거 (paper / upstream 셋업 일치)
- 기존 자체 제작 모델은 학습 단계에서 receptive field(posterior) + future horizon(imagine) 양쪽을 모두 reconstruction loss로 supervise하고 있음을 확인함. MUVO 논문과 upstream MUVO는 학습 시 전체 시퀀스를 known data로 처리하고 imagine은 검증 단계에서만 평가함.
- `_observe_full` 메서드를 신규 추가하고 `training_step`의 호출을 `_observe_and_imagine` → `_observe_full`로 변경하여 학습 단계의 imagine 출력 loss 계산을 제거함.
- 이에 따라 `WEIGHT_FUTURE=1.0` cfg key 및 `self.weight_future` 로딩을 주석 처리함.

### Multi-sample evaluation 도입
- prior 분포에서 1회만 sampling할 경우 imagine 결과의 variance가 크게 나타나 검증 metric이 epoch 간 흔들리는 현상을 확인함.
- `_observe_and_imagine`에 `n_samples` parameter 추가, `validation_step`/`test_step`에서 `getattr(cfg.PREDICTION, "N_SAMPLES", 1)` fallback으로 n_samples를 읽어 전달하고, imagine을 반복 호출한 뒤 loss를 평균하도록 수정.
- 단, 현재 working tree의 `config_muvo_2D.py`에는 아직 `PREDICTION` namespace / `N_SAMPLES` cfg key가 정의되어 있지 않음. 따라서 현재 기본 동작은 fallback 값 `1`이며, cfg 추가 또는 CLI override가 별도로 필요함.
- 시각화 및 하위 호환을 위해 첫 sample만 figure로 반환.
- `validation_step`의 dataset_name 결정 로직을 `_dataset_name()` 호출에서 `"RL"/"DS"` 직접 매핑으로 단순화하고, 반환값도 `{f"val_{dataset_name}_loss": total_loss}` dict에서 bare `total_loss` tensor로 변경함.

### Speed token concat 패턴 정렬
- `ca9f350` 시점에는 `SensorFusionTransformer` 내부에서 `speed_token`을 cam/lidar token 각각에 element-wise add (broadcast-add) 하는 패턴이었음을 확인함.
- upstream MUVO 쪽 speed token 처리와 맞추기 위해 alpha-2D도 `tokens = [cam_tokens, lidar_tokens]` → `tokens.append(speed_token)` → `torch.cat(tokens, dim=0)`로 수정함.
- 같은 함수에서 lidar_out slice를 `out[L_cam:]`에서 `out[L_cam:L_cam + L_lidar]`로 정확화하여 speed token이 fusion 출력 뒤에 붙을 때 lidar slice가 오염되지 않도록 fix함. `L_lidar = Hl * Wl` 변수도 명시 추가.
- 단, 현재 `models_muvo_2D.py`에는 아직 stale speed 표현이 여러 곳 남아 있음: 상단 docstring의 type embedding modulation, `SensorFusionTransformer.forward` docstring의 "speed modulation", concat 직전의 "speed broadcast add"/"element-wise add", model init 주석의 "modulation, not separate token", `encode_fuse_sequence` docstring/comment의 "broadcast add"/"for modulation" 등. 구현은 이미 separate token concat이므로 커밋 전 주석 정정 필요.

### Optimizer weight decay grouping 도입
- `trainer_muvo_2D.py`의 `configure_optimizers()`에서 단일 `AdamW(self.parameters(), weight_decay=...)` 호출을 no_decay/decay parameter group 방식으로 변경함.
- BN/LayerNorm scale/bias 및 1D parameter에는 weight decay를 적용하지 않고, 나머지 weight에만 cfg `OPTIMIZER.WEIGHT_DECAY` 값을 적용하도록 `add_weight_decay()` helper를 추가함.
- `add_weight_decay()` 호출 시 `skip_list=("relative_position_bias_table",)`도 함께 전달하여, 해당 이름의 parameter는 weight decay 제외 group으로 들어가도록 구성함.
- 현재 `AdamW(parameters, lr=self.lr, weight_decay=0.01)`로 호출되지만 parameter group이 각 group의 `weight_decay`를 들고 있으므로 실질 weight decay는 group 설정을 따름. 다만 hardcoded `0.01` 인자는 혼동 여지가 있어 정리 필요.

### 기타 부수 정비
- `WEIGHT_LIDAR_RE`를 0.0 → 0.1로 조정하여 LiDAR reconstruction loss 활성화.
- 디버그/속도용으로 train dataset을 `reduction_factor=4`로 축소하는 임시 로직을 `train_muvo_2D.py`에 추가.
- TensorBoard figure의 scalar tag 탐색 목록을 `train_action` → `train_action_step`으로 수정함. 실제 `self.log(f"train_{name}", ...)` 호출을 rename한 것은 아니고, Lightning on_step 자동 suffix와 figure 매칭을 맞춘 fix.
- `trainer_muvo_2D.py` 모듈 docstring에 "upload" → "uploasd" typo가 우발적으로 들어감 (커밋 전에 복원 필요).
- 현재 working tree에는 `scripts/model_variants` 기준 trailing whitespace도 남아 있음: `train_muvo_2D.py:243`, `trainer_muvo_2D.py:621`, `trainer_muvo_2D.py:627` (`git diff --check -- ':(glob)scripts/model_variants/*.py'` 기준). 기능 변경은 아니지만 커밋 전 정리 필요.

---

## 검증 메타데이터

**재검산된 항목 (commit message / diff 확인 범위 구분)**:
- 5/14~5/19 commit timestamp는 git 원본 (KST)
- diff로 직접 확인 가능한 수치: SSIM weight 0.6→0.06, 2,581 lines, 5/14 `data.py`/`train.py` 변경, `scripts/model_variants` 범위의 `series` 적용 대상 2 trainer, fusion 입력 token 368 / RSSMTD query token 369
- commit message 및 실험 기록 기반 수치: 77M → 112M → 114M param 수, 0.105 plateau, KL ~4.5 nats, experiment_log row 수, SSIM/dim 2×2 ablation 수렴 관찰
- working tree는 현재 `git diff HEAD -- scripts/model_variants` 기준 7 path 변경: `.py` 4개(`config_muvo_2D.py`, `models_muvo_2D.py`, `train_muvo_2D.py`, `trainer_muvo_2D.py`) + `.pyc` 3개
- commit history 기준 5/16, 5/17, 5/21, 5/22, 5/23 에는 `scripts/model_variants/*.py` 변경 없음
- 5/20에는 `scripts/model_variants/*.py`에 대한 새 commit은 없지만, working tree source 4개(`config_muvo_2D.py`, `models_muvo_2D.py`, `train_muvo_2D.py`, `trainer_muvo_2D.py`)의 mtime이 2026-05-20 19:37~23:05 KST이므로 현재 미커밋 코드 변경은 5/20 작업분으로 보는 것이 맞음
- 참고로 `.pyc`까지 포함하면 `trainer_muvo_2D.cpython-38.pyc` mtime이 2026-05-20 23:14:54 KST로 더 늦지만, 본문 제목의 "최종 편집"은 source `.py` 기준임

**환각 오류 정정 내역 (이전 초안 → 최종)**:
- "5/14에는 `scripts/model_variants/*.py` 변경 없음" → `87bd950`에서 `data.py`/`train.py` 변경이 있었으므로 5/14 섹션 신설
- 5/15 새벽 commit 7건의 시각을 오후로 잘못 표기 → 정확한 새벽 timestamp(01:58–03:13)로 정정
- `0ad74bf`를 `_muvo` 변경으로 잘못 분류 → 원본 alpha 파일(`config.py`/`trainer.py`/`models.py`) 수정으로 정정
- `0ad74bf` ↔ `07b5aa2` 순서 잘못 → 실제 commit 순서(0ad74bf → 07b5aa2) 반영
- `646775c` ↔ `7bf7df9` 순서: 14초 차이로 두 commit 가까이 있음 명시
- "5개 trainer에 series 적용" → `scripts/model_variants/*.py` 범위에서는 `trainer.py`/`trainer_muvo.py` 2개만 해당함을 명시
- `cfg.PREDICTION.N_SAMPLES` cfg가 존재하는 것처럼 쓴 표현 → 현재는 fallback read만 있고 cfg key는 미정의임을 명시
- "working tree 10건" → 현재 `scripts/model_variants` 기준 7 path, source `.py` 기준 4 file로 정정
- speed token concat 구현과 stale broadcast-add 주석의 불일치 명시
- optimizer weight decay grouping 및 validation_step return 변경 누락 추가
- 5/15 요일 오기(목 → 금), `a89c9de`의 "import만 교체" 과소기록, `08336bc`의 `change_summary` 1차 동적화 누락 정정
- `ca9f350`의 기존 speed 처리 구조 설명 오류, ClearML prediction artifact upload 누락, 2D reconstruction figure cfg/동작 누락 정정
- `431d588` periodic reconstruction 파일명 패턴을 실제 `<run_name>_e0001_reconstruction_s0.png` 형태로 정정
- working tree trailing whitespace 3건을 커밋 전 정리 필요 항목으로 추가
- `models_muvo_2D.py:446-454` 라인만으로 prior/posterior action 사용을 모두 가리키던 line reference 제거, `train_action_step`을 실제 logging rename처럼 표현한 문구 정정, 5/21~5/22 변경 없음 명시 추가
- `7bf7df9`의 `SensorFeatureConv` shape에서 stale width token 25를 7099f87 이후 실제 width token 24로 정정
