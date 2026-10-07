# alpha-1D -> alpha-2D 작업 보고서 (5/14 ~ 5/20, 2026-05-23 검증 기준)

## 검증 기준
- 5/14~5/20 기간의 `scripts/model_variants/*.py` 코드 변경을 git commit, diff, current working tree 기준으로 교차검증하여 정리함.
- 커밋 시각은 git timestamp(KST)를 기준으로 유지함.
- diff로 직접 확인 가능한 수치와 commit message/실험 기록 기반 수치는 구분하여 작성함.
- 대표적으로 SSIM weight 0.6->0.06, alpha-2D 5개 파일 2,581줄, 5/14 `data.py`/`train.py` 변경, `series` 적용 대상 2 trainer, fusion 입력 token 368/RSSMTD query token 369는 diff 기반 기록으로 정리함.
- parameter 수 77M->112M->114M, RGB plateau 0.105, KL 약 4.5 nats, experiment log row 수, SSIM/dim 2x2 ablation 수렴 관찰은 commit message 및 실험 기록 기반 기록으로 정리함.
- 5/16, 5/17, 5/21, 5/22, 5/23에는 commit history 기준 `scripts/model_variants/*.py` 변경이 없음을 확인함.
- 5/20 작업은 `scripts/model_variants/*.py`에 대한 새 커밋은 없지만, 현재 미커밋 working tree의 `_muvo_2D` 소스 4개 파일 수정시각이 2026-05-20 19:37~23:05 KST 범위에 있어 5/20 작업분으로 정리함.
- 현재 working tree 변경은 `scripts/model_variants` 기준 소스 `.py` 4개와 `.pyc` 3개이며, 본문은 소스 `.py` 4개 변경을 중심으로 작성함.

## 5/14 (목) 17:53:39
- 주요 커밋은 `87bd950`임.

### 원본 alpha one-window overfit 및 resume 코드 정비
- 기존 원본 alpha 학습 코드에 임시로 걸려 있던 "8프레임만 받기" 제한이 남아 있음을 확인함.
- 전체 Arrow table을 정상적으로 학습에 사용하기 위해 `data.py`의 dataset length 계산을 `self.table.num_rows` 기준으로 복원함.
- 이 시점의 원본 alpha image transform은 encoder input과 reconstruction target 모두 `transforms.Resize` 기반이었고, crop 기반 transform은 아직 도입되지 않은 상태였음.
- 원본 alpha 설정은 `IMAGE_INPUT_SIZE=(300, 400)`, `RGB_RECON_SIZE=(216, 288)`였으므로, 두 transform 모두 resize였지만 해상도는 서로 달랐음.
- 원본 alpha 모델을 한 window에 대해 정확히 overfit시키며 reconstruction 품질을 확인할 필요가 있다고 판단함.
- 이를 위해 `train.py`에 `--one-window-overfit` flag를 추가함.
- 해당 flag를 켜면 전체 train run 사용을 끄고, overfit sample 수와 batch size를 모두 1로 강제하도록 구현함.

- 장시간 학습 중 checkpoint에서 이어 학습할 때 OneCycleLR scheduler의 total step 수가 기존 checkpoint와 맞지 않는 문제가 생길 수 있음을 확인함.
- 이를 해결하기 위해 `--resume-from-checkpoint` flag와 `maybe_extend_onecycle_resume_checkpoint()` 함수를 추가함.
- OneCycleLR checkpoint를 더 긴 epoch로 이어 학습할 수 있도록 scheduler의 `total_steps`와 `_schedule_phases`를 새 총 step 수에 맞게 patch한 임시 checkpoint를 생성하도록 구현함.
- `trainer.fit()` 호출에는 `ckpt_path=resume_ckpt_path`를 전달하여 validation 사용 여부와 상관없이 resume 경로가 반영되도록 수정함.

## 5/14 (목) 밤 ~ 5/15 (금) 03:12:56
- 실제 작업 흐름상 5/14 밤 작업으로 정리함. 단, 주요 커밋은 KST 기준 5/15 01:58:07~03:12:56에 남아 있음.
- 주요 커밋은 `a89c9de`, `0ad74bf`, `07b5aa2`, `7099f87`, `646775c`, `7bf7df9`, `0be4871`임.
- `0ad74bf`는 `_muvo` 스캐폴딩 직후 12초 후에 들어간 원본 alpha baseline 기록이고, `7bf7df9`는 SSIM weight 정정 직후 14초 후에 들어간 residual block 정리임.

### MUVO 스타일 1D baseline 분기 생성
- 기존 자체 제작 모델은 image transform이 `transforms.Resize` 기반이라 종횡비 왜곡이 발생할 수 있고, decoder/head 구조도 upstream MUVO와 차이가 있음을 확인함.
- 기존 alpha baseline을 보존한 채 MUVO 스타일 구조를 따로 실험하기 위해 `_muvo` 접미사 5개 파일을 새로 생성함.
- `config_muvo.py`에는 image size `(320, 800)`, LiDAR range-view size `(32, 1024)`, embedding dim 256, RSSM hidden/state dim 512/256, RGB reconstruction 중심 loss weight를 정의함.
- `data_muvo.py`에는 height crop 기반 image transform, `LIDAR_SCALE=40.0`, Arrow dataset pipeline을 구성함.
- `models_muvo.py`에는 `FPNDecoder`, `SensorFeatureConv`, `RSSM`, `SensorDecoder`, `SensorHead`, `Model` 구조를 구성하고 BEV/voxel 관련 출력은 제거함.
- encoder feature aggregation에 쓰는 `FPNDecoder`도 MUVO-style로 수정함.
- 기존 alpha `FPNDecoder`는 가장 작은 feature map `xs[2]`에서 시작해 `F.interpolate`로 위로 올리는 top-down 구조라 upstream 1D MUVO의 `Decoder`(TRANSFORMER.LARGE=True branch)에 가까웠음.
- upstream `muvo.yml`의 기본값은 `TRANSFORMER.LARGE=False`이고, 이 경우 `mile.py`는 `DecoderDS`를 사용함.
- 이에 맞춰 `_muvo`의 `FPNDecoder`를 `xs[0]`에서 시작해 `adaptive_max_pool2d`로 downsample하며 skip을 더하는 DecoderDS 계열의 bottom-up 구조로 변경함.
- 기존 alpha는 `IMAGE_TOKEN_DOWNSAMPLE=(20, 20)`, `LIDAR_TOKEN_DOWNSAMPLE=(4, 16)`을 사용하여 FPN 이후 `F.adaptive_avg_pool2d`로 token grid를 명시적으로 맞추고 있었음.
- `_muvo`에서는 MUVO-style FPN output 해상도 자체를 transformer token grid로 사용하도록 해당 pooling 단계를 제거하고, fusion cfg도 `TRANSFORMER_CHANNELS/LAYERS/HEADS/DROPOUT` 중심으로 단순화함.
- `train_muvo.py`와 `trainer_muvo.py`는 기존 alpha 학습 코드를 기반으로 `_muvo` 파일들을 import하도록 분기함.
- `train_muvo.py`의 `change_summary`는 MUVO-style alpha 설명으로 교체하고, `WorldModelTrainer(..., embedding_n_channels=128)` 명시를 제거하여 cfg embedding default를 사용하도록 수정함.
- `config_muvo.py`는 처음부터 RGB reconstruction 중심 baseline으로 들어와 `WEIGHT_LIDAR_RE=0.0`, `WEIGHT_LIDAR_EMPTY=0.0`, `WEIGHT_RGB=1.0`으로 설정함.
- 이 과정에서 5/14에 추가한 `--one-window-overfit`, `--resume-from-checkpoint`, OneCycleLR resume patch 흐름도 `train_muvo.py`에 함께 반영함.
- `trainer_muvo.py`는 full-resolution head인 `rgb_1`, `lidar_reconstruction_1` 기준으로 reconstruction figure를 그리도록 수정함.
- LiDAR depth visualization과 Chamfer/Euclidean/range MAE metric도 `LIDAR_SCALE`을 반영하여 meter scale로 환산하도록 정리함.
- `embedding_n_channels` default도 고정 128이 아니라 cfg `MODEL.EMBEDDING_DIM` 기반으로 바꾸고, CSV `base_file` fallback을 `train_muvo.py`로 정리함.
- 즉 5/14 원본 alpha에는 crop이 없었고, 5/15 `_muvo` 분기 생성 시점에 encoder input에는 `VerticalCropToSize(320, 800)`가 새로 들어갔으나 reconstruction target은 여전히 `Resize(216, 288)`로 남아 있었음.
- 따라서 `_muvo` 초기 구현에서는 transform 종류와 output 해상도 양쪽에서 encoder input과 reconstruction target이 아직 일치하지 않는 상태였음.

### 원본 alpha RGB-only one-window baseline 기록
- `_muvo` 분기 생성 직후, SSIM 도입 전 원본 alpha의 control 상태를 남겨 둘 필요가 있다고 판단함.
- `_muvo` 파일은 건드리지 않고 원본 alpha의 `config.py`, `trainer.py`, `models.py`만 수정함.
- LiDAR loss 영향을 배제하기 위해 `WEIGHT_LIDAR_RE`와 `WEIGHT_LIDAR_EMPTY`를 0으로 두고 RGB-only 모드로 실험할 수 있도록 설정함.
- reconstruction figure에서는 stride-2 head가 아니라 full-resolution head인 `rgb_1`, `lidar_re_1`을 표시하도록 수정함.
- `SensorDecoder` 레이어 정의 순서는 forward 흐름과 맞도록 정리함.
- 이때 h512/z256, h256/z128 RGB-only 1000 epoch overfit 및 `muvo_one_window_overfit_1000` 결과를 experiment log와 figure로 기록함.
- 확인된 기록 기준으로 experiment log에는 5개 row가 추가되고, `results/figures`에는 PNG 7개가 추가됨.

### SSIM loss 실험 분기 추가
- alpha reconstruction이 흐릿하게 보이는 원인 중 하나가 SSIM term 부재일 수 있다고 판단함.
- `_muvo` 본 trainer를 바로 변경하지 않고, SSIM 효과만 따로 검증하기 위해 별도 trainer 분기를 추가함.
- `train_ssim.py`는 기존 `train.py`를 import한 뒤 `WorldModelTrainer`를 `SSIMWorldModelTrainer`로 monkey-patch하는 방식으로 구성함.
- `trainer_ssim.py`에는 upstream `muvo/losses.py:292-348`의 Gaussian window conv2d 기반 `SSIMLoss`를 inline으로 이식함.
- `SSIMWorldModelTrainer`는 기존 trainer를 상속하고, `compute_loss`에서 multi-scale `rgb_1`, `rgb_2`, `rgb_4` 각각에 `(1 - ssim) x weight_rgb x (1/factor) x weight_ssim` term을 더하도록 구현함.

### Image transform 및 Decoder 구조를 upstream MUVO 방식으로 정렬
- `_muvo` pipeline에서 encoder input은 `VerticalCropToSize(320, 800)`, reconstruction target은 `Resize(216, 288)`를 쓰고 있어 두 경로가 일치하지 않음을 확인함.
- upstream MUVO와 동일하게 input과 reconstruction target이 같은 spatial transform을 거치도록 `transforms.Resize`를 제거하고 `CenterCropToSize`로 통일함.
- 기존 `VerticalCropToSize`는 height만 crop했으나, width도 center crop하도록 바꾸어 좌우 16px씩 crop되도록 하고 image input을 `(320, 768)`로 정렬함.
- 기존 `(320, 800)`은 width 800이 `2^6=64`로 깔끔하게 떨어지지 않아 decoder 6-doubling이 어렵고 5-doubling만 가능했음.
- 반면 `(320, 768)`은 `768 = 2^6 x 12` 형태라 6-doubling 구조에 맞는다고 판단함.
- `RGB_RECON_SIZE`도 `(320, 768)`로 변경하여 encoder input과 decoder output이 같은 해상도를 갖도록 수정함.

- 기존 `SensorDecoder`는 head feature를 공유하고 마지막에 bilinear interpolate로 보정하는 구조였음.
- upstream MUVO의 `ConvDecoder` 패턴을 참고하여 `pre_transpose_conv`, `trans_conv1/2/3`, multi-scale `head_4/2/1` 구조로 decoder를 재작성함.
- RGB는 6단계 doubling, LiDAR는 5단계 doubling이 되도록 `_decoder_base_size`와 `_auto_n_pre_doublings` 로직을 추가함.
- transposed conv는 초기 `k=5/s=2/p=2/output_padding=1`, 마지막 `k=6/s=2/p=2` 패턴을 사용하여 checkerboard artifact를 줄이는 방향으로 정렬함.
- `SensorHead`는 1x1 conv만 남기고 interpolate 분기는 제거함.
- 결과적으로 RGB output은 `80x192`, `160x384`, `320x768` multi-scale로, LiDAR output은 `8x256`, `16x512`, `32x1024` multi-scale로 맞춰짐.
- 이 변경으로 모델 파라미터 수는 확인된 기록 기준 77M에서 112M으로 증가함.

### SSIM weight 정정
- 초기 `WEIGHT_SSIM=0.6` 설정은 upstream effective weight보다 10배 강하게 들어가는 값임을 확인함.
- 1000 epoch one-window overfit에서 step 250 근처 loss 폭발과 checkerboard artifact가 나타나는 것도 확인함.
- 이에 따라 원본 `config.py`에 `WEIGHT_SSIM=0.06`을 추가하고, `trainer_ssim.py`의 fallback default도 `0.6`에서 `0.06`으로 수정함.

### SensorFeatureConv residual block 도입
- `_muvo`의 `SensorFeatureConv`가 plain `Conv-BN-ReLU x 2` 구조라 gradient flow와 학습 안정성 측면에서 한계가 있다고 판단함.
- upstream MUVO의 `BasicBlock`을 alpha 내부에 inline으로 이식함.
- `SensorFeatureConv`의 conv block을 `BasicBlock x 2`로 교체하고, pooling과 flatten 흐름은 유지함.
- image width가 768로 정렬된 이후 feature spatial은 `(10, 24)`가 되며, `(B, 8, 256, 10, 24)`에서 `(B, 8, 256)`으로 정상 압축되도록 구성함.
- 이 변경 이후 모델 파라미터 수는 확인된 기록 기준 112M에서 114M으로 증가함.

### RSSM action_in_gru option 추가
- `_muvo`의 RSSM은 Dreamer 계열처럼 action을 GRU 입력에 직접 concat하는 구조였음.
- 반면 upstream MUVO는 action을 GRU에 직접 넣지 않고 prior mu/sigma 생성 쪽에서만 사용하는 구조임을 확인함.
- 두 방식의 차이를 ablation할 수 있도록 `RSSM.__init__`에 `action_in_gru` option을 추가함.
- default는 alpha 기존 구조를 유지하기 위해 True로 두고, upstream 비교가 필요할 때 False로 끌 수 있게 함.
- `config_muvo.py`에도 `MODEL.TRANSITION.ACTION_IN_GRU=True` key를 추가함.

## 5/15 (금) 13:51:59 - 20:10:57
- 주요 커밋은 `08336bc`, `515a213`, `33d3faa`임.

### SSIM을 trainer_muvo에 정식 통합
- 새벽에 별도 분기로 검증한 SSIM loss를 `_muvo` 본 trainer에도 통합할 필요가 있다고 판단함.
- `trainer_muvo.py`에 `SSIMLoss` class를 inline으로 이식함.
- `compute_loss`의 RGB block에서 `WEIGHT_SSIM > 0`일 때만 SSIM term이 추가되도록 구현함.
- `config_muvo.py`에는 default off 상태인 `LOSSES.WEIGHT_SSIM=0.0`을 추가함.
- `train_muvo.py`에는 `--h-dim`, `--z-dim`, `--weight-ssim` CLI flag를 추가함.
- `change_summary`도 hidden/state dim과 SSIM weight를 cfg에서 읽어 기록하도록 1차 동적화함.

### 1-window 2x2 ablation 실험
- hidden/state dim과 SSIM 유무가 one-window overfit에서 reconstruction 품질에 영향을 주는지 확인하기 위해 2x2 ablation을 수행함.
- h256/z128, h512/z256 각각에 대해 SSIM off와 SSIM 0.06 조건을 비교함.
- h256/z128 SSIM off 조건은 loss가 약 0.2 수준까지 단조롭게 수렴하고 KL도 step 300 근처에서 안정됨을 확인함.
- h256/z128 SSIM 0.06 조건은 loss가 약 0.55 수준에서 진동하고 KL은 step 600 근처에서 안정됨을 확인함.
- h512/z256 SSIM off 조건은 loss가 약 0.2 수준까지 단조롭게 수렴하고 KL도 step 300 근처에서 안정됨을 확인함.
- h512/z256 SSIM 0.06 조건은 loss가 약 0.55 수준에서 진동하고 KL은 step 600 근처에서 안정됨을 확인함.
- 네 조건 모두 최종적으로 유사한 수렴점에 도달하여, one-window overfit은 SSIM/dim 효과 검증 testbed로 부적절하다고 판단함.
- SSIM은 one-window 조건에서는 뚜렷한 품질 개선보다 instability를 더 추가하고, h256 capacity도 이미 충분하므로 multi-window 환경에서 다시 평가해야 한다고 정리함.

### change_summary 동적화
- `train_muvo.py`의 `change_summary`에 기존 image size 등 hardcoded 값이 남아 있음을 확인함.
- decoder/input size를 변경한 이후에도 실험 로그 문구가 stale하게 남을 수 있어, cfg에서 값을 직접 읽도록 수정함.
- `IMAGE_INPUT_SIZE`, `LIDAR_RANGE_VIEW_SIZE`, `LIDAR_SCALE`, `EMBEDDING_DIM`, fusion transformer layer 수가 매 run마다 실제 cfg 기준으로 기록되도록 구현함.

### experiment_log series 컬럼 추가
- trainer와 실험 분기가 늘어나면서 공통 `experiment_log.csv`에서 실험 계열을 구분하기 어려운 상태가 됨.
- 이를 해결하기 위해 `series` 컬럼을 새로 추가함.
- 기존 experiment log row들도 실험 성격에 따라 여러 category로 분류함.
- 확인된 기록 기준으로는 19번째 컬럼으로 `series`를 신설하고, 기존 36 row 전체를 9개 category로 분류함.
- `scripts/model_variants` 범위에서는 `trainer.py`, `trainer_muvo.py`의 `FIELDNAMES`와 row dict에 `series` 값을 추가함.
- commit message의 "5 trainer" 표현은 `scripts/trainer11_*` 등 `scripts/model_variants` 밖 파일까지 포함한 범위로 해석해야 하며, `scripts/model_variants` 범위에서는 2 trainer가 직접 해당됨.
- `config.py`, `config_muvo.py`에는 `LOGGING.SERIES=""`를 추가하고, `train_muvo.py`에는 `--series` CLI flag를 추가함.

## 5/18 (월) 19:35:14 - 21:34:10
- 주요 커밋은 `ca9f350`, `431d588`, `813882c`, `8458af7`임.

### MUVO 기반 2D latent 구조 분석 및 코드 구현
- 논문의 핵심 발견 중 하나는, 2D latent space가 카메라 이미지 예측 및 공간 voxel occupancy 예측에 큰 이득을 준다는 것.
- MUVO github는 1D latent 방식으로 구현되어 있었기에, 이를 기반으로 구현한 our 모델 또한 1D latent space 방식이었음.
- README의 하이퍼링크를 타고 들어가, 2D latent state 모델을 구현해 놓은 repo 발견 (이하 MUVO_2D).
- 이를 기반으로 our 모델을 2D latent 구조로 확장하기 위해 전체 코드 구조를 수정함.

- MUVO_2D 코드와 현재 모델 구조를 비교하며 학습 구조 차이를 분석함.
- 기존 `_muvo` 5개 파일은 baseline 보존을 위해 그대로 두고, `_muvo_2D` 접미사 5개 파일로 새 분기를 구성함.
- 확인된 diff 기준으로 5개 파일, 2,581줄 규모의 alpha-2D 신규 구현이 추가됨.
- 2D latent 구조를 자체 제작 모델에 반영하기 위해 MUVO_2D의 Convolution 기반 GRU 모델인 `ConvGRUCellGlo` 구조를 분석하고 구현함.
- MUVO와 최대한 유사한 학습 환경 및 성능을 도출하기 위해, MUVO 논문 evaluation의 training setup 부분을 확인함.
  - 논문은 0.2초 간격으로 샘플링하여 sequence length 12를 학습 입력으로 사용함.
  - voxel reconstruction을 포함하는 경우, training 속도를 높이기 위해 sequence length를 6으로 축소함.
  - 현재 구현에서는 upstream MUVO config와 맞추기 위해 `RECEPTIVE_FIELD=4`, `FUTURE_HORIZON=2`로 설정함.

### 자체 제작 모델의 Action 및 Speed 반영 코드 수정
- RSSMTD(RSSM Transformer Decoder)의 구조를 분석한 결과, action이 GRU에 직접적으로 입력되고 있지 않은 것과 달리, our 모델에서는 action이 GRU에 직접적으로 입력되고 있음을 확인함.
- MUVO_2D 코드 구현과 동일하게 prior/posterior의 mu, sigma를 만들 때만 action을 사용하도록 코드를 수정함.
  - posterior 쪽은 `observe_step`에서 `posterior_action_module(action_t)`를 token으로 추가함.
  - prior 쪽은 `imagine_step`에서 `prior_action_module(action_t)`를 token으로 추가함.
- 기존 자체 제작 1D 계열에서는 `SensorFusionTransformer.forward`가 image/LiDAR feature만 받고, fusion 이후 `cam_emb`/`lidar_emb`/speed embedding을 `features_combine`에서 concat하는 구조였음을 확인함.
- 2D 구현에서는 type embedding을 image/LiDAR/policy/action/speed의 5 slot으로 확장하고, `SensorFusionTransformer` forward에 `speed_embed` 인자를 추가함.
- 5/18 최초 구현에서는 speed token을 cam/lidar token에 broadcast-add하는 방식으로 반영함.
- speed token을 별도 token으로 concatenate하는 정렬 작업은 이후 5/20 working tree 정비에서 추가 진행함.

### 자체 제작 모델의 Encoder 코드 수정
- 기존 자체 제작 모델의 encoder는 fusion 이후 `SensorFeatureConv`와 `features_combine`을 통해 image/LiDAR feature를 프레임당 1D vector로 압축하고 있었음.
- 해당 압축 단계에서 spatial 정보가 사라져 2D latent 구조와 양립할 수 없음을 확인함.
- 이에 따라 `SensorFeatureConv`와 `features_combine`을 제거하고, token-shape `(B, S, C, N_tokens)`를 RSSMTD까지 그대로 전달하도록 수정함.
- `encode_fuse_sequence`의 반환 shape을 기존 `(B, S, C)`에서 `(B, S, C, N_tokens)`로 변경함.
- MUVO_2D의 LiDAR encoder가 ResNet18 `out_indices=[1, 2, 3]`으로 설정되어 있음을 확인함.
- 기존 자체 제작 모델은 image/LiDAR 모두 `[2, 3, 4]`를 사용하여 stride-32 종료였으나, MUVO_2D와 동일하게 LiDAR encoder만 `out_indices=[1, 2, 3]`으로 변경하여 stride-16 종료로 수정함.
- 이로 인해 LiDAR token spatial이 `(1, 32)=32`에서 `(2, 64)=128`로 확장됨.
- fusion encoder 입력 token은 image 240 + LiDAR 128 = 368개이며, RSSMTD query/output 쪽에서는 policy token 1개를 더해 369개 구조가 됨.
- 기존 SpeedEncoder는 학습 코드 내부에 inline `nn.Sequential`로 정의되어 있었고 output channel이 16으로 transformer embedding channel 256과 일치하지 않았음.
- MUVO_2D와 동일하게 `SpeedEncoder`를 독립 클래스로 분리하고 output channel을 256으로 정렬하여 fusion 단계에서 image/LiDAR token과 동일한 차원으로 활용 가능하도록 구현함.

### RepresentationModel 및 Decoder 코드 수정
- MUVO 논문은 prior/posterior mu, sigma 생성 모듈을 transformer decoder 기반 구조로 사용하는 것을 확인함.
- 이를 반영하여 기존 mu, sigma 생성 모듈인 `RepresentationModel` class를 transformer decoder 구조인 `RepresentationModelTD`로 수정함.
- `RepresentationModelTD`에서는 image/LiDAR/policy별 learnable query embedding을 만들고, 각 query에 type embedding을 추가한 뒤 concatenate하여 transformer decoder의 query 입력으로 사용함.
- transformer decoder의 key/value는 fusion된 observation token을 사용하도록 구성함.
- MUVO 논문의 2D latent 구조를 유지하기 위해 Decoder 입력 및 출력을 `(B, S, C, H, W)` token shape으로 유지하도록 수정하여 latent의 2D 구조가 유지될 수 있도록 구현함.
- 기존 1D decoder처럼 Linear + Unflatten으로 공간 구조를 다시 만드는 방식이 아니라, token state를 직접 `TokenConvDecoder2D`의 ConvTranspose ladder에 투입하도록 구현함.
- 현재 모델의 task는 RGB 및 LiDAR observation을 기반으로 latent state를 학습하고 future observation을 예측하는 것이므로, action을 별도의 output으로 예측하는 head는 필수적이지 않다고 판단함.
- 다만 5/18 최초 구현에는 upstream 구조를 따라 `PolicyDecoder`, `action_pred`, `WEIGHT_ACTION=1.0` 기반 action loss가 포함된 상태로 들어감.
- 따라서 action prediction 관련 head/loss 제거는 완료된 변경사항이 아니라, 추후 정리 대상으로 남겨둠.

### 자체 제작 모델의 차원 및 hyperparameter 정렬
- RSSMTD는 학습 안정성을 위해 `embedding_dim == hidden_state_dim == state_dim` 제약을 요구함.
- 기존 `_muvo` 설정은 `HIDDEN_STATE_DIM=512`, `STATE_DIM=256`이었으나, `_muvo_2D`에서는 세 값을 모두 256으로 정렬함.
- `SensorFusionTransformer`는 기존 `nhead=8`이었으나, d_model=256에서 64 dim/head 컨벤션을 맞추기 위해 `TRANSFORMER_HEADS=4`로 변경함.
- 학습 entrypoint에는 `--h-dim`/`--z-dim` override를 추가함.
- 두 값을 모두 지정했는데 서로 다를 때만 warning 후 z-dim으로 강제하도록 conflict resolution을 넣음.
- 단, `--h-dim`만 단독 지정하면 `STATE_DIM`은 그대로 남을 수 있어 완전한 보호 로직은 아니므로, 추후 CLI guard 보강이 필요함.

### 자체 제작 모델의 손실 함수 수정
- 기존 자체 제작 모델의 LiDAR loss는 전체 point cloud에 대한 global Chamfer만 사용하고 있었음.
- 자율주행에서는 자기 차량 근방의 정확도가 더 중요하다고 판단하여, `[x: -20~20, y: -20~20, z: -2~6]` 박스 내 point만 masking하여 Chamfer를 계산하는 near-field Chamfer metric을 추가함.
- cfg key `NEARFIELD_PC_RANGE`로 near-field box 범위를 조정할 수 있도록 구현함.
- 기존 KL free-bits가 항상 적용되어 있어 upstream과의 controlled comparison이 어려웠으므로, `KL_FREE_BITS_ENABLED` cfg flag를 추가함.
- 5/18 최초 구현 시점에는 upstream과 맞추기 위해 default를 OFF로 둠.

### 자체 제작 모델의 데이터 파이프라인 및 학습 인프라 수정
- `data_muvo_2D.py`를 `data_muvo.py` 기반으로 새로 만들고, `config_muvo_2D`를 import하도록 전환함.
- upstream `muvo_2d`의 preprocess 구현을 참고하여 로컬 `PixelAugmentation`을 추가하고, Blur / Sharpen / ColorJitter를 `cfg.DATA.AUGMENTATION.ENABLED` flag로 제어하도록 구현함.
- augmentation은 encoder input transform에만 적용하고, reconstruction target인 `image_raw`에는 augmentation과 normalisation을 적용하지 않도록 구성함.
- `MUVODataset`, `StreamWindowDataset`, `MultiArrowStreamDataset` 구조와 dataset 반환 key/shape는 기존 `_muvo` 계열과 동일하게 유지함.
- 반환 dict도 `image`, `image_raw`, `lidar`, `action`, `speed`, `run_id`, `start_row` 구조를 유지함.
- 데이터 샘플링이 `SAMPLE_EVERY_N`을 직접 지정해야 했던 구조를 수정함.
- CARLA 20Hz, Arrow 저장 `FRAME_STEP=5`, 학습 시 `EFFECTIVE_HZ` 명시를 기반으로 `SAMPLE_EVERY_N`을 자동 도출하도록 변경함.
- CLI에서 `--effective-hz N`으로 override 가능하도록 구현함.
- Augmentation 설정이 학습 코드에 hardcode되어 있어 toggle이 어려웠으므로, `cfg.DATA.AUGMENTATION` namespace를 추가하고 `--no-augmentation` CLI flag를 제공함.
- 기존 TensorBoard만으로는 hyperparameter, git commit, artifact를 통합 관리하기 어려워 ClearML `Task.init`을 학습 entrypoint에 추가함.
- cfg에 `CML_ENABLED`, `CML_PROJECT`, `CML_TASK`, `CML_TYPE`, `CML_TAGS`를 추가하고, CLI에 `--no-clearml`, `--cml-project`, `--cml-task`를 추가함.
- `trainer_muvo_2D.py`의 `test_step`에는 ClearML prediction artifact dump를 추가함.
- `Task.current_task()`가 존재하고 `batch_idx < 4`일 때 `rgb_pred`, `lidar_pred`, `action_pred`를 `task.upload_artifact()`로 업로드하도록 구현함.

### Reconstruction figure 기본 동작 확장
- 2D 모델의 reconstruction 품질을 학습 중 확인하기 위해 6 frame 전체를 표시하는 reconstruction figure 기능을 추가함.
- `LOGGING.RECON_FIG_ALL_FRAMES=True`, `LOGGING.RECON_FIG_BOTH_VIEWS=True` cfg를 추가함.
- LiDAR reconstruction은 scaled view와 unscaled view를 함께 볼 수 있도록 구성함.

### Periodic reconstruction callback 추가
- overfit 검증 시 loss curve만으로는 reconstruction 품질 변화 추세를 즉시 파악하기 어렵다고 판단함.
- 매 N epoch마다 한 sample을 `model.eval()`로 decode하여 reconstruction figure를 PNG로 저장하는 `PeriodicReconstructionCallback`을 추가함.
- figure 저장 직후에는 train mode로 복원하여 학습 흐름이 끊기지 않도록 구현함.
- CLI에 `--recon-every-n-epochs`, `--recon-sample-idx`를 추가함.
- 실제 저장 파일명은 `<run_name>_e0001_reconstruction_s0.png` 형태로 생성됨.

### RGB reconstruction plateau 진단 및 decoder capacity 보강
- periodic callback으로 1000 epoch overfit 결과를 추적한 결과, alpha-2D의 RGB reconstruction loss가 epoch 500 이후 약 0.105에서 plateau에 빠지는 현상을 확인함.
- 동일 setup의 alpha-1D baseline은 약 0.01까지 떨어졌으므로, 2D decoder 쪽 capacity 문제가 의심된다고 판단함.
- KL은 약 4.5 nats 수준으로 정상이라 KL collapse 문제는 아닌 것으로 정리함.
- `TokenConvDecoder2D`의 capacity가 upstream `ConvDecoder` 대비 절반 수준이라 RGB 정보를 충분히 표현하지 못한다고 판단함.
- `cfg.MODEL.RSSM_2D.DECODER_CHANNELS=512` key를 추가함.
- decoder 첫 stage가 채널을 즉시 256으로 줄이지 않고 512로 보존하도록 수정함.
- transposed conv channel도 `512 -> 256 -> 128 -> 64` 흐름으로 정렬하고, head input channel을 기존보다 2배로 확대함.
- embedding dim은 256으로 유지하여 RSSM 메모리가 불필요하게 4배 증가하지 않도록 함.

### KL free-bits default 복원
- 5/18 최초 구현에서는 upstream과의 구조 일치를 위해 `KL_FREE_BITS_ENABLED` default를 OFF로 두었음.
- 하지만 alpha-1D `_muvo` baseline은 free-bits ON 상태로 학습되어 왔기 때문에, alpha-2D와 controlled comparison을 하려면 같은 조건이 필요하다고 판단함.
- 이에 따라 alpha-2D의 `KL_FREE_BITS_ENABLED` default를 True로 복원함.
- upstream MUVO는 free-bits 없이 학습되지만, alpha 계열은 KL weight 1e-2와 free-bits 1.0 조합으로 posterior collapse를 막아 왔으므로 해당 설정을 유지하는 것이 더 일관적이라고 정리함.

## 5/19 (화) 00:09:36
- 주요 커밋은 `ce77b48`임.

### 1D baseline에 reconstruction callback 포팅
- 5/18에 alpha-2D에 추가한 `PeriodicReconstructionCallback`을 alpha-1D `_muvo` baseline에도 동일하게 적용할 필요가 있다고 판단함.
- `train_muvo.py`, `trainer_muvo.py`에 callback 흐름을 이식하여 1D와 2D의 reconstruction figure를 같은 학습 과정에서 비교할 수 있도록 수정함.
- 1D는 4-column 2-row 단순 figure로 출력되고, 2D는 6-frame full view와 scaled/unscaled LiDAR view를 포함하는 형태라 출력 형식은 약간 다름.
- 다만 callback 호출 방식과 주기 설정 방식은 최대한 맞춰 apples-to-apples 비교가 가능하도록 정리함.

## 5/19~5/20 Working tree 정비 (미커밋, 소스 최종 수정 5/20 23:05)
- 소스 `.py` 수정시각 범위는 5/20 19:37~23:05 KST로 확인됨.
- `.pyc`까지 포함하면 `trainer_muvo_2D.cpython-38.pyc` 수정시각은 5/20 23:14:54 KST이나, 본문 제목의 "최종 수정"은 소스 `.py` 기준으로 정리함.

### 학습 단계 imagine supervise 제거
- 기존 자체 제작 모델은 학습 단계에서 receptive field posterior와 future horizon imagine 양쪽을 모두 reconstruction loss로 supervise하고 있었음.
- MUVO 논문과 upstream MUVO는 학습 시 전체 시퀀스를 known data로 처리하고, imagine은 validation/test 단계에서 평가하는 구조임을 확인함.
- 이에 맞춰 `_observe_full` 메서드를 추가함.
- `training_step`에서는 `_observe_and_imagine` 대신 `_observe_full`을 호출하도록 변경함.
- 이로써 학습 단계에서 future imagine 출력에 대한 reconstruction loss 계산을 제거함.
- 기존 `WEIGHT_FUTURE=1.0` cfg key와 `self.weight_future` 로딩도 더 이상 학습 loss에 직접 사용되지 않도록 주석 처리함.

### Multi-sample evaluation 도입
- prior 분포에서 1회만 sampling할 경우 validation imagine 결과 variance가 커져 metric이 epoch마다 크게 흔들릴 수 있음을 확인함.
- `_observe_and_imagine`에 `n_samples` parameter를 추가함.
- `validation_step`과 `test_step`에서는 `getattr(cfg.PREDICTION, "N_SAMPLES", 1)` fallback으로 sample 수를 읽어 전달하도록 수정함.
- imagine을 n번 반복 호출한 뒤 loss를 평균하여 validation/test metric이 sampling variance에 덜 흔들리도록 구현함.
- 시각화와 하위 호환을 위해 figure에는 첫 sample만 반환하도록 유지함.
- 단, 현재 `config_muvo_2D.py`에는 아직 `PREDICTION.N_SAMPLES` cfg key가 정의되어 있지 않아 기본 동작은 fallback 값 1임.
- validation dataset name 결정 로직도 `_dataset_name()` 호출 대신 `"RL"`/`"DS"` 직접 매핑으로 단순화함.
- validation return 값은 `{f"val_{dataset_name}_loss": total_loss}` dict 대신 bare `total_loss` tensor로 변경함.

### Speed token concat 패턴 정렬
- 5/18 최초 구현에서는 `SensorFusionTransformer` 내부에서 speed token을 cam/lidar token 각각에 element-wise add하는 broadcast-add 패턴이었음을 확인함.
- upstream MUVO 쪽 speed token 처리와 더 유사하게 만들기 위해 alpha-2D도 speed token을 별도 token으로 생성하도록 수정함.
- Fusion 모델에 들어가기 전 image token, LiDAR token, speed token을 `torch.cat`으로 concatenate하도록 구현함.
- 이 과정에서 `lidar_out` slice가 `out[L_cam:]`로 되어 있으면 뒤에 붙은 speed token이 LiDAR feature reshape에 섞일 수 있음을 확인함.
- `L_lidar = Hl * Wl` 변수를 명시하고, slice를 `out[L_cam:L_cam + L_lidar]`로 수정함.
- 구현은 separate speed token concat으로 바뀌었지만, `models_muvo_2D.py`에는 아직 "broadcast add", "modulation" 등 stale 주석이 남아 있어 커밋 전 주석 정정이 필요함.

### Optimizer weight decay grouping 도입
- 기존 `configure_optimizers()`는 단일 `AdamW(self.parameters(), weight_decay=...)` 호출 구조였음.
- BN, LayerNorm, bias, 1D parameter에는 weight decay를 적용하지 않는 것이 일반적으로 더 안정적이라고 판단함.
- `add_weight_decay()` helper를 추가하여 decay/no_decay parameter group을 나누도록 수정함.
- normalization scale/bias 및 1D parameter는 no_decay group에 넣고, 나머지 weight에만 cfg `OPTIMIZER.WEIGHT_DECAY`를 적용하도록 구현함.
- `add_weight_decay()` 호출 시 `skip_list=("relative_position_bias_table",)`를 전달하여, 해당 이름의 parameter는 weight decay 제외 group에 들어가도록 구성함.
- 현재 `AdamW(parameters, lr=self.lr, weight_decay=0.01)` 호출의 hardcoded `0.01` 인자는 남아 있으나, parameter group이 각 group의 weight decay 값을 들고 있어 실질 적용은 group 설정을 따름.
- 다만 혼동 여지가 있으므로 해당 hardcoded 인자는 추후 정리 대상으로 남김.

### LiDAR loss 및 debug 학습 설정 정리
- 2D 실험에서 LiDAR reconstruction도 함께 학습시키기 위해 `WEIGHT_LIDAR_RE`를 0.0에서 0.1로 조정함.
- overfit/debug 속도를 높이기 위해 train dataset을 `reduction_factor=4`로 축소하는 임시 로직을 `train_muvo_2D.py`에 추가함.
- 이 로직은 실험 편의용이므로 정식 학습 전에는 의도한 dataset size인지 다시 확인이 필요함.

### TensorBoard figure 및 logging tag 정리
- TensorBoard figure에서 scalar tag를 찾는 목록이 실제 Lightning logging suffix와 맞지 않는 부분을 확인함.
- figure 탐색 목록의 `train_action`을 `train_action_step`으로 수정함.
- 실제 `self.log(f"train_{name}", ...)` 호출 자체를 rename한 것은 아니며, Lightning on_step suffix와 figure 매칭을 맞춘 수정임.

### 현재 남아 있는 정리 사항
- speed token concat 구현은 완료됐지만, `models_muvo_2D.py`에는 아직 "broadcast add", "modulation" 등 stale 주석이 여러 곳 남아 있음.
- `trainer_muvo_2D.py` 모듈 docstring에 "upload"가 "uploasd"로 잘못 들어간 typo가 있어 커밋 전 복원 필요함.
- `scripts/model_variants` 기준 `train_muvo_2D.py`, `trainer_muvo_2D.py`에는 trailing whitespace 3건이 남아 있어 `git diff --check -- ':(glob)scripts/model_variants/*.py'` 기준 커밋 전 정리 필요함.
- `AdamW(parameters, lr=self.lr, weight_decay=0.01)` 호출에 hardcoded `0.01` 인자가 남아 있어, parameter group의 weight decay 설정과 혼동될 수 있음.
- action prediction head/loss는 현재 구현에 남아 있으므로, 제거하려면 `PolicyDecoder`, `action_pred`, `WEIGHT_ACTION`, trainer action loss, ClearML artifact dump의 `action_pred` 항목을 함께 정리해야 함.
