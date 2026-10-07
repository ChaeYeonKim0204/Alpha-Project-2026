# Upstream `muvo` Default Config Audit

작성일: 2026-05-15

`muvo/configs/muvo.yml` + `muvo/muvo/config.py` defaults 를 직접 읽어 정리한, **upstream muvo가 default로 무엇을 학습하는지**의 정확한 표. 알파 `_muvo` branch 설계 결정/SSIM 실험 가설 점검의 근거.

## TL;DR

upstream muvo default config는 RGB recon + LiDAR 재구성 + 3D voxel 분류를 기본 학습한다. SSIM, BEV, depth, semantic seg는 모두 **default OFF**. 즉 알파의 원본 가설 "muvo는 SSIM 덕분에 sharp" 는 **틀린 전제**. muvo의 sharp recon은 SSIM이 아니라 (a) ConvDecoder 디자인 + (b) LIDAR_RE / VOXEL_SEG 가 RGB encoder/backbone에 흘려주는 추가 gradient pressure 에서 옴.

## 정확한 default ON/OFF

### Reconstruction 헤드

| 항목 | default | weight | 비고 |
|---|---|---|---|
| **RGB recon (L1)** | ON | 0.1 hardcoded (`trainer.py:296`) | 항상 학습됨. main image objective |
| **LIDAR_RE** (range view: xyz + depth) | **ON** | `WEIGHT_LIDAR_RE=0.1` | `LIDAR_RE.ENABLED: True` |
| BEV semantic | OFF | `WEIGHT_SEGMENTATION=0.1` (코드만) | `SEMANTIC_SEG.ENABLED: False` |
| Depth (mono) | OFF | `WEIGHT_DEPTH=0.1` (코드만) | `DEPTH.ENABLED: False` |
| LiDAR seg | OFF | `WEIGHT_LIDAR_SEG=0.1` (코드만) | `LIDAR_SEG.ENABLED: False` |
| Semantic image | OFF | `WEIGHT_SEM_IMAGE=0.1` (코드만) | `SEMANTIC_IMAGE.ENABLED: False` |

### Auxiliary structural 헤드

| 항목 | default | weight | 비고 |
|---|---|---|---|
| **VOXEL_SEG (3D voxel 분류)** | **ON** | `WEIGHT_VOXEL=0.1` | `VOXEL_SEG.ENABLED: True`. 3D occupancy 학습 |
| ROUTE | input ON | (loss 없음) | `MODEL.ROUTE.ENABLED: True`. route_map → backbone_route → encoder features에 append. supervised loss 아니라 입력 추가 |
| BEV (transformer feature) | OFF | — | `TRANSFORMER.BEV: False` |
| POINT_PILLAR (LiDAR encoder 대안) | OFF | — | `LIDAR.POINT_PILLAR.ENABLED: False` |
| Policy | always (action head) | `WEIGHT_ACTION=1.0` | action 예측 |

### Probabilistic / regularization

| 항목 | default | weight |
|---|---|---|
| KL (probabilistic) | ON | `WEIGHT_PROBABILISTIC=1e-3` (alpha는 1e-2로 10× 큼) |
| KL balancing alpha | 0.75 | (동일) |

### RGB loss augmentation (`LOSSES.*` flags)

| 항목 | default | 비고 |
|---|---|---|
| **SSIM** | **OFF** | `LOSSES.SSIM: False`. trainer.py:97에서 `if cfg.LOSSES.SSIM:` gated. SSIM 코드는 있지만 default off |
| RGB_INSTANCE | OFF | instance mask 기반 RGB loss |
| PERCEPTUAL | OFF | VGG-style perceptual loss |

### Data augmentation (`IMAGE.AUGMENTATION` + `ROUTE.AUGMENTATION_*`)

`muvo/models/preprocess.py:201-215`에서 `if self.training:` gate로 학습 step에만 적용,
val/test는 bypass. **default 전부 ON** (확률 기반):

| 항목 | default 확률 / 강도 | 비고 |
|---|---|---|
| Gaussian blur | **30%** (`BLUR_PROB=.3`) | window=5, std=[0.1, 1.7] |
| Sharpen | **30%** (`SHARPEN_PROB=.3`) | factor=[1, 5]; blur과 mutually exclusive (assert 합 ≤ 1) |
| ColorJitter | **30%** (`COLOR_PROB=.3`) | brightness/contrast/saturation ±0.3, hue ±0.1 |
| Route dropout | 2.5% | route_map 전체 dropout |
| Route end-of-route | 2.5% | route_map 끝부분 잘라냄 |
| Route small/large rotation | 각 2.5% | `AUGMENTATION_DEGREES=8°` |
| Route translate / scale / shear | (0.1, 0.1) / (0.95, 1.05) / (0.1, 0.1) | 매 step affine 변형 |

즉 매 학습 frame: blur or sharpen 60% 확률 (30+30), color jitter 독립 30%. → recon
target은 augmented 입력으로부터 원본을 복원해야 하는 셈. 단순 "input 그대로 출력"
도피가 막힘.

## 알파 `_muvo`와의 비교

| 항목 | upstream default | alpha `_muvo` | 의도 |
|---|---|---|---|
| RGB L1 | ON, weight 0.1 | ON, weight 1.0 | weight 10× 더 큼 |
| LIDAR_RE | ON, weight 0.1 | **OFF** (`WEIGHT_LIDAR_RE=0.0`) | _ |
| VOXEL_SEG | ON, weight 0.1 | OFF (의도적 제거) | _muvo 정책: BEV/3D 빼고 가볍게 ([[alpha26-muvo-branch]]) |
| ROUTE 입력 | ON | OFF (의도적 제거) | 동상 |
| BEV/Depth/Sem | OFF | OFF | 양쪽 다 OFF (차이 없음) |
| SSIM | OFF | OFF (default), 실험에서만 ON | 우리 SSIM 실험은 upstream을 잘못 따라간 케이스 |
| KL weight | 1e-3 | 1e-2 (10×) | alpha가 stronger regularization |
| Image augmentation | **ON** (blur 30% / sharpen 30% / ColorJitter 30%) | **없음** | upstream blur/sharpen이 학습 frame의 60%를 변형 → 모델이 단순 mean-color 도피 못함. alpha는 augmentation 0 |
| Route augmentation | ON (dropout/rotation/translate/scale/shear) | **없음** (route input 자체가 없음) | _muvo branch는 route 제거 |

## 핵심 결론

1. **upstream muvo는 SSIM을 default로 사용하지 않는다.** 알파의 SSIM 실험은 "muvo는 SSIM 쓰니까 sharp" 라는 잘못된 전제 위에서 시작되었음. (정정: 실제 sharp 비결은 ConvDecoder + LIDAR_RE + VOXEL_SEG **+ 학습 augmentation**.)
2. **upstream RGB recon 자체도 L1 단독** (RGB_INSTANCE, SSIM, PERCEPTUAL 다 OFF). 차이는 RGB encoder/backbone에 **LiDAR + Voxel gradient가 같이 흐르는지** 여부.
3. **알파 `_muvo`는 RGB recon gradient만 받는 setting**: VOXEL_SEG 의도적 제거 + LIDAR_RE 비활성화 (`WEIGHT_LIDAR_RE=0.0`). 이게 blur 도피 쉬운 근본 원인 후보 1순위.
4. KL weight도 alpha가 10× 강함 (1e-2 vs 1e-3) — posterior collapse 압력이 더 셀 가능성. 별도 점검 후보.
5. **upstream은 학습 augmentation default ON** (blur 30% + sharpen 30% + ColorJitter 30%). 알파는 augmentation 0 → 모델이 "입력 그대로 출력" smooth 도피가 쉬움. anti-blur implicit pressure 부재. 우선순위 2순위 후보.

## 알파 blur 해결의 우선순위 재구성

이전 가설 ("SSIM 추가") 폐기. 새 후보 순서 ([[muvo_paper_summary]] paper finding 반영):

1. **`WEIGHT_LIDAR_RE` 복원** (0.0 → 0.1). 가장 cheap한 시도. LiDAR head는 이미 `_muvo` 모델에 구현돼 있음 (`models_muvo.py:lidar_re`). 단순히 weight만 켜기.
2. **Image augmentation 도입** — blur/sharpen/ColorJitter을 `data_muvo.py` transform에 추가. _muvo branch 정책 위반 없음, 코드 변경 작음 (`torchvision.transforms` 기본 활용). upstream의 anti-blur implicit pressure 복제.
3. (1)(2)가 부분적으로 도움 되면 — VOXEL/BEV 헤드를 _muvo branch 정책 위반 없이 어떻게 보강할지 고민. 가능한 형태: LiDAR range view에 semantic 채널 하나 추가, 또는 RGB에 depth 보조 head 추가 (모두 BEV/3D는 아님).
4. **KL weight 점검** (1e-2 → 1e-3 upstream과 매칭). posterior collapse 여부 확인.
5. **RSSM 2D state 도입** (paper §IV-B "largest boost in performance"). bogdoll/2D 브랜치의 `transition_td.py:RSSMTD` (Conv-based GRU + transformer decoder for prior/posterior) 가 reference 구현. 알파 cfg/dataset interface에 맞춰 포팅. 중대형 작업이나 paper가 가장 강하게 endorse한 변경.
6. (낮은 우선순위) Perceptual loss (VGG) 또는 GAN 계열 — paper IV-B는 perceptual "produces visually poorer reconstructions"로 부정. 마지막 fallback.
7. SSIM은 검증 데이터로 한 번 더 partial-dataset에서 시험하되, 우선순위 낮음.

## 참고: trainer.py 라인 reference

- RGB recon block: `muvo/trainer.py:294-322`
- SSIM gate: `muvo/trainer.py:97, 312`
- LIDAR_RE block: `muvo/trainer.py:325-336`
- VOXEL block: `muvo/trainer.py` (검색 키 `voxel`)
- Augmentation 진입점: `muvo/models/preprocess.py:201-215` (`self.training` gated), `PixelAugmentation` 정의 `:295-333`, `RouteAugmentation` 정의 `:336-`
- Default config: `muvo/configs/muvo.yml` (YAML overrides) + `muvo/muvo/config.py:275-288` (base losses), `:123-160` (augmentation defaults)
