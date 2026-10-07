# MUVO 4-way 비교: alpha 1D / upstream 1D / alpha 2D / upstream 2D

**생성일:** 2026-05-19
**스코프:** 1D와 2D 두 latent state shape에 대해, alpha26 포팅 버전과 upstream 원본 버전 4개를 횡단 비교.

| 라벨 | 위치 | 베이스 |
|---|---|---|
| **alpha 1D** | `alpha26/scripts/model_variants/{config,data,models,train,trainer}_muvo.py` | upstream 1D에서 BEV/voxel/policy/route/aug 제거한 alpha 포팅 |
| **upstream 1D** | `muvo/` (1D RSSM 모드, `TRANSFORMER_TRANSITION.ENABLED=False`) | author 원본, `Mile` + 1D `RSSM` |
| **alpha 2D** | `alpha26/scripts/model_variants/{config,data,models,train,trainer}_muvo_2D.py` | alpha 1D 베이스에 upstream 2D의 RSSMTD/RepresentationModelTD를 축소 포팅 |
| **upstream 2D** | `muvo_2d/` (별도 repo, author 2D 브랜치) | author 원본, `RSSMTD` + voxel head + perceptual |

참고:
- `muvo/` repo는 `Mile`(1D)과 `MUVO`(2D) 둘 다 구현. `MODEL.TRANSFORMER_TRANSITION.ENABLED` flag로 분기. 이 표의 "upstream 1D" 행은 flag=False 모드 기준.
- `muvo_2d/`의 `muvo.yml` 기본은 voxel + BEV + perceptual 모두 ON이지만 코드상 voxel loss는 `trainer.py:192-199`에서 silent 주석처리됨 (실제 동작에서는 voxel head가 forward는 되지만 loss 기여 X).
- 기존 `muvo_full_comparison.md`(2026-05-18)는 alpha 2D 생성 전이라 본 doc이 갱신·확장.

---

## 0. TL;DR

- **latent state shape는 1D ↔ 2D의 본질적 차이.** 1D는 flat `(B, S, D)` (D=256~512), 2D는 token-shape `(B, S, N_tokens, D)` (N=273~1093).
- **alpha 포팅은 둘 다 "주변부 modality 제거" 정책 동일** — voxel head, BEV head, semseg, depth, route encoder, measurement encoder 모두 drop. RGB + LiDAR range view + (1D는 speed concat / 2D는 speed drop + policy token slot only)만 남김.
- **upstream 1D vs alpha 1D 핵심 차이**: alpha는 BEV/voxel/route/aug 모두 OFF + KL weight `1e-2`로 ↑ + free-bits `1.0` 추가. upstream은 보조 head들이 RGB gradient를 보강하지만 alpha는 RGB L1만 남아 blur escape에 취약.
- **upstream 2D vs alpha 2D 핵심 차이**: alpha는 voxel/BEV/depth/semseg/route 모두 제거 → token 1093 → 369로 축소 (image 240 + lidar 128 + policy 1). embedding/state도 512→256으로 축소. TransformerDecoder도 6L/8H → 3L/4H.
- **1-window overfit (5000ep, RTX 4070 Ti)에서 alpha 1D가 alpha 2D보다 RGB recon 선명도 우위.** 1D는 MLP RSSM이라 단일 window 외우기 쉬움; 2D는 TransformerDecoder + learned query라 DETR과 같은 optimization 난이도. 본 평가는 multi-window full-data 서버 학습으로 옮긴 후 페이퍼 §IV-B 주장이 alpha dataset에서 재현되는지 검증해야 함.

---

## 1. 데이터 / 입력

| 항목 | alpha 1D | upstream 1D | alpha 2D | upstream 2D |
|---|---|---|---|---|
| Dataset | Arrow offline (alpha CARLA dump) | CARLA on-the-fly | Arrow offline | CARLA on-the-fly |
| `RECEPTIVE_FIELD` | 4 | 1 (default), **4** (muvo.yml), 6 (test_*.yml) | 4 | 1 (default), **2** (muvo.yml stub), 1 (predict.yml) |
| `FUTURE_HORIZON` | 4 | 1 (default), **2** (muvo.yml), 10 (test_*.yml) | 2 (v2 fix) | 1 (default), **0** (muvo.yml stub), 6 (predict.yml) |
| sequence length (RF+FH) | 8 | 6 (muvo.yml), 16 (test_*.yml) | 6 | 2 (muvo.yml stub) — sequence 2는 world model로서 의미 X |
| Paper §IV cfg (참고) | — | §IV-B: RF=6, FH=6 (seq=12) — sensor fusion ablation; §IV-C: RF=4, FH=2 (seq=6) — voxel | — | (alpha 2D는 paper §IV-C voxel cfg와 매칭) |
| Image input | (320, 768) | (600, 960) → crop (320, 832) | (320, 768) | (600, 960) → crop (320, 832) |
| LiDAR RV size | (32, 1024) | (varies, alpha set is 32-ch) | (32, 1024) | (64, 1024) |
| LiDAR in_chans | 4 (xyzd) | 4 | 4 | 4 (range-view) / 32 (pillar option) |
| Sample every n | 2 | upstream default | 2 (CARLA_HZ/FRAME_STEP/EFFECTIVE_HZ로 유도) | upstream default |
| Frame step | 5 | — | 5 | — |
| Effective Hz | 2 | — | 2 (override 가능, `--effective-hz`) | — |
| Speed normalization | 50.0 (km/h) | 5.0 (m/s) | 50.0 (km/h; 사용 안 함) | 5.0 (m/s) |
| Action vector | (throttle_brake, steering) | 동일 | 동일 | 동일 |
| 이미지 augmentation | **OFF** | ON (ColorJitter+blur+sharpen, p=0.3) | **ON** (v2 신규, 동일 조합) | ON |
| Route map augmentation | 없음 (route 자체 없음) | ON | 없음 | ON |
| LIDAR_SCALE divide | `/40`, loss항 multiply-back X (footgun) | — (upstream은 m unit 유지) | `/40`, eval/figure는 multiply-back, loss 미보정 | — |

---

## 2. Top-level 모델 구조

| 항목 | alpha 1D | upstream 1D | alpha 2D | upstream 2D |
|---|---|---|---|---|
| 클래스 | `Model` (alpha) | `Mile` | `Model` (alpha) | `MUVO` |
| RSSM | 1D `RSSM` (MLP) | 1D `RSSM` (MLP) | `RSSMTD` (token-shape) | `RSSMTD` |
| Representation model | MLP `RepresentationModel` | MLP | `RepresentationModelTD` (3-modality query) | `RepresentationModelTD` (4-modality query: image/lidar/voxel/policy) |
| Recurrent core | `nn.GRUCell` | `nn.GRUCell` | `ConvGRUCellGlo` (Conv1d gates + global modulation) | `ConvGRUCellGlo` |
| Image encoder | timm ResNet18 + FPN, `out_indices=[2,3,4]` | ResNet18, `out_indices=[2,3,4]` | 동일 (ResNet18 + FPN) | ResNet18, `out_indices=[2,3,4]` |
| LiDAR encoder | timm ResNet18 in_chans=4, `out_indices=[2,3,4]` | ResNet18 in_chans=4, `out_indices=[2,3,4]` | timm ResNet18, **`out_indices=[1,2,3]`** (v3 fix, upstream 2D 매칭) | ResNet18 in_chans=4, `out_indices=[1,2,3]` (range-view) |
| BEV encoder (frustum) | 없음 | 있음 | 없음 | 있음 |
| PointPillars LiDAR option | 없음 | 없음 | 없음 | 있음 (in_chans=32) |
| Route encoder | 없음 | 있음 | 없음 | 있음 |
| Speed encoder | features_combine으로 concat | 있음 (별도 path) | 없음 (drop, 추후 token slot 가능) | 있음 (별도 token) |
| Measurement encoder | 없음 | 있음 (optional) | 없음 | 있음 (optional) |
| Sensor-type embedding | 2 slot (cam, lidar) | 모든 modality slot | 4 slot (image, lidar, policy, action) | 5 slot (image, lidar, voxel, policy, action) |
| Sensor fusion transformer | 3-layer, d=256, h=8 | 6-layer, d=384 | 3-layer, d=256, h=4 | 6-layer, d=384 |
| RepresentationModel TransformerDecoder | N/A | N/A | 3-layer, h=4 (v2 축소) | 6-layer, h=8 |
| `SensorFeatureConv` (flatten) | 사용 | 사용 | 제거 (token-shape 유지) | 없음 (애초에 token-shape) |
| `features_combine` (cam+lidar+speed concat) | 사용 | 사용 | 제거 (speed drop 연동) | 없음 |

---

## 3. RSSM 차원

| 항목 | alpha 1D | upstream 1D | alpha 2D | upstream 2D |
|---|---|---|---|---|
| `EMBEDDING_DIM` | 256 | 512 | 256 | 512 |
| `HIDDEN_STATE_DIM` | 512 | 1024 | 256 (v2 축소, RSSMTD 제약 = embedding) | 1024 |
| `STATE_DIM` | 256 | 512 | 256 (= hidden) | 512 (= hidden) |
| `ACTION_LATENT_DIM` | 64 | 64 | 64 (unused) | 64 |
| `ACTION_IN_GRU` | True (alpha 디자인) | False (upstream 디자인) | True (cfg 키 유지, RSSMTD에서 무시) | — (RSSMTD action_module 별도) |
| Latent shape | flat `(B, S, 256)` | flat `(B, S, 512)` | token `(B, S, 369, 256)` | token `(B, S, 1093, 512)` |
| State decompose | 1D | 1D | image 240 + lidar 128 + policy 1 = **369** | image 260 + lidar 256 + voxel 576 + policy 1 = **1093** |
| Image token HW | — | — | (10, 24) | (10, 26) |
| LiDAR token HW | — | — | (2, 64) (stride-16 FPN) | (4, 64) (stride-16) |
| Voxel token HW | — | — | 없음 | (12, 12, 4) |
| Policy token | — | — | 1 (state evolution만, decode 안 함) | 1 (decode O) |

---

## 4. Decoder / Heads

| Head | alpha 1D | upstream 1D | alpha 2D | upstream 2D |
|---|---|---|---|---|
| RGB | `SensorDecoder` (flat → Unflatten → ConvT) | `RgbDecoder` (multi-scale) | `TokenConvDecoder2D` (token → ConvT ladder, 5 stage) | `ConvDecoder2D` (5 stage) |
| LiDAR range | `SensorDecoder` (range view) | `LidarReconstructionDecoder` | `TokenConvDecoder2D` (lidar branch) | `ConvDecoder2D` (range view) |
| BEV semseg | 없음 | 있음 | 없음 | 있음 |
| Voxel (3D occ) | 없음 | 있음 (`Decoder3D`) | 없음 | 있음 (`ConvDecoder3D`, 그러나 loss는 silent 주석) |
| Semseg image | 없음 | 있음 | 없음 | 있음 (default OFF in muvo.yml) |
| Depth | 없음 | 있음 (RGBD가 RGB와 결합) | 없음 | 있음 (default OFF in muvo.yml) |
| Policy / action | 없음 | 있음 (`PolicyDecoder` 4-layer MLP + Tanh) | **있음** (v2 신규, upstream과 동일) | 있음 (`PolicyDecoder`) |
| Route | 없음 | 있음 | 없음 | 있음 |
| Style (AdaIN) | 없음 | 없음 (commented dead code) | 없음 | dead code 보관 |
| Decoder 채널 (working) | 256 (embedding 동일) | 512 | **512** (v4 fix, embedding=256이라도 decoder 내부 채널은 ↑) | 512 (`latent_n_channels`) |

---

## 5. Loss / 가중치

| 항목 | alpha 1D | upstream 1D | alpha 2D | upstream 2D |
|---|---|---|---|---|
| RGB | L1 multi-scale (1,2,4), `WEIGHT_RGB=1.0` | L1, hardcoded `0.1` | L1 multi-scale, `WEIGHT_RGB=1.0` | L1 multi-scale (1,2,4), `WEIGHT_RGB=0.1` 추정 |
| SSIM | `WEIGHT_SSIM=0.0` (off) | off (`LOSSES.SSIM=False`) | `WEIGHT_SSIM=0.0` (off) | off (default), on이면 ssim_weight=0.6 |
| PerceptualLoss | 없음 | 없음 | 없음 (`ssim_addition_notes.md`에 따라 후순위) | 정의 O, default OFF |
| LPIPSLoss | 없음 | 없음 | 없음 | 정의 O, 미사용 dead |
| LiDAR range | `WEIGHT_LIDAR_RE=0.0` (off — footgun) | `0.1` | `0.0` (별도 ablation으로 남김) | `0.1` |
| LiDAR empty | `WEIGHT_LIDAR_EMPTY=0.0` | undefined | `0.0` | undefined |
| LiDAR xyz/seg | 없음 | seg `0.1` 옵션 | 없음 | seg `0.1`, range `0.1` |
| Voxel | 없음 | `0.1` | 없음 | `0.1` (코드 주석으로 silent OFF) |
| Dice / sem_scal / geo_scal | 없음 | scene-class affinity | 없음 | 정의 O, 일부 주석 |
| BEV semseg / instance | 없음 | 있음 | 없음 | `0.1` / `0.1` |
| Depth | 없음 | 있음 | 없음 | `0.1` |
| Future prediction | `WEIGHT_FUTURE=1.0` | 별도 분리 없음 | `WEIGHT_FUTURE=1.0` | 별도 분리 없음 |
| Action | 없음 | `WEIGHT_ACTION=1.0` | **`WEIGHT_ACTION=1.0`** (v2 신규) | `1.0` |
| KL (`WEIGHT_PROBABILISTIC`) | **`1e-2`** | `1e-3` | **`1e-2`** (alpha 1D와 정렬) | `1e-3` |
| KL free-bits | **`1.0`** (ON, alpha 디자인) | 없음 | **`1.0`** ON (v4-2, alpha 1D 베이스라인과 정렬) | 없음 |
| KL_BALANCING_ALPHA | 0.75 | 0.75 | 0.75 | 0.75 |
| Near-field Chamfer | 없음 | 없음 | 있음 (`NEARFIELD_PC_RANGE=[-20,-20,-2,20,20,6]`) | 있음 |
| Near-field voxel IoU | 없음 | 있음 | 없음 (voxel head 없음) | 있음 |
| Global PSNR/SSIM/Chamfer (eval) | 있음 | 있음 | 있음 | 있음 |

---

## 6. 학습 / 평가 인프라

| 항목 | alpha 1D | upstream 1D | alpha 2D | upstream 2D |
|---|---|---|---|---|
| Trainer 클래스 | `WorldModelTrainer` (PL `LightningModule`) | `WorldModelTrainer` (PL) | `WorldModelTrainer` (PL) | `WorldModelTrainer` (PL) |
| Trainer 선택 | 단일 (alpha) | flag `TRANSFORMER_TRANSITION.ENABLED`로 `MUVO` vs `Mile` 분기 | 단일 (별도 _2D 페어 5개) | 단일 |
| Test step | alpha `WorldModelTrainer.test_step` | upstream `trainer.test()` | alpha + ClearML artifact upload | hand-rolled loop, ClearML artifact |
| Logger | TensorBoard + `ExperimentCSVLogger` | ClearML + TensorBoard | TensorBoard + CSV + **ClearML optional** (`--no-clearml` 가능) | ClearML + TensorBoard (import는 주석) |
| Eval split | 단일 val (Arrow) | val0/val1/val2 3-way | 단일 val | val0/val1/val2 |
| Reconstruction figure | `save_reconstruction_figure` (LiDAR scale 보정 ON) | `model.visualise(...)` | alpha 동일 패턴 + 6-frame 전체 + 2-view (scale 보정/미보정) | `model.visualise(...)` (scale 미보정) |
| Periodic recon callback | `PeriodicReconstructionCallback` (commit ce77b48 추가, 2D와 parity) | 없음 | `PeriodicReconstructionCallback` | 없음 |
| CLI flags | `--series`, `--h-dim`, `--z-dim`, `--weight-ssim`, `--recon-every-n-epochs`, `--recon-sample-idx` | upstream argparse | + `--no-clearml`, `--no-augmentation`, `--effective-hz` | upstream argparse |
| Checkpoint contract | alpha cfg에 lock | upstream cfg | alpha 2D cfg에 lock (1D ckpt와 비호환) | upstream 2D cfg |
| h-dim ≠ z-dim 처리 | 가능 (1D RSSM은 분리 OK) | 가능 | **warning** (RSSMTD 제약상 같아야 정상) | hidden=state=embedding 강제 |

---

## 7. Sequence / window 처리

| 항목 | alpha 1D | upstream 1D | alpha 2D | upstream 2D |
|---|---|---|---|---|
| Window source | Arrow + `arrow_window_index.pkl` 캐시 (서버) | CARLA replay | 동일 캐시 OR `data_muvo_2D.MUVODataset.__init__`에서 in-memory build | CARLA replay |
| TBPTT-aligned | yes | yes (replay reset) | yes | yes |
| stride / sample_every_n | `stride=8`, `sample_every_n=2` | upstream 기본 | seq=6에 맞춘 stride, `sample_every_n=2` | upstream 기본 |
| Multi-arrow concat | `MultiArrowStreamDataset` | N/A | `MultiArrowStreamDataset` | N/A |

---

## 8. 의도적 비대칭 (alpha 2D만 v3 이후 새로 도입)

| 항목 | alpha 1D | alpha 2D | 이유 |
|---|---|---|---|
| 이미지 augmentation | OFF | **ON** | upstream 2D와 정렬 + plan v2 Q11. blur escape 완화 |
| Action loss | 없음 | **ON** (`WEIGHT_ACTION=1.0`) | upstream 2D와 정렬 + policy token이 RSSMTD 입력에 있으므로 자연스럽게 supervise |
| ClearML | 없음 | optional (default ON) | plan v3 Q14/Q15. 서버 학습 결과 공유 |
| Near-field Chamfer | 없음 | 있음 | upstream 2D 메트릭 포팅 (`muvo_2d/muvo/metrics.py:253-289`) |
| 6-frame recon figure (all-frame + 2-view) | 없음 (4-col 단일 row × 2) | 있음 (`RECON_FIG_ALL_FRAMES=True`, `RECON_FIG_BOTH_VIEWS=True`) | scale 보정/미보정 둘 다 확인 |
| `EFFECTIVE_HZ` cfg | hardcoded 2 | CLI override 가능 | dataset sampling rate ablation |

---

## 9. upstream에만 있고 양쪽 alpha 모두 없는 것

- Voxel 3D occupancy head + loss (alpha dataset에 GT 없음)
- BEV semseg/instance head (alpha 정책)
- Route encoder + map augmentation (alpha 정책)
- Measurement encoder optional (alpha 정책)
- PointPillars LiDAR encoder option (range-view만 사용)
- `val0/val1/val2` 3-way split (alpha Arrow 단일 val)
- StyleDecoder / AdaIN (dead code, 양쪽 모두 미포팅)
- LPIPSLoss (dead, 미포팅)
- Hydra config (alpha는 `SimpleNamespace`로 충분)
- MobileViTv2 backbone option (upstream 2D ablation에서 marginal)

---

## 10. alpha 양쪽에만 있고 upstream에는 없는 것

- KL free-bits = 1.0 (Dreamer-style anti-posterior-collapse) — upstream은 양쪽 모두 free-bits 없음
- KL weight 1e-2 (upstream 1e-3 대비 10×)
- Arrow offline pipeline + `arrow_window_index.pkl` 캐시
- `ExperimentCSVLogger` (one-row summary per run)
- `LIDAR_SCALE=40` divide (upstream은 raw m)
- `save_reconstruction_figure` with scale 보정 (figure 시각화 단계에서 multiply-back)
- alpha CARLA dataset spec에 맞춘 32-ch LiDAR / 320×768 image
- `RECEPTIVE_FIELD=4` (1D 한정; upstream 1D default는 1)
- alpha 1D 한정: speed가 `features_combine`으로 concat됨 (upstream은 별도 path)

---

## 11. 알려진 footgun / 위험

| 항목 | 위치 |
|---|---|
| `data_muvo.py` LiDAR `/40` divide vs `compute_loss` multiply-back 누락 → `WEIGHT_LIDAR_RE` 재활성 시 train scalar가 baseline 대비 ~1/40~1/1600 배 | alpha 1D, alpha 2D (둘 다 동일 패턴) |
| `config.py` baseline에 `EPOCHS` 없음 → `train.py:203`이 `cfg.EPOCHS` 읽으면 `AttributeError`. `_muvo` 페어는 `EPOCHS=1`로 막아둠 | alpha 1D, alpha 2D |
| Voxel loss `trainer.py:192-199` 주석 → voxel head는 forward되지만 loss 기여 안 함 (silent) | upstream 2D |
| `prediction.py`의 `WorldModelTrainer.load_from_checkpoint` 주석 + 비표준 속성 의존 | upstream 2D |
| RSSM 메모리 `(B,S,256)` → `(B,S,369,256)`로 ~369× 증가. local 4070 Ti는 batch=2까지 OK 확인됨 | alpha 2D |
| KL free-bits를 token 차원에 어떻게 적용할지 (per-dim vs per-token-per-dim). 현재 reduction은 token도 평균 | alpha 2D |
| Action token이 RSSMTD에서 매 frame cross-attn 받음 → alpha 1D의 `action_in_gru=True` 설계와 다른 dynamics | alpha 2D vs alpha 1D |
| Speed embedding이 alpha 2D에서 drop됨 → 1D 베이스라인 대비 회귀 가능 (추후 token slot으로 추가 가능) | alpha 2D |

---

## 12. Recommended next step (alpha 2D 검증용)

`muvo_2D_port_plan.md` v3의 검증 step 4번 (4-GPU 서버 본 학습) 이후, 같은 step 수로 alpha 1D와 alpha 2D를 cross-run:

1. Server에서 alpha 1D `train_muvo.py` 본 학습 (150k step, A5000×4, batch=2/GPU)
2. Server에서 alpha 2D `train_muvo_2D.py` 본 학습 (동일 step, 동일 batch)
3. ClearML Compare로 두 run의 RGB / KL / Chamfer / 카메라 PSNR / near-field Chamfer 곡선 직접 비교
4. 페이퍼 §IV-B의 "2D state이 카메라 recon에 큰 boost" 주장이 alpha dataset(=BEV/voxel head 없는 RGB+LiDAR-only setting)에서도 재현되는지 확인

1-window overfit 결과(2026-05-19, RTX 4070 Ti 5000ep)는 1D가 우위였으나, 이 setting은 2D의 학습 difficulty(DETR-style learned query + token diversity 부족)에 불리하므로 별도 결과로 다뤄야 함. multi-window full-data에서의 비교가 페이퍼 주장 검증의 진짜 baseline.

---

## 참고 문서

- `muvo_full_comparison.md` (2026-05-18) — alpha 2D 생성 전 3-way 비교 (upstream 1D / upstream 2D / alpha 1D)
- `muvo_2D_port_plan.md` (v3) — alpha 2D 포팅 계획 + 결정사항
- `muvo_paper_summary.md` — 페이퍼 §IV-B ablation 결론 ("2D state itself provides the largest boost")
- `muvo_style_alpha_model_notes.md` — alpha 1D 포팅 의도
- `muvo_default_config_audit.md` — alpha 1D blur escape 분석
- `ssim_addition_notes.md` — SSIM/PerceptualLoss 후순위 결정
