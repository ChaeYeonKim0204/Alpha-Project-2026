# MUVO 4-way 비교: upstream / 2D 브랜치 / 디자인 문서 / alpha `_muvo` 페어

**생성일:** 2026-05-18
**스코프:** 다음 4개 출처를 병렬 탐색하여 코드/구현 차이를 정리.

1. `/home/carol/chaeyeon-kim/muvo/` — 업스트림 MUVO `main` 브랜치 (1D-latent reference).
2. `/home/carol/chaeyeon-kim/muvo_2d/` — 같은 코드베이스의 `2D` 브랜치 (2D-token RSSM + perceptual loss + near-field metrics).
3. `/home/carol/chaeyeon-kim/alpha26/docs/muvo_*.md` (+ `three_way_comparison.md`, `ssim_addition_notes.md`) — alpha26 내부 디자인 메모.
4. `/home/carol/chaeyeon-kim/alpha26/scripts/model_variants/{config,data,models,train,trainer}{,_muvo}.py` — alpha26 내부 `_muvo` 브랜치 vs trainer11 baseline.

존재하는 `alpha26/scripts/model_variants/_muvo_comparison.md`(2026-05-14)는 (4)의 부분만 다루며, 이 문서는 그것을 포함·갱신·확장한다.

---

## 0. TL;DR

- **`muvo/`는 1D-latent RSSM이 default인 upstream main**, **`muvo_2d/`는 2D-token RSSM(`RSSMTD`) + Transformer-decoder representation model + PerceptualLoss + near-field 메트릭이 default인 `2D` 브랜치**. 두 repo는 같은 upstream(`bogdoll/MUVO`)의 다른 head. `muvo_2d`는 modality를 *빼는* fork가 아니라 latent state shape를 바꾸는 fork이다.
- **alpha26 `_muvo` 브랜치는 둘 다 아니다.** 업스트림 MUVO의 sensor-token fusion 구조를 alpha26 dataset 조건(32-ch LiDAR, Arrow offline dump)에 맞춰 축소 이식한 형태. **BEV/3D voxel/policy/route head를 제거**하고 RGB+LiDAR fusion + 1D RSSM만 남긴 muvo-style baseline. 페이퍼의 "2D latent state" 변형은 미적용.
- **현 alpha `_muvo`의 blur 문제 root cause는 SSIM 부재가 아니다.** Upstream도 SSIM은 default OFF다. 진짜 원인은 `_muvo`가 BEV/voxel 보조 gradient를 제거 + `WEIGHT_LIDAR_RE=0.0`로 LiDAR head까지 비활성 + augmentation 미적용 → RGB L1만 남음 → blur escape에 counter-pressure 없음. (`muvo_default_config_audit.md`, `ssim_addition_notes.md` 정정).
- **잠재적 footgun 두 가지**:
  - `data_muvo.py`가 LiDAR를 `LIDAR_SCALE=40`으로 나누지만 `trainer_muvo.compute_loss`의 LiDAR L1/MSE 항은 multiply-back을 하지 않음 → LiDAR weight를 재활성하면 train scalar가 baseline 대비 ~1/40~1/1600 배로 보임. CSV log에서 unit mix.
  - `config.py`(baseline)에 `EPOCHS` 필드가 없음에도 `train.py:203`이 `cfg.EPOCHS`를 읽음 → `--epochs` 없이 실행 시 `AttributeError`. `config_muvo.py:82`는 `EPOCHS=1`로 조용히 막아둠.

---

## 1. `muvo/` (1D main) vs `muvo_2d/` (2D 브랜치)

### 1.1 Directory diff

최상위 (`__pycache__/`, `.git/` 제외):

| Path | `muvo/` | `muvo_2d/` | Note |
|------|---------|------------|------|
| `MUVO_…paper.pdf` (1.5 MB) | present | absent | 페이퍼 PDF는 `muvo/`에만 |
| `rl_birdview/` | top-level | `agents/rl_birdview/`로 이동 | layout fix; **`muvo/data_collect.py`는 `from agents.rl_birdview...`를 import 하므로 사실상 깨져 있음** |
| `agents/` | absent | present (rl_birdview만) | 올바른 위치 |
| `README.md` | "Occupancy-Guided Sensor Fusion…" | 원본 "MUVO" | 완전히 다름 (downstream rebrand) |
| `carla_env.yml` | `name: muvo` | `name: carla` | 한 줄 차이 |
| `requirements.txt`, `data_collect.py`, `sim_run.py`, `constants.py`, `train.py` | identical | identical | — |
| `prediction.py` | 121 lines, `trainer.test(...)` | 122 lines, hand-rolled loop + ClearML upload | 본질적 차이 |

`muvo/muvo/configs/` 차이:

| File | `muvo/` | `muvo_2d/` |
|------|---------|------------|
| `debug.yml` | present | present (small diff) |
| `muvo.yml` | 99 줄 2D+transformer 풀 레시피 | 34 줄 stub (DATAROOT 개인 경로) |
| `predict.yml` | absent | **present** (rewritten prediction.py용) |
| `test_base_1d.yml`, `test_base_1d_without_voxel.yml`, `test_base_2d.yml`, `test_mobilevit_2d.yml` | **present** | absent |

`muvo/muvo/models/` 차이:

| File | `muvo/` (LOC) | `muvo_2d/` (LOC) |
|------|---------------|------------------|
| `common.py` | 786 | 838 (+`PositionEmbeddingSine3D`, ~52 lines) |
| `mile.py` | 1032 | 1033 (한 줄: `nn.init.uniform_(self.type_embedding)`) |
| `frustum_pooling.py`, `preprocess.py`, `transition.py`, `utils.py` | identical | identical |
| `decoder.py` | absent | **present (365 LOC)** — `PolicyDecoder`, `ConvDecoder2D/3D`, `StyleDecoder2D/3D`, `AdaptiveInstanceNorm{,3d}` |
| `muvo.py` | absent | **present (785 LOC)** — `class MUVO(nn.Module)` |
| `transition_td.py` | absent | **present (300 LOC)** — `RSSMTD`, `RepresentationModelTD`, `ConvGRUCellGlo` |

기타:

| File | muvo | muvo_2d | Delta |
|---|---|---|---|
| `config.py` | 369 | 375 | +3 새 knob |
| `losses.py` | 375 | 488 | +`DiceLoss`, `PerceptualLoss`, `LPIPSLoss` |
| `metrics.py` | 317 | 378 | Chamfer rewrite + near-field 변형 |
| `trainer.py` | 1095 | 1156 | model selector switch + perceptual + near-field |
| `data/dataset.py` | base | 3개 `val0/1/2` test set 분할 | 68 line diff |

### 1.2 Config 차이

`config.py` 기본값 (양쪽 공유 후 `_C` 수정):

- `muvo_2d` 추가:
  - `_C.CML_DATASET_VERSION = ''` (line 38) — ClearML dataset pinning.
  - `_C.MODEL.TRANSFORMER_TRANSITION = CN(); .ENABLED = True` (lines 182-184) — **이 한 줄이 `Mile`(1D) vs `MUVO`(2D) 모델 선택을 결정.** Default ON.
  - `_C.LOSSES.PERCEPTUAL = CN(); .ENABLED = False; .MODEL = 'resnet18'` (lines 292-294).
- `muvo_2d` 제거:
  - `_C.MODEL.LIDAR.BACKBONE = 'resnet18'` (`muvo/muvo/config.py:194`).
- 숫자 재조정:
  - `_C.VOXEL.EV_POSITION`: `[32, 96, 12]` → `[96, 96, 12]` — ego를 192×192×64 voxel grid 중앙에 재배치. `RSSMTD`의 12×12×4 voxel-token query embedding 및 near-field crop `[48:144, 48:144, 8:24]`과 정합.

`configs/muvo.yml`:

- `muvo/`는 99 줄 본격 레시피: `TRANSFORMER_TRANSITION.ENABLED=True`, `TRANSFORMER.ENABLED=True`, `CHANNELS=384, BEV=False`, `VOXEL_SEG.DIMENSION=64, N_CLASSES=2`. SEMANTIC/DEPTH/LIDAR_SEG 모두 `False`. `LIDAR_RE=True`, `VOXEL_SEG=True`. `RECEPTIVE_FIELD=4, FUTURE_HORIZON=2`. `ACCUMULATE_GRAD_BATCHES=16, N_WORKERS=16`.
- `muvo_2d/`는 34 줄 stub + 개인 절대 경로 (`DATAROOT: '/mnt/d/python/dataset/test/'`) — dev snapshot에 가깝다.

`debug.yml`, `one_frame.yml`은 둘 다 거의 동일. `muvo_2d/configs/predict.yml`은 `RECEPTIVE_FIELD=1, FUTURE_HORIZON=6, PREDICTION.N_SAMPLES=3`, `MODEL.EMBEDDING_DIM=512`, hard-pinned `weights/epoch=3-step=50000.ckpt`.

### 1.3 Model / architecture 차이

핵심: **`muvo_2d`는 두 번째 모델 family(`MUVO`)를 *추가*, 기존(`Mile`)도 유지.**

`muvo_2d/muvo/models/muvo.py` (`class MUVO`, 785 LOC):
- `cfg.MODEL.TRANSFORMER_TRANSITION.ENABLED=True` 일 때 인스턴스화 (`muvo_2d/muvo/trainer.py:46`). False면 `Mile`로 fallback.
- Encoder는 MUVO 그대로 (image FPN `Decoder`/`DecoderDS`, optional frustum-pool BEV, range-view LiDAR encoder 또는 PointPillars, route encode, optional measurement encoder, speed encoder). 다만 sensor-type embedding의 `n_type` 슬롯 수가 enable된 modality에 비례 (`muvo.py:57-134`).
- Token fusion은 6-layer `nn.TransformerEncoder`(d_model=`MODEL.TRANSFORMER.CHANNELS=384`, nhead=8, dropout=0.1) — `Mile`과 동일.
- **Transition은 `RSSMTD`** (`transition_td.py`). 핵심 디자인:
  - `n_out_tokens = 26·10 + 64·4 + 12·12·4 + 1 = 837` (image grid 10×26 + lidar grid 4×64 + voxel grid 12×12×4 + 1 policy token).
  - `RepresentationModelTD`가 `(μ, σ)`를 6-layer `nn.TransformerDecoder`로 산출. Queries는 modality별 learned positional embedding (`query_embed_image`, `query_embed_lidar`, `query_embed_voxel`, `query_embed_policy`) + `PositionEmbeddingSine`/`Sine3D` + type embedding.
  - Recurrent core가 `ConvGRUCellGlo` — token-wise 1D conv-GRU + global gating branch. MUVO/Mile의 linear GRU와 다름.
- Reconstruction: `MUVO.decode`가 latent state에서 modality별 token block을 슬라이스 → `ConvDecoder2D` (image/lidar range/BEV semseg/semseg image/depth), `ConvDecoder3D` (voxel, 192×192×64). 주석으로 `StyleDecoder{2,3}D`(StyleGAN-AdaIN) 대체안이 남아 있음 — dead but usable.
- 모든 decoder는 `decoder.py`에 있고 `common.py`의 기존 per-resolution head(`RGBHead`, `SegmentationHead` 등)를 import해서 재사용.

기존 모델 유지:
- `mile.py`는 사실상 동일 — `mile.py:94`의 `nn.init.uniform_(self.type_embedding)` 한 줄(init 버그 fix)만 추가. `TRANSFORMER_TRANSITION.ENABLED=False`로 두면 upstream `Mile`과 거의 같은 동작.

`common.py` 델타:
- `muvo_2d/muvo/models/common.py`에 `PositionEmbeddingSine3D` 추가 (lines 681-731) — voxel-token query embedding용.

Frustum pooling, preprocess, 기존 `RSSM`(transition.py), utils, layers, head(BEV/3D voxel/LiDAR) 모두 unchanged. **`muvo_2d`는 modality나 head를 *제거*하지 않았다.**

### 1.4 Training / loss 차이

`muvo_2d/muvo/trainer.py`:

- `from muvo.models.muvo import MUVO`, `DiceLoss`, `PerceptualLoss` 추가 (lines 12, 15-16).
- Model selector (line 46): `self.model = MUVO(self.cfg) if self.cfg.MODEL.TRANSFORMER_TRANSITION.ENABLED else Mile(self.cfg)`.
- `PerceptualLoss` 인스턴스화 (lines 102-106), per-resolution RGB에 `rgb_weight * discount * perceptual_loss * 1.0`로 적용 (lines 333-341).
- SSIM 가중치 retune: `0.6 → 0.3` (line 330).
- **Voxel loss 주석 처리** (`trainer.py:192-212`, `405-412`): `voxel_loss`, `sem_scal_loss`, 새 `dice_loss` 모두 주석. `VOXEL_SEG.ENABLED=True`라도 voxel head는 예측·메트릭만 logging되고 **직접 supervision 없음**. work-in-progress로 보이는 상태.
- Eval mode 처리: `self.train()/self.eval()` → `self.model.train()/.eval()` (lines 432-433, 438, 1126-1127, 1138). PL의 metric submodule이 시각화 hook에서 training mode로 flip되는 것 방지.
- **Near-field voxel metric 추가** (lines 509, 520-528, 599-614): 중앙 `[48:144, 48:144, 8:24]` 96×96×16 sub-cube (≈19.2 m × 19.2 m × 3.2 m) crop으로 `Voxel_*_nearfield` IoU/precision/recall.
- **Chamfer가 global + near-field tuple 반환** (`metrics['cd'].get_stat()[0]`, `[1]`, lines 576-577).
- **Test step 재작성** (lines 1126-1155): `model.forward(batch_rf, deployment=False)` 후 final posterior에서 `model.imagine(state_imagine, ...)`를 `future_horizon` 스텝 roll-out. `_ in range(1)` (one imagine sample). `self.visualise(...)` 주석.

`muvo_2d/muvo/losses.py` 추가:
- `DiceLoss` (378-417) — voxel softmax + empty/non-empty 분해. trainer import는 되지만 주석 처리됨.
- `PerceptualLoss` (419-471) — frozen `timm` backbone (default `resnet18`), feature(MSE/L1) + style(Gram) loss separately weighted. Trainer에 연결됨. SSIM 가중치 손실(0.6→0.3) 보전.
- `LPIPSLoss` (473-488) — `torchmetrics`의 LPIPS 래퍼. **`trainer.py`에서 import 안 됨 → dead code.**

가중치 default(`WEIGHT_*`)는 두 `config.py`에서 byte-identical.

`muvo_2d/muvo/metrics.py`:
- `chamferdist` import 제거, `CDMetric0` 거대 주석 블록(297-351)으로 남김. `torch.cdist` 기반 경로만 유지.
- `CDMetric` per-sample cost를 `try/except` 감싸 빈 PC에서 `tensor([0])` 반환 (280-287).
- Near-field 변형 추가: `PC_RANGE = [-20, -20, -2, 20, 20, 6]`로 제한한 Chamfer를 `self.total_cost_in / .avg_cost_in`에 누적, `get_stat()`이 `(float, float)` 반환 (253-273, 289).
- `SSCMetrics.get_stats()`, `PSNRMetric.get_stat`, `SSIMMetric.get_stat`가 `float(...)` 캐스팅 — PL `.log()`에 텐서 들어가는 것 방지.

### 1.5 기타 주목할 차이

- **`rl_birdview` 이동 후 `muvo/data_collect.py`는 깨져 있다.** 양쪽 동일한 `data_collect.py`가 `from agents.rl_birdview.utils.wandb_callback import WandbCallback` 사용. `muvo_2d`는 `agents/rl_birdview/...`(정상). `muvo`는 top-level `rl_birdview/`만 있고 `agents/` 패키지 없음 → `PYTHONPATH=agents` 트릭 없이는 import 실패.
- **`prediction.py`가 사실상 다른 프로그램.** `muvo`는 `trainer.test(...)` thin wrapper. `muvo_2d`는 hand-rolled, `data.test_dataloader()[2]`(세 번째 test set만), dropout만 eval로 두고 나머지는 train mode, `model.model.forward/imagine`을 `cfg.PREDICTION.N_SAMPLES`회 호출해 voxel/RGB/PCD를 ClearML `task.upload_artifact(...)`로 dump. `WorldModelTrainer.load_from_checkpoint(...)`는 주석. Rough한 상태.
- **`train.py`, `data_collect.py`, `sim_run.py`, `constants.py`, `requirements.txt` byte-identical.**
- **`carla_env.yml` 한 줄 차이**(env name `muvo` vs `carla`).
- **`data/dataset.py`는 주로 test-set 분할.** `muvo_2d`가 단일 `test_dataset`을 `test_dataset_0/1/2`로 분할, sampler stride `50→100`(val) / `(100,100,10)`(test). `__getitem__`에 주석 처리된 `single_element_t['steering']=0`, `single_element_t['throttle_brake']=-1`은 "action freeze" 디버그 잔재 (`muvo_2d/muvo/data/dataset.py:370-371`).
- **MUVO paper PDF**는 `muvo/`에만.
- **stale `__pycache__/`** 가 `muvo_2d/carla_gym/**`에 commit됨.
- **Git history.** `muvo` HEAD `57066bd` (main), `muvo_2d` HEAD `7ffbb84` (branch `2D`, "2D latent space"). 둘 다 upstream `bogdoll/MUVO`.

### 1.6 muvo_2d의 의도 (inferred)

`muvo_2d`는 **modality drop fork가 아니라 2D latent-state 브랜치**다. RGB + LiDAR range view + (optional PointPillars) + route + speed + measurement 모든 입력과 RGB/LiDAR range/voxel 3D semseg/BEV semseg/semseg image/depth 모든 head를 유지한 채, parallel `MUVO` 모델을 도입한다. 차이의 본질:

1. flat-vector RSSM(`Mile`의 512-dim) → `RSSMTD`의 **구조화된 token bag**: image 10×26 + LiDAR 4×64 + voxel 12×12×4 + policy 1 = 837 token per timestep × `MODEL.TRANSFORMER.CHANNELS`.
2. Prior/posterior `(μ, σ)`를 modality별 learned query embedding에 대한 `TransformerDecoder`로 생성.
3. Recurrent core를 `ConvGRUCellGlo` (token-wise conv-GRU + global gate)로 교체. 837 token이 평면화 없이 병렬 evolve.
4. `PerceptualLoss`(timm feature + Gram style), `LPIPSLoss`(미사용) 추가. SSIM 0.6 → 0.3.
5. **Near-field eval**: 40×40×8 m box로 제한한 Chamfer + 96×96×16 sub-cube voxel IoU/precision/recall.
6. ClearML로 per-batch `data_*` artifact(gt + recon + N imagine sample) dump.

`MODEL.TRANSFORMER_TRANSITION.ENABLED` 한 flag가 신구 레시피의 single gate. 같은 ckpt format & 같은 trainer entrypoint로 둘 다 돌아간다. README도 `2D` branch description을 "2D latent states, perceptual losses, transformer backbone"로 명시. **현 코드에서는 `muvo`가 더 *superset* 컨피그(여러 `test_base_*.yml`)이고 `muvo_2d`가 *origin*에 가까운 구현.**

### 1.7 다운스트림 주의점

- **Voxel loss commented out**: `muvo_2d` trainer에서 voxel 직접 supervision 없음 → `VOXEL_SEG.ENABLED=True`라도 voxel head는 shared latent를 통한 reconstruction gradient만 받음. `_muvo_2d`를 시작점으로 copy하면 silently 이 상태.
- **`muvo/data_collect.py`는 out of box로 import 실패.**
- **`muvo_2d/prediction.py`는 `model.rf`, `model.fh`, `model.preprocess`, `model.visualise(..., writer=...)` 같은 PL `LightningModule` 표준 외 속성에 의존** + pinned `weights/` ckpt 필요.
- **`EV_POSITION=[96,96,12]`** (muvo_2d) vs `[32,96,12]` (muvo) — `[48:144]` near-field crop은 muvo_2d 기준 대칭, muvo에서는 비대칭. 1D-recipe ckpt에 near-field metric 재활성 시 EV position 정합 필수.

---

## 2. Design context (from `alpha26/docs/`)

`alpha26/docs/muvo_*.md`, `three_way_comparison.md`, `ssim_addition_notes.md` 6개 메모를 종합. 모든 클레임은 해당 doc에서 직접 인용.

### 2.1 MUVO 페이퍼가 하는 일

페이퍼(Bogdoll et al., IV 2025)는 multimodal(camera + LiDAR range-view) world model을 위해 "MILE의 fundamental architecture를 reduced complexity로 가져온" 포지셔닝. 명시적 차별점: **BEV 기반 fusion 회피** — "MUVO does not require BEV features" (`muvo_paper_summary.md` §II 인용).

**Inputs/encoders (§III-A):**
- Image `I ∈ ℝ^{3×600×960}`.
- LiDAR up to 60,000 points, **lossless bijective 2D cylindrical range view** `R ∈ ℝ^{4×Hr×Wr}` (4 ch).
- ResNet18 default, MobileViT-V2도 평가.

**Fusion (§III-B):** image/LiDAR feature를 token으로 flatten, **2D sinusoidal positional embedding + learnable sensor-type embedding** 부착, concat, `k-layer TransformerEncoder`. Output `t_new ∈ ℝ^{C×(Σi Hi·Wi)}`.

**Transition (§III-C, 페이퍼의 중심 기여):** RSSM with deterministic `h_t = GRU(h_{t-1}, s_{t-1})` + stochastic posterior/prior. **헤드라인 변형: 1D state를 `Ch × (Σ_j T_j)` 모양의 2D state로 대체**, GRU의 FC를 conv로 대체, prior/posterior MLP를 learnable embedding을 query로 쓰는 Transformer decoder로 대체.

**Decoder (§III-D):** `(s_t, h_t)`를 modality별 분리, 각 `C × T_j` 블록을 `C × H₀ × W₀`로 reshape 후 camera/LiDAR는 2D conv upsample, occupancy는 3D conv.

**Losses (Eq. 1):**
```
L = Σ_i λ_i ( λ_img L_img^i + λ_pcd (L_p,xyz^i + L_p,r^i + L_pcd^i) + λ_V L_V,scal^i )
```
`L_img`: multi-scale L1 on RGB (factors 1,2,4). `L_p,xyz`: L2. `L_p,r`: L1. `L_V,scal`: Scene-Class Affinity Loss on 192×192×64 voxel @ 0.5 m.

**Ablation (§IV-B, 중심 outcome):**
- *2D vs 1D*: "The 2D latent state **significantly benefits** predictions for camera images and spatial voxel occupancies, while lidar predictions do not see any benefit."
- *Perceptual*: "we do not see a strong effect of utilizing a perceptual loss, as it produces visually poorer reconstructions."
- *ViT*: 카메라 marginal, 외 효과 없음.
- *전체*: "the 2D latent space itself provides the largest boost in performance, while other changes have little effect."

**Optimal recipe (페이퍼 결론):** "transformer-based architecture with a 2D latent space and lossless range-view representations for point clouds."

**3D occupancy (§IV-C):** 카메라+lidar pre-training 후 occupancy 추가(PTO) > from scratch. Occupancy의 RGB/LiDAR 영향은 "minor".

**Repo mapping 주의 (2026-05-15 정정):** 페이퍼가 cite한 FZI repo(`fzi-forschungszentrum-informatik/muvo`)는 1D RSSM만 포함. 페이퍼가 endorse한 2D 변형은 **저자 personal repo (`daniel-bogdoll/MUVO` `2D` 브랜치)**의 `transition_td.py:RSSMTD` + `ConvGRUCellGlo` + `RepresentationModelTD`에 있다. ⇒ 본 워크스페이스 `/muvo_2d/`가 그 personal `2D` 브랜치.

### 2.2 Upstream MUVO default config — 실제로 켜진 것

`muvo_default_config_audit.md`가 `muvo/configs/muvo.yml` + `muvo/muvo/config.py`를 audit한 결과:

**Reconstruction heads (default ON):** RGB recon (L1, `rgb_weight=0.1` `trainer.py:296`), `LIDAR_RE` (range view xyz + depth, `WEIGHT_LIDAR_RE=0.1`), `VOXEL_SEG` (`WEIGHT_VOXEL=0.1`), Route 입력 (loss 아님, feature feed), Policy/action (`WEIGHT_ACTION=1.0`).

**Reconstruction heads (default OFF):** BEV semantic, mono depth, LiDAR seg, semantic image — 모두 `*.ENABLED: False`.

**RGB loss flags `LOSSES.*` — 전부 OFF:** `LOSSES.SSIM: False`, `RGB_INSTANCE: False`, `PERCEPTUAL: False`. `trainer.py:97, 312`이 `if cfg.LOSSES.SSIM:`로 gate — 코드는 있지만 default training에서는 안 돌아간다.

**Probabilistic:** `WEIGHT_PROBABILISTIC=1e-3`, KL balancing α=0.75.

**Data augmentation (default ON, `self.training` gate):** Gaussian blur 30%, sharpen 30% (mutex, 합 ≤1), ColorJitter 30% (B/C/S ±0.3, H ±0.1), route dropout, end-of-route trim, small/large rotation 2.5%, affine translate/scale/shear. ⇒ "모든 프레임이 ~60% blur/sharpen 확률, 독립 30% ColorJitter → augmented input에서 target을 복원해야 하므로 단순 mean-color escape가 차단된다."

**Audit가 alpha의 prior assumption에 대해 flag한 것:**
1. **"muvo가 sharp한 건 SSIM 덕"은 잘못된 전제.** Upstream default OFF. Alpha의 SSIM 실험은 잘못된 전제 위.
2. **Upstream RGB recon도 L1-only.** Sharpness의 실제 원천은 "ConvDecoder design + LIDAR_RE + VOXEL_SEG + training augmentation" — 보조 손실이 공유 backbone에 gradient 추가.
3. **Alpha `_muvo`는 그 보조 gradient source를 전부 제거함**: VOXEL_SEG 제거, `WEIGHT_LIDAR_RE=0.0`, no augmentation, no route. **blur escape의 #1 root cause 후보.**
4. **KL weight가 alpha에서 10× 강함** (1e-2 vs upstream 1e-3) — secondary posterior collapse risk.

**Audit가 정정한 blur-fix 우선순위:**
1. `WEIGHT_LIDAR_RE` 복원 (0.0 → 0.1) — cheapest, head 이미 구현됨.
2. Image augmentation (blur/sharpen/ColorJitter) 추가 — `_muvo` 정책에 안 어긋남.
3. 비 BEV/3D 보조 head (LiDAR semantic, depth 등).
4. KL weight를 upstream에 맞춤 (1e-2 → 1e-3).
5. `bogdoll/2D` `RSSMTD` 포팅 (페이퍼 "largest boost" 변형) — medium effort.
6/7. Perceptual loss와 SSIM은 의도적 후순위.

### 2.3 MUVO 디코더 head (페이퍼-측)

`muvo_decoder_head_explained.md`가 `muvo/muvo/models/common.py:549-632` 분석, target 320×832:

```
Linear(latent → 512) → Unflatten (B, 512, 1, 1)
pre_transpose_conv (5 stages):
  ConvTranspose(k=(5,13))                → (B, 512,   5,  13)
  ConvTranspose(k=5, s=2, p=2, op=1)     → (B, 512,  10,  26)
  ConvTranspose(k=5, s=2, p=2, op=1)     → (B, 512,  20,  52)
  ConvTranspose(k=6, s=2, p=2)           → (B, 512,  40, 104)
trans_conv1: k=6,s=2,p=2 → (B,256,80,208)   → head_4 (1×1 conv) → rgb_4
trans_conv2: k=6,s=2,p=2 → (B,128,160,416)  → head_2 → rgb_2
trans_conv3: k=6,s=2,p=2 → (B, 64,320,832)  → head_1 → rgb_1
```

핵심:
- **각 head_4/2/1이 자체 learnable upsample(`trans_conv*`) + 1×1 conv head.** Bilinear interpolate 없음.
- **Channel: 512 → 256 → 128 → 64** — 해상도 ↑에 따라 capacity ↓, RGB 직전 64ch로 압축.
- **Target 320×832가 `2^6 × small int`** (320=64×5, 832=64×13) → 6번의 stride-2 doubling이 interpolation 없이 정확히 target에 도달.

대조: alpha의 pre-migration `SensorDecoder` (target 216×288)는 `k=4/s=2/p=1`로 (224, 288)에 도달 후 `F.interpolate(bilinear)`로 (216, 288)로 떨어뜨림 — **blur의 직접 원인**으로 식별됨. rgb_2와 rgb_1이 같은 224×288 feature map을 공유하고 head_1이 자체 upsample 없었음. 216의 인수분해 `2^3 × 27`이 upstream의 `2^N × small int` 패턴에 못 맞춤.

**2026-05-15 NOTE:** commit `7099f87`로 upstream ConvDecoder 패턴으로 마이그레이션, target `(320, 768)` (`768 = 64×12 = 32×24`)로 5- 또는 6-doubling 모두 clean. §3 (216×288 archaeology)는 historical.

### 2.4 "muvo_style alpha model" — `_muvo` 브랜치의 정의

`muvo_style_alpha_model_notes.md`가 `_muvo` 브랜치를 "MUVO full model 그대로 복제가 아니라, MUVO의 핵심 sensor-token fusion 구조를 alpha dataset 조건에 맞춰 축소 이식한 형태"로 정의.

**Upstream에서 keep:**
- ResNet18 image encoder (`out_indices=[2,3,4]`), pretrained, not frozen.
- ResNet18 LiDAR encoder `in_chans=4` (xyzd range-view), pretrained, not frozen — 같은 아키텍처 두 번 인스턴스, separate weight.
- DecoderDS/FPN multi-scale aggregation (alpha의 `FPNDecoder`는 downsample 방향 MUVO style).
- Sensor fusion transformer + sinusoidal pos-embedding + learnable sensor-type embedding + concat-then-attend.
- Range view 4채널 (xyz + depth/range).
- RSSM (deterministic h + stochastic z).
- Multi-scale RGB head `rgb_1 / rgb_2 / rgb_4`.

**Drop (의도적 `_muvo` 정책):**
- BEV decoder, voxel decoder, semantic seg, depth head, policy/action head. "_muvo branch 정책: BEV/3D 빼고 가볍게".
- Route map 입력 및 route augmentation.
- Upstream의 ViT/MobileViT 옵션 (ResNet18 고정).

**Size down:**

| | Upstream MUVO | Alpha `_muvo` |
|---|---|---|
| Transformer channels | 384 | 256 |
| Transformer layers | 6 | 3 |
| Embedding dim | 512 | 256 |
| RSSM hidden / state | 1024 / 512 | 512 / 256 |
| LiDAR RV H | 64 (64-ch sensor) | 32 (Immanuel 32-ch) |
| Image input | 600×960 → crop 320×832 | 600×800 → center-crop 320×768 |

LiDAR vertical 해상도 mismatch는 physical alignment: "Alpha raw LiDAR는 실제 32-channel이므로 height를 64로 늘리는 것은 실제 수직 해상도를 왜곡할 수 있다".

**Vs 원래 alpha26/trainer11 (2026-05-15 NOTE):**
- `7099f87`: `IMAGE_INPUT_SIZE` (320,800)→(320,768), `RGB_RECON_SIZE` (216,288)→(320,768), `SensorDecoder`→upstream ConvDecoder.
- `7bf7df9`: `SensorFeatureConv` plain conv → `BasicBlock × 2`.
- `0be4871`: RSSM에 `action_in_gru` flag (default True = alpha, False = upstream).
- `08336bc`: `trainer_muvo.py`에 SSIM 정식 통합 (`cfg.LOSSES.WEIGHT_SSIM`).

**Speed normalization:** `SPEED_NORMALISATION=50.0` (km/h, vs upstream 5.0 m/s). `train_run_002` 샘플 `[10.805, 9.516, 11.496, 0.102] km/h`로 확인.

**LiDAR scaling:** `LIDAR_SCALE=40.0` — alpha LiDAR 80m vs MUVO 100m → `80/2 = 40` ≈ `100/2 = 50`의 자연 대응.

**Token count:** alpha `_muvo`는 240 image (10×24) + 32 lidar = 272 token/frame, vs upstream 260 + 64 = 324. LiDAR token이 절반인 것은 32-channel sensor의 직접 결과.

**Still-open:** `(32, 1024)` range view 데이터 샘플 재검증, tiny overfit 확인, **`WEIGHT_LIDAR_RE=0.0`이면 LiDAR가 실제 훈련되지 않음** — 결정 보류.

### 2.5 `three_way_comparison.md` 요지

표기: **A = alpha non-muvo** (`*.py`, trainer11), **B = alpha `_muvo`** (`*_muvo.py`), **C = upstream muvo**. A→B는 내부 capacity/구조 evolution, B↛C는 의도적 design 선택이지 버그가 아님.

**A → B (alpha self-evolution):**
- 용량 ratchet: image (300×400→320×768), LiDAR width (256→1024), RSSM (256/128→512/256), FPN/fusion (128→256).
- Decoder 교체: `SensorDecoder` (k=4/s=2 + bilinear) → upstream `ConvDecoder`.
- Encoder fusion 강화: `SensorFeatureConv` plain → `BasicBlock × 2`.
- RSSM `ACTION_IN_GRU` toggle.
- SSIM이 configurable option으로 정식화 (default off).
- Image preproc: `Resize` → `CenterCrop`.
- 그대로 둠: TBPTT 머시너리, Arrow data source, aux head 부재, `WEIGHT_LIDAR_RE=0`.

**B vs C (의도적 델타):**
- C: `VOXEL_SEG + LIDAR_RE + Route + Policy` head; B: BEV/3D 제거 + LiDAR weight 0.
- C RSSM이 B의 2× (1024/512 vs 512/256), dropout p=0.15; B 없음.
- C transformer 6-layer vs B 3-layer.
- C `WEIGHT_RGB=0.1` vs B 1.0 (C는 multi-task balance, B는 RGB-only 효과).
- C `WEIGHT_PROBABILISTIC=1e-3` vs B 1e-2; C는 free-bits 없음, B는 1.0.
- C는 image+route augmentation default ON; B는 없음.
- C: CARLA on-the-fly / disk; B: Arrow offline.
- C: RF=4/FH=2; B: RF=4/FH=4.

**RSSM state shape row 주의:** "FZI main = 1D, alpha = 1D 같음; **bogdoll/2D 브랜치가 페이퍼 §IV-B 'largest boost' 2D 변형을 포함한 `RSSMTD`를 보유**" → 페이퍼 헤드라인 결과를 재현하려면 personal repo 필수.

**A↛C:** "거의 모든 축에서 C가 더 크고 무거움 (model, data pipeline, loss heads, eval metrics)." A와 C는 사실상 "RGB+LiDAR fusion world model" 컨셉만 공유.

### 2.6 SSIM 추가 — 무엇이 제안됐고 왜

`ssim_addition_notes.md` (subclass trial → 정식 통합 → 두 차례 정정):

**Original motivation:** "L1만 쓰면 픽셀 평균을 맞추는 방향이 안전 minimum이라 모델이 '평균색으로 칠하는' 흐릿한 출력을 선호." Upstream MUVO가 `muvo/muvo/trainer.py:97-98, 312-318`에서 SSIM을 사용한다고 *전제*하고 cheap diagonal experiment.

**Initial design (2026-05-15 NOTE로 stale 표시):**
- 새 파일 `trainer_ssim.py`에 `SSIMLoss` (`muvo/muvo/losses.py:292-348`에서 copy) + `SSIMWorldModelTrainer(WorldModelTrainer)` (`__init__`, `compute_loss` override).
- 새 파일 `train_ssim.py`에서 `train.WorldModelTrainer = SSIMWorldModelTrainer` monkey-patch 후 `train.main()`. baseline 720-line trainer / 337-line train 무수정.
- 공식: `loss = self.weight_rgb * discount * self.weight_ssim * (1 - ssim_value)`, posterior + future_horizon 양쪽에 적용.
- Pixel `clamp(0.0, 1.0)` (AMP/FP16 guard).

**Initial 1000-epoch one-window-overfit 결과 — baseline보다 나쁨:**
- `WEIGHT_SSIM=0.6` (upstream `rgb_weight × ssim_weight` 매칭 시도): posterior RGB에 **checkerboard artifact**, total loss 250 step에서 3.5→15 폭주, 최종 6-8 vs baseline 0.2.
- 원인: weight composition mismatch — upstream `0.1 × 0.6 = 0.06` 유효 vs alpha `1.0 × 0.6 = 0.6` (10× too hot) + alpha `SensorDecoder` 자체의 checkerboard bias가 L1에서는 smooth escape로 가려져 있었음.
- `WEIGHT_SSIM=0.06` 재시도: checkerboard 사라졌지만 **baseline보다 더 흐림** (트럭 윤곽 거의 안 보임). "SSIM이 local luminance·contrast·structure 매칭을 시도하면서, 'exact pixel match' 대신 '통계적 평균에 가까운 부드러운' 솔루션을 선호하게 되는 부작용."

**Second-round `_muvo` decoder × SSIM × dim 2×2 matrix (1-window, 1000 epoch):** `(h256/z128, h512/z256) × (SSIM=0, SSIM=0.06)` 4개 fresh run, 모두 새 ConvDecoder + BasicBlock×2 + `action_in_gru=True`. **결과: 4개 모두 시각적으로 indistinguishable.** Decoder swap도, capacity 증가도, SSIM regularization도 1-window-overfit에서는 움직이지 않음. SSIM run의 KL settle time이 L1-only의 ~2배.

**최종 두 정정 (가장 중요):**

*Correction 1:* "**upstream muvo는 SSIM을 사용하지 않는다**" — `LOSSES.SSIM: False` in `muvo/configs/muvo.yml`, gated by `if cfg.LOSSES.SSIM:` in `trainer.py:97`. **실험 전체의 motivating premise가 틀렸음.**

*Correction 2:* upstream sharp recon은 `RGB_L1 + LIDAR_RE + VOXEL_SEG` 보조 gradient가 공유 ResNet18 + ConvDecoder에 흐르는 것이지 SSIM이 아님. Alpha `_muvo`는 이 보조 항을 모두 제거 → RGB가 유일한 gradient source → blur escape에 counter-pressure 없음.

**최종 verdict:** "SSIM 추가 검증은 후순위로, `WEIGHT_LIDAR_RE` 복원 (0.0 → 0.1)이 가장 cheap한 첫 시도." 2×2 matrix의 SSIM 행동 특성 분석은 valid하지만 원래의 "alpha가 muvo만큼 sharp하지 않은 이유" 답은: muvo의 sharpness는 애초에 SSIM 때문이 아니었다.

### 2.7 stale/conflicting 메모

- `muvo_decoder_head_explained.md §3`는 pre-`7099f87` `SensorDecoder` (216×288 + bilinear) 분석 — 현 코드에 없음. blur archaeology로만.
- `muvo_style_alpha_model_notes.md` body의 dim은 post-`7099f87` (320×768) 반영, 그러나 `7bf7df9`/`0be4871`/`08336bc`는 top NOTE block에만, body에는 미반영.
- `muvo_paper_summary.md` "Paper vs Public Code Gap" 섹션은 2026-05-15 정정으로 뒤집힘 — 2D 변형은 author personal repo에 존재.
- `ssim_addition_notes.md`의 초기 subclass+monkey-patch는 `trainer_muvo.py` (commit `08336bc`)의 inline 통합으로 superseded, motivating premise 자체가 틀렸음. 2×2 SSIM 행동 특성은 valid, 최상단 "next steps"는 obsolete; operative priority는 `muvo_default_config_audit.md`에.
- `three_way_comparison.md`는 2026-05-15, 가장 최신 A/B/C 스냅샷.

---

## 3. alpha26 내부 `_muvo` 브랜치 vs original (`scripts/model_variants/`)

Line count: `config.py` 82 / `config_muvo.py` 87; `data.py` 242 / `data_muvo.py` 263; `models.py` 522 / `models_muvo.py` 631; `train.py` 337 / `train_muvo.py` 352; `trainer.py` 721 / `trainer_muvo.py` 789.

### 3.1 `config.py` vs `config_muvo.py`

**Intent delta.** `config.py`는 trainer11 baseline (300×400 image, 216×288 RGB target, 32×256 LiDAR RV, FPN out 128, RSSM h=256/z=128). `config_muvo.py`는 같은 `SimpleNamespace` shape를 muvo-style decoder에 맞춰 retune: input을 `2^6` 정렬 + LiDAR scaling + RSSM 2× 확대 + `EMBEDDING_DIM` 명시 필드 추가. **두 파일 모두 KL weight는 `1e-2` + free-bits `1.0` 유지** — `_muvo`가 upstream `1e-3`을 추적하지 않는다.

**구체적 차이** (line ref는 `config_muvo.py`):
- `DATA.IMAGE_INPUT_SIZE`: `(300, 400)` → `(320, 768)` (line 19). 320=5×64, 768=12×64, 6-doubling clean; 300 불가.
- `DATA.RGB_RECON_SIZE`: `(216, 288)` → `(320, 768)` (line 20). Encoder input과 같음 → reconstruction target에 별도 resize 불필요. **`_muvo` `SensorHead`가 1×1 conv only(bilinear fallback 없음)인 이유.**
- `DATA.LIDAR_RANGE_VIEW_SIZE`: `(32, 256)` → `(32, 1024)` (line 21). 1024=32×32, `2^5` divisible, `_auto_n_pre_doublings(target=(32,1024))=2`에 매칭.
- 신규 `DATA.LIDAR_SCALE=40.0` (line 23). `data_muvo.py:199`에서 `(x,y,z,r)`을 나눠 `[-1,1]` 근처로. baseline은 raw meter.
- 신규 `MODEL.EMBEDDING_DIM=256` (line 31). `train.py:285`/`trainer.py:193,201`의 magic `embedding_n_channels=128` 대체.
- `MODEL.FUSION` 재설계:
  - 제거: `IMAGE_TOKEN_DOWNSAMPLE=(20,20)`, `LIDAR_TOKEN_DOWNSAMPLE=(4,16)` (`config.py:33-34`).
  - 추가: `TRANSFORMER_CHANNELS=256` (line 35).
  - 이유: `models.py:425-436`은 transformer 전에 `F.adaptive_avg_pool2d`로 명시 downsample; `models_muvo.py:537-549`는 그 풀링 없음 — FPN bottom-up이 token count 결정.
- `MODEL.TRANSITION`:
  - `HIDDEN_STATE_DIM`: 256 → 512 (line 42).
  - `STATE_DIM`: 128 → 256 (line 43).
  - 신규 `ACTION_IN_GRU=True` (line 49) — `models_muvo.py:236-238, 320-323`만 honor. baseline RSSM (`models.py:248`)은 무조건 concat.
- `LOSSES`: 키셋 동일. `WEIGHT_SSIM`이 `config.py:56`은 `0.06`, `config_muvo.py:60`은 `0.0`. baseline은 `trainer_ssim.py`용으로 키만 있고 trainer.py가 사용 안 함; muvo는 SSIM 코드 inline이지만 default off.
- `LOGGING.RUN_NAME`/`BASE_FILE`: `muvo_style_alpha_fpn256_lidar32` / `train_muvo.py` (lines 76-77).
- 신규 top-level `EPOCHS=1` (line 82). **`config.py`에는 없는데도 `train.py:203`이 `cfg.EPOCHS` 읽음** → baseline의 latent 버그를 muvo가 우회.

**Dead branch.** `cfg.STEPS=400`은 `trainer_muvo.py:690`에서 `estimated_stepping_batches=0`일 때만 fallback. 양쪽 동일, 코스메틱.

### 3.2 `data.py` vs `data_muvo.py`

**Intent delta.** 같은 `MUVODataset`/`StreamWindowDataset`/`MultiArrowStreamDataset` plumbing on 같은 Arrow tables. `_muvo`는 (1) image cropping 정책을 resize → center-crop으로 변경(aspect 유지, pixel drop), (2) LiDAR RV를 `LIDAR_SCALE`로 나누고 tensorize. 나머지(window indexing, run-id 경계, TBPTT-aligned stream order, dict key contract) byte-identical.

**구체적 차이:**
- `data_muvo.py:14`: `from torchvision.transforms import functional as TF` 추가.
- `data_muvo.py:16`: `from config import cfg` → `from config_muvo import cfg`.
- 신규 `CenterCropToSize` (`data_muvo.py:23-37`). PIL image가 target보다 작으면 `ValueError`. **`_muvo` default `(320, 768)`이면 source가 height 320 이상, width 768 이상이어야** — CARLA 800×600 capture는 width OK / height OK, 그러나 1280×720, 800×600 등 표준 capture와의 호환은 항상 확인.
- `_make_img_transform` (`data_muvo.py:40-46`)와 `_make_img_transform_raw` (`data_muvo.py:49-56`)이 `transforms.Resize(...)` → `CenterCropToSize(...)`. `_raw` 변환은 docstring으로 normalization asymmetry 명시 (encoder input: mean/std-normalize, recon target: plain `[0,1]`).
- `MUVODataset.__getitem__` (`data_muvo.py:198-200`): `lidar_rv = point_cloud_to_range_view(...) / float(cfg.DATA.LIDAR_SCALE)`. baseline (`data.py:179`)은 raw meter.

**Dict key/shape는 byte-identical** (`image`, `image_raw`, `lidar`, `action`, `speed`, `run_id`, `start_row`; T=8 from RF+FH).

**Bug/footgun.**
- LIDAR_SCALE 나누기가 **모든 consumer에서 보정되지 않음**. `trainer_muvo.py:117-126`은 figure depth용으로 `× lidar_scale`, `trainer_muvo.py:416-424`는 eval metric용으로 `× lidar_scale`. 그러나 `compute_loss`의 LiDAR L1/MSE 항(`trainer_muvo.py:471-509`, `trainer.py:414-440`에서 verbatim 상속)은 **multiply-back 없음** → train scalar `lidar_xyz_*`/`lidar_depth_*`/`lidar_empty_depth_*`가 `(meters/40)` 단위. `WEIGHT_LIDAR_RE=0.0`이라 dormant이지만 재활성 시 baseline 대비 ~1/40~1/1600 배로 보임 → `experiment_log.csv` silently mix unit.

### 3.3 `models.py` vs `models_muvo.py`

**Intent delta.** 5개 페어 중 **유일하게 실질적 rewrite.** `_muvo_comparison.md §4`가 이미 모듈별 status를 표로 정리. `models.py`는 trainer11 (top-down FPN, transformer 전 explicit `adaptive_avg_pool2d`, `nn.Sequential(AdaptiveAvgPool2d + Flatten)` compression, `k=4 s=2 p=1` ConvTranspose ladder + bilinear). `models_muvo.py`는 같은 top-level `Model.forward`/`Model.imagine` contract를 muvo-style 내부로 재구현: bottom-up FPN, pre-transformer 풀링 없음, `BasicBlock×2 + AdaptiveAvgPool + Flatten`, `k=5/p=2/op=1` + `k=6/p=2` ConvTranspose ladder + `_auto_n_pre_doublings`, 1×1 conv head + decoder가 final resolution 책임. RSSM에 `action_in_gru` toggle (default True = alpha). **BEV/3D, policy/route 양쪽 모두 없음.**

**구체적 차이:**
- Imports/utility byte-identical (`sigmoid2`, `PositionEmbeddingSine`, `SensorFusionTransformer`, `RepresentationModel` 동일).
- `FPNDecoder` (`models.py:25-48` vs `models_muvo.py:25-50`):
  - Channel default: 128 → 256.
  - Top-down (`x = conv1(xs[2])`, iterate `[1,0]`, `F.interpolate`) → **bottom-up** (`x = conv1(xs[0])`, iterate `[1,2]`, `F.adaptive_max_pool2d`).
  - 양쪽 모두 timm `resnet18` `out_indices=[2,3,4]` 3-element 기대.
- `BasicBlock` (`models_muvo.py:134-164`) 신규 — manually inlined ResNet basic block (optional `1×1` downsample), `SensorFeatureConv`만 사용.
- `SensorFeatureConv` (`models_muvo.py:167-185`) 신규, `nn.Sequential(AdaptiveAvgPool2d + Flatten)` (`models.py:373-380`) 대체. 새 구조: `BasicBlock(in→out, stride=2, downsample=True) → BasicBlock(out, out) → AdaptiveAvgPool2d((1,1)) → Flatten(1)`. 풀링 전 residual stage 추가.
- `RSSM` (`models.py:154-267` vs `models_muvo.py:210-342`): **유일한 차이는 `action_in_gru` flag**.
  - `__init__`에 `action_in_gru=True` 파라미터, `pre_gru_input_dim = state_dim + (action_latent_dim if action_in_gru else 0)` (`models_muvo.py:238`).
  - `imagine_step` (`models_muvo.py:315-327`): True면 `gru_input = torch.cat([sample_t, latent_action_t], dim=-1)`, False면 `gru_input = sample_t`.
  - `observe_step`, `forward`, `sample_from_distribution`, `stack_list_of_dict_tensor`은 byte-identical.
  - Prior MLP는 양 모드 모두 latent action concat (`models_muvo.py:325`), upstream 동작 매칭.
- Decoder geometry (`models.py:274-345` vs `models_muvo.py:349-468`):
  - `_decoder_base_size`: baseline은 `ceil(h/32), ceil(w/32)`; `_muvo`는 `n_pre_doublings=3` enforce + `target % 2^(n_pre_doublings+3) == 0` 검증, 미만 시 raise. 6-doubling default for RGB, 5-doubling for LiDAR via `_auto_n_pre_doublings`.
  - `SensorHead`:
    - Baseline: `output_size` + `F.interpolate(..., mode='bilinear')` fallback.
    - `_muvo`: 1×1 conv only, no `output_size`. Decoder가 exact target 산출 (위 divisibility 검사 의존).
  - `SensorDecoder`:
    - Baseline `base_conv`: 1× constant-size convT + 3× `k=4, s=2, p=1` doublings.
    - `_muvo` `pre_transpose_conv`: 1× constant-size convT + `n_pre_doublings` × (non-last: `k=5, s=2, p=2, op=1`; last: `k=6, s=2, p=2`).
    - 이어 3× `trans_conv1/2/3` (`k=6, s=2, p=2`), channel halving (`decoder → /2 → /4 → /8`). 각 단계가 `SensorHead` 공급.
    - Baseline은 head_4 input과 channel-halving 모두 `decoder // 2`; `_muvo`는 `tiny_ch = decoder // 8` (`models_muvo.py:414`) — head_1이 baseline보다 narrower. **실질적 아키텍처 변경.**
  - Baseline은 `head_1` 앞에 `refine` (`Conv2d(low, low, k=3, p=1) + ELU`, `models.py:330-333`); `_muvo`는 없음 — 세 번째 `trans_conv` upsample이 대체.
- `Model.__init__`:
  - Baseline: `embedding_n_channels=128` hardcoded (`models.py:353`). FPN channel = `embedding_n_channels`.
  - `_muvo` (`models_muvo.py:476-484`): `embedding_n_channels=None` → `cfg.MODEL.EMBEDDING_DIM` resolve. 신규 `transformer_channels = cfg.MODEL.FUSION.TRANSFORMER_CHANNELS or embedding_n_channels`. FPN out = `transformer_channels`. Default cfg에서 둘 다 256.
  - Baseline: `image_feature_compress = nn.Sequential(AdaptiveAvgPool2d + Flatten)`; `_muvo`: `SensorFeatureConv(transformer_channels, embedding_n_channels)`.
- `Model.encode_fuse_sequence`:
  - Baseline: 명시 `F.adaptive_avg_pool2d(cam_feat, (IMAGE_INPUT_SIZE / IMAGE_TOKEN_DOWNSAMPLE))` + LiDAR 동일.
  - `_muvo`: FPN 출력을 transformer에 직접 전달. **Token 수가 FPN bottom-up 결과**. 320×768 image @ resnet18 `out_indices=[2,3,4]` → `xs[0]` shape `(B, 128, 40, 96)` → 3840 cam token. 32×1024 LiDAR → `xs[0]` `(B, 128, 4, 128)` → 512 LiDAR token. **총 ~4352 token/frame/batch — baseline `(20×20)+(4×16)=464`보다 메모리 훨씬 무거움.**
- `Model.forward`/`Model.imagine` (`models_muvo.py:562-631`)은 `models.py:453-522`와 byte-identical (같은 kwargs, return contract, TBPTT-friendly `h_init`/`s_init`/`continuation`/`init_action`).

**Bug/dead.**
- `models_muvo.py:3`이 `Optional`을 import만 하고 사용 안 함 (baseline도 동일, `_muvo` regression 아님).
- `SensorFeatureConv` 첫 `BasicBlock`이 `stride=2`이므로 320×768 token grid를 `(20, 48) → AdaptiveAvgPool((1,1))`로 반감 후 collapse — correctness OK, perf smell.

### 3.4 `train.py` vs `train_muvo.py`

**Intent delta.** Verbatim sibling. import 슬롯, 몇 개 CLI flag, `WorldModelTrainer` 생성자 호출, `change_summary` 문자열만 다름. 신규 flag(`--series`, `--h-dim`, `--z-dim`, `--weight-ssim`)는 `config_muvo.py` 수정 없이 ablation 가능하게 한 것.

**구체적 차이:**
- L18-29: import → `config_muvo`, `data_muvo`, `trainer_muvo`.
- L35-37: `--series` flag → `cfg.LOGGING.SERIES`.
- L41-46: `--h-dim`, `--z-dim`, `--weight-ssim` → `cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM/STATE_DIM`, `cfg.LOSSES.WEIGHT_SSIM` (`train_muvo.py:101-106`).
- L301: `model = WorldModelTrainer(cfg=run_cfg, lr=run_cfg.OPTIMIZER.LR)` — baseline (`train.py:285`)의 `embedding_n_channels=128` kwarg 제거. 이제 `cfg.MODEL.EMBEDDING_DIM` 경유.
- L337-341: `WorldModelTrainer.load_from_checkpoint(...)`도 `embedding_n_channels=128` 제거.
- L199-208: `change_summary`가 cfg 값 f-string templating으로 변경:
  ```
  f"MUVO-style alpha: {IMAGE_INPUT_SIZE} center-cropped image input, "
  f"{LIDAR_RANGE_VIEW_SIZE} LiDAR range-view with {LIDAR_SCALE:g}m scaling, "
  f"ResNet18 multi-scale FPN to {EMBEDDING_DIM} channels, "
  f"{TRANSFORMER_LAYERS}-layer sensor fusion transformer, "
  f"RSSM h{HIDDEN_STATE_DIM}/z{STATE_DIM}, "
  f"WEIGHT_SSIM={WEIGHT_SSIM}."
  ```
- 나머지(`resolve_runtime`, `apply_overrides` 외 flag, `build_dataloaders`, `build_callbacks` 외 summary, `build_trainer`, `maybe_extend_onecycle_resume_checkpoint`, `main`)는 byte-identical.

**Bug/footgun.**
- `train.py:203`이 `cfg.EPOCHS`를 읽는데 `config.py`에 `EPOCHS` 없음 → `--epochs` 없이 실행 시 `AttributeError`. `config_muvo.py:82`가 `EPOCHS=1`로 우회. **baseline latent bug.**
- `embedding_n_channels=128` 제거는 cfg 필드 도입과 정합, 그러나 baseline ckpt (FPN out 128)를 `_muvo` `WorldModelTrainer`에 load하면 `EMBEDDING_DIM=256`으로 build → 모든 FPN conv에서 shape mismatch. silent re-init 가능성 — `cfg=run_cfg` + new default 조합.

### 3.5 `trainer.py` vs `trainer_muvo.py`

**Intent delta.** `trainer_muvo.py`는 upstream `muvo/muvo/trainer.py` 포트가 **아니다**. baseline에서 정확히 세 가지: (1) `SSIMLoss` inline + RGB loss 루프에 SSIM 항 추가, (2) `LIDAR_SCALE` divide를 eval metric과 figure에서 보정, (3) `embedding_n_channels` 생성자 인자를 optional + `cfg.MODEL.EMBEDDING_DIM`에서. 손실 공식, TBPTT, metric set, callback, optimizer, checkpointing은 byte-identical. **"trainer.py + SSIM + lidar-scale"이 정확한 한 줄 요약.**

**구체적 차이** (line ref는 `trainer_muvo.py`):
- L9/L13: `import torch.nn as nn` + `from models import Model` → `from models_muvo import Model`.
- L16-65: `SSIMLoss` class verbatim from `muvo/muvo/losses.py:292-348`. `trainer_ssim.py:17-63`과 byte-identical 복사본 (drift risk).
- L102-104 (`save_reconstruction_figure`): `lidar_scale = float(getattr(run_cfg.DATA, "LIDAR_SCALE", 1.0))`, `lidar_vmax = 2.0 * lidar_scale`. `depth(tensor)` closure가 `× lidar_scale` (L106). 4개 `imshow(..., vmin=0, vmax=...)` 호출이 baseline의 `vmax=80` literal → `lidar_vmax` (cfg-driven, default 80 m).
- L246: `__init__` `embedding_n_channels=128` → `None`.
- L254-255: `None`이면 `getattr(self.cfg.MODEL, "EMBEDDING_DIM", 256)`로 resolve.
- L266-269: `self.weight_ssim = getattr(self.cfg.LOSSES, "WEIGHT_SSIM", 0.0)`, `self.ssim_loss = SSIMLoss(channel=3) if self.weight_ssim > 0.0 else None`. 주석에 "0.06 = upstream effective (0.1 × 0.6)" 명시.
- L416-424 (`compute_eval_metrics`): LiDAR metric 3개에 `* lidar_scale`. `lidar_chamfer_xyz`, `lidar_xyz_euclidean`, `lidar_range_mae` 모두 meter 단위로 보고. RGB PSNR은 `[0,1]` 범위 그대로.
- L512-518 (`compute_loss`, RGB only): `for factor in [1,2,4]` RGB 루프 안에서 `self.ssim_loss is not None`이면 `ssim_term = 1.0 - self.ssim_loss(pred.clamp(0,1), target.clamp(0,1))` 계산, `losses[f"{prefix}ssim_{factor}"] = self.weight_rgb * (1/factor) * self.weight_ssim * ssim_term`.
- L762: `ExperimentCSVLogger.on_fit_end` default `base_file` `"trainer11_reviewed.ipynb"` → `"train_muvo.py"` (cfg가 override하므로 cosmetic).
- LiDAR loss 분기(L468-509)는 baseline과 byte-identical. **`× lidar_scale` 보정 없음** → `_muvo` data divide의 영향이 loss 값에 직격. `WEIGHT_LIDAR_RE=0.0`이라 dormant.

**Bug/dead.**
- SSIM이 코드 presence는 ON, cfg default는 OFF (`config_muvo.py:60 WEIGHT_SSIM=0.0` → `self.ssim_loss = None`). baseline `config.py:56 WEIGHT_SSIM=0.06` 있지만 baseline `trainer.py`는 SSIM 로직 자체 없음 → field도 dead (`trainer_ssim.SSIMWorldModelTrainer`만 읽음, `train_ssim.py:14` monkey-patch 경유).
- `SSIMLoss` 중복: `trainer_muvo.py:17-65`와 `trainer_ssim.py:17-63`. 한쪽 패치 시 drift.
- `SSIMLoss.window`가 `__init__` (L29)에 만들어지고 buffer registered 아님; `_ssim`이 매 call에 `torch.as_tensor(self.window, ...)`로 device/dtype 복사 (L41). 동작 OK이지만 GPU 이동 안 됨 → host→device 복사 낭비. `trainer_ssim.py:41`도 동일.
- `chamfer_distance_range_view` (`trainer.py:326-345`): Python 이중 루프, slow. `_muvo` 변화 없음. L420의 `* lidar_scale` multiply로 chamfer가 meter 단위 유지.
- `embedding_n_channels` resolution이 `WorldModelTrainer.__init__` (L254-255)와 `models_muvo.Model.__init__` (L479-480) 양쪽에서 독립적으로 일어남. Default 둘 다 256 → mismatch 없음, 그러나 caller가 `embedding_n_channels=128`을 explicit pass하면 `Model.__init__`의 "if None, read cfg" 분기는 skip되고 값 그대로 — 두 redundant resolution path.

### 3.6 Cross-cutting (alpha26 내부)

- **마이그레이션 모양 확인.** 5 페어 중 **`models_muvo.py`만 실질 rewrite** (FPN 방향, decoder kernel 패턴, head simplification, sensor-feature ResNet, RSSM toggle). `config_muvo.py`는 retuning + 소규모 schema 추가. `data_muvo.py`/`train_muvo.py`/`trainer_muvo.py`는 surgical edit한 sibling clone. 기존 `_muvo_comparison.md` 표(23-31)와 일치.
- **기존 doc (2026-05-14) 이후 변경:**
  - `trainer_muvo.py`가 더 이상 기존 doc 145-160의 "SSIM weight 0.6 ❌ 없음"과 매칭 안 됨. SSIM이 verbatim inline + `cfg.LOSSES.WEIGHT_SSIM` gate, `trainer_muvo.py:267-268` 주석이 "0.06 = upstream effective" 명시. ⇒ §6 row "SSIM loss on RGB"를 `❌ 없음` → `✅ inlined, default off in config_muvo.py, enable via cfg or --weight-ssim`로 갱신 필요.
  - `trainer_muvo.py`가 `LIDAR_SCALE`-aware figure rendering (L102-126) + eval-metric rescaling (L416-424) 획득 — 기존 doc 작성 시점에 `LIDAR_SCALE` 없었음.
  - `train_muvo.py`가 `--series`, `--h-dim`, `--z-dim`, `--weight-ssim` flag 획득 — 기존 doc §5의 "import slot + `embedding_n_channels=128` 제거 + change_summary 변경"은 incomplete.
- **변경 안 됨.** §6 표의 `add_weight_decay`, N_SAMPLES imagination, action loss, dropout-only eval, active_inference, tensorboard image/video push 모두 여전히 부재. `trainer_muvo.py`는 여전히 strictly trainer11 + SSIM + lidar-scale.
- **EMBEDDING_DIM threading.** `_muvo`가 magic `embedding_n_channels=128`을 두 곳(`train_muvo.py:301, 339`)에서 제거, 두 곳(`trainer_muvo.py:246, 254`; `models_muvo.py:476, 479-480`)에서 optional + cfg fallback, cfg 필드 한 곳(`config_muvo.py:31`) 추가. data-driven으로 변환 OK, 그러나 **resolution이 두 번 발생** (trainer 한 번, model 한 번). Default 256으로 일치 시 무해.
- **LIDAR_SCALE units footgun (재강조).**
  - `data_muvo.py:199`이 LiDAR RV를 40으로 나눔.
  - `trainer_muvo.py:106` (figure depth) + `trainer_muvo.py:417-424` (chamfer/euclidean/range_mae)가 `× lidar_scale`로 보정.
  - **`trainer_muvo.py:471-509` LiDAR L1/MSE/smooth-L1 loss는 보정 안 함** (`trainer.py:414-440`에서 verbatim 상속). 결과: training `lidar_xyz_*`/`lidar_depth_*`/`lidar_empty_depth_*` scalar가 `(meters/40)` 단위, eval metric은 meter. `WEIGHT_LIDAR_RE=0.0`이라 dormant이지만 ablation에서 LiDAR weight 재활성 시 train scalar가 baseline 대비 ~1/40~1/1600 배. `results/experiment_log.csv`가 silently mix unit.
- **EPOCHS 필드.** `config_muvo.py:82`가 `EPOCHS=1` 추가; `config.py`는 없음. `train.py:203 max_epochs=cfg.EPOCHS`가 `--epochs` 없이 실행 시 `AttributeError`. baseline에 `EPOCHS=1` one-line fix 권장.
- **`config_muvo.py`의 `WEIGHT_PROBABILISTIC=1e-2`** (line 54) — baseline (`config.py:50`)과 동일. 기존 `_muvo_comparison.md §2`는 upstream `1e-3`과의 divergence로, §7.4는 posterior collapse 후보로 flag하지만 `_muvo` cfg가 따라가지 않음. 의도적 KL knob은 양쪽 모두 `KL_FREE_BITS=1.0`뿐.
- **Unpaired 파일은 `_muvo`에 영향 없음.** `train_ssim.py:11-14`가 baseline `train.WorldModelTrainer = SSIMWorldModelTrainer` monkey-patch — `_muvo` run에 무관. `trainer_ssim.py`는 baseline subclass. `find_lr.py`/`viz_from_ckpt.py`는 CLI utility. 4개 파일 모두 삭제해도 `_muvo` 페어는 build & train OK.

---

## 4. 4-way 종합

### 4.0 병렬 탐색에서 새로 드러난 발견

4개 출처를 *동시에* 비교하지 않았다면 보이지 않았을, 본 작업에서 처음 surface된 3가지:

1. **`/muvo_2d/`는 modality drop fork가 *아니라* 2D-latent 브랜치 + voxel loss가 silent로 꺼져 있다.**
   - 기존 alpha 메모 (`three_way_comparison.md`, `muvo_style_alpha_model_notes.md`)는 upstream을 단일 "MUVO C"로만 다뤘기에 1D main vs 2D branch 구분이 없었다. §1.3, §1.6 참조.
   - `muvo_2d/muvo/trainer.py:192-212, 405-412`에서 `voxel_loss`, `sem_scal_loss`, `dice_loss` 전부 주석 처리. `VOXEL_SEG.ENABLED=True`라도 voxel head는 shared latent의 reconstruction gradient만 받음 → `muvo_2d`를 "2D 페이퍼 검증 시작점"으로 copy하면 default로 voxel supervision이 빠져 있는 상태로 출발하게 된다. `DiceLoss`가 새로 정의·import되었음에도 wiring이 commented이라 work-in-progress로 보이며, 외부 사용 시 명시적으로 uncomment 필요.

2. **alpha `_muvo`의 `LIDAR_SCALE`이 loss / eval metric / figure 사이에서 단위 일관성 없음 — silent footgun.**
   - `data_muvo.py:199`가 LiDAR RV를 40으로 divide → `trainer_muvo.py:106, 417-424`는 figure / eval metric에서 `× lidar_scale`로 보정 → 그러나 `trainer_muvo.py:471-509`의 LiDAR L1/MSE/smooth-L1 loss는 `trainer.py:414-440`에서 verbatim 상속되어 **보정 없음**.
   - 현재 `WEIGHT_LIDAR_RE=0.0`이라 dormant. `muvo_default_config_audit.md`가 권하는 #1 fix("`WEIGHT_LIDAR_RE` 0.0→0.1 복원")를 실행하는 순간 train scalar `lidar_xyz_*`/`lidar_depth_*`/`lidar_empty_depth_*`가 baseline 대비 ~1/40 (L1) ~ 1/1600 (MSE) 배로 보이고, `results/experiment_log.csv`가 unit을 silently mix하게 된다.
   - 결정 옵션: (a) `data_muvo.py:199`의 divide를 제거하고 trainer의 multiply-back도 동시에 제거, 또는 (b) `trainer_muvo.compute_loss`의 LiDAR 항에 `× lidar_scale` 곱셈 추가. (a)가 변경 면적이 작고 trainer11 baseline과 호환.

3. **baseline `config.py`에 `EPOCHS` 필드 누락 — latent `AttributeError`.**
   - `train.py:203`이 `max_epochs=cfg.EPOCHS`를 읽지만 `config.py`에는 `EPOCHS` 키 없음 → `--epochs` 없이 baseline 실행 시 raise. `config_muvo.py:82`가 `EPOCHS=1`로 조용히 우회 중이라 `_muvo` 브랜치에서는 발현 안 됨.
   - one-line fix 가능 (`config.py`에 `EPOCHS=1` 추가). `_muvo`의 책임 아니지만 4-way diff에서 surface됨.

### 4.1 Repo/branch 지형도

```
upstream MUVO (bogdoll/MUVO)
├── main (1D RSSM)                    → /home/carol/chaeyeon-kim/muvo/
└── 2D   (2D-token RSSMTD + perceptual + near-field) → /home/carol/chaeyeon-kim/muvo_2d/

alpha26 (trainer11 line)
├── baseline (config.py, models.py, trainer.py, …)        ← trainer11 reviewed notebook
└── _muvo    (config_muvo.py, models_muvo.py, trainer_muvo.py, …)
              ← upstream main의 sensor-fusion 구조를 축소 이식
                (BEV/3D/policy/route drop, 1D RSSM 유지)
                현재 페이퍼의 2D 헤드라인 변형은 미적용
```

### 4.2 어디서 baseline `_muvo`로의 다음 변경을 가져올 수 있는가

- **페이퍼가 "largest boost"라고 한 2D latent state**를 alpha에 들이려면 `muvo_2d/muvo/models/transition_td.py` (`RSSMTD`, `RepresentationModelTD`, `ConvGRUCellGlo`)와 `muvo_2d/muvo/models/common.py:681-731` (`PositionEmbeddingSine3D`)를 참조. alpha `_muvo`의 token count는 image 240 + LiDAR 32 = 272로 muvo_2d의 837(image+lidar+voxel+policy)보다 훨씬 작아 query embedding 그리드도 조정 필요.
- **PerceptualLoss**는 `muvo_2d/muvo/losses.py:419-471`. `ssim_addition_notes.md`의 verdict는 후순위지만, `WEIGHT_LIDAR_RE` 복원 이후 다음 단계 후보.
- **Near-field eval metric**은 alpha `_muvo`가 voxel head를 안 가지므로 voxel IoU 변형은 그대로 못 가져오지만 **Chamfer near-field 변형** (`muvo_2d/muvo/metrics.py:253-289`)은 그대로 적용 가능. PC range만 alpha의 LiDAR coverage에 맞춤.
- **Augmentation 부재**가 blur fix priority 2번. `muvo`/`muvo_2d` 양쪽 모두 같은 augmentation pipeline (blur/sharpen/ColorJitter, route augmentation). alpha는 route 안 쓰니 image augmentation 부분만 포팅.

### 4.3 alpha `_muvo` 현 상태 ↔ upstream 1D / 2D / alpha baseline 한눈 표

| 항목 | upstream main (muvo) | upstream 2D (muvo_2d) | alpha baseline | alpha `_muvo` |
|---|---|---|---|---|
| RSSM | 1D, `Mile.RSSM` | 1D 또는 2D-token `RSSMTD` (gated) | 1D | 1D (`action_in_gru` toggle) |
| Transformer fusion | 6-layer, ch 384 | 6-layer, ch 384 | 3-layer, ch 128 | 3-layer, ch 256 |
| Image encoder | ResNet18 / MobileViT | ResNet18 / MobileViT | ResNet18 | ResNet18 |
| LiDAR encoder | ResNet18 (xyzd RV) / PointPillars | 동일 | ResNet18 (xyzd RV) | ResNet18 (xyzd RV) |
| RGB target | 320×832 | 320×832 | 216×288 (pre `7099f87`) / 320×768 | 320×768 |
| RGB loss | L1 multi-scale (factor 1,2,4) | L1 + SSIM(0.3) + Perceptual(opt) | L1 multi-scale | L1 multi-scale + (opt) SSIM |
| LiDAR loss | xyz L2 + range L1 + empty L1, w=0.1 | 동일 | xyz/depth/empty L1 (`WEIGHT_LIDAR_RE`) | 동일 framework, **default `WEIGHT_LIDAR_RE=0.0`** |
| Voxel head | ON (`WEIGHT_VOXEL=0.1`, SCAL loss) | ON 예측 + ConvDecoder3D, **loss 주석 처리** | OFF | OFF (`_muvo` 정책) |
| BEV head | OFF (페이퍼 의도) | OFF | OFF | OFF |
| Policy/Route | ON | ON | OFF | OFF |
| Image augmentation | ON (60% blur/sharpen, 30% ColorJitter, …) | ON | OFF | OFF |
| KL weight | 1e-3 | 1e-3 | 1e-2 + free-bits 1.0 | 1e-2 + free-bits 1.0 |
| Free-bits | 없음 | 없음 | 1.0 | 1.0 |
| Sequence | RF=4, FH=2 | RF=4, FH=2 | RF=4, FH=4 | RF=4, FH=4 |
| Data source | CARLA on-the-fly | 동일 | Arrow offline | Arrow offline |
| LiDAR sensor | 64-ch | 64-ch | 32-ch | 32-ch |
| LiDAR RV size | (64, 1024) | 동일 | (32, 256) | (32, 1024) |
| LiDAR data scaling | raw meter | raw meter | raw meter | divide by 40 (LIDAR_SCALE) |
| Decoder pattern | upstream ConvDecoder (k=5,6/s=2 + op) | 동일 + ConvDecoder2D/3D | 구 `SensorDecoder` (k=4/s=2 + bilinear) | upstream ConvDecoder 패턴 |
| Near-field metric | 없음 | Chamfer + voxel IoU near-field | 없음 | 없음 |

### 4.4 가장 우선순위 높은 actionable item

`muvo_default_config_audit.md`의 정정된 우선순위를 본 4-way 비교로 강화:

1. **`WEIGHT_LIDAR_RE`를 0.0 → 0.1**. Head 이미 구현, audit가 #1로 지목. **단 LIDAR_SCALE footgun 해결을 동시에**: `trainer_muvo.compute_loss`의 LiDAR loss 항에 `× lidar_scale` 곱셈을 추가하거나, `data_muvo.py`의 divide를 제거하고 trainer의 figure/metric 보정을 제거(둘 중 하나로 unit 일관성).
2. **Image augmentation 포팅** — `muvo`/`muvo_2d` 둘 다 동일. `_muvo` 정책에 안 어긋남.
3. **EPOCHS field**를 baseline `config.py`에도 추가 (one-line, latent bug fix).
4. KL weight를 `1e-2 → 1e-3`. Free-bits가 alpha에 있고 upstream에 없는 점 고려해 단계적.
5. 페이퍼 헤드라인 검증을 위해 `RSSMTD` 포팅 — medium effort, `muvo_2d/muvo/models/transition_td.py`가 reference.
6. (후순위) PerceptualLoss, SSIM은 `ssim_addition_notes.md` 정정에 따라 마지막.

### 4.5 알려진 footgun 요약

- **alpha `_muvo` LIDAR_SCALE unit mix** — loss vs metric vs figure 사이 일관성 없음. LiDAR weight 재활성 직전 fix 필수.
- **alpha baseline `EPOCHS` 누락** — `--epochs` 없이 baseline 실행 시 `AttributeError`.
- **`muvo/data_collect.py` 깨짐** — `agents/` package 부재로 import fail. `muvo_2d/agents/`에서 layout 가져오거나 import path 조정.
- **`muvo_2d/prediction.py`는 stock PL 외 속성 의존 + pinned ckpt 필요** — 그대로 실행 가능한 entrypoint가 아님.
- **`muvo_2d/muvo/trainer.py`의 voxel loss 주석 처리** — `VOXEL_SEG.ENABLED=True`라도 voxel head는 shared latent를 통한 reconstruction gradient만 받음. starting point로 copy 시 silent issue.
- **alpha `_muvo` ckpt 호환성** — baseline ckpt (FPN out 128)를 `_muvo` `WorldModelTrainer`에 load 시 `EMBEDDING_DIM=256` build로 FPN conv shape mismatch.
- **`SSIMLoss.window`가 buffer 미등록** — GPU 이동 시 매 forward에 host→device copy. `trainer_muvo.py`와 `trainer_ssim.py` 양쪽 동일.
- **`SSIMLoss` 두 사본** — `trainer_muvo.py:17-65`와 `trainer_ssim.py:17-63`. 한쪽 패치 시 drift.
- **`EV_POSITION` 비대칭** — `muvo`의 `[32, 96, 12]`로 학습된 ckpt에 `muvo_2d`의 `[48:144]` near-field crop을 적용하면 비대칭. ckpt와 함께 EV position 정합 검증.
