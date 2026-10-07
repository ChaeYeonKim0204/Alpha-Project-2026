# MUVO 2D (RSSMTD) 코드 리뷰 수정 사항 — 2026-05-25

`scripts/model_variants/`의 MUVO 2D-token RSSM 파이프라인(`config_muvo_2D.py`,
`data_muvo_2D.py`, `models_muvo_2D.py`, `train_muvo_2D.py`, `trainer_muvo_2D.py`,
`viz_from_ckpt.py`)에 대한 코드 리뷰에서 발견한 문제와 수정 내역.

대상 파일:
- `scripts/model_variants/config_muvo_2D.py`
- `scripts/model_variants/data_muvo_2D.py`
- `scripts/model_variants/models_muvo_2D.py`
- `scripts/model_variants/train_muvo_2D.py`
- `scripts/model_variants/trainer_muvo_2D.py`
- `scripts/model_variants/viz_from_ckpt.py`

---

## 1. Validation/데이터 분할 (#1, #2)

- **#1**: `build_dataloaders`에서 `val_rl_ds`/`val_ds_ds`가 `train_ds`를 재사용하던 것을
  `cfg.DATA.VAL_RL_RUN`/`VAL_DS_RUN`(manifest의 `validation` split)에서 만든 **held-out
  validation dataset**으로 변경. overfit/full 모드와 무관하게 항상 validation을 사용.
- **#2**: full-training 경로의 `train set 1/4 축소`(`reduction_factor=4`) 제거 →
  전체 train set 사용 (overfit subset 로직은 유지).
- validation dataset은 `augment=False`로 생성.

## 2. `--steps`/`--epochs` 의미 명확화 (#3)

- 학습 길이는 `Trainer(max_epochs=cfg.EPOCHS)`로만 제어됨(`max_steps` 미설정 = 무제한).
- `--steps`/`cfg.STEPS`는 `OneCycleLR.total_steps`의 fallback(estimated_stepping_batches<=0)
  으로만 쓰이며 학습 길이를 제어하지 않음. 혼동 방지용 help/주석 추가 (동작 변경 없음).

## 3. Action 예측 제거 (요청 사항)

alpha 데이터는 `[throttle, steer]`만 있고 brake가 없어 MUVO의 signed `throttle_brake`를
재현할 수 없음. action을 **RSSM 입력으로만** 쓰고 예측 헤드는 제거 (Scope B).

- `PolicyDecoder` 클래스, `action_pred` 출력, action loss, `LOSSES.WEIGHT_ACTION`,
  `MODEL.POLICY` namespace, `RSSM_2D.POLICY_TOKENS` 제거.
- `RSSMTD`: policy token 제거 → `n_out_tokens = image(240) + lidar(128) = 368`
  (기존 369). `policy_tokens` 배선 전체 삭제.
- `RepresentationModelTD`: policy query 제거 (image/lidar 2-modality query),
  `type_embeddings` 3 slot → 2 slot.
- `_slice_token_state`/`_decode_state`: image/lidar만 디코드.
- ClearML connect/CSV/figure/change_summary의 action 항목 제거.
- action은 `prepare_custom_batch`·`forward`·`imagine(future_actions)`·RSSM
  `prior/posterior_action_module`에서 **입력으로 그대로 유지**.

## 4. Augmentation (#8, #9)

- **#8**: `--no-augmentation`이 `run_cfg`(deepcopy)만 바꾸고, 실제 transform은
  `data_muvo_2D.py`의 **module-level `cfg`**를 읽어 train augmentation이 안 꺼지던 버그.
  → `aug_cfg`를 `_make_img_transform`/`MUVODataset`에 **명시 인자로 전달**(Approach 2).
  `build_dataloaders`가 `run_cfg.DATA.AUGMENTATION`과 `train_augment(=ENABLED)`를 전달.
- **#9**: validation augmentation은 `augment=False`로 이미 차단됨(#1). val transform이
  `[CenterCrop, ToTensor, Normalize]`로 deterministic임을 확인.

## 5. Checkpoint 경로 / logger version (#10)

- `ModelCheckpoint`의 `dirpath`(`logs/<run>/checkpoints`, 버전 없음) 제거 →
  Lightning이 logger의 versioned dir `logs/<run>/version_N/checkpoints`에 저장.
  같은 `run_name` 재실행 시 다른 실험 checkpoint와 섞이지 않음.
- reconstruction fallback을 `trainer.checkpoint_callback.dirpath`(이번 run) 기준으로 변경.
- `build_callbacks`의 미사용 `save_dir` 파라미터 제거.

## 6. Resume 시 OneCycleLR total_steps (#11)

- `maybe_extend_onecycle_resume_checkpoint`가 `len(train_loader)*EPOCHS`로 total_steps를
  재계산하던 것이 DDP에서 `world_size`배 과대 추정. → `world_size`로 나눠 per-rank 기준
  (`estimated_stepping_batches`와 일치). patched checkpoint를 **rank별 파일**로 저장해
  동시 write 손상 방지.

## 7. LiDAR FOV/scale 설정 분리 (#12)

- `point_cloud_to_range_view`(fov)와 `__getitem__`(LIDAR_SCALE)이 module-level `cfg`를
  직접 읽던 것을, `MUVODataset(lidar_fov, lidar_scale)` 인자로 받아 `run_cfg`에서 전달.
  Chamfer metric(run_cfg scale)과 데이터 정규화(이전 module scale)가 어긋날 위험 제거.

## 8. DDP 중복 기록 (#13, #20)

- **#13**: `ExperimentCSVLogger.on_fit_end`에 `trainer.is_global_zero` 가드 → rank 0만 CSV 기록.
- **#20**: `maybe_init_clearml`에 `LOCAL_RANK/RANK == 0` 가드 → rank 0만 ClearML `Task.init`.

## 9. 평가지표 정직성 (#14, #15, #16)

- **#14**: LiDAR Chamfer가 예측 point를 **모델 자신의 depth>0**에서 뽑아, 어려운 영역을
  depth<=0(empty)으로 예측하면 평가에서 빠져 점수가 낙관적이던 문제. → 예측 point를
  **target-valid 픽셀**에서 추출하도록 변경(숨기기 불가). garbage를 empty로 숨긴 테스트
  케이스에서 Chamfer 24.6(이전 방식 ~0.75)로 페널티 정상 반영 확인.
- **#15**: `--overfit-samples` 기본값 8 → **0**. 기본 실행이 overfit 모드라 validation이
  안 돌고 train으로 checkpoint를 고르던 문제 해결 → 기본 = 전체 train + held-out validation,
  checkpoint는 `val/RL_loss`로 선택. overfit sanity check는 `--overfit-samples N`로 opt-in.
- **#16**: future rollout(prior imagine) 지표가 요약 산출물에서 누락되던 문제. 요약 figure에
  **Future LiDAR Chamfer** 패널 추가(future RGB PSNR은 Camera PSNR 패널에 recon과 함께 표시),
  CSV 요약(`_collect_final_val_metrics`)에 future PSNR·Chamfer 추가. recon/future를 나란히 보고.

## 10. Reconstruction / 시각화 (#17, #18, #19)

- **#18**: 학습 종료 후 reconstruction figure가 항상 `train_ds`를 쓰던 것을, 일반 모드는
  **held-out `val_rl_ds`**, overfit 모드는 `train_ds`(subset)로 선택.
- **#17/#19**: `viz_from_ckpt.py`가 base/trainer11 모듈(`config`/`data`/`trainer`) +
  `embedding_n_channels=128` 하드코딩이라 MUVO 2D ckpt를 잘못된 구조로 로드하던 문제.
  → `*_muvo_2D` 모듈로 교체, **checkpoint hparams의 saved cfg**로 모델·데이터셋을 재구성하고
  dataset의 모든 DATA 값(`lidar_fov`/`lidar_scale` 포함)을 saved cfg에서 명시 전달해
  module-level cfg 혼입 차단. (base/trainer11 viz는 더 이상 지원 안 함.)

---

## 검증

- 6개 파일 모두 `py_compile` 통과 (`kcy-alpha`, torch 2.0.0).
- RSSMTD forward: `n_out_tokens=368`, 2-slot type_embeddings 정상.
- Augmentation: 전달된 `aug_cfg.ENABLED`만 따름(module cfg 무시), val transform deterministic.
- Checkpoint dir: 재실행 시 `version_0`/`version_1`로 분리, logger version과 일치.
- `#11` per-rank total_steps, `#14` Chamfer 회피 불가, `#12` `__getitem__`이 module
  `cfg.DATA.LIDAR_SCALE` 미참조 — 각각 런타임 테스트로 확인.

## 비고

- 로컬 torch 환경: `/home/carol/exit/envs/kcy-alpha/bin/python` (conda root는
  `~/miniconda3`가 아니라 `/home/carol/exit`).
- alpha 데이터 스키마에는 `brake` 컬럼 없음 (`run_id, frame, timestamp, image_front,
  lidar, steer, throttle, speed_kmh`).
