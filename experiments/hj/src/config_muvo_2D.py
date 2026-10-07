"""config_muvo_2D.py — MUVO 2D-token latent-diffusion 미래 생성 config.

원래 RSSMTD(transition) 베이스에서 dynamics를 latent diffusion으로 교체한 변형이다.
인코더/디코더/데이터 설정은 유지하고, MODEL.DIFFUSION / PREDICTION / LOSSES.WEIGHT_DIFFUSION
을 추가했다. TRANSITION/KL_* 키는 호환을 위해 잔존하나 dynamics에는 미사용.
설계·근거: `alpha26/archive/hj/DIFFUSION_FUTURE_DESIGN.md`.
"""
import os
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(os.environ.get("ALPHA26_ROOT", "/home/carol/chaeyeon-kim/alpha26")).expanduser()
DATA_ROOT = Path(os.environ.get("CARLA_ARROW_ROOT", str(PROJECT_ROOT.parent / "processed"))).expanduser()


def _derive_sample_every_n(carla_hz, frame_step, effective_hz):
    """CARLA 20Hz → Arrow (every FRAME_STEP) → effective_hz training rate."""
    denom = frame_step * effective_hz
    if denom <= 0:
        raise ValueError(f"frame_step * effective_hz must be > 0, got {denom}")
    if carla_hz % denom != 0:
        raise ValueError(
            f"carla_hz={carla_hz} not divisible by frame_step({frame_step}) * effective_hz({effective_hz})={denom}. "
            f"Pick effective_hz from divisors of {carla_hz // frame_step} (e.g. 1, 2, 4)."
        )
    return carla_hz // denom


_CARLA_HZ = 20
_FRAME_STEP = 5
_EFFECTIVE_HZ = 2  # default 2Hz training rate. override with --effective-hz.

cfg = SimpleNamespace(
    RECEPTIVE_FIELD=4,
    FUTURE_HORIZON=4,  # 미래 4프레임 생성. seq_len = RF+FH = 8. (v2에서 2로 줄였다가 복원)

    DATA=SimpleNamespace(
        ROOT=str(DATA_ROOT),
        MANIFEST_PATH=str(DATA_ROOT / "arrow_manifest.pkl"),
        TRAIN_RUN="train_run_002.arrow",
        VAL_RL_RUN="validation_run_008.arrow",
        VAL_DS_RUN="validation_run_024.arrow",
        USE_ALL_TRAIN_RUNS=True,
        IMAGE_INPUT_SIZE=(320, 768),
        RGB_RECON_SIZE=(320, 768),
        LIDAR_RANGE_VIEW_SIZE=(32, 1024),
        LIDAR_FOV_DEGREES=(-30.0, 10.0),
        LIDAR_SCALE=40.0,
        # Effective sampling Hz (v3, Q13). 다른 값 원하면 --effective-hz로 override.
        CARLA_HZ=_CARLA_HZ,
        FRAME_STEP=_FRAME_STEP,
        EFFECTIVE_HZ=_EFFECTIVE_HZ,
        SAMPLE_EVERY_N=_derive_sample_every_n(_CARLA_HZ, _FRAME_STEP, _EFFECTIVE_HZ),
        NUM_WORKERS=2,
        # Image augmentation (v2, Q11). 단일 ENABLED flag로 on/off.
        AUGMENTATION=SimpleNamespace(
            ENABLED=True,
            BLUR_PROB=0.3,
            BLUR_WINDOW=5,
            BLUR_STD=(0.1, 1.7),
            SHARPEN_PROB=0.3,
            SHARPEN_FACTOR=(1.0, 5.0),
            COLOR_PROB=0.3,
            BRIGHTNESS=0.3,
            CONTRAST=0.3,
            SATURATION=0.3,
            HUE=0.1,
        ),
    ),

    MODEL=SimpleNamespace(
        ACTION_DIM=2,
        EMBEDDING_DIM=256,
        SPEED_CHANNELS=16,
        SPEED_NORMALISATION=50.0,  # alpha km/h 기준. upstream m/s 5.0과 다름.
        FUSION=SimpleNamespace(
            TRANSFORMER_CHANNELS=256,
            TRANSFORMER_LAYERS=3,
            TRANSFORMER_HEADS=4,  # v2: d_model=256 + nhead=4 → 64 dim/head (upstream convention 매칭)
            TRANSFORMER_DROPOUT=0.1,
        ),
        TRANSITION=SimpleNamespace(
            # NOTE: RSSMTD는 제거되고 latent diffusion으로 대체됨 (DIFFUSION_FUTURE_DESIGN.md).
            # 아래 키는 인코더 채널 정렬·실험로그 호환을 위해 유지되며 dynamics에는 미사용.
            ENABLED=True,
            HIDDEN_STATE_DIM=256,
            STATE_DIM=256,
            ACTION_LATENT_DIM=64,
            USE_DROPOUT=False,
            DROPOUT_PROBABILITY=0.0,
            ACTION_IN_GRU=True,
        ),
        # ── Latent diffusion dynamics (RSSMTD 대체). DIFFUSION_FUTURE_DESIGN.md §6 ──
        DIFFUSION=SimpleNamespace(
            ENABLED=True,
            TRAIN_DIFFUSION=True,        # False면 미래 diffusion 끄고 reconstruction만 학습 (--no-diffusion)
            NUM_TRAIN_STEPS=1000,        # T: 학습 noise step 수
            NUM_SAMPLING_STEPS=50,       # DDIM 역확산 step 수 (추론 기본)
            SCHEDULE="cosine",           # 현재 cosine만 구현
            PARAMETERIZATION="v",        # {"v","eps"}
            DEPTH=6,                     # DiT 블록 수
            HEADS=8,                     # attention head
            MLP_RATIO=4,                 # FFN 확장 비율
            P_UNCOND=0.1,                # CFG 학습: action 조건 드롭 확률 (§2.5)
            LATENT_NORM="ema",           # {"ema","fixed","none"} latent 정규화
            LATENT_NORM_MOMENTUM=0.99,
            LATENT_SCALE=1.0,            # LATENT_NORM=="fixed"일 때 std
            FREEZE_ENCODER_FOR_DIFFUSION=False,  # True면 2단계 학습(encoder freeze)
        ),
        # 2D-token transformer 전용 (image-only).
        RSSM_2D=SimpleNamespace(
            IMAGE_TOKEN_HW=(10, 24),       # FPN stride-32 of (320, 768)
            POLICY_TOKENS=1,
            TRANSFORMER_DECODER_LAYERS=3,  # v2: encoder와 일치 (was 6)
            TRANSFORMER_DECODER_HEADS=4,   # v2: 64 dim/head
            # decoder 내부 working channel. upstream `latent_n_channels=512` 매칭.
            DECODER_CHANNELS=512,
        ),
        # PolicyDecoder + action loss (v2, Q6/Q9).
        POLICY=SimpleNamespace(
            ENABLED=True,
            # in_channels는 cfg.MODEL.EMBEDDING_DIM 자동 사용. upstream 코드 그대로.
        ),
    ),

    LOSSES=SimpleNamespace(
        # Diffusion 손실 (KL probabilistic 대체).
        WEIGHT_DIFFUSION=1.0,
        WEIGHT_FUTURE_DECODE=0.0,   # z0_hat 디코딩 후 미래 GT와 비교 (Phase 2에서 ↑)
        # 아래 KL_* / WEIGHT_PROBABILISTIC은 diffusion 경로에서 미사용 (deprecated, 호환 유지).
        WEIGHT_PROBABILISTIC=1e-2,
        KL_FREE_BITS=1.0,
        KL_FREE_BITS_ENABLED=True,   # v4-2: alpha _muvo 1D 베이스라인과 정렬 (controlled comparison).
                                     # upstream MUVO는 free-bits 없으나 alpha 1D는 항상 ON으로 학습됨.
        KL_BALANCING_ALPHA=0.75,
        WEIGHT_RGB=1.0,
        WEIGHT_SSIM=0.0,
        WEIGHT_FUTURE=1.0,
        WEIGHT_ACTION=1.0,  # v2: action loss 도입 (Q9). upstream과 동일.
    ),

    OPTIMIZER=SimpleNamespace(
        LR=1e-4,
        WEIGHT_DECAY=0.01,
        ACCUMULATE_GRAD_BATCHES=1,
    ),

    SCHEDULER=SimpleNamespace(
        NAME='OneCycleLR',
        PCT_START=0.1,
    ),

    # Counterfactual 미래 생성 검증/시각화 (DIFFUSION_FUTURE_DESIGN.md §5).
    PREDICTION=SimpleNamespace(
        N_SAMPLES=4,    # noise 다양성 검증 K (부차)
        DDIM_STEPS=50,  # 검증 샘플링 step
        GUIDANCE_SCALES=(0.0, 1.0, 2.0, 4.0),  # CFG w-sweep (핵심)
        CF_ACTIONS="default",  # probe action 세트: 직진/좌/우/급제동 (trainer에서 합성)
    ),

    LOGGING=SimpleNamespace(
        RUN_NAME="muvo_2D_diffusion_future",
        BASE_FILE="train_muvo_2D.py",
        EXPERIMENT_LOG_PATH=str(PROJECT_ROOT / "results" / "experiment_log.csv"),
        SERIES="",
        # Reconstruction figure 설정.
        RECON_FIG_ALL_FRAMES=True,    # RF+FH=8 frame 전체 column 표시 (미래 4 column 포함)
        RECON_FIG_PER_FUTURE_FRAME=False,  # 미래 FH 프레임을 각각 독립 디코딩해 시점별 PNG 1장씩 저장
    ),

    # ClearML 실험 트래킹 (v3, Q14/Q15).
    CML_ENABLED=True,
    CML_PROJECT="alpha26_muvo_2D",
    CML_TASK="muvo_2D_diffusion_future",
    CML_TYPE="training",
    CML_TAGS=("diffusion", "multi-future", "token-shape", "alpha26"),

    EPOCHS=1,
    STEPS=400,
    PRETRAINED=SimpleNamespace(
        PATH=None,
    ),
)
