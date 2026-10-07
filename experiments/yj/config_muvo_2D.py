"""config_muvo_2D.py — MUVO 2D-token RSSM (RSSMTD) variant.

`config_muvo.py`를 베이스로 RSSM_2D namespace, augmentation, action loss,
ClearML, EFFECTIVE_HZ-driven sampling, KL free-bits on/off, recon figure
flags를 추가한다. 자세한 차이는 `alpha26/docs/muvo_2D_port_plan.md` v3 참조.
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
    FUTURE_HORIZON=2,  # v2: upstream과 일치 (was 4). seq_len = 6.

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
        # TransFuser trajectory를 이미지에 투영할 때 쓰는 front-camera calibration.
        # 실제 CARLA 센서 rig 값에 맞춰야 정확. (data_collect 카메라 설정 참조)
        CAMERA=SimpleNamespace(
            FOV_DEG=90.0,      # 수평 시야각
            HEIGHT_M=1.6,      # 지면 위 카메라 높이
            X_OFFSET_M=1.2,    # ego 원점 대비 전방 장착 오프셋
            PITCH_DEG=0.0,     # 아래로 +
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
            ENABLED=True,
            # RSSMTD는 hidden=state=embedding 단일 채널 가정. v2: 256/256 (was 512/256).
            HIDDEN_STATE_DIM=256,
            STATE_DIM=256,
            ACTION_LATENT_DIM=64,  # unused by RSSMTD (action_module 자체적으로 hidden_state_dim으로 mapping)
            USE_DROPOUT=False,
            DROPOUT_PROBABILITY=0.0,
            ACTION_IN_GRU=True,  # RSSMTD에서는 unused. 키 유지 (호환성).
        ),
        # 2D-token RSSM 전용 (v3 신규).
        RSSM_2D=SimpleNamespace(
            IMAGE_TOKEN_HW=(10, 24),       # FPN stride-32 of (320, 768)
            LIDAR_TOKEN_HW=(2, 64),        # v3: FPN stride-16 of (32, 1024). upstream MUVO와 매칭.
            LIDAR_OUT_INDICES=(1, 2, 3),   # v3: upstream MUVO LiDAR encoder out_indices.
            POLICY_TOKENS=1,
            TRANSFORMER_DECODER_LAYERS=3,  # v2: encoder와 일치 (was 6)
            TRANSFORMER_DECODER_HEADS=4,   # v2: 64 dim/head
            # v4 (RGB plateau fix): decoder 내부 working channel. upstream `latent_n_channels=512` 매칭.
            # embedding=256이라도 decoder는 더 크게 유지 → 첫 Conv2d(in=512, out=512)로 채널 보존.
            DECODER_CHANNELS=512,
            NEARFIELD_PC_RANGE=(-20.0, -20.0, -2.0, 20.0, 20.0, 6.0),  # [xmin,ymin,zmin,xmax,ymax,zmax]
        ),
        # PolicyDecoder + action loss (v2, Q6/Q9).
        POLICY=SimpleNamespace(
            ENABLED=True,
            # in_channels는 cfg.MODEL.EMBEDDING_DIM 자동 사용. upstream 코드 그대로.
        ),
        # TransFuser-style trajectory head (additive; 기존 decoder/RSSM 그대로 유지).
        # 융합 latent → GRU autoregressive waypoint decoder.
        WAYPOINT=SimpleNamespace(
            ENABLED=True,
            N_WAYPOINTS=2,         # 예측 future waypoint 수 (default = FUTURE_HORIZON)
            GRU_HIDDEN=64,         # TransFuser waypoint GRU hidden size
            # ── kinematic GT 파라미터 (pose 컬럼 부재 → speed+steer로 근사 생성) ──
            DT=0.5,                # 샘플 간 시간 간격 [s] = 1/EFFECTIVE_HZ
            WHEELBASE=2.85,        # [m] CARLA 차량 wheelbase 근사
            MAX_STEER_RAD=1.221,   # [rad] ≈ 70° front-wheel 최대 조향각 근사
            Y_RIGHT_POSITIVE=True, # ego +y가 오른쪽(CARLA 관례)인지
        ),
    ),

    LOSSES=SimpleNamespace(
        WEIGHT_PROBABILISTIC=1e-2,
        KL_FREE_BITS=1.0,
        KL_FREE_BITS_ENABLED=True,   # v4-2: alpha _muvo 1D 베이스라인과 정렬 (controlled comparison).
                                     # upstream MUVO는 free-bits 없으나 alpha 1D는 항상 ON으로 학습됨.
        KL_BALANCING_ALPHA=0.75,
        WEIGHT_LIDAR_RE=0.0,
        WEIGHT_LIDAR_EMPTY=0.0,
        WEIGHT_RGB=1.0,
        WEIGHT_SSIM=0.0,
        WEIGHT_FUTURE=1.0,
        WEIGHT_ACTION=1.0,  # v2: action loss 도입 (Q9). upstream과 동일.
        WEIGHT_WAYPOINT=1.0,  # TransFuser trajectory(waypoint) L1 loss 가중치.
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

    LOGGING=SimpleNamespace(
        RUN_NAME="muvo_style_alpha_fpn256_lidar128_rssmtd",
        BASE_FILE="train_muvo_2D.py",
        EXPERIMENT_LOG_PATH=str(PROJECT_ROOT / "results" / "experiment_log.csv"),
        SERIES="",
        # Reconstruction figure 설정 (v3, Q18).
        RECON_FIG_ALL_FRAMES=True,    # RF+FH=6 frame 전체 column 표시
        RECON_FIG_BOTH_VIEWS=True,    # LiDAR scale 보정·미보정 두 row
    ),

    # ClearML 실험 트래킹 (v3, Q14/Q15).
    CML_ENABLED=True,
    CML_PROJECT="alpha26_muvo_2D",
    CML_TASK="muvo_2D_default",
    CML_TYPE="training",
    CML_TAGS=("rssmtd", "token-shape", "alpha26"),

    EPOCHS=1,
    STEPS=400,
    PRETRAINED=SimpleNamespace(
        PATH=None,
    ),
)
