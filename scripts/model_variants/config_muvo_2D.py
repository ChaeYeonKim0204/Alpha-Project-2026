"""config_muvo_2D.py — MUVO 2D-token RSSM (RSSMTD) variant.

`config_muvo.py`를 베이스로 RSSM_2D namespace, augmentation, action loss,
ClearML, EFFECTIVE_HZ-driven sampling, KL free-bits on/off, recon figure
flags를 추가한다. 자세한 차이는 `alpha26/docs/muvo_2D_port_plan.md` v3 참조.
"""
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PROJECT_ROOT = os.environ.get("ALPHA26_ROOT")
if _ENV_PROJECT_ROOT:
    _ENV_PROJECT_ROOT = Path(_ENV_PROJECT_ROOT).expanduser()
    _ENV_CONFIG_PATH = (_ENV_PROJECT_ROOT / "scripts" / "model_variants" / Path(__file__).name).resolve()
    if _ENV_PROJECT_ROOT.exists() and _ENV_CONFIG_PATH == Path(__file__).resolve():
        PROJECT_ROOT = _ENV_PROJECT_ROOT.resolve()
    else:
        warnings.warn(
            f"ALPHA26_ROOT does not match this script checkout, using script project root instead: "
            f"{_ENV_PROJECT_ROOT}"
        )
        PROJECT_ROOT = _DEFAULT_PROJECT_ROOT
else:
    PROJECT_ROOT = _DEFAULT_PROJECT_ROOT
DATA_ROOT = Path(os.environ.get("CARLA_ARROW_ROOT", str(PROJECT_ROOT.parent / "processed"))).expanduser().resolve()


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


def _resolve_experiment_log_path(project_root):
    # ALPHA26_OUTPUT_ROOT가 있으면 그 아래 experiment_log.csv. 없으면 PROJECT_ROOT/results 기본값.
    output_root = os.environ.get("ALPHA26_OUTPUT_ROOT")
    if output_root:
        return str(Path(output_root).expanduser().resolve() / "experiment_log.csv")
    return str(project_root / "results" / "experiment_log.csv")

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
        LIDAR_SCALE=40.0,  # alpha LiDAR range=80m / 2.0 = 40. MUVO(range=100m, SCALE=50 → [0,2]) 정규화 *특성* 정렬.
        #                  센서 사거리가 다르므로(80 vs 100m) 상수(50) 복사가 아니라 80/2로 맞춤. 5/25 50 변경은 상수 복사 착오 → revert.
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
            ENABLED=True,
            # RSSMTD는 hidden=state=embedding 단일 채널 가정. v2: 256/256 (was 512/256).
            HIDDEN_STATE_DIM=256,
            STATE_DIM=256,
            ACTION_LATENT_DIM=64,  # unused by RSSMTD (action_module 자체적으로 hidden_state_dim으로 mapping)
            # MUVO 재현: scheduled sampling — 학습 중 15% 확률로 posterior 대신 prior sample을
            # forward에 흘려 미래 rollout 강건화 (nn.Dropout이 아님).
            USE_DROPOUT=True,
            DROPOUT_PROBABILITY=0.15,
            ACTION_IN_GRU=True,  # RSSMTD에서는 unused. 키 유지 (호환성).
        ),
        # 2D-token RSSM 전용 (v3 신규).
        RSSM_2D=SimpleNamespace(
            IMAGE_TOKEN_HW=(10, 24),       # FPN stride-32 of (320, 768)
            LIDAR_TOKEN_HW=(2, 64),        # v3: FPN stride-16 of (32, 1024). upstream MUVO와 매칭.
            LIDAR_OUT_INDICES=(1, 2, 3),   # v3: upstream MUVO LiDAR encoder out_indices.
            TRANSFORMER_DECODER_LAYERS=3,  # v2: encoder와 일치 (was 6)
            TRANSFORMER_DECODER_HEADS=4,   # v2: 64 dim/head
            # v4 (RGB plateau fix): decoder 내부 working channel. upstream `latent_n_channels=512` 매칭.
            # embedding=256이라도 decoder는 더 크게 유지 → 첫 Conv2d(in=512, out=512)로 채널 보존.
            DECODER_CHANNELS=512,
            NEARFIELD_PC_RANGE=(-20.0, -20.0, -2.0, 20.0, 20.0, 6.0),  # [xmin,ymin,zmin,xmax,ymax,zmax]
        ),
    ),

    LOSSES=SimpleNamespace(
        WEIGHT_PROBABILISTIC=1e-3,   # MUVO 재현: upstream과 동일 (was 1e-2, alpha-1D 정렬용).
        KL_FREE_BITS=1.0,
        KL_FREE_BITS_ENABLED=False,  # MUVO 재현: upstream은 free-bits 없음 (was True, alpha-1D 정렬용).
        KL_BALANCING_ALPHA=0.75,
        WEIGHT_LIDAR_RE=0.1,
        WEIGHT_RGB=1.0,
        WEIGHT_SSIM=0.0,
        #WEIGHT_FUTURE=1.0,
    ),

    OPTIMIZER=SimpleNamespace(
        LR=1e-4,
        WEIGHT_DECAY=0.01,
        ACCUMULATE_GRAD_BATCHES=1,
    ),

    SCHEDULER=SimpleNamespace(
        NAME='OneCycleLR',
        PCT_START=0.1,
        # 이전엔 미지정 → torch 기본값(25, 1e4) 사용. final_div_factor=1e4는 종료 LR을
        # max_lr/1e4=1e-8로 만들어 마지막 구간을 dead tail로 만들었음. 아래로 완화.
        DIV_FACTOR=10.0,          # 시작 LR = max_lr / DIV_FACTOR
        FINAL_DIV_FACTOR=1e2,     # 종료 LR = max_lr / FINAL_DIV_FACTOR
    ),

    LOGGING=SimpleNamespace(
        RUN_NAME="muvo_style_alpha_fpn256_lidar128_rssmtd",
        BASE_FILE="train_muvo_2D.py",
        EXPERIMENT_LOG_PATH=_resolve_experiment_log_path(PROJECT_ROOT),
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
