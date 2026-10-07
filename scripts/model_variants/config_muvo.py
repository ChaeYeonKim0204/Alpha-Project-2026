import os
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(os.environ.get("ALPHA26_ROOT", "/home/carol/chaeyeon-kim/alpha26")).expanduser()
DATA_ROOT = Path(os.environ.get("CARLA_ARROW_ROOT", str(PROJECT_ROOT.parent / "processed"))).expanduser()

cfg = SimpleNamespace(
    RECEPTIVE_FIELD=4,
    FUTURE_HORIZON=4,

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
        SAMPLE_EVERY_N=2,
        FRAME_STEP=5,
        NUM_WORKERS=2,
    ),

    MODEL=SimpleNamespace(
        ACTION_DIM=2,
        EMBEDDING_DIM=256,
        SPEED_CHANNELS=16,
        SPEED_NORMALISATION=50.0,
        FUSION=SimpleNamespace(
            TRANSFORMER_CHANNELS=256,
            TRANSFORMER_LAYERS=3,
            TRANSFORMER_HEADS=8,
            TRANSFORMER_DROPOUT=0.1,
        ),
        TRANSITION=SimpleNamespace(
            ENABLED=True,
            HIDDEN_STATE_DIM=512,
            STATE_DIM=256,
            ACTION_LATENT_DIM=64,
            USE_DROPOUT=False,
            DROPOUT_PROBABILITY=0.0,
            # True (default, alpha 디자인): pre_gru에 [sample, action] concat. 표준 Dreamer.
            # False (upstream muvo): pre_gru에 sample만. action은 GRU 이후 prior에만.
            ACTION_IN_GRU=True,
        ),
    ),

    LOSSES=SimpleNamespace(
        WEIGHT_PROBABILISTIC=1e-2,
        KL_FREE_BITS=1.0,
        KL_BALANCING_ALPHA=0.75,
        WEIGHT_LIDAR_RE=0.0,
        WEIGHT_LIDAR_EMPTY=0.0,
        WEIGHT_RGB=1.0,
        WEIGHT_SSIM=0.0,
        WEIGHT_FUTURE=1.0,
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
        RUN_NAME="muvo_style_alpha_fpn256_lidar32",
        BASE_FILE="train_muvo.py",
        EXPERIMENT_LOG_PATH=str(PROJECT_ROOT / "results" / "experiment_log.csv"),
        SERIES="",  # experiment_log.csv의 'series' 컬럼 값. e.g. "muvo_branch_post_migration".
    ),

    EPOCHS=1,
    STEPS=400,
    PRETRAINED=SimpleNamespace(
        PATH=None,
    ),
)
