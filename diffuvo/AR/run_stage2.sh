#!/usr/bin/env bash
# Diffuvo_AR Stage2 (AR latent diffusion) 학습 런처. tmux 세션 'hj'에서 실행.
# (conda activate가 미설정 변수를 참조하므로 set -u는 쓰지 않는다.)

AR_ROOT=/home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_AR
AE_CKPT=/home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_WM/logs/diffuvo_ae_fulldata_fh4/version_0/checkpoints/last.ckpt
RUN_NAME=run_001_fh2_ar_baseline

source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
set -eo pipefail

cd "$AR_ROOT/src"
export DIFFUVO_LOG_DIR="$AR_ROOT/log"

python -u train_Diffuvo.py --stage 2 \
  --ae-checkpoint "$AE_CKPT" \
  --epochs 40 --batch-size 2 --devices 1 --precision 16-mixed \
  --figure-dir "$AR_ROOT/figure" \
  --run-name "$RUN_NAME" \
  --no-clearml \
  2>&1 | tee "$AR_ROOT/log/${RUN_NAME}.train.log"
