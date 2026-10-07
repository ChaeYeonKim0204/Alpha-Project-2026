set -e
source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
cd /home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_AR
CKPT=log/run_001_fh2_ar_baseline/version_0/checkpoints/last.ckpt
RUN=run_001_fh2_ar_baseline_eval_ddim50
echo "######## eval AR baseline epoch39 DDIM50 START $(date) ########"
# last.ckpt = epoch 39 / step 149,120 (max_epochs=40 완주, train_loss_epoch=0.308).
# 통합 ckpt(model.*/diffusion.*) → --checkpoint. FH2/RF4는 AR config 기본값과 일치.
# --split train: stage2 평가 관례. --num-streams 2 = 학습 global batch(2×1).
python src/predict_Diffuvo.py \
  --checkpoint "$CKPT" \
  --ddim-steps 50 --cfg-scale 1.0 \
  --split train --num-streams 2 \
  --sample-indices 0,1,2 \
  --figure-dir figure/$RUN \
  --run-name $RUN
echo "######## DONE rc=$? $(date) ########"
