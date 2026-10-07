set -e
source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
cd /home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_WM
CKPT=logs/diffuvo_diff_fulldata_fh4_rebal150/version_4/checkpoints/epoch=93-step=078490.ckpt
RUN=diffuvo_diff_fulldata_fh4_rebal150_eval_ddim50
echo "######## eval rebal150 epoch93 DDIM50 START $(date) ########"
# best val/RL_loss=0.2886 @ epoch 93 (save-top-k=1, monitor val/RL_loss).
# 통합 ckpt(model.*/diffusion.*) → --checkpoint. FH4/RF4는 config 기본값과 일치.
# --split train: stage2 평가 관례. --num-streams 4 = 학습 global batch(4×1).
python src/predict_Diffuvo.py \
  --checkpoint "$CKPT" \
  --ddim-steps 50 --cfg-scale 1.0 \
  --split train --num-streams 4 \
  --sample-indices 0,1,2 \
  --figure-dir figures/$RUN \
  --run-name $RUN
echo "######## DONE rc=$? $(date) ########"
