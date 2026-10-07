set -e
source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
cd /home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_WM
AE=logs/diffuvo_ae_fulldata_fh4/version_0/checkpoints/last.ckpt
echo "######## full-data stage2 FH4 OneCycle-150ep + rebalance START $(date) ########"
# A안: OneCycle 유지(--constant-lr 안 줌) + total_steps는 estimated_stepping_batches로
#       150 epoch에 맞춰 자동 산정 → epoch 40 조기 anneal underfit 해소.
# 데이터 리밸런싱: 정지 x0.3 / 직진 x1.0 / turn x3.0 (WeightedRandomSampler).
# 이전 비교군 diffuvo_diff_fulldata_fh4: FH4, batch4, OneCycle, 40ep, train≈val≈0.31(underfit).
python src/train_Diffuvo.py --stage 2 --ae-checkpoint $AE \
  --future-horizon 4 --epochs 150 \
  --batch-size 8 --num-workers 4 --lr 2e-4 --devices 1 \
  --rebalance --w-stationary 0.3 --w-turn 3.0 \
  --stationary-speed 1.0 --turn-steer 0.1 \
  --save-top-k 1 --monitor val/RL_loss --monitor-mode min \
  --no-clearml \
  --run-name diffuvo_diff_fulldata_fh4_rebal150 \
  2>&1 | tee run_logs/fulldata_fh4_rebal150.log
echo "######## DONE rc=${PIPESTATUS[0]} $(date) ########"
