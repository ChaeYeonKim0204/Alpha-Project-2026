set -e
source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
cd /home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_WM
AE=logs/diffuvo_ae_fulldata_fh4/version_0/checkpoints/last.ckpt
echo "######## 16-window full-batch constLR START $(date) ########"
python src/train_Diffuvo.py --stage 2 --ae-checkpoint $AE \
  --overfit-samples 16 --batch-size 16 --future-horizon 2 \
  --lr 2e-4 --constant-lr --epochs 8000 \
  --devices 1 --save-top-k 0 \
  --no-reconstruction --no-metric-export --no-clearml \
  --run-name diffuvo_overfit16_constlr2e4 \
  > run_logs/overfit16_constlr.log 2>&1
echo "######## DONE rc=$? $(date) ########"
