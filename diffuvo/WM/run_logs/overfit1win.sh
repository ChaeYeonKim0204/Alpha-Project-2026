set -e
source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
cd /home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_WM
AE=logs/diffuvo_ae_fulldata_fh4/version_0/checkpoints/last.ckpt
echo "######## 1-window long overfit START $(date) ########"
python src/train_Diffuvo.py --stage 2 --ae-checkpoint $AE \
  --overfit-samples 1 --future-horizon 2 --epochs 5000 \
  --batch-size 1 --devices 1 --save-top-k 1 \
  --no-reconstruction --no-metric-export --no-clearml \
  --run-name diffuvo_overfit1win_long \
  > run_logs/overfit1win.log 2>&1
echo "######## DONE rc=$? $(date) ########"
