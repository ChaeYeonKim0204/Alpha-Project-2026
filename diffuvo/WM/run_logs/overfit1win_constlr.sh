set -e
source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
cd /home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_WM
AE=logs/diffuvo_ae_fulldata_fh4/version_0/checkpoints/last.ckpt
echo "######## 1-window constLR overfit START $(date) ########"
# A/B vs overfit1win_long: same 1 window / FH2 / batch1 / 5000ep,
# only LR schedule changed (OneCycle max_lr=1e-4 -> constant 1e-4).
python src/train_Diffuvo.py --stage 2 --ae-checkpoint $AE \
  --overfit-samples 1 --future-horizon 2 --epochs 5000 \
  --lr 1e-4 --constant-lr \
  --batch-size 1 --devices 1 --save-top-k 1 \
  --no-reconstruction --no-metric-export --no-clearml \
  --run-name diffuvo_overfit1win_constlr \
  > run_logs/overfit1win_constlr.log 2>&1
echo "######## DONE rc=$? $(date) ########"
