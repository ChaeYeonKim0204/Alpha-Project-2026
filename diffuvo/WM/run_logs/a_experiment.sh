set -e
source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
cd /home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_WM
AE=logs/diffuvo_ae_fulldata_fh4/version_0/checkpoints/last.ckpt
COMMON="--stage 2 --ae-checkpoint $AE --overfit-samples 8 --future-horizon 2 --epochs 1000 --batch-size 2 --devices 1 --no-reconstruction --no-clearml"

echo "######## RUN 1/2: norm OFF  $(date) ########"
python src/train_Diffuvo.py $COMMON --no-latent-norm --run-name diffuvo_normoff_fh2_overfit8 \
  > run_logs/normoff.log 2>&1
echo "######## RUN 1 done rc=$? $(date) ########"

echo "######## RUN 2/2: norm ON   $(date) ########"
python src/train_Diffuvo.py $COMMON --run-name diffuvo_normon_fh2_overfit8 \
  > run_logs/normon.log 2>&1
echo "######## RUN 2 done rc=$? $(date) ########"
echo "ALL DONE $(date)"
