set -e
source /home/carol/exit/etc/profile.d/conda.sh
conda activate kcy-alpha
cd /home/carol/chaeyeon-kim/alpha26/archive/Diffuvo_AR
CKPT=log/run_001_fh2_ar_baseline/version_0/checkpoints/last.ckpt
RUN=run_001_fh2_ar_baseline_eval_ddim50_cfg3
echo "######## eval AR baseline epoch39 DDIM50 CFG3.0 START $(date) ########"
# cfg-scale 1.0(무증폭)에서 straight vs no-action latentMSE가 동일 → action conditioning
# 존재 여부 판별용 A/B. P_DROP=0.1로 학습됐고 config 권장 CFG_SCALE=3.0.
# 그 외 설정은 eval_ar_baseline_ep39_ddim50.sh와 동일.
python src/predict_Diffuvo.py \
  --checkpoint "$CKPT" \
  --ddim-steps 50 --cfg-scale 3.0 \
  --split train --num-streams 2 \
  --sample-indices 0,1,2 \
  --figure-dir figure \
  --run-name $RUN
echo "######## DONE rc=$? $(date) ########"
