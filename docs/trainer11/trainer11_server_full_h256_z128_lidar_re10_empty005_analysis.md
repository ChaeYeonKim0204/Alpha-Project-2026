# Trainer11 Full Run Analysis

Run: `trainer11_server_full_h256_z128_lidar_re10_empty005`

Result directory:
`/home/carol/chaeyeon-kim/server_results/transfer_trainer11_result`

## 1. Run Configuration

This run trained `trainer11` on the full Arrow train set with the following important settings.

| item | value |
|---|---:|
| train data | full train Arrow dataset, 22 train files |
| validation data | RL: `validation_run_008.arrow`, DS: `validation_run_024.arrow` |
| receptive field | 4 |
| future horizon | 4 |
| saved data rate | 4 Hz |
| sample setting | `SAMPLE_EVERY_N=2`, approximately 2 Hz sequence |
| RGB reconstruction size | 216 x 288 |
| LiDAR range view size | 32 x 256 |
| hidden state dim | 256 |
| stochastic state dim | 128 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| scheduler | OneCycleLR |
| precision | 16-mixed |
| monitor | `val/DS_loss`, min |
| max steps | 150,000 |
| effective global batch size | 8 |

Loss weights used in the actual logged run:

| loss setting | value |
|---|---:|
| `WEIGHT_PROBABILISTIC` | 0.01 |
| `KL_FREE_BITS` | 1.0 |
| `KL_BALANCING_ALPHA` | 0.75 |
| `WEIGHT_LIDAR_RE` | 1.0 |
| `WEIGHT_LIDAR_EMPTY` | 0.05 |
| `WEIGHT_RGB` | 1.0 |
| `WEIGHT_FUTURE` | 1.0 |

## 2. Artifacts Checked

Primary files used for this analysis:

- TensorBoard scalar CSV: `trainer11_server_full_h256_z128_lidar_re10_empty005_tensorboard_scalars.csv`
- Loss and metric plot: `trainer11_server_full_h256_z128_lidar_re10_empty005_loss_metrics.png`
- Reconstruction figure: `trainer11_server_full_h256_z128_lidar_re10_empty005_reconstruction_s0.png`
- Experiment log: `experiment_log.csv`
- Checkpoints: `trainer11_server_full_h256_z128_lidar_re10_empty005/checkpoints/`

Additional analysis artifacts generated locally:

- `trainer11_server_full_h256_z128_lidar_re10_empty005_analysis_summary.csv`
- `trainer11_server_full_h256_z128_lidar_re10_empty005_gap_summary.csv`
- `trainer11_server_full_h256_z128_lidar_re10_empty005_analysis_focus.png`
- `trainer11_server_full_h256_z128_lidar_re10_empty005_analysis_report.md`

## 3. Overall Training Behavior

The loss curves look healthy. The model converged without obvious instability.

| metric | first validation | final | best | interpretation |
|---|---:|---:|---:|---|
| `val/RL_loss` | 237.8678 | 32.6385 | 32.5779 at step 125,399 | strong convergence, plateau after about 120k |
| `val/DS_loss` | 203.3267 | 23.2331 | 23.1576 at step 123,599 | strong convergence, plateau after about 120k |
| `train_loss_epoch` | 247.9785 | 16.0448 | 15.5946 | train loss continued slightly lower than validation |

The best `val/DS_loss` appeared around step 123,599. The final value at step 149,999 is only slightly worse, so the late-stage training did not collapse, but it also did not add much.

Representative validation progress:

| step | DS loss | RL loss | DS RGB PSNR | RL RGB PSNR |
|---:|---:|---:|---:|---:|
| 599 | 203.3267 | 237.8678 | 11.3249 | 10.7420 |
| 29,999 | 31.4491 | 40.4599 | 14.0403 | 12.0930 |
| 59,999 | 25.6741 | 36.3612 | 15.6202 | 13.6226 |
| 89,999 | 24.1030 | 33.3243 | 16.2287 | 14.0788 |
| 119,999 | 23.3726 | 32.8788 | 16.4440 | 14.2986 |
| 149,999 | 23.2331 | 32.6385 | 16.4701 | 14.4340 |

Conclusion: training was mostly complete by roughly 120k-130k steps.

## 4. Train and Validation Gap

Final train loss was `16.0448`.

| validation split | final val loss | gap vs train | gap ratio |
|---|---:|---:|---:|
| RL | 32.6385 | +16.5937 | +103.4% |
| DS | 23.2331 | +7.1883 | +44.8% |

There is a meaningful train/validation gap. However, this should not be interpreted as plain overfitting only.

Likely contributors:

- Training uses stream ordering and TBPTT-style hidden carry.
- Validation is evaluated on fixed windows, effectively harder than the train continuation path.
- RL validation appears to be a harder distribution than DS validation.
- LiDAR reconstruction terms dominate total loss, so domain differences in LiDAR geometry strongly affect total loss.

## 5. RL vs DS Validation Difference

DS validation is consistently easier than RL validation.

| metric | RL final | DS final | DS vs RL |
|---|---:|---:|---:|
| loss | 32.6385 | 23.2331 | DS lower by 28.8% |
| RGB PSNR | 14.4340 dB | 16.4701 dB | DS higher by 2.04 dB |
| future RGB PSNR | 13.7471 dB | 16.2976 dB | DS higher by 2.55 dB |
| LiDAR Chamfer xyz | 1.5298 | 1.2998 | DS lower by 15.0% |
| future LiDAR Chamfer xyz | 1.5742 | 1.3203 | DS lower by 16.1% |
| LiDAR range MAE | 3.1286 m | 2.5021 m | DS lower by 20.0% |
| future LiDAR range MAE | 3.3453 m | 2.5798 m | DS lower by 22.9% |

Interpretation: the model generalizes better to DS validation than RL validation. RL likely contains harder visual/LiDAR geometry or dynamics for this training setup.

## 6. RGB and LiDAR Metric Trends

RGB PSNR improved steadily and plateaued late.

| metric | first | final | improvement |
|---|---:|---:|---:|
| RL RGB PSNR | 10.7420 dB | 14.4340 dB | +3.69 dB |
| DS RGB PSNR | 11.3249 dB | 16.4701 dB | +5.15 dB |
| RL future RGB PSNR | 10.9165 dB | 13.7471 dB | +2.83 dB |
| DS future RGB PSNR | 11.6071 dB | 16.2976 dB | +4.69 dB |

LiDAR metrics also improved strongly.

| metric | first | final | reduction |
|---|---:|---:|---:|
| RL LiDAR Chamfer | 7.2344 | 1.5298 | -78.9% |
| DS LiDAR Chamfer | 6.6788 | 1.2998 | -80.5% |
| RL future LiDAR Chamfer | 7.2353 | 1.5742 | -78.2% |
| DS future LiDAR Chamfer | 6.6628 | 1.3203 | -80.2% |
| RL LiDAR range MAE | 15.2312 m | 3.1286 m | -79.5% |
| DS LiDAR range MAE | 14.0224 m | 2.5021 m | -82.2% |

Conclusion: both RGB and LiDAR improved meaningfully. The final curves are mostly flat, not diverging.

## 7. KL / Probabilistic Loss

The weighted probabilistic loss did not collapse and did not explode.

| metric | first | final | last 10% mean |
|---|---:|---:|---:|
| `val/RL_probabilistic` | 0.7863 | 0.9686 | 0.9618 |
| `val/DS_probabilistic` | 0.7461 | 0.8647 | 0.8624 |
| `train_probabilistic_step` | 0.0104 | 1.0312 | 0.9297 |

Because `WEIGHT_PROBABILISTIC=0.01`, the final unweighted KL scale is roughly around 86-97 nats for validation. This is not posterior collapse. It also does not show runaway KL growth.

Recommendation: keep `WEIGHT_PROBABILISTIC=0.01` for the next main run unless a specific KL capacity ablation is planned.

## 8. Reconstruction Quality

The reconstruction figure shows the following behavior.

RGB:

- Posterior and future predictions capture the coarse scene layout.
- Road, sky, buildings, and large geometry are recognizable.
- Fine details are strongly blurred.
- Future RGB is plausible but not sharp.

LiDAR depth:

- Target LiDAR depth is sparse and structured.
- Predicted LiDAR depth is much denser and smoother than the target.
- The model reconstructs broad depth bands, but it tends to fill empty cells.

Interpretation: the model learned useful world structure, but the output still behaves like a smooth reconstruction model rather than a sparse LiDAR occupancy/range model.

## 9. Loss Component Balance

Final validation loss is dominated by LiDAR xyz reconstruction.

Final RL loss component share:

| component | value | share |
|---|---:|---:|
| future LiDAR xyz 2x | 8.8239 | 27.0% |
| LiDAR xyz 2x | 7.8480 | 24.0% |
| future LiDAR xyz 4x | 4.4640 | 13.7% |
| LiDAR xyz 4x | 3.9459 | 12.1% |
| probabilistic | 0.9686 | 3.0% |
| all RGB terms combined | about 0.2131 | less than 1% |

Final DS loss component share:

| component | value | share |
|---|---:|---:|
| future LiDAR xyz 2x | 5.8640 | 25.2% |
| LiDAR xyz 2x | 5.4165 | 23.3% |
| future LiDAR xyz 4x | 3.0006 | 12.9% |
| LiDAR xyz 4x | 2.7553 | 11.9% |
| probabilistic | 0.8647 | 3.7% |
| all RGB terms combined | about 0.1582 | less than 1% |

The total loss is therefore mostly a LiDAR xyz objective. RGB PSNR improves, but RGB has little influence on total loss scale.

## 10. Runtime Check

The server run reportedly took about 12 hours. TensorBoard wall-time confirms this.

| runtime item | value |
|---|---:|
| TensorBoard wall-time span | 12.094 hours |
| train steps | 150,000 |
| average throughput | 3.45 steps/sec |
| average step time | 0.290 sec/step |
| normal train interval median | 0.272 sec/step |
| validation-near interval p95 | 1.36 sec/step |

This runtime is normal for this configuration.

Reasons:

- 150k training steps is large.
- The run used DDP with global batch size 8, likely per-GPU batch size 2 on 4 GPUs.
- Each sample contains RF 4 + FH 4 sequence frames.
- RGB reconstruction and LiDAR range-view reconstruction are both active.
- Validation runs every 600 steps, for about 250 validation rounds.
- Each validation round evaluates both RL and DS loaders.
- Validation computes expensive LiDAR metrics, especially Chamfer distance using `torch.cdist`.
- Checkpoints are large, about 456 MB each, but checkpoint writing is not the main runtime cost.

Approximate interpretation:

- Pure training at median speed would take about 11.3 hours.
- Validation, checkpointing, logging, and occasional slower intervals bring the run to about 12.1 hours.

Conclusion: there is no evidence that the run was abnormally slow. The 12-hour duration is consistent with the model, data, metric, and validation settings.

## 11. Recommended Next Experiments

Highest priority:

1. Reduce max steps to 125k-135k for the next comparable run.
   The best `val/DS_loss` happened around 123.6k, and later improvement was tiny.

2. Increase validation interval.
   Use `--val-check-interval 1200` or `2400` instead of 600 unless dense validation curves are needed.

3. Keep `WEIGHT_PROBABILISTIC=0.01`.
   Current KL behavior is healthy.

LiDAR sparsity / empty-cell issue:

4. Try `WEIGHT_LIDAR_EMPTY=0.1` or `0.2`.
   Current reconstruction tends to fill empty LiDAR cells too densely.

5. Consider separating LiDAR occupancy/valid-mask prediction from range regression.
   The current smooth dense output suggests that range-only penalties are not enough to preserve sparse structure.

RGB quality:

6. If RGB sharpness matters, try `WEIGHT_RGB=2.0` or `4.0`.
   Current RGB terms are less than 1% of total loss, so RGB may be underweighted relative to LiDAR xyz.

Validation protocol:

7. Re-evaluate validation with a stream/TBPTT-carry protocol if the deployment path uses recurrent state carry.
   The current validation gap may be partially caused by zero-init window evaluation.

Runtime:

8. If wall-clock time is a concern, reduce validation frequency first.
   Disabling or reducing Chamfer computation during intermediate validation would likely save more time than checkpoint tweaks.

## 12. Bottom Line

This run is successful as a full-data trainer11 baseline.

The model converges cleanly, improves RGB and LiDAR metrics substantially, and avoids KL collapse. The main remaining issues are the train/validation gap, weaker RL validation performance, blurred RGB predictions, and overly dense/smooth LiDAR depth predictions.

For the next run, the most practical setting is:

- `steps=125000` to `135000`
- `val_check_interval=1200` or `2400`
- keep `WEIGHT_PROBABILISTIC=0.01`
- try `WEIGHT_LIDAR_EMPTY=0.1` or `0.2`
- optionally try `WEIGHT_RGB=2.0` if RGB quality is important
