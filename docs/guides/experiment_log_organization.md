# `experiment_log.csv` Organization

Generated to make `alpha26/results/experiment_log.csv` (36 rows) easier to navigate.
The original CSV is preserved unchanged. A chronologically sorted copy lives at
`alpha26/results/experiment_log_sorted.csv` (rows sorted by `date` ASC, ties broken
by `run_name` ASC). This document classifies each row into an `experiment_series`
label — **the CSV schema is unchanged**; the series labels live only in this doc.

## Series summary

| series_name | count | date range | description |
|---|---|---|---|
| `trainer11_original` | 9 | 2026-05-07 .. 2026-05-13 | Earliest trainer11 ablations: lowres/2 Hz baseline plus the May-13 sweep of pooling, decoder width, and hidden-state size. `train_files` is either `train_run_002` alone or the 22-file alldata manifest. `rgb_recon_size=216x288`, `lidar_size=32x256`. |
| `trainer11_v2_v5` | 6 | 2026-05-07 .. 2026-05-13 | The v2-v5 versioned `trainer11_reviewed.ipynb` line: loss rebalance (RGB×10, LiDAR×0.2), TBPTT stride + per-sample continuation mask, KL free-bits, OneCycleLR. Includes the `_v3_alldata` / `_v3_alldata_warmup` / `_v3_alldata_warmup_noearlystop` rollouts and the `_v4_alldata_40epoch` long run (`max_steps=150000`). |
| `alpha_overfit_baselines` | 4 | 2026-05-14 | One-window overfit sanity checks on the alpha (non-`_muvo`) `trainer11_reviewed.ipynb` stack at `h256/z128`. Step budget sweep at 100/200/400/1000. Two rows share `run_name=one_window_overfit_train002_400epoch` (see "Duplicates" below). |
| `alpha_overfit_dim_sweep` | 2 | 2026-05-14 | h/z dimension ablation on the same alpha overfit harness: `h256_z128_rgbonly` (LiDAR weight off) vs `h512_z256`. Both 1000-epoch. |
| `alpha_ssim_ablation` | 2 | 2026-05-15 | SSIM weight ablation on alpha non-`_muvo` h256/z128: `ssim_*` (per the v2-v5 baseline change_summary, paired with the 0.06 variant the next day) and `ssim06_*` (explicit `WEIGHT_SSIM=0.06`). The `ssim06_*` row's change_summary explicitly notes "weight 0.6 직전 run과 동일 setup, weight만 1/10", so the unprefixed `ssim_*` row represents the 0.6 weight setting. |
| `muvo_branch_initial` | 2 | 2026-05-14 | First `_muvo` (`base_file=train_muvo.py`) runs with the pre-migration `rgb_recon_size=216x288`. `muvo_smoke_1window` (max_steps=1) and `name=muvo_one_window_overfit_100` (max_steps=800, the leaked `name=` prefix is a stray CLI artefact in the run_name). |
| `muvo_branch_post_migration` | 5 | 2026-05-15 | `_muvo` runs after migration to the MUVO-faithful `rgb_recon_size=320x768`. Includes h/z dim sweep (h256/z128 vs h512/z256) crossed with SSIM weight (0.0 vs 0.06), plus an early `muvo_one_window_overfit_1000` (still at 216x288 — see "Anomalies"). |
| `smoke_throwaway` | 4 | 2026-05-11 | Developer smoke tests with `max_steps=1`, `lr=0.0003`, `scheduler=none`, no `batch_size`. All four rows were reconstructed from `hparams.yaml` on 2026-05-15 because the live CSV did not capture them at the time. They validate the server-script export from the notebook. |
| `server_runs` | 2 | 2026-05-13 .. 2026-05-13 | Rows reconstructed from `hparams.yaml`. `trainer11_recon150x200` is the long server run (`max_steps=124200`, `base_file=trainer11_reviewed_alldata_40epoch_server.ipynb`, `rgb_recon_size=150x200`). Co-classifying the `*_tiny_overfit_*_smoke` runs as `smoke_throwaway` keeps the "server vs smoke" split coherent. |

Total: **36 rows** classified, 0 unclassified.

## Series detail (rows in sorted CSV order)

Format: `date | run_name | (h/z, rgb_recon, max_steps)` — one-line summary.

### trainer11_original (9)

- `2026-05-07 trainer11_lowres_2hz` (h128/z64, 216x288, 6000) — meeting ablation: low-res reconstruction target, 32x256 range view, 2 Hz sampling, pkl manifest. **Empty `final_val_metrics` and `notes` — appears to be a header-only entry; row 2 below is the actual completed run with same name.**
- `2026-05-07 trainer11_lowres_2hz` (h128/z64, 216x288, 6000) — completed lowres/2 Hz baseline with full val metrics.
- `2026-05-13 trainer11_files_epoch200_bestckpt_viz` (h128/z64, 216x288, 2200) — pre-`hiddenstate_256` v2-v5 stack at h128/z64 with best-ckpt visualization. (Included here rather than v2-v5 because the run name does not carry a `_v?_` tag and it kicks off the May-13 ablation series.)
- `2026-05-13 hiddenstate_256_loss_weight` (h256/z128, 216x288, 2200) — bump hidden state from 128 to 256 (z 64→128). Start of the May-13 width sweep.
- `2026-05-13 epochs_400_hiddenstate_256_loss_weight` (h256/z128, 216x288, 4400) — same as above but doubled budget (400 epochs).
- `2026-05-13 switch_pooling__epochs_200_hiddenstate_256_loss_weight` (h256/z128, 216x288, 2200) — pooling layout change on top of h256/z128.
- `2026-05-13 pooling44_switch_pooling_epochs_200_hiddenstate_256_loss_weight` (h256/z128, 216x288, 2200) — pooling 4x4 variant.
- `2026-05-13 decoder512_pooling44_switch_pooling_epochs_200_hiddenstate_512_loss_weight` (h256/z128, 216x288, 2200) — adds decoder width 512 on top of pooling44. (Run-name says `_hiddenstate_512_` but the row's `h_dim` column is 256 — see "Anomalies".)
- `2026-05-13 overfitting_decoder512_pooling44_switch_pooling_epochs_200_hiddenstate_512_loss_weight` (h256/z128, 216x288, 800) — overfitting variant: `val_files == train_files` (the same 22-file manifest appears in `val_files`, repeated 3 times — likely a CSV emitter bug). Same hiddenstate_512 vs h_dim=256 inconsistency.

### trainer11_v2_v5 (6)

- `2026-05-07 trainer11_reviewed_v2_lossfix` (h128/z64, 216x288, 15000) — v2 loss rebalance + KL free-bits + OneCycleLR, single arrow file.
- `2026-05-08 trainer11_reviewed_v3_alldata` (h128/z64, 216x288, 15000) — v3 with `USE_ALL_TRAIN_RUNS=True` (22 arrow files). `train_files` column still shows only `train_run_002` (manifest enable was via flag, not file list).
- `2026-05-08 trainer11_reviewed_v3_alldata_warmup` (h128/z64, 216x288, 15000) — v3 alldata + current-run viz/log lookup.
- `2026-05-08 trainer11_reviewed_v3_alldata_warmup_noearlystop` (h128/z64, 216x288, 15000) — EarlyStopping removed, full max_steps run.
- `2026-05-08 trainer11_reviewed_v4_alldata_40epoch` (h128/z64, 216x288, 150000) — ~40 epoch longrun, ModelCheckpoint top-5, best-ckpt visualization. The flagship pre-server run.
- `2026-05-13 trainer11_reviewed_v3_alldata_warmup_noearlystop` (×2, h128/z64, 216x288, 400) — two short (400-step) re-runs of the v3 warmup config; both have empty `final_val_metrics`. Distinguished only by `version_1` / `version_2` log paths in `notes` (see "Duplicates").

### alpha_overfit_baselines (4)

All h256/z128, `rgb_recon_size=216x288`, batch_size=1, train=`train_run_002.arrow` only.

- `2026-05-14 one_window_overfit_train002_100epoch` (max_steps=100).
- `2026-05-14 one_window_overfit_train002_200epoch` (max_steps=200).
- `2026-05-14 one_window_overfit_train002_400epoch` (max_steps=400, log `version_1`).
- `2026-05-14 one_window_overfit_train002_400epoch` (max_steps=1000, log `version_3`) — same `run_name` but a different max_steps and a different log version; the user kept the directory name when bumping the budget (see "Duplicates").

### alpha_overfit_dim_sweep (2)

Same overfit harness as baselines, different model dims.

- `2026-05-14 one_window_overfit_train002_h256_z128_rgbonly_1000epoch` (h256/z128, RGB-only, 1000) — control with LiDAR weight off.
- `2026-05-14 one_window_overfit_train002_h512_z256_1000epoch` (h512/z256, 1000) — large-model variant.

### alpha_ssim_ablation (2)

Same overfit harness, h256/z128, `rgb_recon_size=216x288`.

- `2026-05-15 ssim_one_window_overfit_train002_h256_z128_1000epoch` — copies the standard v2-v5 change_summary (implicitly WEIGHT_SSIM=0.6 — the "weight 0.6 직전 run" referenced by the next row).
- `2026-05-15 ssim06_one_window_overfit_train002_h256_z128_1000epoch` — explicit WEIGHT_SSIM=0.06 (1/10 of the 0.6 setting).

### muvo_branch_initial (2)

`base_file=train_muvo.py`, `rgb_recon_size=216x288` (pre-migration), h512/z256.

- `2026-05-14 muvo_smoke_1window` (max_steps=1) — first MUVO-style smoke test, has populated `final_val_metrics`.
- `2026-05-14 name=muvo_one_window_overfit_100` (max_steps=800) — the `name=` prefix is a leaked CLI flag in the run_name; train_files and val_files are both the full 22-file manifest (val concatenated 2x — emission artifact).

### muvo_branch_post_migration (5)

`base_file=train_muvo.py`, all 1000 max_steps, single-window overfit (`train_run_002.arrow`).

- `2026-05-15 muvo_one_window_overfit_1000` (h512/z256, 216x288) — **still at the pre-migration 216x288 resolution**; on the date boundary between initial and post-migration. See "Anomalies".
- `2026-05-15 muvo_one_window_overfit_h256_z128_1000epoch` (h256/z128, 320x768, WEIGHT_SSIM=0.0) — post-migration baseline at smaller model.
- `2026-05-15 muvo_one_window_overfit_h512_z256_1000epoch_v2` (h512/z256, 320x768, WEIGHT_SSIM=0.0) — post-migration baseline at large model. The `_v2` suffix suggests a previous attempt was abandoned.
- `2026-05-15 muvo_ssim06_one_window_overfit_h256_z128_1000epoch` (h256/z128, 320x768, WEIGHT_SSIM=0.06).
- `2026-05-15 muvo_ssim06_one_window_overfit_h512_z256_1000epoch` (h512/z256, 320x768, WEIGHT_SSIM=0.06).

  (This block is effectively a 2×2 grid: {h256/z128, h512/z256} × {SSIM=0.0, SSIM=0.06}.)

### smoke_throwaway (4)

All 2026-05-11, `max_steps=1`, `lr=0.0003`, `scheduler=none`, batch_size empty.
All four were reconstructed from `hparams.yaml` on 2026-05-15.

- `trainer11_server_tiny_overfit_defs_smoke` — exercises `trainer11_server_defs.py` import path.
- `trainer11_server_tiny_overfit_scripts_smoke` — exercises the CLI script wrapper.
- `trainer11_server_tiny_overfit_smoke` — generic server-side smoke.
- `trainer11_tiny_overfit_smoke` — local (non-server) smoke; note the `log path` lives at `notebooks/training/logs/...` rather than `logs/...`.

### server_runs (2)

Reconstructed from `hparams.yaml` on 2026-05-15.

- `2026-05-13 trainer11_recon150x200` (h256/z128, **150x200** RGB, max_steps=124200, `base_file=trainer11_reviewed_alldata_40epoch_server.ipynb`) — the long 4-GPU server training run. Has full `final_val_metrics` including `val/RL_rgb_psnr=15.0031`, `val/DS_lidar_chamfer_xyz=1.3213`. This is the row whose TensorBoard event file's hostname is `user-ESC4000A-E12` per the prompt brief.

  *Note: only one server run after retroactive reconstruction; the smoke variants are kept separate in `smoke_throwaway` since they don't represent comparable training runs.*

## Duplicates and near-duplicates

- **`trainer11_lowres_2hz` (2026-05-07, ×2)** — both rows have the same `run_name`, `h_dim`, `z_dim`, `lr`, `max_steps`. The first row's `final_val_metrics` is empty and `notes` is empty; the second row has full metrics and `log_dir=./logs/trainer11_lowres_2hz/version_0`. Likely the first row was emitted before training completed (a "started" stub) and the second was emitted on completion. Treat the second as authoritative.

- **`trainer11_reviewed_v3_alldata_warmup_noearlystop` (2026-05-13, ×2)** — identical schema rows; only `notes` differs (`version_1` vs `version_2` log paths). Both have empty `final_val_metrics`. These look like two short (400-step) replays of the same config — probably for visualization regeneration or a quick sanity-check rerun. Keep both rows but treat them as a pair.

- **`one_window_overfit_train002_400epoch` (2026-05-14, ×2)** — same `run_name`, different `max_steps` (400 vs 1000) and different log `version_` (1 vs 3). The user retained the original directory name when extending the step budget from 400 to 1000; the run name "400epoch" no longer matches the second row's actual max_epochs=1000. The second row is effectively part of the 1000-epoch sweep — comparable to `one_window_overfit_train002_h256_z128_rgbonly_1000epoch` and `one_window_overfit_train002_h512_z256_1000epoch`.

## Anomalies

1. **`decoder512_*` and `overfitting_decoder512_*` rows have `h_dim=256` but `_hiddenstate_512_` in the run_name.** The run name suggests a hidden state of 512, but the recorded `h_dim` column is 256 (and `z_dim=128`). Either the run-name is misleading (the `_hiddenstate_512_` token may refer to a decoder-internal width, not the RSSM hidden state) or the CSV row is wrong. Cross-check by reading the actual `hparams.yaml` in `logs/decoder512_pooling44_switch_pooling_epochs_200_hiddenstate_512_loss_weight/version_0/`.

2. **`overfitting_decoder512_*` has `val_files == train_files × 3`.** The `val_files` field repeats the 22-file train manifest three times (66 file tokens). This appears to be an emission artefact from the overfit-on-train code path; the actual validation set is the train set (overfit by design). The doubled/tripled repetition is curious — possibly an `'\n'.join` over multiple loader objects.

3. **`muvo_one_window_overfit_1000` (2026-05-15) sits at `rgb_recon_size=216x288`** while all four sibling rows on the same date use `320x768`. The change_summary is identical to the initial-branch text, so this row likely reflects the *last* pre-migration `_muvo` configuration that happened to be run after midnight, before the resolution bump. Classified under `muvo_branch_post_migration` by date but flagged as a transitional row — strict comparability with the other 2026-05-15 `_muvo` rows is limited.

4. **`name=muvo_one_window_overfit_100`** has a CLI artifact in its run name (`name=` prefix). The `val_files` field is also unusually shaped: the 22-file train manifest concatenated twice (44 tokens), suggesting val loader was misconfigured (it likely re-used the train set). Final val metrics are empty.

5. **Row 1 of the date-sorted CSV (the first `trainer11_lowres_2hz`)** is the only row in the whole log where `sample_hz` is the integer `2` rather than `2.0`, `train_files` and `val_files` are quoted with `"..."`, and `lr` is `1e-4` rather than `0.0001`. Stylistic only — values are equivalent — but it confirms this row was written by a different code path than the rest (probably the original notebook's emitter before standardization).

6. **`trainer11_recon150x200` (server long run)** has an empty `batch_size` cell and `base_file` pointing at the `trainer11_reviewed_alldata_40epoch_server.ipynb`. This is the only "real" server run in the log; the four `tiny_overfit_*_smoke` reconstructions are smokes from the same server export but separate config.

7. **Schema observations (not anomalies, just notes for future emitter work):** `lr` is stored as a free-form string (`1e-4` vs `0.0001`); `train_files`/`val_files` are `;`-joined paths inside a CSV field, which makes the column ungainly in spreadsheet viewers; `final_val_metrics` packs many key=value pairs into a single semicolon-joined string. None of these need fixing in this pass — the user asked only for sort + classification.
