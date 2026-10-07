# Experiment: lr_baseline_muvo2d — OneCycleLR annealing tail fix

- **Date:** 2026-05-29
- **Run name:** `lr_basline_muvo2d`
- **Base run compared against:** `muvo2D_full_muvoaligned_lidar01_h256z256_2hz_bs6x4_ep200`
- **Files changed:** `scripts/model_variants/config_muvo_2D.py`, `scripts/model_variants/trainer_muvo_2D.py`

## Motivation

Loss/metric curves from the previous `ep200` run showed a **dead tail**: after
~45k steps (≈158 epochs) most metrics flattened, yet LiDAR Chamfer was *still*
improving right up to that point. Root cause was the OneCycleLR scheduler being
constructed **without** `div_factor` / `final_div_factor`, so PyTorch defaults
applied silently:

- `div_factor=25`   → start LR = `max_lr/25`  = 4e-6
- `final_div_factor=1e4` → **end LR = `max_lr/1e4` = 1e-8** (effectively zero)

With `max_lr=1e-4`, the cosine anneal drove the LR to 1e-8 over the last ~12k
steps, so those steps contributed almost nothing — the productive LiDAR
improvement was being cut off.

## Change

Made `div_factor` and `final_div_factor` explicit (and configurable via cfg).

### `config_muvo_2D.py` — `SCHEDULER` namespace

```python
SCHEDULER=SimpleNamespace(
    NAME='OneCycleLR',
    PCT_START=0.1,
    DIV_FACTOR=10.0,        # NEW: start LR = max_lr / DIV_FACTOR
    FINAL_DIV_FACTOR=1e2,   # NEW: end LR  = max_lr / FINAL_DIV_FACTOR
),
```

### `trainer_muvo_2D.py` — `configure_optimizers()`

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=self.lr, total_steps=total_steps,
    pct_start=getattr(self.cfg.SCHEDULER, "PCT_START", 0.3),
    div_factor=getattr(self.cfg.SCHEDULER, "DIV_FACTOR", 10.0),
    final_div_factor=getattr(self.cfg.SCHEDULER, "FINAL_DIV_FACTOR", 1e2),
)
```

## Effect (LR schedule, max_lr=1e-4)

| Parameter | Before (torch default) | After |
|---|---|---|
| `div_factor` | 25 | **10** |
| start LR | 4e-6 | **1e-5** |
| `final_div_factor` | 1e4 | **1e2** |
| **end LR** | **1e-8** | **1e-6** |

End LR is now 100× higher, so the final ~12k steps keep fine-tuning instead of
sitting at a near-zero LR. Expected to mainly help the LiDAR Chamfer metrics,
which were still descending at the point the old schedule died.

> Note: `total_steps` is bound to `trainer.estimated_stepping_batches`
> (EPOCHS-driven), so the anneal curve auto-scales with `--epochs`. Keep epochs
> at 200 to make this run directly comparable to the baseline.

## Launch command (server)

```bash
cd ~/chaeyeon-kim/alpha26
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kcy-alpha

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 \
python scripts/train_muvo_2D.py \
  --epochs 200 \
  --batch-size 6 \
  --devices 4 \
  --strategy ddp_find_unused_parameters_true \
  --num-workers 4 \
  --run-name lr_basline_muvo2d
```

## What to check in results

- `val/*_lidar_chamfer_xyz` — does the tail (after ~45k steps) keep descending now?
- `val/*_rgb_psnr` / total loss — expected ~flat (already saturated by ~20k); no regression.
- `train_probabilistic` (KL) — should stay healthy (~0.035), watch for drift.
