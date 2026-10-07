# TensorBoard 사용법 (alpha26, ClearML 보조용)

> **`_muvo_2D` v3부터 primary 실험 트래킹은 ClearML.** ClearML이 TensorBoard scalar/figure를 자동으로 web UI에 mirror하므로, 대부분의 비교/공유 작업은 `clearml_guide.md`만 보면 됨. 본 가이드는 다음 두 경우에 필요:
> - 인터넷 없는 환경 또는 ClearML 비활성 (`--no-clearml`) run의 로컬 디버깅
> - 빠른 loss curve 확인 (브라우저 1개로 끝내고 싶을 때)
>
> 둘 다 안 해당되면 `clearml_guide.md`만 읽으면 됨.

---

## 1. 어디에 로그가 쌓이나

PyTorch Lightning `TensorBoardLogger` 자동 출력:

```
alpha26/logs/<run-name>/
  ├── events.out.tfevents.*      # scalar/figure/text 모두 여기
  ├── hparams.yaml                # PL이 자동 dump한 hparams
  └── (체크포인트는 별도 /checkpoints/)
```

- **서버**: `/home/user/chaeyeon-kim/alpha26/logs/<run-name>/`
- **로컬**: `/home/carol/chaeyeon-kim/alpha26/logs/<run-name>/` (server_results에서 sync한 경우)

## 2. 로컬에서 띄우기

```bash
cd /home/carol/chaeyeon-kim/alpha26
conda activate kcy-alpha
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

브라우저: <http://localhost:6006>

여러 run 비교 (UI 좌측 체크박스):
```bash
tensorboard --logdir logs --port 6006
```

라벨로 명시 비교:
```bash
tensorboard --logdir_spec=baseline:logs/_muvo_baseline,exp:logs/_muvo_2D_v1
```

## 3. 서버 → 로컬 브라우저 (SSH 포트포워딩)

**서버 셸**:
```bash
ssh user@<server-ip>
cd /home/user/chaeyeon-kim/alpha26
conda activate kcy-alpha
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

**로컬 셸** (별도 창):
```bash
ssh -N -L 6006:localhost:6006 user@<server-ip>
```

로컬 브라우저: <http://localhost:6006> — 서버의 TensorBoard.

## 4. 주요 카테고리 (`_muvo_2D` 기준)

### Scalars
- `train_loss_total`, `train_loss_rgb_{1,2,4}`, `train_loss_lidar_*`, `train_loss_kl_*`
- `train_loss_action` (v2 신규), `train_loss_ssim_*` (cfg 켜면)
- `future_*` 접두사: imagine horizon에 대한 loss
- `val_*` / `test_*`: 동일 키셋
- `val_psnr`, `val_ssim`, `val_chamfer_xyz`, `val_lidar_chamfer_xyz_nearfield` (v2)
- `lr`, `epoch`

### Images
- `recon_figure_epoch_N` — RF + FH = 6 frame 전체 column × LiDAR-scale 보정/미보정 두 row (v3)

### Text
- `change_summary` — `train_muvo_2D.py`가 push하는 1줄 run 요약
- `hparams`

## 5. 자주 쓰는 패턴

### `_muvo` baseline vs `_muvo_2D` 비교
```bash
tensorboard --logdir_spec=baseline:logs/_muvo_baseline,exp:logs/_muvo_2D_v1 --port 6006
```

### archive
```bash
tar -czvf <run-name>.tar.gz logs/<run-name>
```
CLAUDE.md 가이드 따라.

## 6. 자주 묻는 문제

**Q. 아무것도 안 보임.** → (a) `ls logs/<run-name>/events.*` 확인, (b) 1 step 이상 돌았는지, (c) 브라우저 강력 새로고침.

**Q. 두 run이 한 곡선으로 겹침.** → `--logdir`을 부모 폴더로 주거나 `--logdir_spec=name1:path1,name2:path2`.

**Q. step axis가 epoch이 아니라 global_step.** → PL 기본. `self.log(..., on_epoch=True, on_step=False)`로 push한 metric만 epoch 단위.

**Q. 서버 6006 포트 막힘.** → SSH 포트포워딩 사용 (§3).

**Q. 자동 갱신.** → `tensorboard --logdir logs --reload_interval 10` (10초마다 새로 읽음).

## 7. 관련 도구 / 문서

- `alpha26/docs/clearml_guide.md` — **primary 실험 트래킹** (대부분 여기 보면 됨)
- `alpha26/scripts/export_tensorboard_metrics.py` — events에서 scalar CSV export
- `alpha26/results/experiment_log.csv` — run-level overview
- `alpha26/docs/experiment_log_organization.md` — CSV 컬럼 / `series` 분류
