"""tune_muvo_2D.py — hyperparameter sweep driver for alpha26 MUVO 2D (RSSMTD).

train_muvo_2D.py를 trial마다 subprocess로 호출. 각 trial이 독립 Python
프로세스로 돌아서 한 trial이 죽어도 다음 trial은 영향 없음.

사용법 (서버에서):
    cd /home/user/chaeyeon-kim/scripts
    export CARLA_ARROW_ROOT=/home/user/chaeyeon-kim/processed
    export ALPHA26_OUTPUT_ROOT=/home/user/chaeyeon-kim/alpha26/results
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kcy-alpha
    python tune_muvo_2D.py

trial 추가/제거: 아래 TRIALS 리스트만 편집.
공통 학습 인자 변경: SHARED dict 편집.
"""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── trial 정의 ──────────────────────────────────────────────────────────────
# 각 entry는 (run_name_suffix, override_dict). override_dict의 key는
# train_muvo_2D.py CLI 인자(`--weight-lidar-re` → `weight_lidar_re`)와 매칭.
# 값이 bool이면 store_true flag로, 그 외엔 `--flag VALUE` 형태로 전달.
TRIALS = [
    # ── Round 2: Round 1 승자 wp=3e-3 고정 + kl_balancing_alpha 스윕 ──
    # (Round 1 wp 스윕 결과: wp3e3 승. baseline α=0.75는 이미 결과 있어 생략.)
    # α↑ = prior-fitting 집중 = future RGB 직접 레버.
    ("wp3e3_a08",  {"weight_probabilistic": 3e-3, "kl_balancing_alpha": 0.8}),   # DreamerV2 표준값
    ("wp3e3_a09",  {"weight_probabilistic": 3e-3, "kl_balancing_alpha": 0.9}),
    # 마감(5h30) 때문에 α=0.95(상한)는 이번엔 제외. 필요하면 후속으로 추가.
]

# 모든 trial이 공유하는 학습 인자.
SHARED = dict(
    epochs=50,                                  # screening 길이 (full은 200)
    batch_size=8,
    num_workers=4,
    devices=4,
    strategy="ddp_find_unused_parameters_true",
    no_clearml=True,
)

# Run 이름 접두사. 최종 run_name = f"{RUN_PREFIX}_{suffix}".
RUN_PREFIX = "kl_balance_muvo_2d"

# ── 승자 선택 설정 ───────────────────────────────────────────────────────────
# future RGB 품질이 목표 → RL/DS 두 val셋의 future RGB PSNR 평균이 가장 높은 trial이 승자.
# (PSNR은 높을수록 좋음.)
WINNER_METRICS = ["val/RL_future_rgb_psnr", "val/DS_future_rgb_psnr"]
WINNER_HIGHER_IS_BETTER = True
# 승자 판정에는 안 쓰지만 같이 기록하는 가드레일 (obs RGB가 안 무너졌는지 확인용).
GUARDRAIL_METRICS = ["val/RL_rgb_psnr", "val/DS_rgb_psnr"]
# 승자 정보를 저장할 파일명 (학습은 안 돌리고 "누가 이겼는지"만 남김).
# <ALPHA26_OUTPUT_ROOT>/experiments/ 아래에 markdown으로 저장 (resolve_winner_dir 참고).
WINNER_OUT_NAME = "{prefix}_winner.md"
# 승자를 나중에 따로 돌릴 때 쓸 epoch 수 (명령어 문자열 생성용).
WINNER_FINAL_EPOCHS = 200

# trial별 stdout/stderr 로그 디렉토리. CWD 기준 상대경로.
SWEEP_LOG_DIR = "logs/_sweep_console"

# True면 RUN_PREFIX_<suffix> 디렉토리가 이미 logs/에 존재하면 skip.
# 중간에 멈췄다가 재실행할 때 유용.
SKIP_IF_EXISTS = True

# 한 trial 실패해도 다음 trial로 진행 (subprocess.run check=False와 동일).
CONTINUE_ON_FAIL = True
# ────────────────────────────────────────────────────────────────────────────


def args_from(override):
    """override dict → train_muvo_2D.py에 넘길 CLI 인자 list."""
    out = []
    for key, value in override.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(flag)
        else:
            out += [flag, str(value)]
    return out


def run_name_exists(run_name, logs_root="logs"):
    return (Path(logs_root) / run_name).exists()


# ── 승자 선택 헬퍼 ───────────────────────────────────────────────────────────
def resolve_csv_path():
    """experiment_log.csv 경로를 train_muvo_2D.py와 동일하게 해석."""
    try:
        from config_muvo_2D import cfg as _cfg
        p = getattr(_cfg.LOGGING, "EXPERIMENT_LOG_PATH", None)
        if p:
            return str(p)
    except Exception:
        pass
    root = os.environ.get("ALPHA26_OUTPUT_ROOT")
    if root:
        return os.path.join(os.path.abspath(os.path.expanduser(root)), "experiment_log.csv")
    return "results/experiment_log.csv"


def resolve_winner_dir():
    """승자 markdown 저장 디렉토리 = <ALPHA26_OUTPUT_ROOT>/experiments.
    (예: /home/user/chaeyeon-kim/alpha26/results/experiments)
    env 미설정 시 resolve_csv_path()의 부모(=results)/experiments, 최후엔 ./results/experiments."""
    root = os.environ.get("ALPHA26_OUTPUT_ROOT")
    if root:
        return Path(os.path.abspath(os.path.expanduser(root))) / "experiments"
    csv_path = resolve_csv_path()
    return Path(csv_path).resolve().parent / "experiments"


def parse_metric_string(s):
    """'key=val; key=val' 형태의 final_val_metrics 컬럼 → {key: float}."""
    out = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        try:
            out[key.strip()] = float(val.strip())
        except ValueError:
            pass
    return out


def mean_of(metrics, keys):
    """metrics dict에서 keys 중 존재하는 값들의 평균. 하나도 없으면 None."""
    vals = [metrics[k] for k in keys if k in metrics]
    return sum(vals) / len(vals) if vals else None


def select_and_save_winner():
    """이번 sweep의 trial들 중 future RGB PSNR이 가장 좋은 1개를 골라 파일로 저장.
    학습은 돌리지 않는다. 승자를 찾으면 (run_name, override) 반환, 못 찾으면 None."""
    csv_path = resolve_csv_path()
    if not os.path.exists(csv_path):
        print(f"\n[winner] CSV가 없어 승자 선택 불가: {csv_path}")
        return None

    suffix_by_run = {f"{RUN_PREFIX}_{suffix}": (suffix, override) for suffix, override in TRIALS}

    # CSV는 append-only라 같은 run_name이 여러 줄일 수 있음 → 마지막 줄이 최신.
    latest = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            run_name = (row.get("run_name") or "").strip()
            if run_name in suffix_by_run:
                latest[run_name] = row

    candidates = []
    for run_name, row in latest.items():
        metrics = parse_metric_string(row.get("final_val_metrics", ""))
        score = mean_of(metrics, WINNER_METRICS)
        if score is None:
            continue  # future RGB 지표가 없는 run(중도 실패 등)은 후보에서 제외
        candidates.append({
            "run_name":  run_name,
            "suffix":    suffix_by_run[run_name][0],
            "override":  suffix_by_run[run_name][1],
            "score":     score,
            "guardrail": mean_of(metrics, GUARDRAIL_METRICS),
            "metrics":   {k: metrics[k] for k in (WINNER_METRICS + GUARDRAIL_METRICS) if k in metrics},
            "date":      (row.get("date") or "").strip(),
        })

    if not candidates:
        print("\n[winner] future RGB 지표가 있는 trial이 없어 승자 선택 불가.")
        return None

    candidates.sort(key=lambda c: c["score"], reverse=WINNER_HIGHER_IS_BETTER)
    winner = candidates[0]

    # 나중에 따로 돌릴 200ep 명령어 문자열 생성 (실행은 하지 않음).
    final_run_name = f"{RUN_PREFIX.replace('50ep', f'{WINNER_FINAL_EPOCHS}ep')}_winner_{winner['suffix']}"
    final_cfg = {**SHARED, "epochs": WINNER_FINAL_EPOCHS, **winner["override"]}
    final_cmd = [sys.executable, "train_muvo_2D.py", "--run-name", final_run_name] + args_from(final_cfg)

    final_command = " ".join(final_cmd)

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

    # ── 승자 요약을 docs/experiments/<prefix>_winner.md 로 저장 ──────────────
    metric_label = " + ".join(WINNER_METRICS) + " 평균"
    lines = []
    lines.append(f"# Sweep winner — {RUN_PREFIX}")
    lines.append("")
    lines.append(f"- 선정 기준: **{metric_label}** ({'높을수록' if WINNER_HIGHER_IS_BETTER else '낮을수록'} 좋음)")
    lines.append(f"- 후보 trial 수: {len(candidates)} / {len(TRIALS)}")
    lines.append(f"- 출처 CSV: `{csv_path}`")
    if winner.get("date"):
        lines.append(f"- 승자 학습일: {winner['date']}")
    lines.append("")
    lines.append("## 승자")
    lines.append("")
    lines.append(f"- **run_name**: `{winner['run_name']}`")
    lines.append(f"- **override**: `{json.dumps(winner['override'], ensure_ascii=False)}`")
    lines.append(f"- **future RGB (선정 점수)**: {_fmt(winner['score'])}")
    lines.append(f"- **obs RGB (가드레일)**: {_fmt(winner['guardrail'])}")
    lines.append("")
    lines.append("### 나중에 따로 돌릴 명령어 (이 sweep은 학습만, 결승은 수동 실행)")
    lines.append("")
    lines.append("```bash")
    lines.append(f"# {WINNER_FINAL_EPOCHS}ep 결승 — 승자 설정으로")
    lines.append(final_command)
    lines.append("```")
    lines.append("")
    lines.append("## 전체 랭킹")
    lines.append("")
    lines.append("| 순위 | run_name | future RGB | obs RGB (가드레일) |")
    lines.append("|---|---|---|---|")
    for rank, c in enumerate(candidates, start=1):
        mark = " 🏆" if rank == 1 else ""
        lines.append(f"| {rank} | `{c['run_name']}`{mark} | {_fmt(c['score'])} | {_fmt(c['guardrail'])} |")
    lines.append("")

    out_path = resolve_winner_dir() / WINNER_OUT_NAME.format(prefix=RUN_PREFIX)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print("\n=== winner (future RGB PSNR 기준) ===")
    print(f"  랭킹 ({'높을수록' if WINNER_HIGHER_IS_BETTER else '낮을수록'} 좋음, "
          f"지표={'+'.join(WINNER_METRICS)} 평균):")
    for rank, c in enumerate(candidates, start=1):
        mark = "  <-- WINNER" if rank == 1 else ""
        guard = f", obs_rgb={c['guardrail']:.3f}" if c["guardrail"] is not None else ""
        print(f"    {rank}. {c['run_name']:45s} future_rgb={c['score']:.3f}{guard}{mark}")
    print(f"\n  승자 저장: {out_path}")
    print(f"  나중에 따로 돌릴 {WINNER_FINAL_EPOCHS}ep 명령어:")
    print(f"    {final_command}")
    return winner["run_name"], winner["override"]


def main():
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_muvo_2D.py"
    if not train_script.exists():
        raise FileNotFoundError(f"train_muvo_2D.py not found beside this file: {train_script}")

    Path(SWEEP_LOG_DIR).mkdir(parents=True, exist_ok=True)

    summary = []
    started = time.time()
    for i, (suffix, override) in enumerate(TRIALS, start=1):
        run_name = f"{RUN_PREFIX}_{suffix}"

        if SKIP_IF_EXISTS and run_name_exists(run_name):
            print(f"\n[{i}/{len(TRIALS)}] SKIP {run_name} (logs/{run_name} already exists)")
            summary.append((run_name, "skipped", 0.0))
            continue

        combined = {**SHARED, **override}
        cmd = [sys.executable, str(train_script), "--run-name", run_name] + args_from(combined)

        console_log = Path(SWEEP_LOG_DIR) / f"{run_name}.log"
        print(f"\n[{i}/{len(TRIALS)}] === {run_name} ===")
        print("  cmd  :", " ".join(cmd))
        print("  log  :", console_log)
        t0 = time.time()
        with open(console_log, "w") as f:
            f.write(f"# {run_name}\n# cmd: {' '.join(cmd)}\n\n")
            f.flush()
            ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        status = "ok" if ret.returncode == 0 else f"fail (rc={ret.returncode})"
        print(f"  done : {status} in {elapsed/60:.1f} min")
        summary.append((run_name, status, elapsed))

        if ret.returncode != 0 and not CONTINUE_ON_FAIL:
            print(f"  ABORTING sweep (CONTINUE_ON_FAIL=False).")
            break

    total = time.time() - started
    print(f"\n=== sweep summary ({total/60:.1f} min total) ===")
    for run_name, status, elapsed in summary:
        print(f"  {run_name:50s}  {status:20s}  {elapsed/60:6.1f} min")
    print()
    print(f"TB:  tensorboard --logdir logs --port 6006")
    print(f"CSV: $ALPHA26_OUTPUT_ROOT/experiment_log.csv")

    # 모든 trial이 끝난 뒤 승자 1개만 골라 파일로 저장 (학습은 돌리지 않음).
    select_and_save_winner()


if __name__ == "__main__":
    main()
