# Repository Guide

[← README](../README.md) · **English** | [한국어](repository_ko.md)

## Folder Tree

```text
Alpha-Project-2026/
├── README.md / README_ko.md       Project overview (English / Korean)
├── kcy-alpha.yml                  Conda environment (CUDA 11.8 + PyTorch)
├── assets/                        README figures not stored elsewhere (full-data server run)
│
├── notebooks/                     [team] Where trainer11 was written
│   ├── training/                      trainer11*.ipynb — source of the definitions
│   ├── prototypes/                    image encoder, decoder/sequence, fusion transformer
│   └── evaluation/                    checkpoint metrics, loss plots, frame inspection
│
├── scripts/                       [team] Runnable code
│   ├── trainer11_server_defs.py       auto-exported copy of the notebook setup cells
│   ├── train_full_trainer11_server.py, tiny_overfit_trainer11_server.py   server CLI
│   ├── build_arrow_manifest.py, build_arrow_window_index.py   dataset index (run once)
│   ├── eval_psnr_from_ckpt.py, export_tensorboard_metrics.py, analyze_lidar_scale.py
│   └── model_variants/                *_muvo.py   1D MUVO-style line
│                                      *_muvo_2D.py 2D-token RSSMTD line (MUVO-2D), tune/viz/find_lr
│
├── diffuvo/                       [team] Latent-diffusion world model
│   ├── WM/                            joint denoiser over the future frames; docs/diffuvo_plan.md
│   └── AR/                            frame-wise autoregressive denoiser, per-step action
│
├── experiments/                   [team] Side lines built on MUVO-2D
│   ├── hj/                            single-stage diffusion overfits; DIFFUSION_FUTURE_DESIGN.md
│   └── yj/                            RSSMTD overfit + GRU waypoint head; TRAJECTORY_TRANSFUSER.md
│
├── results/
│   ├── experiment_log.csv             one row per run instance (133 rows, 103 distinct runs) — the authoritative record
│   └── figures/, checkpoint_eval_*    reconstructions and loss curves
│
└── docs/
    ├── contributions / postmortem / timeline / repository (+ _ko)   this write-up
    ├── paper/                         MUVO paper text, summary and Korean translation
    ├── code_audit/, improvement_plan/ 1D vs 2D, ours vs upstream audits
    ├── muvo2d/, trainer11/, journal/, experiments/   working notes from the semester
    └── guides/                        TensorBoard, ClearML, experiment-log conventions
```

- `[team]` written by our team
- Paths in `diffuvo/*/run_logs/*.sh` and older notes: original workstation layout (`alpha26/archive/...`), kept as a record of what was run
- Code paths from `ALPHA26_ROOT`, `CARLA_ARROW_ROOT`, `ALPHA26_LOG_ROOT` / `ALPHA26_OUTPUT_ROOT`

## Workflow

### How the Workflow Changed

```mermaid
flowchart LR
    A["① Study<br/>03.02–03.27<br/>Notion pages,<br/>DAVE-2 notebooks"] --> B["② Fork attempt<br/>03.23–04.06<br/>adapt MUVO code"]
    B --> C["③ From the paper<br/>04.13–04.30<br/>notebooks passed<br/>around in chat"]
    C --> D["④ GitHub<br/>04.30–05.12<br/>one repo, review-driven<br/>restructure"]
    D --> E["⑤ Scripts + server<br/>05.11–05.29<br/>.py modules, DDP,<br/>experiment log"]
    E --> F["⑥ Parallel lines<br/>05.24–06.06<br/>diffuvo/, hj, yj"]
    F -.->|2026.06–10| G["Cleanup and<br/>this write-up"]

    classDef final fill:#d8f0dc,stroke:#3c8a4f,color:#1b3d24;
    class E,F final;
```

### Phase by Phase

| Phase | Code sharing | Running | Version control | Division of work |
|---|---|---|---|---|
| ① Study | Notion pages | Laptops, Jupyter notebooks | None | One paper per person |
| ② Fork attempt | MUVO repository | Lab GPU server | Upstream clone | Together |
| ③ From the paper | `.ipynb` files posted in the chat | Laptops | File names (`trainer11.ipynb`) | By module: image / LiDAR / transformer / RSSM, then decoders |
| ④ GitHub | This repository (folders sorted 05.07) | Laptops + server | `main` only | Team lead integrates, reviews by Codex / Claude |
| ⑤ Scripts + server | `scripts/`, exported defs | 4-GPU DDP server, tmux, Tailscale SSH via campus WSL | `main`, run-scoped logs | Experiments split by question |
| ⑥ Parallel lines | Per-person folders | Server; Diffuvo on one 12 GB workstation GPU | Folders, not branches | Diffusion, waypoints, MUVO-2D in parallel |

### The Experiment Loop (phases ⑤–⑥)

1. Change one setting, give the run a descriptive `--run-name`
2. Overfit 1–16 windows first; full data only when the overfit looks right
3. Watch TensorBoard; reconstructions saved to `results/figures/` at the end of the run
4. `ExperimentCSVLogger` appends a row to `results/experiment_log.csv`
5. Compare in the table, write a note under `docs/`, bring it to the next advisor meeting

## Not in the Repository

| What | Where it lives |
|---|---|
| Dataset (~81 GB of Arrow files) | [`immanuelpeter/carla-autopilot-multimodal-dataset`](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset), converted with our scripts |
| Checkpoints and TensorBoard logs | Kept offline, one checkpoint per run |
| DAVE-2 notebooks, meeting notes, reports | Team Notion |

## Third-party Sources

| Source | Used for |
|---|---|
| [MUVO](https://github.com/fzi-forschungszentrum-informatik/muvo) (repository cited by the paper, 1D latent) | Architecture reference; the 1D `_muvo` line ports its RSSM, representation model and sine position embedding almost line by line |
| [MUVO, first author's GitHub](https://github.com/daniel-bogdoll/MUVO) (`2D` branch, released 2D weights) | MUVO-2D modules (`ConvGRUCellGlo`, positional embedding) follow the 2D code and cite it in docstrings |
| MUVO paper (IEEE IV) | Text and translation under `docs/paper/` for study |
| [TransFuser](https://github.com/autonomousvision/transfuser) | GRU waypoint decoder design |
| DWM, DiT | Diffusion world-model design (`diffuvo/`, `experiments/hj`) |
