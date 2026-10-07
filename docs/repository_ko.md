# 저장소 안내

[← README](../README_ko.md) · [English](repository.md) | **한국어**

## 폴더 구조

```text
Alpha-Project-2026/
├── README.md / README_ko.md       프로젝트 개요(영어 / 한국어)
├── kcy-alpha.yml                  Conda 환경(CUDA 11.8 + PyTorch)
├── assets/                        다른 위치에 없는 README 그림(전체 데이터 서버 run)
│
├── notebooks/                     [team] trainer11 작성 위치
│   ├── training/                      trainer11*.ipynb — 모델 정의 원본
│   ├── prototypes/                    이미지 encoder · decoder/sequence · fusion transformer
│   └── evaluation/                    checkpoint 지표 · loss 그래프 · 프레임 점검
│
├── scripts/                       [team] 실행 코드
│   ├── trainer11_server_defs.py       notebook 설정 cell 자동 추출본
│   ├── train_full_trainer11_server.py, tiny_overfit_trainer11_server.py   서버 CLI
│   ├── build_arrow_manifest.py, build_arrow_window_index.py   데이터셋 index(최초 1회)
│   ├── eval_psnr_from_ckpt.py, export_tensorboard_metrics.py, analyze_lidar_scale.py
│   └── model_variants/                *_muvo.py   1D MUVO 방식 계열
│                                      *_muvo_2D.py 2D-token RSSMTD 계열(MUVO-2D) · tune/viz/find_lr
│
├── diffuvo/                       [team] latent-diffusion world model
│   ├── WM/                            미래 프레임을 함께 처리하는 denoiser · docs/diffuvo_plan.md
│   └── AR/                            프레임별 autoregressive denoiser · step별 action
│
├── experiments/                   [team] MUVO-2D 기반 별도 실험
│   ├── hj/                            single-stage diffusion overfit · DIFFUSION_FUTURE_DESIGN.md
│   └── yj/                            RSSMTD overfit + GRU waypoint head · TRAJECTORY_TRANSFUSER.md
│
├── results/
│   ├── experiment_log.csv             run 실행당 1행(133행 · 서로 다른 run 103개) — 기준 기록
│   └── figures/, checkpoint_eval_*    복원 결과 · loss 곡선
│
└── docs/
    ├── contributions / postmortem / timeline / repository (+ _ko)   프로젝트 정리 문서
    ├── paper/                         MUVO 논문 원문 · 요약 · 한국어 번역
    ├── code_audit/, improvement_plan/ 1D vs 2D · 자체 구현 vs upstream 비교 점검
    ├── muvo2d/, trainer11/, journal/, experiments/   학기 중 작업 노트
    └── guides/                        TensorBoard · ClearML · 실험 로그 규칙
```

- `[team]`: 팀 직접 작성
- `diffuvo/*/run_logs/*.sh` · 이전 노트의 경로: 원래 workstation 구조(`alpha26/archive/...`) · 당시 실행 기록으로 보존
- 코드 경로 기준: `ALPHA26_ROOT` · `CARLA_ARROW_ROOT` · `ALPHA26_LOG_ROOT` / `ALPHA26_OUTPUT_ROOT`

## 작업 방식

### 작업 방식 변화

```mermaid
flowchart LR
    A["① 학습<br/>03.02–03.27<br/>Notion 페이지<br/>DAVE-2 notebook"] --> B["② fork 시도<br/>03.23–04.06<br/>MUVO 코드 수정"]
    B --> C["③ 논문 기반 구현<br/>04.13–04.30<br/>채팅으로<br/>notebook 공유"]
    C --> D["④ GitHub<br/>04.30–05.12<br/>단일 저장소<br/>리뷰 기반 구조 개편"]
    D --> E["⑤ 스크립트 + 서버<br/>05.11–05.29<br/>.py 모듈 · DDP<br/>실험 로그"]
    E --> F["⑥ 병렬 실험<br/>05.24–06.06<br/>diffuvo/ · hj · yj"]
    F -.->|2026.06–10| G["저장소 정리<br/>문서 작성"]

    classDef final fill:#d8f0dc,stroke:#3c8a4f,color:#1b3d24;
    class E,F final;
```

### 단계별 비교

| 단계 | 코드 공유 | 실행 환경 | 버전 관리 | 역할 분담 |
|---|---|---|---|---|
| ① 학습 | Notion 페이지 | 노트북 PC · Jupyter notebook | 없음 | 1인 1논문 |
| ② fork 시도 | MUVO 저장소 | 연구실 GPU 서버 | upstream clone | 공동 작업 |
| ③ 논문 기반 구현 | 채팅으로 `.ipynb` 파일 공유 | 노트북 PC | 파일명(`trainer11.ipynb`) | 이미지 / LiDAR / transformer / RSSM별 분담 후 decoder 구현 |
| ④ GitHub | 현재 저장소 · 05.07 폴더 정리 | 노트북 PC + 서버 | `main`만 사용 | 팀장 통합 · Codex / Claude 리뷰 |
| ⑤ 스크립트 + 서버 | `scripts/` · 추출한 정의 파일 | 4-GPU DDP 서버 · tmux · 교내 WSL 경유 Tailscale SSH | `main` · run별 로그 | 연구 질문별 실험 분담 |
| ⑥ 병렬 실험 | 개인별 폴더 | 서버 · Diffuvo는 12 GB workstation GPU 1개 | branch 대신 폴더 구분 | diffusion · waypoint · MUVO-2D 병행 |

### 실험 절차 (⑤–⑥단계)

1. 설정 하나 변경 · 실험을 설명하는 `--run-name` 지정
2. 1–16 window overfit 우선 · 정상 결과 확인 후 전체 데이터 학습
3. TensorBoard 확인 · run 종료 시 `results/figures/`에 복원 결과 저장
4. `ExperimentCSVLogger`로 `results/experiment_log.csv`에 행 추가
5. 표 비교 · `docs/`에 기록 · 다음 교수님 미팅에서 공유

## 저장소 외부 자료

| 항목 | 보관 위치 |
|---|---|
| 데이터셋(Arrow 파일 약 81 GB) | [`immanuelpeter/carla-autopilot-multimodal-dataset`](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset) · 자체 스크립트로 변환 |
| checkpoint · TensorBoard 로그 | 오프라인 보관 · run당 checkpoint 하나 |
| DAVE-2 notebook · 미팅 기록 · 보고서 | 팀 Notion |

## 외부 출처

| 출처 | 활용 |
|---|---|
| [MUVO](https://github.com/fzi-forschungszentrum-informatik/muvo) (논문에서 인용한 저장소 · 1D latent) | 구조 참고 · 1D `_muvo` 계열에서 RSSM · representation model · sine position embedding을 거의 줄 단위로 이식 |
| [MUVO 제1저자 GitHub](https://github.com/daniel-bogdoll/MUVO) (`2D` branch · 공개 2D 가중치) | MUVO-2D 모듈(`ConvGRUCellGlo` · positional embedding)에 2D 코드 반영 · docstring에 출처 명시 |
| MUVO 논문(IEEE IV) | 학습용 원문 · 번역을 `docs/paper/`에 보관 |
| [TransFuser](https://github.com/autonomousvision/transfuser) | GRU waypoint decoder 설계 |
| DWM · DiT | diffusion world model 설계(`diffuvo/` · `experiments/hj`) |
