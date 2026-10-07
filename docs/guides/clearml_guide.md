# ClearML 사용법 (alpha26 / `_muvo_2D` 학습 실험 관리용)

> **이 문서는 ClearML을 처음 접하는 사람 기준으로 작성**됨. 머신러닝 학습 코드는 돌릴 줄 알지만 실험 관리 툴은 모르는 단계.

---

## 0. ClearML이 뭔지 한 문단으로

학습 코드를 돌릴 때마다 어떤 hyperparameter로 어떤 결과 (loss curve, recon figure, 메트릭)가 나왔는지 일일이 기록하기 귀찮음. **ClearML**은 학습 스크립트에 한 줄 (`Task.init(...)`)만 추가하면, 그 run의 hyperparam·console 출력·TensorBoard 그래프·git commit hash·환경 변수까지 전부 **자동으로 웹 UI에 기록**해주는 도구. 나중에 web에서 여러 run을 한꺼번에 비교하거나, 동료에게 링크 한 번에 공유하는 데 쓰임.

비유: 학습 run마다 "노트북 한 페이지" (=Task)가 자동으로 생기고, 그 페이지 안에 hyperparam 설정/그래프/체크포인트 위치/실행 로그가 다 들어 있는 셈.

---

## 1. 첫 셋업 (한 번만)

### 1-1. ClearML 계정 만들기

가장 쉬운 길은 무료 SaaS 사용. 자체 서버 운영 안 해도 됨.

1. <https://app.clear.ml> 접속
2. 회원가입 (이메일 또는 Google login)
3. 로그인 후 우상단 프로필 → **Settings** → **Workspace** → **Create new credentials** 버튼 클릭
4. 팝업에 표시되는 큰 텍스트 블록 (다음과 같이 생김)을 복사해둠:
   ```
   api {
       web_server: https://app.clear.ml
       api_server: https://api.clear.ml
       files_server: https://files.clear.ml
       credentials {
           "access_key" = "ABC...XYZ"
           "secret_key" = "abc...xyz"
       }
   }
   ```
   ⚠️ secret_key는 한 번만 보임 — 닫으면 다시 못 봄. 안전한 곳에 저장.

### 1-2. 로컬·서버에 ClearML 설치

GPU 서버에서 학습할 거니, 학습이 돌아갈 머신마다 한 번씩:

```bash
conda activate kcy-alpha   # alpha26 env
pip install clearml
```

### 1-3. ClearML credential 파일 생성

서버 (또는 학습 머신)에서:

```bash
clearml-init
```

대화형 프롬프트가 뜸. 1-1에서 복사해둔 큰 텍스트 블록을 그대로 붙여넣기. 그러면 `~/.clearml/clearml.conf` 파일이 자동 생성됨.

확인:
```bash
cat ~/.clearml/clearml.conf
# 위에서 입력한 access_key / secret_key가 들어있으면 OK
```

이걸로 한 번에 끝. 학습 스크립트가 이 conf 파일을 자동으로 읽음.

### 1-4. 동작 테스트 (선택)

```bash
python -c "from clearml import Task; t = Task.init(project_name='test', task_name='hello'); t.close(); print('ok')"
```

`https://app.clear.ml`에서 좌측 메뉴 **Projects** → `test` → `hello` task가 나타나면 성공.

---

## 2. 학습 스크립트에서 ClearML 사용 (`_muvo_2D` 기준)

`train_muvo_2D.py`는 plan에 따라 이미 `Task.init(...)`이 들어가 있음. 그래서 별도로 코드 손볼 필요 없이 그냥 평소처럼 실행하면 됨:

```bash
conda activate kcy-alpha
cd /home/user/chaeyeon-kim/alpha26

python scripts/model_variants/train_muvo_2D.py \
  --steps 150000 --devices 4 \
  --strategy ddp_find_unused_parameters_true \
  --batch-size 2 --num-workers 2 \
  --window-index /home/user/chaeyeon-kim/processed/arrow_window_index_seq6.pkl \
  --cml-task muvo_2D_full_v1 --run-name muvo_2D_full_v1
```

실행하면 콘솔 첫 줄에 다음과 같은 ClearML 메시지가 뜸:

```
ClearML Task: created new task id=abc123... 
ClearML results page: https://app.clear.ml/projects/.../experiments/abc123...
```

그 URL을 브라우저로 열면 web UI에서 실시간으로 진행상황 추적 가능.

### 2-1. CLI flag 정리

| Flag | 설명 |
|---|---|
| `--cml-task <NAME>` | 이 run의 이름 (web UI에 표시됨). 미지정 시 `cfg.CML_TASK` default 사용 |
| `--cml-project <NAME>` | 이 run이 들어갈 project 이름. 미지정 시 `cfg.CML_PROJECT` (`alpha26_muvo_2D`) |
| `--no-clearml` | ClearML 비활성. 디버깅용 빠른 run에 사용 |
| `--run-name <NAME>` | 로컬 log 디렉토리 이름 (`logs/<run-name>/`). ClearML task 이름과 일치시키면 편함 |

### 2-2. 무엇이 자동으로 기록되나

`Task.init(...)` 한 줄로 다음이 **자동 capture**됨:

- **Hyperparameters**: `task.connect(cfg)` 호출로 cfg의 모든 키값이 web UI의 **HYPER PARAMETERS** 탭에 자동 등록.
- **Scalars (loss/metric curve)**: PyTorch Lightning이 `TensorBoardLogger`로 push한 모든 scalar가 ClearML web UI의 **SCALARS** 탭에 mirror됨.
- **Reconstruction figure**: 마찬가지로 TensorBoard로 push한 모든 image가 **DEBUG SAMPLES** 탭에 자동 mirror.
- **Console 출력**: stdout/stderr가 **CONSOLE** 탭에 실시간 기록.
- **Git commit hash + 변경된 파일 diff**: code reproducibility를 위해 자동 등록.
- **Python 환경**: `pip freeze` 결과가 자동 등록.
- **GPU/CPU usage**: 시스템 메트릭이 **SYSTEM METRICS** 탭에 기록.

### 2-3. 명시적으로 artifact 올리기 (선택)

`trainer_muvo_2D.test_step` 끝에서 prediction batch를 dump하고 싶을 때:

```python
from clearml import Task
task = Task.current_task()
if task is not None:
    task.upload_artifact(f'pred_batch_{batch_idx}', np.array(prediction_data))
```

→ web UI의 **ARTIFACTS** 탭에 binary로 저장. 다운로드 버튼 1 클릭으로 받을 수 있음.

---

## 3. Web UI 사용법 (실험 비교가 핵심)

### 3-1. Project / Task / Dataset 용어

| 용어 | 뜻 | alpha26 예시 |
|---|---|---|
| **Project** | 관련된 task들을 묶는 폴더 | `alpha26_muvo_2D` |
| **Task (= Experiment)** | 학습 run 하나 | `muvo_2D_full_v1`, `muvo_2D_no_aug` 등 |
| **Dataset** | 학습용 데이터를 버전 관리 (선택 사용) | alpha26은 Arrow 직접 사용, ClearML Dataset 안 씀 |

### 3-2. 한 task 살펴보기

좌측 메뉴 **Projects** → `alpha26_muvo_2D` → run 이름 클릭. 다음 탭들이 보임:

- **EXECUTION** — task 이름, git hash, 실행 commands, 환경
- **HYPER PARAMETERS** — cfg 값 전부 (sort/filter 가능)
- **CONFIGURATION** — `task.connect(cfg)`로 등록한 config 트리
- **CONSOLE** — stdout/stderr 로그 (검색 가능)
- **SCALARS** — loss/metric curve. 마우스 hover로 값 확인, smoothing slider 사용 가능
- **PLOTS** — 기타 plotly figure 등 (alpha26은 거의 사용 안 함)
- **DEBUG SAMPLES** — reconstruction figure 모음 (epoch별로 정렬)
- **ARTIFACTS** — 명시적으로 upload한 binary
- **SYSTEM METRICS** — GPU 메모리, CPU usage 시간별 그래프
- **INFO** — 작성자, 시작/종료 시간, status

### 3-3. 여러 run 비교 (핵심 기능)

1. 좌측 **Projects** → `alpha26_muvo_2D` 열고 비교할 task 두 개 이상 체크박스로 선택
2. 상단 **COMPARE EXPERIMENTS** 버튼 클릭
3. 자동으로 다음을 비교 표시:
   - **HYPER PARAMETERS**: 다른 값만 강조 표시
   - **SCALARS**: 같은 metric 키를 여러 곡선으로 overlay
   - **DEBUG SAMPLES**: 같은 epoch의 figure를 옆으로 나란히

예시: `_muvo` baseline run과 `_muvo_2D_v1` run을 비교하면 RGB recon이 sharp해졌는지 한 화면에서 확인 가능.

### 3-4. Task 검색 / 필터

상단 검색창에 `tags:rssmtd` 또는 `name:muvo_2D` 등으로 검색. cfg 변경했지만 결과 안 좋았던 run을 빠르게 찾을 때 유용.

### 3-5. 동료에게 공유

비교 화면 또는 task 상세 페이지 URL을 그대로 복사해서 Slack/email로 보내면 됨. 같은 workspace 멤버는 로그인 후 동일 view 접근.

---

## 4. 자주 쓰는 패턴

### 4-1. 빠른 디버깅 run (ClearML 비활성)

```bash
python scripts/model_variants/train_muvo_2D.py --no-clearml --steps 100 ...
```

web UI에 안 올라감. 로컬 TensorBoard만 보임. PR 머지 전 sanity check할 때.

### 4-2. tag로 ablation 묶기

```bash
python scripts/model_variants/train_muvo_2D.py --cml-task muvo_2D_lidar_re_0.0 --tags ablation_lidar_re ...
python scripts/model_variants/train_muvo_2D.py --cml-task muvo_2D_lidar_re_0.1 --tags ablation_lidar_re ...
```

(`--tags`는 cfg.CML_TAGS에 append하도록 train_muvo_2D.py에서 처리)

web UI에서 `tags:ablation_lidar_re`로 필터링 → 비교.

### 4-3. 죽은 task 정리

학습이 OOM/Ctrl-C 등으로 중간에 죽은 task는 status가 "Aborted"로 표시됨. 좌측 체크 → 우상단 **Delete** 또는 **Archive**로 정리.

### 4-4. 체크포인트 다운로드

ClearML이 자동으로 `epoch=N-step=M.ckpt`를 ARTIFACTS에 register함 (PyTorch Lightning + ClearML 연동). 다운로드 버튼 클릭 → 로컬로 받음.

⚠️ alpha26의 큰 ckpt는 자동 업로드되지 않을 수 있음 — 명시적 `task.upload_artifact('best_ckpt', '/path/to/best.ckpt')` 호출 권장.

---

## 5. 자주 겪는 문제

### Q. `Task.init()`에서 hang됨
A. credential 못 찾음. `clearml-init` 다시 실행해서 conf 재생성. 또는 인터넷이 막혀있는 서버라면 `--no-clearml` 사용.

### Q. web UI에서 scalar가 안 보임
A. (a) `--no-clearml`로 실행했거나, (b) 학습이 1 step 이상 돌지 않음. 콘솔에 `ClearML results page: ...` URL 떴는지 확인.

### Q. "Task is offline" 에러
A. 인터넷 끊김. 다시 연결 후 재실행하면 offline 동안 쌓인 데이터가 자동 sync됨.

### Q. credential을 잘못 입력함
A. `~/.clearml/clearml.conf` 파일 직접 삭제 → `clearml-init` 다시 실행.

### Q. ClearML 무료 사용량 한도가 있나요
A. SaaS 무료 plan은 1 workspace + 일정 data storage. alpha26 정도 규모는 대체로 충분. 한도 넘으면 self-hosted server (`clearml-server` Docker) 운영 옵션.

### Q. team끼리 share하고 싶음
A. SaaS 우상단 프로필 → **Workspace** → 다른 사람을 멤버로 invite. 같은 workspace 안에서는 모든 project/task가 공유됨.

---

## 6. ClearML vs 기존 alpha26 인프라

| 항목 | alpha26 기존 | ClearML 도입 후 |
|---|---|---|
| Hyperparam 기록 | `results/experiment_log.csv` 1줄 row | ClearML HYPER PARAMETERS 탭 (cfg 전체) |
| Scalar 추적 | TensorBoard 로컬 | TensorBoard + ClearML web mirror |
| Figure 보기 | TensorBoard 로컬 | TensorBoard + ClearML DEBUG SAMPLES |
| run 비교 | TensorBoard `--logdir_spec` | ClearML COMPARE EXPERIMENTS (훨씬 편함) |
| 공유 | tar.gz로 logs/ 전달 | ClearML URL 공유 |
| Git/env reproducibility | 수동 (commit hash 기록) | 자동 capture |
| 큰 artifact 저장 | `server_results/` rsync | ClearML ARTIFACTS |

**기존 인프라는 그대로 유지**: TensorBoard 로컬도 여전히 사용 가능 (`tensorboard_guide.md`), `experiment_log.csv`도 그대로. ClearML이 그 위에 web UI + 비교 + 공유 기능을 추가하는 것.

---

## 7. 관련 자료

- 공식 docs: <https://clear.ml/docs/latest/docs/>
- alpha26 plan: `alpha26/docs/muvo_2D_port_plan.md` §F (실험 트래킹)
- TensorBoard 보조 사용: `alpha26/docs/tensorboard_guide.md`
- `muvo_2d` 코드에서 ClearML 사용 예: `muvo_2d/prediction.py:23-25, 98`, `muvo_2d/sim_run.py:23-25`
