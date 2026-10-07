# Upstream 1D Code Audit — `/home/carol/chaeyeon-kim/muvo/`

본 문서는 upstream MUVO 1D 레퍼런스 리포지토리(`/muvo/`)를 모든 leaf 디렉터리 단위로 감사한 결과를 종합한다. 각 leaf는 별도 Explore 서브에이전트로 조사되었고, 결과를 모달리티/책임별로 묶었다.

---

## 0. 디렉터리 트리 요약

```
muvo/
├── train.py / prediction.py / data_collect.py / sim_run.py / constants.py
├── muvo/                            ← world model core
│   ├── config.py, losses.py, metrics.py, trainer.py, visualisation.py
│   ├── data/    (carlagym_utils.py, dataset.py, dataset_utils.py)
│   ├── layers/  (layers.py)
│   ├── models/  (common.py, frustum_pooling.py, mile.py, preprocess.py, transition.py, utils.py)
│   └── utils/   (carla_utils.py, geometry_utils.py, instance_utils.py, network_utils.py)
├── data/        (data_preprocessing.py, dataset_utils.py, generate_voxels.py, pcd.py)
├── utils/       (saving_utils.py, server_utils.py)
├── rl_birdview/
│   ├── rl_birdview_agent.py
│   ├── models/   (distributions.py, ppo.py, ppo_buffer.py, ppo_policy.py, torch_layers.py, torch_util.py)
│   └── utils/    (rl_birdview_wrapper.py, wandb_callback.py)
└── carla_gym/   ← CARLA env scaffolding
    ├── __init__.py, carla_multi_agent_env.py
    ├── envs/{__init__.py, suites/{endless_env.py, leaderboard_env.py}}
    ├── utils/   (birdview_map.py, config_utils.py, dynamic_weather.py, gps_utils.py,
    │             hazard_actor.py, traffic_light.py, transforms.py)
    └── core/
        ├── obs_manager/
        │   ├── obs_manager.py, obs_manager_handler.py
        │   ├── actor_state/ (control.py, route.py, speed.py, velocity.py)
        │   ├── birdview/    (chauffeurnet.py, chauffeurnet_label.py)
        │   ├── camera/      (depth_semantic.py, depth_semantic_m.py, rgb.py)
        │   ├── lidar/       (ray_cast.py, ray_cast_multi.py, ray_cast_semantic.py)
        │   ├── navigation/  (gnss.py, waypoint_plan.py)
        │   └── object_finder/(ego.py, pedestrian.py, stop_sign.py, traffic_light_new.py, vehicle.py)
        ├── task_actor/
        │   ├── common/     (task_vehicle.py, criteria/*, navigation/*)
        │   ├── ego_vehicle/ (ego_vehicle_handler.py, reward/valeo_action.py, terminal/*)
        │   └── scenario_actor/(scenario_actor_handler.py, agents/{basic_agent.py, constant_speed_agent.py, utils/*})
        ├── zombie_vehicle/ (zombie_vehicle.py, zombie_vehicle_handler.py)
        └── zombie_walker/  (zombie_walker.py, zombie_walker_handler.py)
```

총 32 leaf 디렉터리 (Python 모듈 보유). 모델 핵심은 `muvo/muvo/`, 데이터 수집은 `data/` + `utils/` + `carla_gym/`, RL agent는 `rl_birdview/`.

---

## 1. World Model Core — `muvo/muvo/`

### 1.1 Top-level (`muvo/muvo/{config,losses,metrics,trainer,visualisation}.py`)

| 파일 | 책임 | 주요 export |
|---|---|---|
| `config.py` (370줄) | fvcore CfgNode 기반 hyperparameter 스키마 정의 | `CfgNode`, `get_cfg()`, `LOSSES.WEIGHT_*`, `MODEL.LIDAR/ROUTE/TRANSITION/REWARD.*` 등 |
| `losses.py` (376줄) | 11종 loss 모듈 | `SegmentationLoss, RegressionLoss, SpatialRegressionLoss, ProbabilisticLoss, KLLoss, VoxelLoss, SemScalLoss, GeoScalLoss, SSIMLoss, CDLoss, DiceLoss` |
| `metrics.py` (318줄) | 평가 지표 | `SSCMetrics, SSIMMetric, CDMetric, CDMetric0, PSNRMetric` + `get_iou`, `get_accuracy` |
| `trainer.py` (1096줄) | PL LightningModule | `WorldModelTrainer` — shared_step, compute_loss(11+ terms), add_metrics, visualise, configure_optimizers(AdamW+OneCycleLR) |
| `visualisation.py` (342줄) | BEV/RGB/flow/heatmap 렌더링 유틸 | `prepare_final_display_image`, `convert_bev_to_image`, `add_ego_vehicle`, `flow_to_image` 등 |

Paper 매핑: `config.py` ↔ §IV-A 하이퍼파라미터, `losses.py` ↔ §IV-A 식 (1), `metrics.py` ↔ §IV-B 평가 지표(PSNR, Chamfer, IoU+/-), `trainer.py` ↔ §III + §IV-A 학습 루프.

### 1.2 `muvo/muvo/data/` — 데이터 로더

| 파일 | 책임 |
|---|---|
| `carlagym_utils.py` (67줄) | CARLA 좌표 변환(`vec_global_to_ref`, `carla_rot_to_mat`, `gps_to_location`) |
| `dataset.py` (386줄) | `DataModule` + `CarlaDataset` — RGB/depth/LiDAR/voxel/BEV/route 멀티모달 sample 제공 |
| `dataset_utils.py` (129줄) | BEV 라벨 디코딩, GPS 전처리, instance mask, binary↔integer 변환 |

`CarlaDataset.__getitem__` dict 키: `image, route_map, semantic_image, range_view_pcd_xyzd, range_view_pcd_seg, points_raw, num_points, voxel, birdview, birdview_label, instance_label, image_instance_mask, depth, depth_color, steering, throttle_brake, speed, intrinsics, extrinsics` — paper §IV-A 입력군과 일치 (RGB + depth + LiDAR + route + voxel + action + speed).

LiDAR range view 변환: `PointCloud.do_range_projection()` (in `muvo/muvo/utils/geometry_utils.py`).

### 1.3 `muvo/muvo/layers/`

`layers.py` (358줄): `BasicBlock, RestrictionActivation, ConvBlock, Bottleneck, Interpolate, Upsampling, UpsamplingAdd, UpsamplingConcat, ActivatedNormLinear, Flatten, VoxelsSumming(autograd.Function)`. 표준 ResNet 빌딩 블록 + voxel 합산 autograd op.

### 1.4 `muvo/muvo/models/` ← Paper §III 핵심

| 파일 | 책임 | Paper 매핑 |
|---|---|---|
| `mile.py` (49KB) | `Mile` — 메인 멀티모달 world model. ResNet18(cam) + ResNet18(LiDAR) + 선택적 transformer fusion + frustum pooling(BEV 변형) + point pillars(LiDAR 변형) + route encoder + RSSM + 8개 디코더(BEV, RGB, LiDAR re, LiDAR seg, Sem image, Depth, Voxel) + Policy | §III 전체 |
| `transition.py` (192줄) | `RSSM` — 1D `(B,S,D)` 상태, `nn.GRUCell` 사용. observe_step/imagine_step, use_dropout(prior 대체) | §III-C |
| `common.py` (29KB) | `RouteEncode, GRUCellLayerNorm, Policy, Decoder, DecoderDS, BevDecoder, VoxelDecoder1, ConvDecoder, SegmentationHead, RGBHead, DepthHead, LidarReHead, LidarSegHead, SemHead, VoxelSemHead, PositionEmbeddingSine, PointPillarNet` | §III-A/B/D 빌딩블록 |
| `frustum_pooling.py` (8.5KB) | `FrustumPooling` — Lift-Splat-Shoot 스타일 이미지→BEV voxel pooling. `get_geometry, voxel_pooling, get_depth_map` | §IV-B "BEV mapping [73]" 구현 |
| `preprocess.py` (17KB) | `PreProcess`(이미지 정규화 + multi-scale 라벨), `PixelAugmentation`(blur/sharpen/colorjitter), `RouteAugmentation` | §IV-A 자기지도 학습 셋업 |
| `utils.py` (287B) | `Concat`, `PixelNorm` | – |

RSSM 핵심 (1D):
- `h_{t+1} = GRUCell(pre_gru_net(s_t), h_t)` — action은 prior/posterior에만 (default: `action_in_gru=False`).
- prior `p(s_t | h_t, a_{t-1}) ~ N_θ(h_t, a_{t-1})` 와 posterior `q(s_t | o≤t, a<t) ~ N_φ(o_t, h_t, a_t)` 모두 MLP.
- 출력 상태: `[hidden_state(B,H), sample(B,S)]` concat → 디코더 입력.

### 1.5 `muvo/muvo/utils/`

| 파일 | 책임 |
|---|---|
| `carla_utils.py` (24줄) | CARLA `Vector3D`, wheelbase, steering→curvature, GPS dict 변환 |
| `geometry_utils.py` (358줄) | 카메라 intrinsics/extrinsics, `PointCloud` 클래스(range projection), `lidar_to_histogram_features`, `compute_pcd_transformation` (Open3D ICP) |
| `instance_utils.py` (36줄) | `convert_instance_mask_to_center_and_offset_label` (인스턴스 라벨→center heatmap+offset) |
| `network_utils.py` (145줄) | `set_bn_momentum, preprocess_batch, pack/unpack_sequence_dim, freeze_network, NormalizeInverse, calculate_birds_eye_view_parameters` 등 |

---

## 2. Offline Data Processing — `muvo/data/`

| 파일 | 책임 | Paper 매핑 |
|---|---|---|
| `data_preprocessing.py` (248줄) | depth+LiDAR → 3D point → ego frame 변환 → `voxel_filter`로 occupancy grid 생성. 23-class semantic 라벨 매핑. | §IV-A voxel target 생성 |
| `dataset_utils.py` (121줄) | BEV bit-packing/argmax, GPS 전처리, route map 추출 | – |
| `generate_voxels.py` (169줄) | Hydra-기반 멀티프로세스 voxelization 파이프라인 (`voxelize_dir`, `voxelize_one`) | §IV-A occupancy 데이터 생성 |
| `pcd.py` (10줄) | 빈 placeholder | – |

Voxel filter 알고리즘: 점들을 voxel hash로 정렬 → 셀별로 가장 가까운 점의 라벨 할당 → roadlines 라벨은 우선권. `cfg.voxel_size`, `cfg.center`, `cfg.fov`, `cfg.mask_ego`로 파라미터화.

---

## 3. Common Utils — `muvo/utils/`

| 파일 | 책임 |
|---|---|
| `saving_utils.py` (343줄) | `DataWriter` — 에피소드 단위 데이터 수집/검증/저장. image/depth/LiDAR/birdview/route+ supervision+reward 누적 후 PNG/NPY/DataFrame 작성. 종결 시 위반 시나리오의 마지막 프레임 제거. |
| `server_utils.py` (66줄) | `CarlaServerManager.start/stop` — CarlaUE4 프로세스 관리. `CARLA_FPS=10`. |

---

## 4. Top-level scripts

| 파일 | 책임 | 비고 |
|---|---|---|
| `constants.py` (230줄) | 전역 상수: `CARLA_FPS=10`, `EGO_VEHICLE_DIMENSION`, `ROUTE_COMMANDS`, `BIRDVIEW_COLOURS`, `SEMANTIC_SEG_WEIGHTS`, `VOXEL_SEG_WEIGHTS`, `VOXEL_LABEL_CARLA`(23-class), `VOXEL_LABEL`(2-class), `LABEL_MAP`. | – |
| `train.py` (120줄) | PL 학습 진입점: `get_cfg` → DataModule → WorldModelTrainer → TensorBoard logger + `MyModelCheckpoint`. `cfg.STEPS, PRECISION, VAL_CHECK_INTERVAL, LOG_DIR, TAG`. | – |
| `prediction.py` (120줄) | 모델 추론 + ClearML artifact 업로드. `rgb_label/pcd_label/voxel_label` vs `rgb_re/pcd_re/voxel_re` vs `rgb_im/pcd_im/voxel_im` 비교. | – |
| `sim_run.py` (120줄) | `prediction.py`와 동일. | – |
| `data_collect.py` (302줄) | `run_single()` 에피소드 루프 — CARLA env step + obs/sup/reward 수집 + `DataWriter`로 저장 + 재개 체크포인트. `cfg.test_suites, n_episodes, run_time` 등. | – |

---

## 5. RL Birdview Agent — `muvo/rl_birdview/`

| Leaf | Files | Role |
|---|---|---|
| `rl_birdview/` | `rl_birdview_agent.py` | RL agent 진입점 |
| `rl_birdview/models/` | `distributions.py, ppo.py, ppo_buffer.py, ppo_policy.py, torch_layers.py, torch_util.py` | PPO 정책/buffer/torch util |
| `rl_birdview/utils/` | `rl_birdview_wrapper.py, wandb_callback.py` | env wrapper + wandb |

데이터 수집 시 expert RL agent로 사용됨 (paper §IV-A "expert reinforcement learning agent [24], [83]").

---

## 6. CARLA Gym Environment — `muvo/carla_gym/`

세부 leaf 인벤토리(파일 목록 + 1줄 역할):

### Root + envs
- `carla_gym/` : `__init__.py, carla_multi_agent_env.py` — multi-agent env wrapper
- `carla_gym/envs/` : `__init__.py` — env 등록
- `carla_gym/envs/suites/` : `endless_env.py, leaderboard_env.py` — endless 주행 + LeaderBoard 시나리오 스위트
- `carla_gym/utils/` : `birdview_map.py, config_utils.py, dynamic_weather.py, gps_utils.py, hazard_actor.py, traffic_light.py, transforms.py` — 지도/날씨/GPS/신호/변환 유틸

### core/obs_manager (센서 관찰자)
- `core/obs_manager/` : `obs_manager.py, obs_manager_handler.py`
- `.../actor_state/` : `control.py, route.py, speed.py, velocity.py`
- `.../birdview/` : `chauffeurnet.py, chauffeurnet_label.py`
- `.../camera/` : `depth_semantic.py, depth_semantic_m.py, rgb.py`
- `.../lidar/` : `ray_cast.py, ray_cast_multi.py, ray_cast_semantic.py` ← paper의 LiDAR sensor 원본
- `.../navigation/` : `gnss.py, waypoint_plan.py`
- `.../object_finder/` : `ego.py, pedestrian.py, stop_sign.py, traffic_light_new.py, vehicle.py`

### core/task_actor
- `task_actor/common/` : `task_vehicle.py`
- `.../common/criteria/` : 7 files (collision, blocked, red_light, stop_sign, route_deviation 등 종료 조건)
- `.../common/navigation/` : `global_route_planner.py, map_utils.py, route_manipulation.py`
- `.../ego_vehicle/` : `ego_vehicle_handler.py`
- `.../ego_vehicle/reward/` : `valeo_action.py`
- `.../ego_vehicle/terminal/` : `leaderboard.py, leaderboard_dagger.py, valeo.py, valeo_no_det_px.py`
- `.../scenario_actor/` : `scenario_actor_handler.py`
- `.../scenario_actor/agents/` : `basic_agent.py, constant_speed_agent.py`
- `.../scenario_actor/agents/utils/` : `controller.py, local_planner.py, misc.py`

### core/zombie
- `zombie_vehicle/` : `zombie_vehicle.py, zombie_vehicle_handler.py`
- `zombie_walker/` : `zombie_walker.py, zombie_walker_handler.py`

→ 이 모든 carla_gym leaf는 **CARLA 시뮬레이션 환경 스캐폴딩**이며 world model 학습 코드와 직접 결합되지 않는다. 데이터 수집(`data_collect.py`)에서만 사용. alpha26은 이미 Arrow로 변환된 데이터셋을 사용하므로 carla_gym 코드는 alpha 비교 범위 밖.

---

## 7. Paper §III/§IV ↔ upstream 1D 트레이서빌리티

| Paper | upstream 1D 위치 |
|---|---|
| §III.A 이미지 600×960 + LiDAR cylindrical | `muvo/data/data_preprocessing.py`(데이터 생성) + `muvo/muvo/utils/geometry_utils.py:PointCloud.do_range_projection` |
| §III.A ResNet18 multi-layer feature | `muvo/muvo/models/mile.py` (timm ResNet18) |
| §III.B 2D sinusoidal PE + sensor embedding | `muvo/muvo/models/common.py:PositionEmbeddingSine` + mile.py 내 type embedding |
| §III.B k-layer transformer encoder | `mile.py` 내 nn.TransformerEncoder |
| §III.C 1D state RSSM | `muvo/muvo/models/transition.py:RSSM` |
| §III.D 카메라 디코더 | `muvo/muvo/models/common.py:ConvDecoder` + `RGBHead` |
| §III.D LiDAR 디코더 | `common.py:ConvDecoder` + `LidarReHead/LidarSegHead` |
| §III.D Voxel 3D 디코더 | `common.py:VoxelDecoder1` + `VoxelSemHead` |
| §IV-A 식 (1) 손실 | `muvo/muvo/losses.py` + `trainer.py:compute_loss` (11+ terms) |
| §IV-B 8 fusion combos | `mile.py` 내 옵션 (BEV/PP/RV, AVG/FC/TR) |
| §IV-C PTF/PTO/NPT | `trainer.py:configure_optimizers` 내 freeze 옵션 |

---

## 8. 핵심 관찰

1. **`muvo/muvo/models/mile.py`가 단일 거대 모델 클래스** — fusion/encoder/RSSM/decoder를 모두 조립. paper §III 전체를 한 파일로 구현.
2. **RSSM은 별도 파일(`transition.py`)** — 1D 상태(`(B,S,D)`), GRUCell, MLP `RepresentationModel`. action은 prior/posterior MLP 입력에만(기본).
3. **Decoder는 `common.py`에 헤드별로 산재** — RGB/Depth/Lidar/Sem image는 ConvDecoder + head, Voxel은 별도 VoxelDecoder1(3D ConvT) + VoxelSemHead.
4. **데이터 수집 + 학습이 한 리포지토리에 결합** — CARLA env + data collection + world model 학습이 모두 들어있다. alpha26은 이미 수집된 Arrow 데이터를 사용하므로 학습 파트만 비교 대상.
5. **`muvo/data/generate_voxels.py`는 offline voxelization 파이프라인** — depth map + LiDAR 점을 융합해 occupancy 격자 생성. alpha에는 부재.
