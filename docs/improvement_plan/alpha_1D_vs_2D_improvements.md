# Alpha 1D vs Alpha 2D Improvement Plans — 비교

`alpha_1D_improvements.md` 와 `alpha_2D_improvements.md`의 두 plan을 옆으로 펼친 비교. 어느 쪽을 먼저 진행할지, 공통 P0 항목은 무엇인지, 충돌하는 항목은 무엇인지 정리한다.

---

## 1. 공통 P0 (두 갈래에 모두 적용)

| ID | 갭 | 1D 적용 | 2D 적용 | 권장 동시 진행? |
|---|---|---|---|---|
| **Common-P0-A: LiDAR loss 활성화** | `WEIGHT_LIDAR_RE=0.1, WEIGHT_LIDAR_EMPTY=0.05` | ✓ G1 | ✓ G1 | **예** — 1D-2D controlled 비교를 위해 반드시 동시 |
| **Common-P0-B: KL free-bits 처리 통일** | alpha-1D는 free_bits 항상 적용; 2D는 cfg 게이트. | (free_bits 처리 명시 cfg화) | ✓ G4 (ablation 후 default 결정) | **예** — 비교 시 한쪽만 활성이면 KL trace 비교 무의미 |
| **Common-P0-C: Augmentation 동등화** | 1D에 PixelAugmentation 포팅 | ✓ G12 | (이미 적용) | **예** — augmentation 차이를 변수로 두지 않기 |
| **Common-P0-D: SSIM warmup** | `WEIGHT_SSIM=0.06` warmup | ✓ G6 | (옵션) | 1D 우선 — 2D는 RGB plateau 이미 부분 해결 |
| **Common-P0-E: 학습 imagine supervise 제거 (paper 일치)** | `WEIGHT_FUTURE=0` (옵션 A) 또는 training_step 에서 imagine 부분 제거 (옵션 B) | ✓ G13 | ✓ G15 | **예** — 양쪽 동시. paper L372 "All 12 frames as known data" 와 upstream 코드 (`transition_td.py:RSSMTD.forward` 전체 seq posterior, `muvo.py:observe_and_imagine` 검증 전용) 와 일치시키기 |

---

## 2. 1D만 해당 (alpha-2D는 해당 없음)

| ID | 갭 | 우선순위 |
|---|---|---|
| 1D-only-A: `ACTION_IN_GRU=True` ablation | P0 (1D 디자인 검증) |
| 1D-only-B: TBPTT carry 효과 검증 | P1 (1D는 TBPTT 사용, 2D는 미사용 — 효과 ablation) |
| 1D-only-C: SensorFeatureConv (1D 임베딩 압축) 효과 검증 | P2 |

---

## 3. 2D만 해당

| ID | 갭 | 우선순위 |
|---|---|---|
| 2D-only-A: Transformer capacity (6L/8H/512) 확장 | P0 (paper 매치) |
| 2D-only-B: decoder_channels grid (256/512/768) | P1 |
| 2D-only-C: LiDAR out_indices (1,2,3) vs (0,1,2,3) vs (2,3,4) ablation | P1 |
| 2D-only-D: PerceptualLoss/LPIPS optional | P1 |
| 2D-only-E: decoder_channels=embedding 통일 변형 | P1 |

---

## 4. 두 갈래 공통 P1/P2

| 갭 | 1D | 2D | 권장 |
|---|---|---|---|
| EFFECTIVE_HZ=5 (paper-aligned 0.2s) | G3 P1 | G8 P1 | **동시** — 데이터 인덱스 재빌드 1회로 양쪽 가능 |
| Validation split 명시 분리 ($`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$) | G4 P1 | G9 P1 | **동시** — Arrow schema 확장은 양쪽 공유 |
| 이미지 해상도 paper-aligned (600×960) | G2 P1 | (직접 포함되지 않음, 2D는 토큰 grid 영향) | 1D 우선 → 2D는 token grid 재산정 후 |
| 3D Occupancy 헤드 추가 | G9 P2 | G10 P2 | **2D 우선** (paper §IV-C는 2D context에서 평가). 1D는 baseline 비교용 |
| PointPillars/BEV 분기 | G10/G11 P2 | G11/G12 P2 | 우선순위 낮음, 별도 evaluation 필요 시만 |

---

## 5. 서로 충돌하는 항목

현재 plan 사이에 직접적 충돌은 없으나, 다음 사항은 주의가 필요하다:

| 항목 | 1D | 2D | 충돌 가능성 |
|---|---|---|---|
| `KL_FREE_BITS_ENABLED` default | 항상 적용 (코드 자체) | cfg 게이트 (현재 True) | 양쪽 default를 통일하지 않으면 1D-2D KL trace 비교 시 노이즈. **결정: 양쪽 True 유지** (controlled 비교 우선). |
| Augmentation default | OFF (적용 안 됨) | ON | 1D에 포팅 후 default ON으로 통일 권장. 비교 ablation 시에는 OFF run도 1회. |
| TBPTT carry | 1D 적용, 2D 미적용 | (그대로) | 2D에 TBPTT 도입하면 비교 복잡. **결정: 그대로 두고 1D-2D 비교 시 TBPTT 효과는 분리 변수로 인식**. |
| `ACTION_IN_GRU` | True (1D 디자인) | dead config (RSSMTD 미소비) | 그대로 유지. 1D에서 True/False ablation 후 1D-2D 비교 시 1D는 결정값 사용. |
| Encoder capacity | (1D 기본) | 확장 검토 | 2D만 변경 — 1D-2D 비교 시 capacity 차이가 변수가 되지만 이는 paper §IV-B 비교(1D vs 2D)와 같은 의도이므로 OK. |

---

## 6. 어느 쪽을 먼저 진행할지

### 결론: **alpha-2D를 우선** (단 공통 P0는 양쪽 동시)

근거:
- **paper §IV-B headline: "2D latent significantly benefits camera predictions"** — alpha-2D가 alpha 프로젝트의 main bet.
- alpha-1D는 paper의 1D baseline 비교군 역할이 일차적 목표. 따라서 alpha-1D는 alpha-2D 결과의 reference point로 기능.
- 단 1D single-window overfit에서 1D > 2D 결과(2026-05-19, `docs/muvo_4way_1D_2D_comparison.md`)가 multi-window/server에서도 유지되는지 검증은 1D를 디버그 테스트벤치로 유지할 가치 있음.

### 동시 진행 (1D + 2D)

Sprint 1 — 양쪽 동시:
- Common-P0-A (LiDAR loss)
- Common-P0-B (KL free-bits 통일)
- Common-P0-C (Augmentation 통일)

Sprint 1 — 2D 단독:
- 2D-only-A (Transformer capacity 6L)
- 2D-only-D (KL free-bits ablation default 결정 — 2D 단독)

Sprint 1 — 1D 단독:
- 1D-only-A (ACTION_IN_GRU ablation)
- Common-P0-D (SSIM warmup)

→ Sprint 1 종료 시 alpha-1D-vs-2D 4-GPU server run 비교 가능.

Sprint 2 — paper-aligned 변형 (양쪽 동시):
- EFFECTIVE_HZ=5 (양쪽)
- Validation split 분리 (양쪽)
- Image 해상도 paper-aligned (1D 우선)
- 2D-only-B/C (decoder_channels, out_indices ablation)

Sprint 3 — paper §IV-C 재현 (2D 우선):
- 3D Occupancy 헤드 + SCAL loss (2D 먼저, 효과 확인 후 1D)

---

## 7. 1D를 유지하는 이유 (debug baseline)

1. **단일 window overfit 빠름** — 1D MLP-GRUCell이 memorization 빠르고 디버깅이 쉬움.
2. **paper 1D baseline 비교군** — paper Fig. 4의 1D vs 2D 비교를 alpha 도메인에서 재현.
3. **변경 검증 sanity check** — 새 loss / 새 augmentation 도입 시 1D에서 먼저 효과 확인.
4. **capacity 작아 학습 빠름** — Sprint 1 변형들이 1D에서 빠르게 수렴.

→ alpha-1D는 production model이 아니라 **debug + paper 1D baseline 비교** 용도로 명시 유지.

---

## 8. 통합 우선순위 표 (양 plan 종합)

| 우선순위 | 적용 | ID/갭 | 설명 |
|---|---|---|---|
| **P0** | 1D+2D | Common-P0-A | LiDAR loss 활성화 (둘 다 0→0.1) |
| **P0** | 1D+2D | Common-P0-B | KL free-bits 게이트 통일 |
| **P0** | 1D+2D | Common-P0-C | Augmentation default 통일 (1D에 포팅) |
| **P0** | 1D | Common-P0-D | SSIM warmup 0.06 (1D 우선) |
| **P0** | 1D | 1D-only-A | ACTION_IN_GRU ablation |
| **P0** | 2D | 2D-only-A | Transformer 6L 확장 |
| **P0** | 2D | 2D-only-D (G4) | KL free-bits default 결정 |
| **P1** | 1D+2D | EFFECTIVE_HZ=5 | paper-aligned 0.2s |
| **P1** | 1D+2D | Validation split 분리 | $`𝒟^{\text{RL}}`$ / $`𝒟^{\text{DS}}`$ 명시 |
| **P1** | 1D | G2 | 이미지 해상도 paper-aligned |
| **P1** | 2D | 2D-only-B/C/E | decoder_channels / out_indices ablation |
| **P1** | 2D | 2D-only-D (G7) | PerceptualLoss optional |
| **P2** | 2D 먼저, 1D 후속 | 3D Occupancy + SCAL | paper §IV-C 재현 |
| **P2** | 1D+2D | BEV / PointPillars | paper §IV-B baseline 비교 (skip 가능) |
| **P2** | 2D | ViT 백본 | paper Fig. 4 PL/ViT 변형 |

---

## 9. 검증 매트릭스 (1D vs 2D 정량 비교)

각 sprint 종료 시 다음 메트릭으로 비교:

| 메트릭 | 1D | 2D | 비고 |
|---|---|---|---|
| `val/RL_loss` | ✓ | ✓ | 총 loss |
| `val/RL_psnr` | ✓ | ✓ | RGB recon |
| `val/RL_lidar_chamfer_xyz` | ✓ | ✓ | LiDAR Chamfer |
| `val/RL_lidar_chamfer_xyz_nearfield` | – | ✓ | 2D 전용 |
| `val/DS_*` (도메인 이동) | ✓ | ✓ | – |
| `val/future_*` | ✓ | ✓ | future prediction |
| KL trace | ✓ | ✓ | free-bits 영향 |
| 학습 시간 / step | 비교 | 비교 | capacity 차이 반영 |
| GPU memory | 비교 | 비교 | 2D 더 큼 |

---

## 10. 핵심 의사 결정 트리

```
[paper §IV-B "2D latent benefits camera"를 alpha 도메인에서 재현하고 싶은가?]
   ├─ 예 (default)
   │   → Sprint 1-2 양쪽 동시 진행
   │       1D는 paper 1D baseline, 2D는 paper headline
   │       4-GPU server run으로 1D-vs-2D 비교
   │
   └─ 아니오 (alpha 2D만 production 목표)
       → 1D는 SSIM/aug debug용으로만 유지
       → 2D Sprint 1-2-3에 집중
       → Sprint 3 (occupancy) 더 빨리 진입
```

→ 추천: **default 트랙** (1D + 2D 동시, paper §IV-B 재현 목표).
