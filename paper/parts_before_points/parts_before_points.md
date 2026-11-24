# Parts Before Points: Chirality-Free Anatomical Part Segmentation and Zero-Label Self-Distillation for Sim-to-Real Multi-Animal Pose Estimation

*Anonymous CVPR 2026 CV4Animals Workshop Submission (Paper ID \*\*\*\* — Strict 8-Page Main Body + References)*

> **Compiled Camera-Style Review PDF:** [`parts_before_points.pdf`](parts_before_points.pdf) (9 pages: 8-page main body + Page 9 references)
> **LaTeX Source:** [`parts_before_points.tex`](parts_before_points.tex)
> **Unified Statistical & Kinematics Bundle:** [`paper_stats_and_kinematics.json`](paper_stats_and_kinematics.json)

---

## Abstract

Markerless multi-animal pose estimation in behavioral neuroscience typically requires costly manual keypoint labeling whenever camera geometry, fur contrast, lighting, or cranial implants (headstages, miniscopes, tethers) change. Synthetic 3D rodent simulation offers a zero-annotation alternative, yet directly regressing keypoints from synthetic renders degrades sharply on real footage due to (i) local texture overfitting around discrete landmarks and (ii) *chirality entanglement*, where predicting left/right points couples mirror-sensitive labels to hardware-occluded appearance cues.

We propose **Parts Before Points**, a sim-to-real paradigm decoupling pose estimation into *chirality-free anatomical body-part segmentation* (`head`, `snout`, `ear`, `torso`, `tail`) followed by a *scale-invariant, occlusion-aware geometric pose solver* that resolves left/right handedness analytically from the spine vector. We instantiate this principle in single-frame `YOLO11n-seg` and a 580 KB temporal VOS head (`Cutie-BodyPartSeg`), and introduce:
1. **Parts2Points-Distill**, a zero-label transductive cross-paradigm self-distillation framework transferring anatomy-gated unlabeled real poses into a 38 FPS keypoint student; and
2. **Zero-GT-Seed Hybrid VOS Fusion**, which seeds frame 0 and re-anchors dark-corner drift purely from zero-shot part detections.

On real CalMS21/MARS tethered footage (`mouse083`, 4,881 frames), zero-shot part segmentation cuts median error by **34.4%** (22.34 → 14.66 px) and lifts PCK@16 by **+20.1 pts**. Zero-label **Parts2Points-Distill** cuts RMSE **50.0%** (54.31 → 27.20 px, $p = 3.66 \times 10^{-23}$), collapses L/R swaps **8.4×** (12.46% → 1.48%), and decisively beats a same-paradigm **Points2Points-Distill** control ($p = 1.90 \times 10^{-24}$), proving cross-paradigm part supervision breaks self-training confirmation bias. On an unseen second real rig (`OFT_54`, `192×192`, 0 target labels/tuning), **Parts2Points-Distill** transfers inductively at **97.0%** recall and **4.33 px** (9.3% body-length) centroid error, while **Zero-GT-Seed Hybrid Fusion** outperforms 20-real-label active learning on dyadic nose-to-nose correlation ($r = 0.936$ vs. $0.875$) and social contact F1 ($0.814$ vs. $0.739$).

---

## Figures

1. **Figure 1 (`figures/fig1_pipeline.pdf`):** Full *Parts Before Points* architecture (3D simulation + muzzle cap expansion → `YOLO11n-seg` & `Cutie-BodyPartSeg` → scale-invariant geometric pose solver → `Parts2Points-Distill` & `Zero-GT-Seed Hybrid VOS`).
2. **Figure 2 (`figures/fig2_pareto_sim2real.pdf`):** Zero-label progression & `Points2Points` control with 95% frame-cluster bootstrap CIs + Accuracy–Speed Pareto frontier on Apple M2 MPS.
3. **Figure 3 (`figures/fig3_per_keypoint_and_cdf.pdf`):** Per-keypoint median errors and 0–64 px cumulative PCK curves across `Sim v3`, `Pts2Pts`, `YOLO-Seg`, `P2P-Distill`, and `AL-20*`.
4. **Figure 4 (`figures/fig4_ablations_and_darkcorner.pdf`):** Controlled ablations: (a) scale-invariant headstage-aware grouping, (b) 3-way distillation paradigm & gating control (`Pts2Pts` vs. `P2P Unfiltered` vs. `P2P Gated`), and (c) VOS vs. Zero-GT-Seed & f0-Seed Hybrid Part Fusion across both 300-frame clips.
5. **Figure 5 (`figures/fig5_qualitative_and_kinematics.pdf`):** Qualitative overlays on `mouse083` (`f=613`, `f=3041`), unseen cross-rig `OFT_54` (`192×192`, `f=11255`), and downstream dyadic nose-to-nose social interaction trajectory (`clip_0_299`).

---

## Key Quantitative Tables (Unified Hungarian Matching — 0.00 px Discrepancy)

### Table 1: Full-Video 200-Frame Real Benchmark (`mouse083_task1_annotator1.mp4`, `1024×570`)

| Method | Regime | Real Lbls | FPS (M2) | Rec. (%) | Cls (%) | RMSE (px) ↓ | MAE (px) ↓ | p50 [95% CI] ↓ | p90 (px) ↓ | PCK@16 (%) ↑ | PCK@32 (%) ↑ | mOKS ↑ | L/R Swap (%) ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| *Oracle GT Part Masks → Pose* | *Ceiling* | — | — | 100 | 100 | — | 6.21 | 5.91 | 10.28 | — | — | — | 0.00 |
| `YOLO11n-pose` Sim v1 (base) | Zero-shot | 0 | 45.9 | 86.8 | **100.0** | 73.23 | 52.95 | 31.05 [28.4, 34.6] | 130.47 | 21.7 | 51.0 | 0.142 | 27.81 |
| `YOLO11n-pose` Sim v2 (2D DR) | Zero-shot | 0 | 39.7 | 90.2 | 99.5 | 74.62 | 38.82 | 22.32 [21.3, 23.6] | 79.13 | 30.0 | 71.4 | 0.201 | 11.50 |
| `YOLO11n-pose` Sim v3 (3D rig) | Zero-shot | 0 | 42.3 | 87.2 | 98.3 | 54.31 | 36.51 | 22.34 [21.2, 23.9] | 91.04 | 34.3 | 66.5 | 0.232 | 12.46 |
| `Points2Points-Distill` (Ctrl) | Transd. | 0 | 40.2 | 91.0 | 99.5 | 42.87 | 30.72 | 22.45 [21.3, 23.5] | 61.71 | 34.4 | 68.3 | 0.231 | 6.59 |
| `YOLO11n-seg` Body-Parts → Pose | Zero-shot | 0 | 7.7 | 92.0 | 99.5 | 41.91 | 26.50 | **14.66** [13.9, 15.4] | 66.18 | **54.4** | 75.4 | **0.379** | 7.61 |
| **Parts2Points-Distill (Unfilt.)** | Transd. | 0 | **42.3** | 92.5 | **100.0** | 28.13 | 21.38 | 17.40 [16.6, 18.3] | 38.99 | 44.8 | 83.1 | 0.305 | 1.62 |
| **Parts2Points-Distill (Gated)** | Transd. | 0 | 38.0 | **92.8** | 99.5 | **27.20** | **19.85** | 15.50 [14.6, 16.2] | **36.57** | 51.8 | **86.0** | 0.350 | **1.48** |
| `YOLO11n-pose` Sim v3 + AL-20\* | Sup.\* | 20 | 44.9 | 94.8 | 99.7 | 21.75 | 15.85 | 12.57 [12.1, 13.1] | 28.53 | 64.0 | 92.5 | 0.438 | 0.66 |

**Two-Sided Paired Wilcoxon Signed-Rank Tests (`eval_200`):**
- `Sim v3` → `YOLO11n-seg`: $n = 197$, $\Delta = -9.67\text{ px}$, $p = 4.67 \times 10^{-8}$
- `Points2Points-Distill (Ctrl)` → `Parts2Points-Distill (Gated)`: $n = 196$, $\Delta = -10.87\text{ px}$, **$p = 1.90 \times 10^{-24}$**
- `Sim v3` → `Parts2Points-Distill (Gated)`: $n = 197$, $\Delta = -16.41\text{ px}$, $p = 3.66 \times 10^{-23}$
- `Parts2Points-Distill (Unfilt.)` → `Parts2Points-Distill (Gated)`: $n = 199$, $\Delta = -1.19\text{ px}$, $p = 3.72 \times 10^{-8}$
- `YOLO11n-seg` → `Parts2Points-Distill (Gated)`: $n = 198$, $\Delta = -5.89\text{ px}$, $p = 1.74 \times 10^{-11}$

---

### Table 3: Complete Contiguous 300-Frame Clip Benchmarks (`0..299` Filtered & Dark-Corner `4400..4699`)

| Clip | Model | Real Seed / Labels | Rec. (%) | RMSE ↓ | MAE ↓ | p50 ↓ | PCK@16 ↑ | PCK@32 ↑ | OKS mAP ↑ | Swap (%) ↓ | Res / Int p50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0..299` | `YOLO11n-pose` Sim v3 | 0 | 88.2 | 45.27 | 31.89 | 21.76 | 37.2 | 67.3 | 0.0076 | 8.41 | 34.5 / 13.5 |
| `0..299` | `Points2Points-Distill` (Ctrl) | 0 transd. | 92.8 | 36.25 | 27.00 | 20.43 | 39.7 | 71.5 | 0.0085 | 4.26 | 31.4 / 13.0 |
| `0..299` | `Cutie-Pose` (582 KB) | 0 + f0 GT | **100.0** | 75.46 | 51.79 | 28.53 | 32.5 | 53.7 | 0.0069 | 29.92 | 14.5 / 72.7 |
| `0..299` | `Cutie-BodyPartSeg` (580 KB) | 0 + f0 GT | **100.0** | 83.70 | 50.86 | 20.47 | 44.4 | 60.5 | **0.0504** | 26.42 | **10.1** / 63.3 |
| `0..299` | `YOLO11n-seg` Parts → Pose | 0 | **100.0** | 43.57 | 29.77 | 18.34 | 45.0 | 70.7 | 0.0111 | 10.09 | 25.8 / 13.7 |
| `0..299` | **Hybrid Part Fusion (f0 GT)** | 0 + f0 GT | **100.0** | 46.45 | 28.56 | 16.25 | 49.4 | 73.9 | 0.0249 | 11.78 | 15.4 / 17.5 |
| `0..299` | **Hybrid Part Fusion (0-seed)** | **0 (YOLO f0)** | **100.0** | 32.05 | **22.03** | **15.02** | **52.5** | 79.2 | 0.0235 | 6.00 | 15.1 / 15.0 |
| `0..299` | **Parts2Points-Distill (Gated)** | 0 transd. | 98.5 | **29.07** | 22.32 | 17.61 | 45.1 | **80.7** | 0.0251 | **2.28** | 24.2 / **12.0** |
| `0..299` | `YOLO11n-pose` Sim v3 + AL-20\* | 20 real | 97.8 | 26.98 | 17.75 | 12.71 | 64.1 | 90.9 | 0.0412 | 2.31 | 14.0 / 11.5 |
| `4400..4699` | `YOLO11n-pose` Sim v3 | 0 | 72.8 | 44.89 | 29.67 | 19.57 | 39.5 | 74.6 | 0.0194 | 9.04 | 29.2 / 14.8 |
| `4400..4699` | `Points2Points-Distill` (Ctrl) | 0 transd. | 96.7 | 40.37 | 28.91 | 22.10 | 31.6 | 71.8 | 0.0152 | 5.95 | 28.9 / 15.8 |
| `4400..4699` | `Cutie-Pose` (582 KB) | 0 + f0 GT | **100.0** | 61.14 | 46.68 | 34.85 | 27.0 | 46.9 | 0.0022 | 39.83 | 59.0 / 19.0 |
| `4400..4699` | `Cutie-BodyPartSeg` (580 KB) | 0 + f0 GT | **100.0** | 59.94 | 43.58 | 25.13 | 37.8 | 54.9 | 0.0205 | 31.92 | 52.3 / 12.4 |
| `4400..4699` | `YOLO11n-seg` Parts → Pose | 0 | 98.0 | 46.34 | 28.92 | **14.98** | **52.4** | 73.5 | **0.0549** | 9.52 | 22.9 / 10.2 |
| `4400..4699` | **Hybrid Part Fusion (f0 GT)** | 0 + f0 GT | **100.0** | 35.31 | 25.63 | 18.08 | 45.6 | 69.0 | 0.0338 | 5.01 | 31.3 / 9.9 |
| `4400..4699` | **Hybrid Part Fusion (0-seed)** | **0 (YOLO f0)** | **100.0** | 40.39 | 28.07 | 18.36 | 46.0 | 66.4 | 0.0414 | 7.52 | 35.8 / **8.9** |
| `4400..4699` | **Parts2Points-Distill (Gated)** | 0 transd. | 98.7 | **31.30** | **20.69** | 15.27 | 52.1 | **86.9** | 0.0136 | **3.72** | **17.5** / 13.8 |
| `4400..4699` | `YOLO11n-pose` Sim v3 + AL-20\* | 20 real | 100.0 | 35.09 | 21.51 | 14.32 | 56.8 | 88.5 | 0.0070 | 6.58 | 13.8 / 15.1 |

---

### Table 4 & Table 5: Cross-Rig (`OFT_54`, `192×192`, 200f Native / 4-Way Rotated), $N = 4$ Crowded Mice, Real *Drosophila*, & Dyadic Social Kinematics

| Model (`OFT_54`, 0 OFT labels/tuning) | Rec. (%) Nat / Rot | Centroid p50 (px) Nat / Rot | Dir. OK (%) Nat / Rot | Ant / Post p50 (Rot px) ↓ | Anat.+Chiral Valid (Rot %) ↑ |
|---|---:|---:|---:|---:|---:|
| `YOLO11n-pose` Sim v3 (0-shot) | 43.0 / 45.5 | 10.4 / 17.4 | 37.2 / 33.0 | 19.8 / 50.1 | 24.5 |
| `Points2Points` (Ind. Ctrl) | 79.5 / 73.0 | 5.3 / 8.2 | 79.0 / 60.4 | 17.3 / 16.3 | 41.0 |
| `YOLO11n-seg` Parts → Pose (0-shot) | 85.5 / 88.0 | 17.8 / 19.5 | 53.2 / 56.8 | **10.4** / 40.1 | 37.5 |
| **Parts2Points-Distill (Ind., 0 real)** | **97.0 / 91.0** | **4.3 / 5.2** | **80.3 / 72.5** | 13.0 / **7.3** | **71.0** |
| `YOLO11n-pose` Sim v3 + AL-20\* (Ind.) | 97.0 / 90.5 | 4.9 / 4.7 | 87.6 / 82.9 | 6.3 / 4.6 | 84.0 |

- **Crowded $N = 4$ Multi-Mouse Scaling (60 frames, 240 mice, 1,680 3D GT keypoints; models trained strictly on $N = 2$):** `YOLO11n-seg` (`dedup_parents=False`) achieves **100.0% recall / 100.0% precision**, **8.29 px p50**, **12.73 px RMSE**, and **91.1% PCK@16** (+12.3 pts over `Sim v3`'s 78.8% and +13.8 pts over `AL-20*`'s 77.3%), while `Parts2Points-Distill (Gated)` preserves **100.0% recall/precision**, **9.66 px p50**, **98.0% PCK@32**, and **0.00% L/R swap**.
- **Non-Rodent Cross-Species Validation (Real Tethered *Drosophila*, `2019_06_26_fly2.mp4`, `280×230`, 45,000 frames, 0 manual keypoint labels):** `Parts2Points-Distill` (Insect 3-tagma `head`/`thorax`/`abdomen` → 5 canonical landmarks) runs at **117.5 FPS**, achieving **8.65 px p50 / 9.58 px MAE** (`neck_joint = 2.81 px`, `head_tip = 7.65 px`) on strictly held-out manual GT frame `7082` (±500-frame guard band) and **100.0% recall, 0.995 heading alignment, 12.96 px p50** (`abdomen_tip = 7.86 px`) across 200 unseen future frames (`24,000–44,900`).

| Model (`clip_0_299` Dyadic Kinematics) | Frames | NN $r$ ↑ | NN MAE (px) ↓ | Contact F1 ↑ | AG $r$ ↑ | AG MAE (px) ↓ | Heading $\cos\Delta\theta$ (Res / Int) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `YOLO11n-pose` Sim v3 | 228 | 0.768 | 58.4 | 0.326 | 0.745 | 68.4 | 0.740 / **0.935** |
| `Points2Points` (Ctrl) | 246 | 0.898 | 45.1 | 0.444 | 0.824 | 49.1 | 0.889 / 0.899 |
| `YOLO11n-seg` Parts → Pose | 280 | 0.824 | 53.5 | 0.222 | 0.810 | 63.4 | 0.826 / 0.899 |
| `Hybrid Part Fusion` (f0 GT) | 281 | 0.696 | 37.1 | 0.673 | 0.732 | 38.9 | 0.928 / 0.621 |
| **Hybrid Part Fusion (0-seed)** | **300** | **0.936** | **22.9** | **0.814** | **0.908** | **32.2** | 0.978 / 0.843 |
| `P2P-Distill` (Unfiltered) | 279 | 0.884 | 32.3 | 0.459 | 0.903 | 40.8 | 0.981 / 0.925 |
| `P2P-Distill` (Gated) | 288 | 0.895 | 34.5 | 0.436 | 0.896 | 42.1 | **0.984** / 0.927 |
| `Sim v3 + AL-20*` (20 real) | 280 | 0.875 | 20.3 | 0.739 | 0.948 | 18.2 | 0.925 / 0.971 |
