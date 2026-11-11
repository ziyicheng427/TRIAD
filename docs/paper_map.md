# Paper ↔ Code Map

Element-by-element mapping between the TRIAD paper (*TRIAD: Consistency-Guided
LiDAR-Camera Calibration, Point Cloud Registration, and Dynamic Occupancy
Mapping with Cross-Layer Confidence Propagation*, Robotics and Autonomous
Systems) and this repository.  Section/equation/figure/table numbers refer to
the submitted manuscript.

## §3.1 Overview and the C2C operator

| Paper element | Code |
| --- | --- |
| Consistency probe `κ_s(z_s; C_s)` | `triad/common/c2c.py::ConsistencyProbe` (populated per module) |
| Confidence map `Φ: R^k → [0,1]^m` | `triad/common/confidence_calibration.py::IsotonicConfidenceMap` |
| Eq. (1) weighted residual estimation `θ̂ = argmin Σ w_i ρ(r_i)` | `triad/common/c2c.py::C2COperator.objective / irls_weights / weighted_normal_equation` |
| Robust kernel ρ (Huber) | `triad/common/c2c.py::huber_weight` |
| Calibration probe: cross-modal consistency | `triad/cal/association.py::CrossModalAssociation.associate` → `consistency_scores` |
| Registration probe: pairwise geometric compatibility | `triad/reg/cga.py::geometric_compatibility` |
| Mapping probe: endpoint persistence | `triad/map/persistence.py::PersistenceTracker` |
| Fig. 1 architecture | `triad/pipeline.py::TriadPipeline`, README diagram |

## §3.2 TRIAD-Cal

| Paper element | Code |
| --- | --- |
| Voxel partition (1 m outdoor / 0.5 m indoor) | `triad/cal/lidar_edges.py::VoxelPlaneFitter`; defaults `CalConfig.voxel_size`, `configs/cal_outdoor.yaml`, `configs/cal_indoor.yaml` |
| Per-voxel RANSAC planes | `triad/cal/lidar_edges.py::fit_plane_ransac` |
| Dihedral window [30°, 150°] plane intersections | `triad/cal/lidar_edges.py::extract_lidar_edges` (`_dihedral_deg`, `_intersect_planes`) |
| 16×16 SAM grid prompts | `triad/cal/sam_adapter.py::SamMaskExtractor.grid_points / extract` |
| NMS + Sobel thinning | `triad/cal/image_edges.py::_thin_with_nms` |
| Adaptive RGB-std filter (0.7 × mean) | `triad/cal/image_edges.py::filter_boundary_pixels` (`CalConfig.rgb_std_ratio`) |
| Distance-transform attraction field `G(u,v)` | `triad/cal/image_edges.py::build_attraction_field`, `AttractionField` |
| Residual `r_i = G(p_i)` | `triad/cal/association.py::CrossModalAssociation.associate` |
| Direction weight (line direction vs edge normal) | `triad/cal/association.py::CrossModalAssociation._direction_weights` |
| Multi-frame weight (5-frame window) | `triad/cal/association.py::temporal_weights_over_window` (`CalConfig.multi_frame_window`) |
| Noise model σ_i (angular + range + 1.2 px SAM) | `triad/cal/estimator.py::ExtrinsicEstimator.point_sigmas` |
| Jacobian with attraction-field gradient | `triad/cal/estimator.py::ExtrinsicEstimator.estimate` (chain `∇G ← projection ← twist`); `triad/common/se3.py::point_jacobian_right` |
| Covariance `Σ_ξ = (J^T Σ_r⁻¹ J)⁻¹` | `triad/cal/estimator.py::ExtrinsicEstimator.estimate` (final IRLS covariance) |
| Correspondence-rate-driven grid search init | `triad/cal/estimator.py::grid_search_init` |
| Coverage < 0.6 → covariance ×1.5 | `CalConfig.coverage_warn_threshold / coverage_inflation` |
| Fig. 2 pipeline | `triad/cal/pipeline.py::TriadCalPipeline.run` |

## §3.3 TRIAD-Reg

| Paper element | Code |
| --- | --- |
| DW-FPS: maximise `w_i d_i`, `w_i = ln(1 + d_i/d_max) + β` | `triad/reg/sampling.py::distance_weighted_fps` |
| Shared three-level MLP encoder | `triad/reg/encoder.py::ThreeLevelMLP / HierarchicalEncoder` |
| Dual-softmax candidate scoring | `triad/reg/matching.py::dual_softmax_similarity / candidate_correspondences` |
| Seed selection by combined confidence | `triad/reg/matching.py::select_seeds` |
| Compatibility `B_ij = ReLU(1 − |d_pair_ij|/σ_c)` | `triad/reg/cga.py::geometric_compatibility` |
| CGA restricted to seed neighbourhoods | `triad/reg/cga.py::ConsistencyGuidedAttention` |
| Eq. (2) propagation `s_i^(t) = f(Σ_k B_ik s_k^(t−1))` | `triad/reg/propagation.py::ConsistencyPropagation` |
| Weighted Kabsch (coarse transform, deepest level) | `triad/reg/pose.py::weighted_kabsch` |
| Upper levels re-align and refine (local searches) | `triad/reg/pose.py::hierarchical_refine` |
| Weights = converged scores ⊕ similarity | `triad/reg/model.py::TRIADRegModel.weight_head` |
| InfoNCE + weighted BCE + L1 transform loss (all levels) | `triad/reg/losses.py` |
| AdamW, 50 epochs, KITTI splits | `triad/reg/train.py::train / KittiOdometryDataset` (`TRAIN/VAL/TEST_SEQUENCES`) |
| Fig. 3 pipeline | `triad/reg/model.py::TRIADRegModel.forward` |

## §3.4 TRIAD-Map

| Paper element | Code |
| --- | --- |
| Clamped log-odds cells | `triad/map/grid.py::LogOddsGrid.apply_updates` |
| Precomputed lookup tables | `triad/map/grid.py::UpdateTable` |
| Persistence probe (radius-R, consecutive scans, reset) | `triad/map/persistence.py::PersistenceTracker.observe / support_cells` |
| PGC: hits below 0.49 need N = 3 frames | `triad/map/updater.py::TriadMapUpdater._hit_increment` (free band) |
| TBB: boosted hit (0.85) in [0.49, 0.55) | `triad/map/updater.py::TriadMapUpdater._hit_increment` (transition band) |
| Base hits at ≥ 0.55 | `triad/map/updater.py::TriadMapUpdater._hit_increment` (occupied band) |
| Misses on traversed cells incl. unknown | `triad/map/updater.py::TriadMapUpdater.update` + `triad/map/raycasting.py::cast_rays` |
| 2D-3D bridge (voxel + plane reuse, ground removal, column projection) | `triad/map/bridge_3d.py::SolidStateBridge.project` |
| Fig. 4 update rule | `triad/map/updater.py` module docstring + band rule |

## §3.5 TRIAD-Link

| Paper element | Code |
| --- | --- |
| L1: `σ_i² += J_ξ Σ_ξ J_ξ^T` | `triad/link/l1_covariance.py::propagate_to_observation_variance` |
| L1 → map-update gentling | `triad/link/l1_covariance.py::map_gain_from_variance` |
| L2 / Eq. (3) `g_hit = clip(σ₀²/(σ₀²+σ_t²), 0.25, 1)`, `g_miss = clip(·, 0.6, 1)`, σ₀ = 3 cm | `triad/link/l2_gains.py::l2_gains` |
| L3: free cells (P < 0.2) → weight 0.1; occupied/unknown keep | `triad/link/l3_reweight.py::occupancy_reweight` |
| Link wiring with switches | `triad/pipeline.py::TriadPipeline` |

## §5 Experiments

| Paper element | Code |
| --- | --- |
| §5.1 calibration evaluation (6 scenes × 20 runs, ±5°/±10 cm init) | `scripts/calibrate.py`; ranges in `CalConfig.init_rot_range_deg / init_trans_range_m` |
| §5.2 RTE / RRE / RR (Tables 2, 7) | `triad/evaluation/registration_metrics.py`, `scripts/evaluate_registration.py` |
| §5.3 clearing / establishment delay (Tables 5, 6, 8) | `triad/evaluation/map_metrics.py::clearing_delay / establishment_delay` |
| §5.3 reimplemented baselines (Table 6) | `triad/map/baselines.py` (RegionStatistics / ChangeDecay / FrequencyDomain / BeamContribution) |
| §5.4 isotonic calibration, ECE (Table 4) | `triad/common/confidence_calibration.py`, `scripts/calibrate_confidence.py` |
| §5.5 amplification control (Fig. 5) | `scripts/link_controls.py::cmd_amplification` |
| §5.5 L1 control (Table 5: coverage / ECE / RTE / map RMSE) | `triad/evaluation/cal_metrics.py::covariance_coverage`, `scripts/link_controls.py` |
| §5.5 L2 noise sweep (Table 9) | `scripts/link_controls.py::cmd_l2` |
| §5.5 L3 weighting control (Tables 10–11) | `scripts/link_controls.py::cmd_l3 / weighted_icp` |
| §5.5 error budget (Table 14) | `triad/evaluation/budget.py::error_budget / slopes_from_bias_sweep` |
| §5.5 2³ factorial link ablation (Table 13) | `scripts/link_controls.py::cmd_factorial` (`LinkConfig.enable_l1/l2/l3`) |
| §5.5 confidence-transfer strategies (Table 12) | `triad/link/transfer.py`, `scripts/link_controls.py::cmd_transfer` |
| §5.6 end-to-end (Tables 15–16) | `scripts/run_e2e.py`, `triad/pipeline.py` (`use_task_modules`, link switches) |

## Constants named in the paper

| Value | Where | Config field |
| --- | --- | --- |
| voxel 1 m / 0.5 m | §3.2 | `CalConfig.voxel_size` |
| dihedral [30°, 150°] | §3.2 | `CalConfig.dihedral_min_deg / dihedral_max_deg` |
| 16×16 SAM grid | §3.2 | `CalConfig.sam_grid` |
| RGB filter ratio 0.7 | §3.2 | `CalConfig.rgb_std_ratio` |
| SAM edge σ 1.2 px | §3.2 | `CalConfig.sam_edge_sigma_px` |
| 5-frame window | §3.2 | `CalConfig.multi_frame_window` |
| coverage 0.6 → ×1.5 | §3.2 | `CalConfig.coverage_warn_threshold / coverage_inflation` |
| init randomisation ±5° / ±10 cm | §5.1 | `CalConfig.init_rot_range_deg / init_trans_range_m` |
| bands 0.49 / 0.55 | §3.4 | `MapConfig.p_free_threshold / p_transition_threshold` |
| persistence N = 3 | §3.4 | `MapConfig.persistence_frames` |
| TBB hit probability 0.85 | §3.4 | `MapConfig.p_hit_transition` |
| σ₀ = 3 cm; clips 0.25 / 0.6 | §3.5 Eq. (3) | `LinkConfig.sigma_0 / hit_gain_clip_min / miss_gain_clip_min` |
| L3 free threshold 0.2, weight 0.1 | §3.5 | `LinkConfig.l3_free_threshold / l3_free_weight` |
| RR thresholds 2 m / 5° | §5.2 Table 2 | `RegConfig.rte_thresh_m / rre_thresh_deg` |
| 50 epochs, AdamW | §5.2 | `RegConfig.epochs` (optimiser in `triad/reg/train.py`) |
| ≥ 5000 calibration probes | §5.4 | `triad/common/confidence_calibration.MIN_CALIBRATION_OBSERVATIONS` |
