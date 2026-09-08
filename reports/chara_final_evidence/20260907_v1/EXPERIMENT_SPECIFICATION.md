# Experiment Specification: Disentangling Mean Geometry Offsets from Dynamic Fluctuation Reconstruction

## Objective
Determine whether sparse linear sensor decoders trained on residue-pair distance subsets reconstruct genuine dynamic distance fluctuations across independent molecular dynamics trajectories, or whether reconstruction skill is predominantly a static mean-offset correction between simulation runs.

## Protocol & Data
1. **Data**: Three independent, unconsulted all-atom explicit-solvent trajectories of a standard globular protein (DESRES Anton BPTI 1-ms runs; N=10,000 frames/run sampled at 1 ns), partitioned into 2 development runs and 1 held-out test run (3-fold cross-validation).
2. **Task**: Select k=10 candidate residue-pair distances (|i - j| >= 5) from a non-local pool and train an L2-regularized linear ridge decoder to predict a disjoint panel of 500 unobserved target distances.
3. **Measurements**:
   - Exact population MSE decomposition: MSE = Bias^2 + Var_res, recording the fraction of MSE reduction attributable to static mean shift: Delta_Bias^2 / Delta_MSE.
   - Per-target Pearson correlation r between predicted and observed held-out traces.
   - Noncircular lag control at offsets {0, 10, 25, 50, 100} frames on fixed support [100..N-1].
   - Zero-centered evaluation where development and test targets are mean-centered prior to evaluation.

## Falsification Criterion
If on mean-centered target data, or under frame lag, the learned decoder achieves positive skill (MSE_model < MSE_baseline) and median Pearson correlation r > 0.35 on dynamically unconstrained targets, the hypothesis that skill is primarily a static mean-offset artifact is contradicted. If Delta_Bias^2 / Delta_MSE remains > 70% and centered skill collapses to <= 0, the static-offset dominance is established.
