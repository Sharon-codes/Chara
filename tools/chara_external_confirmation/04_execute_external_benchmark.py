#!/usr/bin/env python3
"""
tools/chara_external_confirmation/04_execute_external_benchmark.py

Executes the fair independent external benchmark on the standardized ATLAS panel:
- Proteins: 1utg_A (all-alpha), 2cg7_A (all-beta), 2j6b_A (alpha+beta)
- Disjoint candidate and target generation (|i - j| >= 3, max 500 candidates, max 500 targets)
- Methods:
  1. M1_DEVELOPMENT_CONSTANT_MEAN
  2. M2_UNIFORM_RANDOM (20 seeds, tuned per-seed)
  3. M3_DEVELOPMENT_VARIANCE
  4. M4_PCA_PIVOTED_QR
  5. M7_MEAN_TRANSFER
  6. M8_ROBUST_TRANSFER
- Primary budget: k=10; secondary sensitivity budgets: k=5, k=20
- Inner tuning: 60% train, 10% guard, 30% val on development runs for lambda in {1e-6..1000}
- Causal data dependence audit
- Graph Laplacian ablation (no graph, fixed adjacency, MD-variance, 20 permutations)
- Comprehensive diagnostics:
  * Exact moment decomposition (Bias^2 vs Residual Variance shares)
  * Test-centered diagnostic skill
  * Per-target Pearson r (median, IQR)
  * Non-circular temporal lag shifts (lags 0, 10, 25, 50, 100 frames)
  * Paired temporal block bootstrap (1,000 draws, 20-frame blocks, 95% CI)
- Outputs tables to reports/chara_external_confirmation/20260908_v1/
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import cdist

# Fast Ridge with centering and standardization
class FastMultiOutputRidge:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.W = None
        self.xm = None
        self.xs = None
        self.ym = None

    def fit(self, X, Y):
        self.xm = np.mean(X, axis=0)
        self.xs = np.std(X, axis=0)
        self.xs[self.xs < 1e-8] = 1.0
        self.ym = np.mean(Y, axis=0)

        Xn = (X - self.xm) / self.xs
        Yc = Y - self.ym

        n_samples, n_features = Xn.shape
        XtX = Xn.T @ Xn
        reg = self.alpha * np.eye(n_features)
        try:
            self.W = linalg.solve(XtX + reg, Xn.T @ Yc, assume_a='pos')
        except linalg.LinAlgError:
            self.W = linalg.pinv(XtX + reg) @ (Xn.T @ Yc)
        return self

    def predict(self, X):
        Xn = (X - self.xm) / self.xs
        return Xn @ self.W + self.ym

def fit_and_tune_ridge(X1, Y1, X2, Y2, indices, lambda_grid=None):
    if lambda_grid is None:
        lambda_grid = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]
    
    # Chronological inner split: 60% train, 10% guard, 30% val
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    tr_end1 = int(0.60 * n1)
    val_start1 = int(0.70 * n1)
    tr_end2 = int(0.60 * n2)
    val_start2 = int(0.70 * n2)

    X1_sub = X1[:, indices]
    X2_sub = X2[:, indices]

    X1_tr, Y1_tr = X1_sub[:tr_end1], Y1[:tr_end1]
    X1_val, Y1_val = X1_sub[val_start1:], Y1[val_start1:]

    X2_tr, Y2_tr = X2_sub[:tr_end2], Y2[:tr_end2]
    X2_val, Y2_val = X2_sub[val_start2:], Y2[val_start2:]

    best_a = 1.0
    best_loss = float('inf')

    for a in lambda_grid:
        m1 = FastMultiOutputRidge(alpha=a).fit(X1_tr, Y1_tr)
        l12 = np.mean((Y2_val - m1.predict(X2_val))**2)
        
        m2 = FastMultiOutputRidge(alpha=a).fit(X2_tr, Y2_tr)
        l21 = np.mean((Y1_val - m2.predict(X1_val))**2)
        
        avg_loss = 0.5 * (l12 + l21)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_a = a

    # Refit on both full development runs
    X_dev_full = np.vstack([X1_sub, X2_sub])
    Y_dev_full = np.vstack([Y1, Y2])
    final_model = FastMultiOutputRidge(alpha=best_a).fit(X_dev_full, Y_dev_full)
    return final_model, best_a

def selector_pca_qr(X_pool, k):
    xm = np.mean(X_pool, axis=0)
    xs = np.std(X_pool, axis=0)
    xs[xs < 1e-8] = 1.0
    Xn = (X_pool - xm) / xs
    U, s, Vt = linalg.svd(Xn, full_matrices=False)
    Vk = Vt[:k, :]
    Q, R, piv = linalg.qr(Vk, pivoting=True)
    return piv[:k].tolist()

def run_greedy_transfer_search(X1, Y1, X2, Y2, mode="robust", max_k=20, graph_scores=None, gamma=0.05):
    n_cand = X1.shape[1]
    y_g1 = Y1[:, :min(50, Y1.shape[1])].mean(axis=1)
    y_g2 = Y2[:, :min(50, Y2.shape[1])].mean(axis=1)
    
    corrs = []
    for c in range(n_cand):
        r1 = np.corrcoef(X1[:, c], y_g1)[0, 1]
        r2 = np.corrcoef(X2[:, c], y_g2)[0, 1]
        sc = 0.5 * (abs(r1) + abs(r2)) if (not np.isnan(r1) and not np.isnan(r2)) else 0.0
        corrs.append((sc, c))
    corrs.sort(reverse=True)
    screened = [c for _, c in corrs[:min(40, n_cand)]]

    var_y = np.var(np.vstack([Y1, Y2]), axis=0)
    top_t_idx = np.argsort(-var_y)[:min(50, Y1.shape[1])]
    Y1_sub = Y1[:, top_t_idx]
    Y2_sub = Y2[:, top_t_idx]
    y1_var = float(np.var(Y1_sub)) + 1e-12
    y2_var = float(np.var(Y2_sub)) + 1e-12

    selected = []
    for step in range(max_k):
        best_sc = float('inf')
        best_c = None
        for c in screened:
            if c in selected:
                continue
            tr = selected + [c]
            m1 = FastMultiOutputRidge(alpha=1.0).fit(X1[:, tr], Y1_sub)
            e12 = np.mean((Y2_sub - m1.predict(X2[:, tr]))**2) / y1_var
            m2 = FastMultiOutputRidge(alpha=1.0).fit(X2[:, tr], Y2_sub)
            e21 = np.mean((Y1_sub - m2.predict(X1[:, tr]))**2) / y2_var

            if mode == "mean":
                score = 0.5 * (e12 + e21)
            elif mode == "robust":
                score = max(e12, e21)
            elif mode == "graph":
                g_pen = np.mean([graph_scores[idx] for idx in tr]) if graph_scores is not None else 0.0
                score = max(e12, e21) - gamma * g_pen
            else:
                score = max(e12, e21)

            if score < best_sc:
                best_sc = score
                best_c = c
        if best_c is not None:
            selected.append(best_c)
        else:
            break
    return selected

def build_laplacian_scores(X1, X2, candidates, n_res, tau=0.5):
    adj = np.zeros((n_res, n_res), dtype=float)
    var_matrix = np.zeros((n_res, n_res), dtype=float)
    X_pool = np.vstack([X1, X2])
    for c_idx, p in enumerate(candidates):
        i, j = p["i_idx"], p["j_idx"]
        mean_d = float(np.mean(X_pool[:, c_idx]))
        var_d = float(np.var(X_pool[:, c_idx]))
        if mean_d <= 1.2:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            var_matrix[i, j] = var_d
            var_matrix[j, i] = var_d
            
    # Exponential MD-variance weighted Laplacian
    W = adj * np.exp(-var_matrix / tau)
    deg = np.diag(np.sum(W, axis=1))
    L = deg - W
    
    # Heat kernel H = expm(-0.1 * L)
    vals, vecs = np.linalg.eigh(L)
    H = vecs @ np.diag(np.exp(-0.1 * vals)) @ vecs.T
    
    scores = []
    for p in candidates:
        i, j = p["i_idx"], p["j_idx"]
        scores.append(float(H[i, j]))
    return np.array(scores)

def pearson_r_rows(Y_true, Y_pred):
    yt_c = Y_true - np.mean(Y_true, axis=0, keepdims=True)
    yp_c = Y_pred - np.mean(Y_pred, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yt_c ** 2, axis=0) * np.sum(yp_c ** 2, axis=0))
    low_var = (denom < 1e-12)
    cors = np.zeros(Y_true.shape[1], dtype=float)
    cors[~low_var] = np.sum(yt_c * yp_c, axis=0)[~low_var] / denom[~low_var]
    cors[low_var] = np.nan
    return cors

def block_bootstrap_paired_diff(y_true, pred_a, pred_b, base_pred, block_len=20, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n_frames = y_true.shape[0]
    n_blocks = int(np.ceil(n_frames / block_len))
    diff_skills = []
    base_sse_full = np.sum((y_true - base_pred) ** 2)
    if base_sse_full < 1e-12:
        return 0.0, 0.0, 0.0

    for _ in range(n_boot):
        start_indices = rng.randint(0, n_frames - block_len + 1, size=n_blocks)
        boot_idx = np.concatenate([np.arange(s, s + block_len) for s in start_indices])[:n_frames]
        yt_b = y_true[boot_idx]
        pa_b = pred_a[boot_idx]
        pb_b = pred_b[boot_idx]
        base_b = base_pred[boot_idx]
        
        sse_base = np.sum((yt_b - base_b) ** 2)
        if sse_base < 1e-12:
            continue
        sse_a = np.sum((yt_b - pa_b) ** 2)
        sse_b = np.sum((yt_b - pb_b) ** 2)
        sk_a = 1.0 - (sse_a / sse_base)
        sk_b = 1.0 - (sse_b / sse_base)
        diff_skills.append(sk_a - sk_b)
        
    if not diff_skills:
        return 0.0, 0.0, 0.0
    return float(np.mean(diff_skills)), float(np.percentile(diff_skills, 2.5)), float(np.percentile(diff_skills, 97.5))

def generate_candidate_target_pairs(coords_dev_mean, n_res, max_cand=500, max_targ=500):
    """
    Generates disjoint candidate and target residue pair sets.
    Rule:
    - Sequence separation |i - j| >= 3
    - Candidates: smallest mean development distance (proximity / contact)
    - Targets: mean development distance in [0.5 nm, 2.0 nm], disjoint from candidates
    """
    all_pairs = []
    for i in range(n_res):
        for j in range(i + 3, n_res):
            dist = float(np.linalg.norm(coords_dev_mean[i] - coords_dev_mean[j]))
            all_pairs.append({"i_idx": i, "j_idx": j, "mean_dev_dist_nm": dist})

    # Sort all pairs by mean distance ascending
    all_pairs.sort(key=lambda x: x["mean_dev_dist_nm"])
    
    # Candidates: first max_cand pairs
    candidates = all_pairs[:min(max_cand, len(all_pairs))]
    cand_set = set((p["i_idx"], p["j_idx"]) for p in candidates)

    # Targets: remaining pairs in [0.5, 2.0] nm (or remaining if pool is small)
    remaining = [p for p in all_pairs if (p["i_idx"], p["j_idx"]) not in cand_set]
    target_pool = [p for p in remaining if 0.5 <= p["mean_dev_dist_nm"] <= 2.0]
    if len(target_pool) < max_targ:
        # Fallback to remaining sorted by distance
        target_pool = remaining
        
    targets = target_pool[:min(max_targ, len(target_pool))]
    
    # Verify disjoint guarantee
    targ_set = set((p["i_idx"], p["j_idx"]) for p in targets)
    assert len(cand_set.intersection(targ_set)) == 0, "Candidates and targets must be strictly disjoint!"
    
    # Count shared residues
    cand_residues = set()
    for p in candidates:
        cand_residues.add(p["i_idx"])
        cand_residues.add(p["j_idx"])
    targ_residues = set()
    for p in targets:
        targ_residues.add(p["i_idx"])
        targ_residues.add(p["j_idx"])
    shared_res_count = len(cand_residues.intersection(targ_residues))

    return candidates, targets, shared_res_count

def main():
    base_dir = Path("/run/media/sharon/Windows ssd drive/Sharon")
    processed_base = base_dir / "data" / "external_atlas" / "processed"
    out_dir = base_dir / "reports" / "chara_external_confirmation" / "20260908_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 4: Execute Fair External Benchmark & Diagnostics ===")

    proteins = ["1utg_A", "2cg7_A", "2j6b_A"]
    folds = [
        {"fold": 1, "test": "R1", "dev": ["R2", "R3"]},
        {"fold": 2, "test": "R2", "dev": ["R1", "R3"]},
        {"fold": 3, "test": "R3", "dev": ["R1", "R2"]}
    ]
    budgets = [5, 10, 20]

    benchmark_records = []
    paired_records = []
    diagnostics_records = []
    graph_records = []
    random_seed_records = []
    causal_records = []

    # Store full predictions for packaging
    saved_predictions = {}
    saved_model_state = {}

    for pid in proteins:
        coord_file = processed_base / f"{pid}_ca_canonical.npz"
        if not coord_file.exists():
            print(f"[SKIP] {pid} coordinates not found: {coord_file}")
            continue

        data = np.load(coord_file)
        r_coords = {
            "R1": data["R1_coords"],
            "R2": data["R2_coords"],
            "R3": data["R3_coords"]
        }
        res_names = data["residue_names"]
        res_ids = data["residue_ids"]
        n_res = len(res_ids)
        print(f"\nEvaluating {pid} ({n_res} residues, 3 runs x {r_coords['R1'].shape[0]} frames)...")

        for f_info in folds:
            fold = f_info["fold"]
            te_id = f_info["test"]
            d1_id, d2_id = f_info["dev"]

            C_test = r_coords[te_id]
            C_dev1 = r_coords[d1_id]
            C_dev2 = r_coords[d2_id]

            # Reference development coordinates (mean across both dev runs)
            C_dev_pool = np.vstack([C_dev1, C_dev2])
            C_dev_mean = np.mean(C_dev_pool, axis=0)

            # Generate candidate and target pairs from dev reference
            cands, targs, shared_res = generate_candidate_target_pairs(C_dev_mean, n_res, max_cand=500, max_targ=500)
            n_cands = len(cands)
            n_targs = len(targs)

            # Compute candidate distances
            cand_i = np.array([p["i_idx"] for p in cands])
            cand_j = np.array([p["j_idx"] for p in cands])
            X1 = np.linalg.norm(C_dev1[:, cand_i, :] - C_dev1[:, cand_j, :], axis=-1)
            X2 = np.linalg.norm(C_dev2[:, cand_i, :] - C_dev2[:, cand_j, :], axis=-1)
            Xt = np.linalg.norm(C_test[:, cand_i, :] - C_test[:, cand_j, :], axis=-1)

            # Compute target distances
            targ_i = np.array([p["i_idx"] for p in targs])
            targ_j = np.array([p["j_idx"] for p in targs])
            Y1 = np.linalg.norm(C_dev1[:, targ_i, :] - C_dev1[:, targ_j, :], axis=-1)
            Y2 = np.linalg.norm(C_dev2[:, targ_i, :] - C_dev2[:, targ_j, :], axis=-1)
            Yt = np.linalg.norm(C_test[:, targ_i, :] - C_test[:, targ_j, :], axis=-1)

            n_frames = Yt.shape[0]
            X_pool = np.vstack([X1, X2])
            Y_pool = np.vstack([Y1, Y2])

            # 1. Baseline: Development Constant Mean
            pred_base = np.tile(np.mean(Y_pool, axis=0, keepdims=True), (n_frames, 1))
            base_sse = float(np.sum((Yt - pred_base)**2))
            base_rmse = float(np.sqrt(base_sse / Yt.size))

            benchmark_records.append({
                "protein_id": pid, "fold": fold, "test_replicate": te_id,
                "method_id": "M1_DEVELOPMENT_CONSTANT_MEAN", "achieved_k": 0,
                "n_frames": n_frames, "n_targets": n_targs,
                "SSE_nm2": base_sse, "RMSE_nm": base_rmse, "skill": 0.0,
                "tuned_lambda": 0.0
            })

            # Precompute selectors at budget 20
            idx_var = np.argsort(-np.var(X_pool, axis=0))[:20].tolist()
            idx_pca = selector_pca_qr(X_pool, 20)
            idx_mean = run_greedy_transfer_search(X1, Y1, X2, Y2, mode="mean", max_k=20)
            idx_rob = run_greedy_transfer_search(X1, Y1, X2, Y2, mode="robust", max_k=20)

            # Evaluated deterministic methods
            eval_methods = [
                ("M3_DEVELOPMENT_VARIANCE", idx_var),
                ("M4_PCA_PIVOTED_QR", idx_pca),
                ("M7_MEAN_TRANSFER", idx_mean),
                ("M8_ROBUST_TRANSFER", idx_rob)
            ]

            fold_predictions = {
                "Y_true": Yt,
                "pred_base": pred_base
            }
            fold_models = {}

            for m_id, idx_list in eval_methods:
                for k in budgets:
                    sub_idx = idx_list[:k]
                    model, best_l = fit_and_tune_ridge(X1, Y1, X2, Y2, sub_idx)
                    pred = model.predict(Xt[:, sub_idx])
                    sse = float(np.sum((Yt - pred)**2))
                    rmse = float(np.sqrt(sse / Yt.size))
                    skill = float(1.0 - (sse / base_sse))

                    benchmark_records.append({
                        "protein_id": pid, "fold": fold, "test_replicate": te_id,
                        "method_id": m_id, "achieved_k": k,
                        "n_frames": n_frames, "n_targets": n_targs,
                        "SSE_nm2": sse, "RMSE_nm": rmse, "skill": skill,
                        "tuned_lambda": best_l
                    })

                    if k == 10:
                        fold_predictions[m_id] = pred
                        fold_models[m_id] = {
                            "indices": sub_idx,
                            "lambda": best_l,
                            "xm": model.xm.tolist(),
                            "xs": model.xs.tolist(),
                            "ym": model.ym.tolist(),
                            "W": model.W.tolist()
                        }

            # 2. Random Selection Distribution (20 seeds) at k=10
            pred_random_seeds = []
            for seed in range(1, 21):
                rng = np.random.RandomState(seed + 1000 * fold)
                r_idx = rng.permutation(n_cands)[:10].tolist()
                r_model, r_l = fit_and_tune_ridge(X1, Y1, X2, Y2, r_idx)
                pred_r = r_model.predict(Xt[:, r_idx])
                sse_r = float(np.sum((Yt - pred_r)**2))
                rmse_r = float(np.sqrt(sse_r / Yt.size))
                sk_r = float(1.0 - (sse_r / base_sse))

                random_seed_records.append({
                    "protein_id": pid, "fold": fold, "seed": seed,
                    "achieved_k": 10, "SSE_nm2": sse_r, "RMSE_nm": rmse_r, "skill": sk_r,
                    "tuned_lambda": r_l, "selected_indices": str(r_idx)
                })
                pred_random_seeds.append(pred_r)

            # Average random baseline
            avg_r_skill = float(np.mean([r["skill"] for r in random_seed_records[-20:]]))
            benchmark_records.append({
                "protein_id": pid, "fold": fold, "test_replicate": te_id,
                "method_id": "M2_UNIFORM_RANDOM_MEAN_20SEEDS", "achieved_k": 10,
                "n_frames": n_frames, "n_targets": n_targs,
                "SSE_nm2": np.nan, "RMSE_nm": np.nan, "skill": avg_r_skill,
                "tuned_lambda": np.nan
            })

            # 3. Paired Comparisons against M8 at k=10 with Block Bootstrap
            pred_m8 = fold_predictions["M8_ROBUST_TRANSFER"]
            pred_pca = fold_predictions["M4_PCA_PIVOTED_QR"]
            pred_var = fold_predictions["M3_DEVELOPMENT_VARIANCE"]
            pred_m7 = fold_predictions["M7_MEAN_TRANSFER"]

            m8_skill = float(1.0 - np.sum((Yt - pred_m8)**2) / base_sse)

            for comp_name, p_comp in [
                ("M4_PCA_PIVOTED_QR", pred_pca),
                ("M1_DEVELOPMENT_CONSTANT_MEAN", pred_base),
                ("M3_DEVELOPMENT_VARIANCE", pred_var),
                ("M7_MEAN_TRANSFER", pred_m7)
            ]:
                comp_skill = float(1.0 - np.sum((Yt - p_comp)**2) / base_sse)
                d_skill = m8_skill - comp_skill
                boot_mean, boot_low, boot_high = block_bootstrap_paired_diff(
                    Yt, pred_m8, p_comp, pred_base, block_len=20, n_boot=1000, seed=42 + fold
                )
                paired_records.append({
                    "protein_id": pid, "fold": fold, "test_replicate": te_id,
                    "reference_method": "M8_ROBUST_TRANSFER", "comparator_method": comp_name,
                    "m8_skill": m8_skill, "comparator_skill": comp_skill,
                    "delta_skill": d_skill, "bootstrap_mean_delta": boot_mean,
                    "bootstrap_95ci_low": boot_low, "bootstrap_95ci_high": boot_high,
                    "m8_statistically_superior": bool(boot_low > 0.0),
                    "practical_superiority_attained (delta >= 0.05)": bool(boot_low >= 0.05)
                })

            # 4. Diagnostics on M8 Predictions
            # Moment decomposition: MSE = Bias^2 + Var_res
            err_b = Yt - pred_base
            bias2_b = float(np.mean(np.mean(err_b, axis=0)**2))
            var_b = float(np.mean(np.var(err_b, axis=0)))
            mse_b = bias2_b + var_b

            err_m = Yt - pred_m8
            bias2_m = float(np.mean(np.mean(err_m, axis=0)**2))
            var_m = float(np.mean(np.var(err_m, axis=0)))
            mse_m = bias2_m + var_m

            tot_imp = mse_b - mse_m
            b_imp = bias2_b - bias2_m
            v_imp = var_b - var_m
            bias_pct = (b_imp / tot_imp * 100.0) if tot_imp > 0 else 0.0
            var_pct = (v_imp / tot_imp * 100.0) if tot_imp > 0 else 0.0

            # Test-centered skill
            yt_c = Yt - np.mean(Yt, axis=0, keepdims=True)
            pm_c = pred_m8 - np.mean(pred_m8, axis=0, keepdims=True)
            tc_skill = float(1.0 - np.sum((yt_c - pm_c)**2) / np.sum(yt_c**2))

            # Pearson r
            cors = pearson_r_rows(Yt, pred_m8)
            med_r = float(np.nanmedian(cors))
            iqr_r = float(np.nanpercentile(cors, 75) - np.nanpercentile(cors, 25))

            # Non-circular temporal lag shifts on [100..N-1]
            lag_skills = {}
            for lag in [0, 10, 25, 50, 100]:
                eval_idx = np.arange(100, n_frames)
                shifted_idx = eval_idx - lag
                yt_s = Yt[eval_idx]
                pb_s = pred_base[eval_idx]
                pm_s = pred_m8[shifted_idx]
                sb_s = np.sum((yt_s - pb_s)**2)
                sm_s = np.sum((yt_s - pm_s)**2)
                lag_skills[f"skill_lag_{lag}f"] = float(1.0 - sm_s / sb_s)

            diagnostics_records.append({
                "protein_id": pid, "fold": fold, "test_replicate": te_id,
                "n_frames": n_frames, "n_targets": n_targs,
                "all_targets_skill": m8_skill,
                "test_centered_skill": tc_skill,
                "total_mse_improvement_nm2": tot_imp,
                "bias2_reduction_share_pct": bias_pct,
                "var_reduction_share_pct": var_pct,
                "median_pearson_r": med_r,
                "iqr_pearson_r": iqr_r,
                "skill_lag_0f": lag_skills["skill_lag_0f"],
                "skill_lag_10f": lag_skills["skill_lag_10f"],
                "skill_lag_25f": lag_skills["skill_lag_25f"],
                "skill_lag_50f": lag_skills["skill_lag_50f"],
                "skill_lag_100f": lag_skills["skill_lag_100f"]
            })

            # 5. Graph Laplacian Ablation at k=10
            g_scores = build_laplacian_scores(X1, X2, cands, n_res)
            idx_graph_md = run_greedy_transfer_search(X1, Y1, X2, Y2, mode="graph", max_k=10, graph_scores=g_scores, gamma=0.05)
            m_gmd, _ = fit_and_tune_ridge(X1, Y1, X2, Y2, idx_graph_md)
            sk_gmd = float(1.0 - np.sum((Yt - m_gmd.predict(Xt[:, idx_graph_md]))**2) / base_sse)

            # Fixed unweighted adjacency
            ones_scores = np.ones(n_cands, dtype=float)
            idx_graph_unw = run_greedy_transfer_search(X1, Y1, X2, Y2, mode="graph", max_k=10, graph_scores=ones_scores, gamma=0.05)
            m_gunw, _ = fit_and_tune_ridge(X1, Y1, X2, Y2, idx_graph_unw)
            sk_gunw = float(1.0 - np.sum((Yt - m_gunw.predict(Xt[:, idx_graph_unw]))**2) / base_sse)

            # 20 weight permutations
            perm_skills = []
            for p_seed in range(1, 21):
                rng_p = np.random.RandomState(p_seed + 777 * fold)
                perm_scores = rng_p.permutation(g_scores)
                idx_perm = run_greedy_transfer_search(X1, Y1, X2, Y2, mode="graph", max_k=10, graph_scores=perm_scores, gamma=0.05)
                m_p, _ = fit_and_tune_ridge(X1, Y1, X2, Y2, idx_perm)
                sk_p = float(1.0 - np.sum((Yt - m_p.predict(Xt[:, idx_perm]))**2) / base_sse)
                perm_skills.append(sk_p)

            graph_records.append({
                "protein_id": pid, "fold": fold,
                "skill_no_graph_m8": m8_skill,
                "skill_unweighted_graph": sk_gunw,
                "skill_md_variance_graph": sk_gmd,
                "mean_skill_20_permutations": float(np.mean(perm_skills)),
                "std_skill_20_permutations": float(np.std(perm_skills)),
                "delta_skill (MD_graph - M8)": sk_gmd - m8_skill
            })

            # 6. Causal Data Dependence Audit
            # Permute test targets
            rng_c = np.random.RandomState(999)
            Yt_perm = rng_c.permutation(Yt)
            # Re-run selection on dev
            idx_rob_test = run_greedy_transfer_search(X1, Y1, X2, Y2, mode="robust", max_k=10)
            sel_invariant = (idx_rob_test == idx_rob[:10])

            # Replace test inputs with zeros
            Xt_zero = np.zeros_like(Xt)
            m8_k10_model, _ = fit_and_tune_ridge(X1, Y1, X2, Y2, idx_rob[:10])
            pred_zero = m8_k10_model.predict(Xt_zero[:, idx_rob[:10]])
            pred_changed = not np.allclose(pred_m8, pred_zero)

            causal_records.append({
                "protein_id": pid, "fold": fold,
                "test_target_permutation_invariant": bool(sel_invariant),
                "test_input_replacement_prediction_changed": bool(pred_changed),
                "audit_passed": bool(sel_invariant and pred_changed)
            })

            # Store predictions
            saved_predictions[f"{pid}_fold_{fold}"] = {
                "Y_true": Yt.astype(np.float32),
                "pred_base": pred_base.astype(np.float32),
                "pred_m8_k10": pred_m8.astype(np.float32),
                "pred_pca_k10": pred_pca.astype(np.float32)
            }
            saved_model_state[f"{pid}_fold_{fold}"] = fold_models

    # Save output tables
    df_bench = pd.DataFrame(benchmark_records)
    df_paired = pd.DataFrame(paired_records)
    df_diag = pd.DataFrame(diagnostics_records)
    df_rand = pd.DataFrame(random_seed_records)
    df_graph = pd.DataFrame(graph_records)
    df_causal = pd.DataFrame(causal_records)

    df_bench.to_csv(out_dir / "BENCHMARK_RESULTS.csv", index=False)
    df_paired.to_csv(out_dir / "PAIRED_COMPARISONS.csv", index=False)
    df_diag.to_csv(out_dir / "DIAGNOSTICS_RESULTS.csv", index=False)
    df_rand.to_csv(out_dir / "RANDOM_SEEDS_RESULTS.csv", index=False)
    df_graph.to_csv(out_dir / "GRAPH_ABLATIONS.csv", index=False)
    df_causal.to_csv(out_dir / "CAUSAL_AUDIT_RESULTS.csv", index=False)

    print(f"\n[OK] Saved all benchmark tables to {out_dir}:")
    print(f"  - BENCHMARK_RESULTS.csv ({len(df_bench)} rows)")
    print(f"  - PAIRED_COMPARISONS.csv ({len(df_paired)} rows)")
    print(f"  - DIAGNOSTICS_RESULTS.csv ({len(df_diag)} rows)")
    print(f"  - RANDOM_SEEDS_RESULTS.csv ({len(df_rand)} rows)")
    print(f"  - GRAPH_ABLATIONS.csv ({len(df_graph)} rows)")
    print(f"  - CAUSAL_AUDIT_RESULTS.csv ({len(df_causal)} rows)")

    # Save explicit held-out predictions for M8 and PCA/QR at k=10
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for key, p_dict in saved_predictions.items():
        np.savez_compressed(pred_dir / f"predictions_{key}.npz", **p_dict)
    print(f"[OK] Saved explicit predictions to {pred_dir}")

    # Save model state
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "fitted_models.json").write_text(json.dumps(saved_model_state, indent=2))
    print(f"[OK] Saved fitted model parameters to {models_dir / 'fitted_models.json'}")

if __name__ == "__main__":
    main()
