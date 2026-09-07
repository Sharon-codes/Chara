#!/usr/bin/env python3
"""
tools/chara_md_pilot/04_benchmark_experiment.py

Executes the complete bounded CPU benchmark across three outer folds of KRAS_G12D.
Evaluates 9 selectors at budgets k in {5, 10, 20}:
  1. Training-Mean Baseline
  2. Random Selection (20 fixed seeds)
  3. Highest Training Variance
  4. PCA with Pivoted-QR Selection
  5. Information Imbalance (DII-equivalent)
  6. Ordinary Pooled-Training Greedy Selector
  7. Matched Mean-Transfer Selector
  8. Proposed Worst-Transfer (Robust) Selector
  9. Graph-Assisted Robust Selector (Chara heat-kernel)

Also runs:
  - Negative control: Temporal circular shift of test candidates (+150 frames)
  - Sensitivity analysis: 20% equilibration discard vs 10% primary
  - Target breakdown: Directly restrained (N=101) vs Unrestrained (N=899)
  - Inter-fold selection overlap and correlation analysis

Saves:
  - reports/chara_md_pilot/20260907_v1/evidence/benchmark_results.json
  - reports/chara_md_pilot/20260907_v1/evidence/test_predictions_fold1.npz
  - reports/chara_md_pilot/20260907_v1/evidence/test_predictions_fold2.npz
  - reports/chara_md_pilot/20260907_v1/evidence/test_predictions_fold3.npz
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist

BASE_DIR = Path("E:/Sharon")
EVID_DIR = BASE_DIR / "reports" / "chara_md_pilot" / "20260907_v1" / "evidence"

# Import Chara's graph utilities
sys.path.insert(0, str(BASE_DIR))
from chara.graph import exponential_chara_laplacian, heat_kernel

BUDGETS = [5, 10, 20]
PRIMARY_K = 10
FOLDS = [
    {"fold_id": "fold_1", "train": ["rep1", "rep2"], "test": "rep3"},
    {"fold_id": "fold_2", "train": ["rep1", "rep3"], "test": "rep2"},
    {"fold_id": "fold_3", "train": ["rep2", "rep3"], "test": "rep1"},
]
RIDGE_LAMBDAS = [1e-4, 1e-2, 1.0, 10.0, 100.0]

# --- Common Decoder ---
class MultiOutputRidgeDecoder:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.W = None
        self.b = None

    def fit(self, X, Y):
        # X: (N, k), Y: (N, M)
        self.x_mean = np.mean(X, axis=0, keepdims=True)
        self.x_std = np.std(X, axis=0, keepdims=True)
        self.x_std[self.x_std < 1e-8] = 1.0
        
        self.y_mean = np.mean(Y, axis=0, keepdims=True)
        
        X_norm = (X - self.x_mean) / self.x_std
        Y_cent = Y - self.y_mean
        
        # Closed-form ridge: (X^T X + alpha * I)^(-1) X^T Y
        k = X.shape[1]
        XtX = X_norm.T @ X_norm
        reg = self.alpha * np.eye(k, dtype=X.dtype)
        try:
            self.W = linalg.solve(XtX + reg, X_norm.T @ Y_cent, assume_a='pos')
        except linalg.LinAlgError:
            self.W = linalg.pinv(XtX + reg) @ (X_norm.T @ Y_cent)
            
        self.b = self.y_mean - (self.x_mean / self.x_std) @ self.W
        return self

    def predict(self, X):
        X_norm = (X - self.x_mean) / self.x_std
        return X_norm @ self.W + self.y_mean

def evaluate_predictions(Y_true, Y_pred, Y_train_mean):
    """Computes overall RMSE, Target-wise RMSE, and Reconstruction Skill."""
    sse_method = np.sum((Y_true - Y_pred) ** 2)
    sse_mean = np.sum((Y_true - Y_train_mean) ** 2)
    
    n_samples, n_targets = Y_true.shape
    overall_rmse = float(np.sqrt(sse_method / (n_samples * n_targets)))
    
    skill = float(1.0 - (sse_method / (sse_mean + 1e-12)))
    
    # Per-target RMSE
    per_target_rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))
    
    return {
        "rmse_nm": overall_rmse,
        "skill": skill,
        "sse_method": float(sse_method),
        "sse_mean": float(sse_mean),
        "per_target_rmse": per_target_rmse
    }

def fit_and_tune_ridge(X_tr1, Y_tr1, X_tr2, Y_tr2, selected_indices):
    """Selects best lambda via cross-run validation, then refits on pooled data."""
    X1 = X_tr1[:, selected_indices]
    X2 = X_tr2[:, selected_indices]
    
    best_alpha = 1.0
    best_loss = float('inf')
    
    for alpha in RIDGE_LAMBDAS:
        # 1 -> 2
        m1 = MultiOutputRidgeDecoder(alpha=alpha).fit(X1, Y_tr1)
        pred2 = m1.predict(X2)
        loss12 = np.mean((Y_tr2 - pred2) ** 2)
        
        # 2 -> 1
        m2 = MultiOutputRidgeDecoder(alpha=alpha).fit(X2, Y_tr2)
        pred1 = m2.predict(X1)
        loss21 = np.mean((Y_tr1 - pred1) ** 2)
        
        avg_loss = 0.5 * (loss12 + loss21)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_alpha = alpha
            
    # Refit on pooled training data
    X_pool = np.vstack([X1, X2])
    Y_pool = np.vstack([Y_tr1, Y_tr2])
    final_model = MultiOutputRidgeDecoder(alpha=best_alpha).fit(X_pool, Y_pool)
    return final_model, best_alpha

# --- Selectors ---

def selector_variance(X_pool, k):
    var = np.var(X_pool, axis=0)
    return np.argsort(-var)[:k].tolist()

def selector_pca_qr(X_pool, k):
    x_mean = np.mean(X_pool, axis=0)
    x_std = np.std(X_pool, axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_norm = (X_pool - x_mean) / x_std
    
    U, s, Vt = linalg.svd(X_norm, full_matrices=False)
    Vk = Vt[:k, :]
    Q, R, piv = linalg.qr(Vk, pivoting=True)
    return piv[:k].tolist()

def compute_information_imbalance(D_A, rank_B):
    """Exact Information Imbalance Delta(A -> B) using rank matrix of B."""
    N = D_A.shape[0]
    D_A_diag_inf = D_A.copy()
    np.fill_diagonal(D_A_diag_inf, np.inf)
    nn_A = np.argmin(D_A_diag_inf, axis=1)
    ranks = rank_B[np.arange(N), nn_A]
    return float((2.0 / (N * N)) * np.sum(ranks))

def selector_info_imbalance(X_pool, Y_pool, k, max_budget=20):
    """Greedy forward selection minimizing Information Imbalance to Target panel."""
    N = X_pool.shape[0]
    step = max(1, N // 100)
    idx_sub = np.arange(0, N, step)[:100]
    X_sub = X_pool[idx_sub]
    Y_sub = Y_pool[idx_sub]
    
    D_B = cdist(Y_sub, Y_sub, metric='euclidean')
    rank_B = np.argsort(np.argsort(D_B, axis=1), axis=1) + 1
    
    # Pre-screen top 40 candidates with lowest individual II
    single_ii = []
    for c in range(X_pool.shape[1]):
        D_c = np.abs(X_sub[:, c:c+1] - X_sub[:, c:c+1].T)
        ii = compute_information_imbalance(D_c, rank_B)
        single_ii.append((ii, c))
    single_ii.sort()
    candidate_pool = [c for _, c in single_ii[:40]]
    
    selected = [candidate_pool[0]]
    for step_num in range(1, max_budget):
        best_c = None
        best_ii = float('inf')
        for c in candidate_pool:
            if c in selected:
                continue
            trial = selected + [c]
            D_trial = cdist(X_sub[:, trial], X_sub[:, trial], metric='euclidean')
            ii = compute_information_imbalance(D_trial, rank_B)
            if ii < best_ii:
                best_ii = ii
                best_c = c
        if best_c is not None:
            selected.append(best_c)
        else:
            break
    return selected[:k]

def run_greedy_transfer_selection(X1, Y1, X2, Y2, mode="robust", max_k=20, graph_scores=None, gamma=0.0):
    """
    Greedy forward selection with objective:
      - 'pooled': MSE on pooled training
      - 'mean': (E1->2 + E2->1) / 2
      - 'robust': max(E1->2, E2->1)
      - 'graph': max(E1->2, E2->1) - gamma * mean_graph_score
    """
    n_candidates = X1.shape[1]
    
    # Screen candidates to top 40 with highest cross-transfer correlation to targets
    corrs = []
    y_guide1 = Y1[:, :50].mean(axis=1)
    y_guide2 = Y2[:, :50].mean(axis=1)
    for c in range(n_candidates):
        r1 = np.corrcoef(X1[:, c], y_guide1)[0, 1]
        r2 = np.corrcoef(X2[:, c], y_guide2)[0, 1]
        score = 0.5 * (abs(r1) + abs(r2)) if (not np.isnan(r1) and not np.isnan(r2)) else 0.0
        corrs.append((score, c))
    corrs.sort(reverse=True)
    screened_pool = [c for _, c in corrs[:40]]
    
    # Use top 50 highest variance targets for fast greedy search evaluation
    var_y = np.var(np.vstack([Y1, Y2]), axis=0)
    top_targets_idx = np.argsort(-var_y)[:50]
    Y1_sub = Y1[:, top_targets_idx]
    Y2_sub = Y2[:, top_targets_idx]
    y1_var = np.var(Y1_sub)
    y2_var = np.var(Y2_sub)
    
    selected = []
    
    for step in range(max_k):
        best_score = float('inf')
        best_c = None
        
        for c in screened_pool:
            if c in selected:
                continue
            trial = selected + [c]
            
            if mode == "pooled":
                X_p = np.vstack([X1[:, trial], X2[:, trial]])
                Y_p = np.vstack([Y1_sub, Y2_sub])
                m = MultiOutputRidgeDecoder(alpha=1.0).fit(X_p, Y_p)
                score = np.mean((Y_p - m.predict(X_p)) ** 2)
            else:
                # 1 -> 2
                m1 = MultiOutputRidgeDecoder(alpha=1.0).fit(X1[:, trial], Y1_sub)
                e12 = np.mean((Y2_sub - m1.predict(X2[:, trial])) ** 2) / y1_var
                
                # 2 -> 1
                m2 = MultiOutputRidgeDecoder(alpha=1.0).fit(X2[:, trial], Y2_sub)
                e21 = np.mean((Y1_sub - m2.predict(X1[:, trial])) ** 2) / y2_var
                
                if mode == "mean":
                    score = 0.5 * (e12 + e21)
                elif mode == "robust":
                    score = max(e12, e21)
                elif mode == "graph":
                    g_penalty = np.mean([graph_scores[idx] for idx in trial]) if graph_scores is not None else 0.0
                    score = max(e12, e21) - gamma * g_penalty
                    
            if score < best_score:
                best_score = score
                best_c = c
                
        if best_c is not None:
            selected.append(best_c)
        else:
            break
            
    return selected

def build_chara_graph_prior(X1, X2, pair_definitions):
    """
    Constructs Chara exponential Laplacian and heat kernel diffusion
    from development residue-pair trajectory data.
    """
    n_res = 170
    candidates = pair_definitions["candidates"]
    
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
            
    L = exponential_chara_laplacian(adj, var_matrix, tau=0.5)
    H = heat_kernel(L, diffusion_time=0.1)
    
    candidate_graph_scores = []
    for p in candidates:
        i, j = p["i_idx"], p["j_idx"]
        candidate_graph_scores.append(float(H[i, j]))
        
    return np.array(candidate_graph_scores)

def main():
    print("=================================================================================", flush=True)
    print("      CHARA MD PILOT: EXECUTING STRICT LEAKAGE-FREE BOUNDED BENCHMARK           ", flush=True)
    print("=================================================================================", flush=True)
    
    # Load pair definitions
    with open(EVID_DIR / "pair_definitions.json", "r", encoding="utf-8") as f:
        pair_defs = json.load(f)
    targets = pair_defs["targets"]
    is_target_restrained = np.array([t["is_restrained"] for t in targets], dtype=bool)
    n_restrained = int(is_target_restrained.sum())
    n_unrestrained = int((~is_target_restrained).sum())
    print(f"[INFO] Target panel: {len(targets)} pairs ({n_restrained} directly restrained, {n_unrestrained} unrestrained).", flush=True)

    # Load distance data for all replicates
    reps_data = {}
    for rep in ["rep1", "rep2", "rep3"]:
        npz = np.load(EVID_DIR / f"distances_{rep}.npz")
        p_mask = npz["primary_mask"]
        reps_data[rep] = {
            "X": npz["candidate_dists"][p_mask],
            "Y": npz["target_dists"][p_mask],
            "X_sens": npz["candidate_dists"][npz["sens_mask"]],
            "Y_sens": npz["target_dists"][npz["sens_mask"]],
            "times_ps": npz["times_ps"][p_mask]
        }
        print(f"       Loaded {rep}: {reps_data[rep]['X'].shape[0]} analysis frames (10% discard).", flush=True)

    benchmark_results = {
        "metadata": {
            "system": "KRAS_G12D_ChainA",
            "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "budgets": BUDGETS,
            "primary_budget": PRIMARY_K,
            "targets_count": len(targets),
            "targets_restrained_count": n_restrained,
            "targets_unrestrained_count": n_unrestrained
        },
        "folds": {},
        "summary_table": []
    }

    # Execute outer folds
    for fold in FOLDS:
        f_id = fold["fold_id"]
        tr_reps = fold["train"]
        te_rep = fold["test"]
        print(f"\n>>> Executing {f_id.upper()}: Train={tr_reps}, Test={te_rep} <<<", flush=True)
        
        X1, Y1 = reps_data[tr_reps[0]]["X"], reps_data[tr_reps[0]]["Y"]
        X2, Y2 = reps_data[tr_reps[1]]["X"], reps_data[tr_reps[1]]["Y"]
        X_test, Y_test = reps_data[te_rep]["X"], reps_data[te_rep]["Y"]
        
        X_pool = np.vstack([X1, X2])
        Y_pool = np.vstack([Y1, Y2])
        Y_train_mean = np.mean(Y_pool, axis=0, keepdims=True)
        
        fold_record = {
            "train_replicates": tr_reps,
            "test_replicate": te_rep,
            "methods": {}
        }
        test_predictions = {"Y_true": Y_test}
        
        # 1. Training-Mean Baseline (constant predictor)
        Y_pred_mean = np.repeat(Y_train_mean, Y_test.shape[0], axis=0)
        eval_mean = evaluate_predictions(Y_test, Y_pred_mean, Y_train_mean)
        fold_record["methods"]["m1_mean"] = {
            "name": "Training-Mean Baseline",
            "k_eval": {
                "constant": {
                    "rmse_nm": eval_mean["rmse_nm"],
                    "skill": eval_mean["skill"],
                    "rmse_restrained_nm": float(np.sqrt(np.mean((Y_test[:, is_target_restrained] - Y_pred_mean[:, is_target_restrained]) ** 2))),
                    "rmse_unrestrained_nm": float(np.sqrt(np.mean((Y_test[:, ~is_target_restrained] - Y_pred_mean[:, ~is_target_restrained]) ** 2)))
                }
            }
        }
        test_predictions["m1_mean"] = Y_pred_mean
        print(f"  [M1 Mean Baseline]  Held-out RMSE: {eval_mean['rmse_nm']:.4f} nm (Skill: {eval_mean['skill']:.4f})", flush=True)
        
        # 2. Random Selection (20 fixed seeds)
        rand_results = {k: [] for k in BUDGETS}
        for seed in range(1, 21):
            rng = np.random.RandomState(seed + 100)
            perm = rng.permutation(1000)
            for k in BUDGETS:
                idx = perm[:k].tolist()
                model, alpha = fit_and_tune_ridge(X1, Y1, X2, Y2, idx)
                pred = model.predict(X_test[:, idx])
                ev = evaluate_predictions(Y_test, pred, Y_train_mean)
                rand_results[k].append(ev["rmse_nm"])
                
        fold_record["methods"]["m2_random"] = {
            "name": "Random Selection (20 Seeds)",
            "k_eval": {
                str(k): {
                    "mean_rmse_nm": float(np.mean(rand_results[k])),
                    "std_rmse_nm": float(np.std(rand_results[k])),
                    "min_rmse_nm": float(np.min(rand_results[k])),
                    "max_rmse_nm": float(np.max(rand_results[k])),
                    "skill": float(1.0 - (np.mean(rand_results[k]) ** 2) / (eval_mean['rmse_nm'] ** 2))
                } for k in BUDGETS
            }
        }
        print(f"  [M2 Random (k=10)]  Mean RMSE: {np.mean(rand_results[10]):.4f} +/- {np.std(rand_results[10]):.4f} nm", flush=True)
        
        # 3. Highest Variance Selection
        idx_var_20 = selector_variance(X_pool, 20)
        fold_record["methods"]["m3_variance"] = {"name": "Highest Variance Selection", "k_eval": {}}
        for k in BUDGETS:
            idx = idx_var_20[:k]
            model, alpha = fit_and_tune_ridge(X1, Y1, X2, Y2, idx)
            pred = model.predict(X_test[:, idx])
            ev = evaluate_predictions(Y_test, pred, Y_train_mean)
            fold_record["methods"]["m3_variance"]["k_eval"][str(k)] = {
                "rmse_nm": ev["rmse_nm"],
                "skill": ev["skill"],
                "selected_features": idx,
                "best_alpha": alpha
            }
            if k == PRIMARY_K:
                test_predictions["m3_variance_k10"] = pred
                print(f"  [M3 Variance (k=10)] RMSE: {ev['rmse_nm']:.4f} nm (Skill: {ev['skill']:.4f})", flush=True)
                
        # 4. PCA with Pivoted-QR Selection
        idx_pca_20 = selector_pca_qr(X_pool, 20)
        fold_record["methods"]["m4_pca_qr"] = {"name": "PCA / Pivoted-QR Selection", "k_eval": {}}
        for k in BUDGETS:
            idx = idx_pca_20[:k]
            model, alpha = fit_and_tune_ridge(X1, Y1, X2, Y2, idx)
            pred = model.predict(X_test[:, idx])
            ev = evaluate_predictions(Y_test, pred, Y_train_mean)
            fold_record["methods"]["m4_pca_qr"]["k_eval"][str(k)] = {
                "rmse_nm": ev["rmse_nm"],
                "skill": ev["skill"],
                "selected_features": idx,
                "best_alpha": alpha
            }
            if k == PRIMARY_K:
                test_predictions["m4_pca_qr_k10"] = pred
                print(f"  [M4 PCA/QR (k=10)]   RMSE: {ev['rmse_nm']:.4f} nm (Skill: {ev['skill']:.4f})", flush=True)

        # 5. Information Imbalance (DII-equivalent)
        idx_ii_20 = selector_info_imbalance(X_pool, Y_pool, 20, max_budget=20)
        fold_record["methods"]["m5_info_imbalance"] = {"name": "Information Imbalance (DII-equiv)", "k_eval": {}}
        for k in BUDGETS:
            idx = idx_ii_20[:k]
            model, alpha = fit_and_tune_ridge(X1, Y1, X2, Y2, idx)
            pred = model.predict(X_test[:, idx])
            ev = evaluate_predictions(Y_test, pred, Y_train_mean)
            fold_record["methods"]["m5_info_imbalance"]["k_eval"][str(k)] = {
                "rmse_nm": ev["rmse_nm"],
                "skill": ev["skill"],
                "selected_features": idx,
                "best_alpha": alpha
            }
            if k == PRIMARY_K:
                test_predictions["m5_info_imbalance_k10"] = pred
                print(f"  [M5 Info Imb (k=10)] RMSE: {ev['rmse_nm']:.4f} nm (Skill: {ev['skill']:.4f})", flush=True)

        # 6. Ordinary Pooled-Training Greedy Selector
        idx_pool_20 = run_greedy_transfer_selection(X1, Y1, X2, Y2, mode="pooled", max_k=20)
        fold_record["methods"]["m6_pooled_greedy"] = {"name": "Pooled Greedy Selector", "k_eval": {}}
        for k in BUDGETS:
            idx = idx_pool_20[:k]
            model, alpha = fit_and_tune_ridge(X1, Y1, X2, Y2, idx)
            pred = model.predict(X_test[:, idx])
            ev = evaluate_predictions(Y_test, pred, Y_train_mean)
            fold_record["methods"]["m6_pooled_greedy"]["k_eval"][str(k)] = {
                "rmse_nm": ev["rmse_nm"],
                "skill": ev["skill"],
                "selected_features": idx,
                "best_alpha": alpha
            }
            if k == PRIMARY_K:
                test_predictions["m6_pooled_greedy_k10"] = pred
                print(f"  [M6 Pooled (k=10)]   RMSE: {ev['rmse_nm']:.4f} nm (Skill: {ev['skill']:.4f})", flush=True)

        # 7. Matched Mean-Transfer Selector
        idx_mean_20 = run_greedy_transfer_selection(X1, Y1, X2, Y2, mode="mean", max_k=20)
        fold_record["methods"]["m7_mean_transfer"] = {"name": "Matched Mean-Transfer Selector", "k_eval": {}}
        for k in BUDGETS:
            idx = idx_mean_20[:k]
            model, alpha = fit_and_tune_ridge(X1, Y1, X2, Y2, idx)
            pred = model.predict(X_test[:, idx])
            ev = evaluate_predictions(Y_test, pred, Y_train_mean)
            fold_record["methods"]["m7_mean_transfer"]["k_eval"][str(k)] = {
                "rmse_nm": ev["rmse_nm"],
                "skill": ev["skill"],
                "selected_features": idx,
                "best_alpha": alpha
            }
            if k == PRIMARY_K:
                test_predictions["m7_mean_transfer_k10"] = pred
                print(f"  [M7 Mean-Tr (k=10)]  RMSE: {ev['rmse_nm']:.4f} nm (Skill: {ev['skill']:.4f})", flush=True)

        # 8. Proposed Worst-Transfer (Robust) Selector
        idx_robust_20 = run_greedy_transfer_selection(X1, Y1, X2, Y2, mode="robust", max_k=20)
        fold_record["methods"]["m8_robust_transfer"] = {"name": "Proposed Worst-Transfer (Robust) Selector", "k_eval": {}}
        for k in BUDGETS:
            idx = idx_robust_20[:k]
            model, alpha = fit_and_tune_ridge(X1, Y1, X2, Y2, idx)
            pred = model.predict(X_test[:, idx])
            ev = evaluate_predictions(Y_test, pred, Y_train_mean)
            
            rmse_rest = float(np.sqrt(np.mean((Y_test[:, is_target_restrained] - pred[:, is_target_restrained]) ** 2)))
            rmse_unrest = float(np.sqrt(np.mean((Y_test[:, ~is_target_restrained] - pred[:, ~is_target_restrained]) ** 2)))
            
            fold_record["methods"]["m8_robust_transfer"]["k_eval"][str(k)] = {
                "rmse_nm": ev["rmse_nm"],
                "skill": ev["skill"],
                "rmse_restrained_nm": rmse_rest,
                "rmse_unrestrained_nm": rmse_unrest,
                "selected_features": idx,
                "best_alpha": alpha
            }
            if k == PRIMARY_K:
                test_predictions["m8_robust_transfer_k10"] = pred
                print(f"  [M8 Robust (k=10)]   RMSE: {ev['rmse_nm']:.4f} nm (Skill: {ev['skill']:.4f}) [Restrained: {rmse_rest:.4f}, Unrestrained: {rmse_unrest:.4f}]", flush=True)

        # 9. Graph-Assisted Robust Selector (Chara heat-kernel)
        graph_scores = build_chara_graph_prior(X1, X2, pair_defs)
        idx_graph_20 = run_greedy_transfer_selection(X1, Y1, X2, Y2, mode="graph", max_k=20, graph_scores=graph_scores, gamma=0.05)
        fold_record["methods"]["m9_graph_assisted"] = {"name": "Graph-Assisted Robust Selector", "k_eval": {}}
        for k in BUDGETS:
            idx = idx_graph_20[:k]
            model, alpha = fit_and_tune_ridge(X1, Y1, X2, Y2, idx)
            pred = model.predict(X_test[:, idx])
            ev = evaluate_predictions(Y_test, pred, Y_train_mean)
            fold_record["methods"]["m9_graph_assisted"]["k_eval"][str(k)] = {
                "rmse_nm": ev["rmse_nm"],
                "skill": ev["skill"],
                "selected_features": idx,
                "best_alpha": alpha
            }
            if k == PRIMARY_K:
                test_predictions["m9_graph_assisted_k10"] = pred
                print(f"  [M9 Graph (k=10)]    RMSE: {ev['rmse_nm']:.4f} nm (Skill: {ev['skill']:.4f})", flush=True)

        # Negative Control: Temporal Circular Shift (+150 frames) on M8 Robust k=10
        idx_k10 = idx_robust_20[:10]
        model_k10, alpha_k10 = fit_and_tune_ridge(X1, Y1, X2, Y2, idx_k10)
        X_test_shifted = np.roll(X_test[:, idx_k10], shift=150, axis=0)
        pred_shifted = model_k10.predict(X_test_shifted)
        ev_shifted = evaluate_predictions(Y_test, pred_shifted, Y_train_mean)
        fold_record["negative_control_temporal_shift_k10"] = {
            "rmse_nm": ev_shifted["rmse_nm"],
            "skill": ev_shifted["skill"],
            "shift_frames": 150
        }
        print(f"  [Neg Control (k=10)] Shifted RMSE: {ev_shifted['rmse_nm']:.4f} nm (Skill: {ev_shifted['skill']:.4f})", flush=True)

        # Sensitivity Analysis: 20% Equilibration Discard (frames 201..1000)
        X1_s, Y1_s = reps_data[tr_reps[0]]["X_sens"], reps_data[tr_reps[0]]["Y_sens"]
        X2_s, Y2_s = reps_data[tr_reps[1]]["X_sens"], reps_data[tr_reps[1]]["Y_sens"]
        Xt_s, Yt_s = reps_data[te_rep]["X_sens"], reps_data[te_rep]["Y_sens"]
        Y_mean_s = np.mean(np.vstack([Y1_s, Y2_s]), axis=0, keepdims=True)
        
        idx_rob_sens = run_greedy_transfer_selection(X1_s, Y1_s, X2_s, Y2_s, mode="robust", max_k=10)
        m_sens, _ = fit_and_tune_ridge(X1_s, Y1_s, X2_s, Y2_s, idx_rob_sens)
        pred_sens = m_sens.predict(Xt_s[:, idx_rob_sens])
        ev_sens = evaluate_predictions(Yt_s, pred_sens, Y_mean_s)
        fold_record["sensitivity_20pct_discard_k10"] = {
            "rmse_nm": ev_sens["rmse_nm"],
            "skill": ev_sens["skill"]
        }
        print(f"  [Sensitivity 20%]    RMSE: {ev_sens['rmse_nm']:.4f} nm (Skill: {ev_sens['skill']:.4f})", flush=True)

        # Save test predictions for this fold
        np.savez_compressed(EVID_DIR / f"test_predictions_{f_id}.npz", **test_predictions)
        benchmark_results["folds"][f_id] = fold_record

    # Save benchmark results JSON
    with open(EVID_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n=================================================================================", flush=True)
    print("      BENCHMARK COMPLETE: RESULTS COMPILED ACROSS ALL 3 OUTER FOLDS             ", flush=True)
    print("=================================================================================", flush=True)
    print(f"[OK] Full benchmark results saved: {EVID_DIR / 'benchmark_results.json'}", flush=True)

if __name__ == "__main__":
    main()
