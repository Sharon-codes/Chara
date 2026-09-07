#!/usr/bin/env python3
"""
tools/chara_md_dii/01_dii_software_check.py

Official DADApy DII Software Check (Synthetic Data Verification)
Confirms:
1. Imported official DADApy version, platform, python version.
2. Initialization of FeatureWeighting on source and target coordinates.
3. Computation of initial DII.
4. Execution of differentiable weight optimization (return_weights_optimize_dii).
5. Execution of backward greedy elimination (return_backward_greedy_dii_elimination).
6. Production of finite loss/weights, optimization history, and subset extraction.
7. Saves output to reports/chara_md_dii/20260907_v1/evidence/dii_software_check.json.
"""

import os
import sys
import json
import platform
from pathlib import Path
import numpy as np

BASE_DIR = Path("E:/Sharon")
EVID_DIR = BASE_DIR / "reports" / "chara_md_dii" / "20260907_v1" / "evidence"
EVID_DIR.mkdir(parents=True, exist_ok=True)

import dadapy
from dadapy.feature_weighting import FeatureWeighting

def run_software_check():
    print("=================================================================")
    print("      OFFICIAL DADAPY DII SOFTWARE VERIFICATION CHECK            ")
    print("=================================================================")
    
    np.random.seed(42)
    N = 60
    D = 6
    
    # Informative features + noise
    t = np.linspace(0, 4 * np.pi, N)
    y1 = np.sin(t)
    y2 = np.cos(t)
    Y = np.column_stack([y1, y2])
    
    # X has first 2 features correlated with Y, remaining 4 are pure noise
    x1 = y1 + 0.1 * np.random.randn(N)
    x2 = y2 + 0.1 * np.random.randn(N)
    x_noise = np.random.randn(N, 4)
    X = np.column_stack([x1, x2, x_noise])

    import importlib.metadata
    dadapy_ver = importlib.metadata.version('dadapy')
    print(f"DADApy version: {dadapy_ver}")
    print(f"Python: {sys.version.split()[0]} on {platform.platform()}")
    print(f"Synthetic test dimensions: N={N}, D_source={D}, D_target=2")

    # Step 1: Initialize FeatureWeighting objects
    fw_x = FeatureWeighting(coordinates=X)
    fw_y = FeatureWeighting(coordinates=Y)

    # Step 2: Compute initial DII with equal weights
    initial_dii = float(fw_x.return_dii(target_data=fw_y))
    print(f"Initial unweighted DII: {initial_dii:.6f}")
    assert np.isfinite(initial_dii), "Initial DII must be finite"

    # Step 3: Differentiable optimization of weights
    print("Running return_weights_optimize_dii (n_epochs=30)...")
    opt_weights = fw_x.return_weights_optimize_dii(
        target_data=fw_y, n_epochs=30, learning_rate=0.1
    )
    final_opt_dii = float(fw_x.return_dii(target_data=fw_y))
    print(f"Optimized weights: {opt_weights.round(4)}")
    print(f"Optimized DII: {final_opt_dii:.6f}")
    assert np.all(np.isfinite(opt_weights)), "Weights must be finite"
    assert np.isfinite(final_opt_dii), "Optimized DII must be finite"

    # Step 4: Backward greedy elimination
    print("Running return_backward_greedy_dii_elimination (n_epochs=15)...")
    final_diis, final_weights = fw_x.return_backward_greedy_dii_elimination(
        target_data=fw_y, n_epochs=15, learning_rate=0.1
    )
    print(f"Backward elimination completed. Shapes: final_diis={final_diis.shape}, final_weights={final_weights.shape}")
    assert len(final_diis) == D, f"Expected {D} DII values, got {len(final_diis)}"
    assert final_weights.shape == (D, D), f"Expected ({D},{D}) weight matrix, got {final_weights.shape}"

    # Extract non-zero feature counts and subsets at each stage
    subsets_by_size = {}
    for n_feat in range(1, D + 1):
        w_row = final_weights[n_feat - 1]
        active_indices = np.where(np.abs(w_row) > 1e-8)[0].tolist()
        dii_val = float(final_diis[n_feat - 1])
        subsets_by_size[str(n_feat)] = {
            "active_feature_count": len(active_indices),
            "active_indices": active_indices,
            "dii": dii_val,
            "weights": w_row.tolist()
        }
        print(f"  k={n_feat}: active={active_indices} (count={len(active_indices)}), DII={dii_val:.5f}")

    # History verification
    history_keys = list(fw_x.history.keys())
    print(f"History recorded keys: {history_keys}")
    assert len(history_keys) > 0, "Optimization history must be recorded"

    check_result = {
        "status": "PASSED",
        "dadapy_version": dadapy_ver,
        "dadapy_module_path": dadapy.__file__,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "synthetic_test_params": {"N": N, "D_source": D, "D_target": 2, "random_seed": 42},
        "initial_dii": initial_dii,
        "optimized_dii": final_opt_dii,
        "optimized_weights": opt_weights.tolist(),
        "backward_elimination_subsets": subsets_by_size,
        "history_keys": history_keys,
        "finite_checks": {
            "initial_dii_finite": bool(np.isfinite(initial_dii)),
            "optimized_dii_finite": bool(np.isfinite(final_opt_dii)),
            "weights_finite": bool(np.all(np.isfinite(opt_weights))),
            "elimination_diis_finite": bool(np.all(np.isfinite(final_diis)))
        }
    }

    out_file = EVID_DIR / "dii_software_check.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(check_result, f, indent=2)

    print(f"[OK] DII Software Check successfully passed. Output written to {out_file}")

if __name__ == "__main__":
    run_software_check()
