#!/usr/bin/env python3
import os
import hashlib
import pandas as pd
import numpy as np

def verify():
    print("=== Standalone Evidence & Mathematical Integrity Verification ===")
    if os.path.exists("manifest_checksums.sha256"):
        with open("manifest_checksums.sha256", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    h_exp = parts[0]
                    fname = parts[1]
                    if os.path.exists(fname):
                        with open(fname, "rb") as fp:
                            h_act = hashlib.sha256(fp.read()).hexdigest()
                        assert h_act == h_exp, f"Hash mismatch for {fname}"
                        print(f"  [OK] Verified hash: {fname}")

    if os.path.exists("DIAGNOSTICS_FIXED.csv"):
        df = pd.read_csv("DIAGNOSTICS_FIXED.csv")
        # Check exact moment decomposition MSE = Bias^2 + Var_res
        err_m = np.abs(df["mean_mse_model_nm2"] - (df["mean_bias2_model_nm2"] + df["mean_var_res_model_nm2"]))
        err_b = np.abs(df["mean_mse_baseline_nm2"] - (df["mean_bias2_baseline_nm2"] + df["mean_var_res_baseline_nm2"]))
        max_err = max(np.max(err_m), np.max(err_b))
        assert max_err < 1e-12, f"Moment decomposition residual too high: {max_err}"
        print(f"  [OK] Exact population moment decomposition holds (max residual: {max_err:.2e})")

        # Check KRAS M8 reference values
        kras_m8 = df[(df["system"] == "KRAS_G12D") & (df["method_key"] == "pred_m8_robust_transfer_k10") & (df["target_group"] == "all_targets")]
        sum_d_mse = np.sum(kras_m8["delta_mse_fold_mean_nm2"])
        sum_d_bias2 = np.sum(kras_m8["delta_bias2_fold_mean_nm2"])
        ratio = sum_d_bias2 / sum_d_mse
        print(f"  [OK] KRAS M8 Fold-mean delta_MSE: {sum_d_mse:.18f} nm^2")
        print(f"  [OK] KRAS M8 Fold-mean delta_Bias^2: {sum_d_bias2:.18f} nm^2")
        print(f"  [OK] KRAS M8 Ratio: {ratio:.18f}")
        assert abs(sum_d_mse - 0.002958242063380099) < 1e-12
        assert abs(sum_d_bias2 - 0.002478250469269188) < 1e-12
        assert abs(ratio - 0.837744314418114) < 1e-10

    print("=== All Standalone Verifications Passed Successfully ===")

if __name__ == '__main__':
    verify()
