#!/usr/bin/env python3
"""
tools/chara_md_pilot/05_analyze_and_generate_figures.py

Performs exhaustive analysis of the MD pilot benchmark results, computes paired statistics,
generates publication-quality embedded SVG figures, analyzes selected residue pairs and structural mapping,
and saves consolidated analysis metrics.
"""

import sys
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path("E:/Sharon")
EVID_DIR = BASE_DIR / "reports" / "chara_md_pilot" / "20260907_v1" / "evidence"

def generate_svg_rmse_curve(results):
    """Figure 1: Held-out RMSE vs budget k across folds."""
    methods_to_plot = [
        ("m1_mean", "Mean Baseline", "#94a3b8", "dash"),
        ("m2_random", "Random (20 seeds)", "#64748b", "dot"),
        ("m3_variance", "Highest Variance", "#f59e0b", "solid"),
        ("m4_pca_qr", "PCA / Pivoted-QR", "#3b82f6", "solid"),
        ("m5_info_imbalance", "Information Imbalance (DII)", "#8b5cf6", "solid"),
        ("m6_pooled_greedy", "Pooled Greedy", "#06b6d4", "solid"),
        ("m7_mean_transfer", "Matched Mean-Transfer", "#ec4899", "solid"),
        ("m8_robust_transfer", "Proposed Robust-Transfer", "#10b981", "solid_thick"),
        ("m9_graph_assisted", "Graph-Assisted Robust", "#ef4444", "dash_thick")
    ]
    
    # Calculate mean RMSE across folds for each budget
    budgets = [5, 10, 20]
    plot_data = {}
    for mid, mlabel, color, style in methods_to_plot:
        plot_data[mid] = []
        for k in budgets:
            vals = []
            for fid in ["fold_1", "fold_2", "fold_3"]:
                m_entry = results["folds"][fid]["methods"][mid]
                if mid == "m1_mean":
                    vals.append(m_entry["k_eval"]["constant"]["rmse_nm"])
                elif mid == "m2_random":
                    vals.append(m_entry["k_eval"][str(k)]["mean_rmse_nm"])
                else:
                    vals.append(m_entry["k_eval"][str(k)]["rmse_nm"])
            plot_data[mid].append(float(np.mean(vals)))

    width, height = 750, 420
    pad_l, pad_r, pad_t, pad_b = 80, 240, 40, 60
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    # Y range: 0.055 to 0.085 nm
    y_min, y_max = 0.055, 0.085
    def x_scale(k): return pad_l + ((k - 5) / (20 - 5)) * pw
    def y_scale(v): return pad_t + (1.0 - (v - y_min) / (y_max - y_min)) * ph

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:8px; font-family:sans-serif;">']
    svg.append(f'<text x="{width/2}" y="25" fill="#f8fafc" font-size="14" font-weight="bold" text-anchor="middle">Figure 1: Held-out Reconstruction RMSE vs Measurement Budget (k)</text>')
    
    # Grid lines
    for y_val in [0.060, 0.065, 0.070, 0.075, 0.080, 0.085]:
        y_pos = y_scale(y_val)
        svg.append(f'<line x1="{pad_l}" y1="{y_pos:.1f}" x2="{pad_l+pw}" y2="{y_pos:.1f}" stroke="#334155" stroke-width="1" stroke-dasharray="2,2"/>')
        svg.append(f'<text x="{pad_l-10}" y="{y_pos+4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{y_val:.3f} nm</text>')

    for k in budgets:
        x_pos = x_scale(k)
        svg.append(f'<line x1="{x_pos:.1f}" y1="{pad_t}" x2="{x_pos:.1f}" y2="{pad_t+ph}" stroke="#334155" stroke-width="1"/>')
        svg.append(f'<text x="{x_pos:.1f}" y="{pad_t+ph+20}" fill="#cbd5e1" font-size="12" font-weight="bold" text-anchor="middle">k = {k}</text>')

    # Axis labels
    svg.append(f'<text x="{pad_l+pw/2}" y="{height-15}" fill="#cbd5e1" font-size="12" text-anchor="middle">Observed Distance Budget (k pairs)</text>')
    svg.append(f'<text x="25" y="{pad_t+ph/2}" fill="#cbd5e1" font-size="12" text-anchor="middle" transform="rotate(-90 25 {pad_t+ph/2})">Mean Held-Out RMSE (nm)</text>')

    # Plot lines
    legend_y = pad_t + 10
    for mid, mlabel, color, style in methods_to_plot:
        pts = [f"{x_scale(k):.1f},{y_scale(plot_data[mid][i]):.1f}" for i, k in enumerate(budgets)]
        dash = 'stroke-dasharray="6,4"' if "dash" in style else ('stroke-dasharray="2,2"' if "dot" in style else "")
        sw = "3" if "thick" in style else "2"
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{sw}" {dash}/>')
        for i, k in enumerate(budgets):
            svg.append(f'<circle cx="{x_scale(k):.1f}" cy="{y_scale(plot_data[mid][i]):.1f}" r="4" fill="{color}"/>')

        # Legend
        svg.append(f'<line x1="{pad_l+pw+20}" y1="{legend_y}" x2="{pad_l+pw+50}" y2="{legend_y}" stroke="{color}" stroke-width="{sw}" {dash}/>')
        svg.append(f'<circle cx="{pad_l+pw+35}" cy="{legend_y}" r="3.5" fill="{color}"/>')
        svg.append(f'<text x="{pad_l+pw+58}" y="{legend_y+4}" fill="#e2e8f0" font-size="11">{mlabel}</text>')
        legend_y += 24

    svg.append('</svg>')
    return "\n".join(svg)

def generate_svg_forest_plot(results):
    """Figure 2: Forest plot of paired differences against comparator methods at primary k=10."""
    # Comparators to Robust Selector:
    # Diff = RMSE(Robust) - RMSE(Comparator)
    # Negative value means Robust is BETTER.
    comparisons = [
        ("m7_mean_transfer", "vs Matched Mean-Transfer (M8 - M7)"),
        ("m6_pooled_greedy", "vs Pooled Greedy (M8 - M6)"),
        ("m5_info_imbalance", "vs Info Imbalance / DII (M8 - M5)"),
        ("m4_pca_qr", "vs PCA / Pivoted-QR (M8 - M4)"),
        ("m3_variance", "vs Highest Variance (M8 - M3)"),
        ("m2_random", "vs Random Selection (M8 - M2)"),
        ("m1_mean", "vs Mean Baseline (M8 - M1)"),
        ("graph_gain", "Graph Gain (M9 Graph - M8 Robust)")
    ]

    diff_data = []
    for comp_id, label in comparisons:
        fold_diffs = []
        for fid in ["fold_1", "fold_2", "fold_3"]:
            f_m = results["folds"][fid]["methods"]
            r_rob = f_m["m8_robust_transfer"]["k_eval"]["10"]["rmse_nm"]
            if comp_id == "m1_mean":
                r_comp = f_m["m1_mean"]["k_eval"]["constant"]["rmse_nm"]
            elif comp_id == "m2_random":
                r_comp = f_m["m2_random"]["k_eval"]["10"]["mean_rmse_nm"]
            elif comp_id == "graph_gain":
                r_comp = r_rob
                r_rob = f_m["m9_graph_assisted"]["k_eval"]["10"]["rmse_nm"]
            else:
                r_comp = f_m[comp_id]["k_eval"]["10"]["rmse_nm"]
            fold_diffs.append(r_rob - r_comp)
        mean_d = float(np.mean(fold_diffs))
        min_d = float(np.min(fold_diffs))
        max_d = float(np.max(fold_diffs))
        diff_data.append((label, mean_d, min_d, max_d, fold_diffs))

    width, height = 750, 420
    pad_l, pad_r, pad_t, pad_b = 300, 40, 50, 60
    pw = width - pad_l - pad_r
    row_h = (height - pad_t - pad_b) / len(comparisons)

    # Scale: -0.015 to +0.005 nm
    x_min, x_max = -0.015, +0.005
    def x_scale(v): return pad_l + ((v - x_min) / (x_max - x_min)) * pw

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:8px; font-family:sans-serif;">']
    svg.append(f'<text x="{width/2}" y="25" fill="#f8fafc" font-size="14" font-weight="bold" text-anchor="middle">Figure 2: Paired Differences in Held-Out RMSE at Primary Budget k=10</text>')

    # Zero null line
    x0 = x_scale(0.0)
    svg.append(f'<line x1="{x0:.1f}" y1="{pad_t}" x2="{x0:.1f}" y2="{height-pad_b}" stroke="#ef4444" stroke-width="2" stroke-dasharray="4"/>')
    svg.append(f'<text x="{x0:.1f}" y="{pad_t-8}" fill="#ef4444" font-size="11" font-weight="bold" text-anchor="middle">Null (Δ=0)</text>')

    # 5% Screening Threshold line relative to average baseline (~ -0.0035 nm)
    x_thresh = x_scale(-0.0038)
    svg.append(f'<line x1="{x_thresh:.1f}" y1="{pad_t}" x2="{x_thresh:.1f}" y2="{height-pad_b}" stroke="#10b981" stroke-width="1.5" stroke-dasharray="2,2"/>')
    svg.append(f'<text x="{x_thresh:.1f}" y="{pad_t-8}" fill="#10b981" font-size="10" font-weight="bold" text-anchor="middle">5% Margin</text>')

    # X axis ticks
    for val in [-0.015, -0.010, -0.005, 0.0, 0.005]:
        xp = x_scale(val)
        svg.append(f'<line x1="{xp:.1f}" y1="{height-pad_b}" x2="{xp:.1f}" y2="{height-pad_b+5}" stroke="#64748b" stroke-width="1"/>')
        svg.append(f'<text x="{xp:.1f}" y="{height-pad_b+20}" fill="#94a3b8" font-size="11" text-anchor="middle">{val:+.3f}</text>')

    svg.append(f'<text x="{pad_l+pw/2}" y="{height-15}" fill="#cbd5e1" font-size="12" text-anchor="middle">Paired Difference Δ RMSE (nm) [Left = Proposed Method Wins]</text>')

    # Rows
    for i, (label, mean_d, min_d, max_d, f_diffs) in enumerate(diff_data):
        y_c = pad_t + (i + 0.5) * row_h
        bg_col = "#1e293b" if i % 2 == 0 else "none"
        svg.append(f'<rect x="10" y="{y_c - row_h/2:.1f}" width="{width-20}" height="{row_h:.1f}" fill="{bg_col}" opacity="0.3"/>')
        svg.append(f'<text x="{pad_l-15}" y="{y_c+4:.1f}" fill="#e2e8f0" font-size="11" font-weight="600" text-anchor="end">{label}</text>')
        
        # Horizontal error bar across the 3 folds
        x_left = x_scale(min_d)
        x_right = x_scale(max_d)
        x_pt = x_scale(mean_d)
        svg.append(f'<line x1="{x_left:.1f}" y1="{y_c:.1f}" x2="{x_right:.1f}" y2="{y_c:.1f}" stroke="#38bdf8" stroke-width="2" stroke-linecap="round"/>')
        svg.append(f'<line x1="{x_left:.1f}" y1="{y_c-4:.1f}" x2="{x_left:.1f}" y2="{y_c+4:.1f}" stroke="#38bdf8" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x_right:.1f}" y1="{y_c-4:.1f}" x2="{x_right:.1f}" y2="{y_c+4:.1f}" stroke="#38bdf8" stroke-width="1.5"/>')
        
        pt_color = "#10b981" if mean_d < -0.001 else ("#f59e0b" if abs(mean_d) <= 0.001 else "#f43f5e")
        svg.append(f'<circle cx="{x_pt:.1f}" cy="{y_c:.1f}" r="5" fill="{pt_color}" stroke="#0f172a" stroke-width="1.5"/>')
        svg.append(f'<text x="{x_right+10:.1f}" y="{y_c+4:.1f}" fill="#cbd5e1" font-size="10">{mean_d:+.4f} nm</text>')

    svg.append('</svg>')
    return "\n".join(svg)

def generate_svg_trace_plot(f1_preds, pair_defs):
    """Figure 3: Representative actual vs reconstructed distance trajectory traces."""
    Y_true = f1_preds["Y_true"]
    Y_pred_rob = f1_preds["m8_robust_transfer_k10"]
    Y_pred_mean = f1_preds["m1_mean"]
    
    # Select median-error target among unrestrained pairs
    targets = pair_defs["targets"]
    errs = np.mean((Y_true - Y_pred_rob)**2, axis=0)
    unrest_indices = [i for i, t in enumerate(targets) if not t["is_restrained"]]
    med_target_idx = unrest_indices[int(np.argsort(errs[unrest_indices])[len(unrest_indices)//2])]
    med_target = targets[med_target_idx]
    
    # Also select a restrained target
    rest_indices = [i for i, t in enumerate(targets) if t["is_restrained"]]
    rest_target_idx = rest_indices[int(np.argsort(errs[rest_indices])[len(rest_indices)//2])]
    rest_target = targets[rest_target_idx]

    width, height = 750, 420
    pad_l, pad_r, pad_t, pad_b = 60, 40, 40, 50
    pw = width - pad_l - pad_r
    ph = (height - pad_t - pad_b - 30) / 2

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:8px; font-family:sans-serif;">']
    svg.append(f'<text x="{width/2}" y="25" fill="#f8fafc" font-size="14" font-weight="bold" text-anchor="middle">Figure 3: Actual vs Reconstructed Target Distances (Fold 1: Test Replicate 3, k=10)</text>')

    def plot_subtrace(y_true_s, y_pred_s, y_mean_s, title, y_offset):
        # Frame axis: 0 to 900
        n_frames = len(y_true_s)
        # Subsample every 3 frames for clean SVG rendering
        sub_idx = np.arange(0, n_frames, 3)
        v_min = min(y_true_s.min(), y_pred_s.min()) - 0.02
        v_max = max(y_true_s.max(), y_pred_s.max()) + 0.02
        
        def x_s(idx): return pad_l + (idx / n_frames) * pw
        def y_s(val): return y_offset + (1.0 - (val - v_min) / (v_max - v_min)) * ph
        
        svg.append(f'<rect x="{pad_l}" y="{y_offset}" width="{pw}" height="{ph}" fill="#1e293b" opacity="0.4" rx="4"/>')
        svg.append(f'<text x="{pad_l+10}" y="{y_offset+18}" fill="#38bdf8" font-size="11" font-weight="bold">{title}</text>')
        
        # Grid lines
        for step_v in np.linspace(v_min, v_max, 4):
            yp = y_s(step_v)
            svg.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{pad_l+pw}" y2="{yp:.1f}" stroke="#334155" stroke-width="0.5"/>')
            svg.append(f'<text x="{pad_l-5}" y="{yp+3:.1f}" fill="#94a3b8" font-size="9" text-anchor="end">{step_v:.2f}</text>')

        # Mean line
        mean_y = y_s(float(y_mean_s[0]))
        svg.append(f'<line x1="{pad_l}" y1="{mean_y:.1f}" x2="{pad_l+pw}" y2="{mean_y:.1f}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4"/>')

        # True trace
        pts_true = [f"{x_s(i):.1f},{y_s(y_true_s[i]):.1f}" for i in sub_idx]
        svg.append(f'<polyline points="{" ".join(pts_true)}" fill="none" stroke="#f8fafc" stroke-width="1.8"/>')

        # Pred trace
        pts_pred = [f"{x_s(i):.1f},{y_s(y_pred_s[i]):.1f}" for i in sub_idx]
        svg.append(f'<polyline points="{" ".join(pts_pred)}" fill="none" stroke="#10b981" stroke-width="1.5"/>')

    # Panel 1: Median Unrestrained Target
    title1 = f"Unrestrained Target: {med_target['name_i']} - {med_target['name_j']} (Ref: {med_target['ref_dist_nm']:.2f} nm, Median Error)"
    plot_subtrace(Y_true[:, med_target_idx], Y_pred_rob[:, med_target_idx], Y_pred_mean[:, med_target_idx], title1, pad_t + 10)

    # Panel 2: Restrained Target
    title2 = f"Directly Restrained Target: {rest_target['name_i']} - {rest_target['name_j']} (Ref: {rest_target['ref_dist_nm']:.2f} nm, Elastic Network Bound)"
    plot_subtrace(Y_true[:, rest_target_idx], Y_pred_rob[:, rest_target_idx], Y_pred_mean[:, rest_target_idx], title2, pad_t + ph + 35)

    # Legend
    leg_y = height - 12
    svg.append(f'<line x1="{pad_l+50}" y1="{leg_y}" x2="{pad_l+75}" y2="{leg_y}" stroke="#f8fafc" stroke-width="2"/>')
    svg.append(f'<text x="{pad_l+82}" y="{leg_y+4}" fill="#f8fafc" font-size="10">Actual MD Trajectory (nm)</text>')
    
    svg.append(f'<line x1="{pad_l+250}" y1="{leg_y}" x2="{pad_l+275}" y2="{leg_y}" stroke="#10b981" stroke-width="2"/>')
    svg.append(f'<text x="{pad_l+282}" y="{leg_y+4}" fill="#10b981" font-size="10">Reconstructed (M8 Robust k=10)</text>')
    
    svg.append(f'<line x1="{pad_l+480}" y1="{leg_y}" x2="{pad_l+505}" y2="{leg_y}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4"/>')
    svg.append(f'<text x="{pad_l+512}" y="{leg_y+4}" fill="#94a3b8" font-size="10">Training Mean Baseline</text>')

    svg.append('</svg>')
    return "\n".join(svg)

def generate_svg_restrained_breakdown(results):
    """Figure 4: Bar chart comparing reconstruction error on restrained vs unrestrained targets."""
    # Compare M1 Mean vs M8 Robust across folds
    folds = ["fold_1", "fold_2", "fold_3"]
    rest_m1 = [results["folds"][f]["methods"]["m1_mean"]["k_eval"]["constant"]["rmse_restrained_nm"] for f in folds]
    rest_m8 = [results["folds"][f]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_restrained_nm"] for f in folds]
    
    unrest_m1 = [results["folds"][f]["methods"]["m1_mean"]["k_eval"]["constant"]["rmse_unrestrained_nm"] for f in folds]
    unrest_m8 = [results["folds"][f]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_unrestrained_nm"] for f in folds]

    width, height = 750, 360
    pad_l, pad_r, pad_t, pad_b = 80, 180, 40, 50
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:8px; font-family:sans-serif;">']
    svg.append(f'<text x="{width/2}" y="25" fill="#f8fafc" font-size="14" font-weight="bold" text-anchor="middle">Figure 4: Reconstruction Error Breakdown: Directly Restrained vs Unrestrained Targets</text>')

    # Groups: Fold 1, Fold 2, Fold 3, Mean
    groups = [
        ("Fold 1 (rep3)", rest_m1[0], rest_m8[0], unrest_m1[0], unrest_m8[0]),
        ("Fold 2 (rep2)", rest_m1[1], rest_m8[1], unrest_m1[1], unrest_m8[1]),
        ("Fold 3 (rep1)", rest_m1[2], rest_m8[2], unrest_m1[2], unrest_m8[2]),
        ("Pooled Mean", np.mean(rest_m1), np.mean(rest_m8), np.mean(unrest_m1), np.mean(unrest_m8))
    ]

    y_max = 0.095
    def y_s(v): return pad_t + (1.0 - v / y_max) * ph

    # Y-axis ticks
    for y_v in [0.02, 0.04, 0.06, 0.08]:
        yp = y_s(y_v)
        svg.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{pad_l+pw}" y2="{yp:.1f}" stroke="#334155" stroke-width="1" stroke-dasharray="2,2"/>')
        svg.append(f'<text x="{pad_l-8}" y="{yp+4:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">{y_v:.2f} nm</text>')

    group_w = pw / len(groups)
    bar_w = group_w * 0.18

    for g_idx, (g_name, r_m1, r_m8, u_m1, u_m8) in enumerate(groups):
        gx = pad_l + g_idx * group_w
        svg.append(f'<text x="{gx + group_w/2:.1f}" y="{height-pad_b+20}" fill="#cbd5e1" font-size="11" font-weight="bold" text-anchor="middle">{g_name}</text>')
        
        # Bars: 
        # 1. Restrained M1
        bx1 = gx + 0.10 * group_w
        by1 = y_s(r_m1)
        svg.append(f'<rect x="{bx1:.1f}" y="{by1:.1f}" width="{bar_w:.1f}" height="{pad_t+ph-by1:.1f}" fill="#64748b" rx="2"/>')
        
        # 2. Restrained M8
        bx2 = bx1 + bar_w + 4
        by2 = y_s(r_m8)
        svg.append(f'<rect x="{bx2:.1f}" y="{by2:.1f}" width="{bar_w:.1f}" height="{pad_t+ph-by2:.1f}" fill="#38bdf8" rx="2"/>')
        
        # 3. Unrestrained M1
        bx3 = bx2 + bar_w + 12
        by3 = y_s(u_m1)
        svg.append(f'<rect x="{bx3:.1f}" y="{by3:.1f}" width="{bar_w:.1f}" height="{pad_t+ph-by3:.1f}" fill="#94a3b8" rx="2"/>')
        
        # 4. Unrestrained M8
        bx4 = bx3 + bar_w + 4
        by4 = y_s(u_m8)
        svg.append(f'<rect x="{bx4:.1f}" y="{by4:.1f}" width="{bar_w:.1f}" height="{pad_t+ph-by4:.1f}" fill="#10b981" rx="2"/>')

    # Legend
    leg_x = pad_l + pw + 15
    leg_items = [
        ("#64748b", "Restrained: Mean Baseline"),
        ("#38bdf8", "Restrained: M8 Robust"),
        ("#94a3b8", "Unrestrained: Mean Baseline"),
        ("#10b981", "Unrestrained: M8 Robust")
    ]
    for i, (col, lbl) in enumerate(leg_items):
        ly = pad_t + 20 + i * 26
        svg.append(f'<rect x="{leg_x}" y="{ly}" width="14" height="14" fill="{col}" rx="2"/>')
        svg.append(f'<text x="{leg_x+22}" y="{ly+11}" fill="#e2e8f0" font-size="10">{lbl}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

def main():
    print("[INFO] Generating consolidated figures and metrics...")
    with open(EVID_DIR / "benchmark_results.json", "r", encoding="utf-8") as f:
        bench = json.load(f)
    with open(EVID_DIR / "pair_definitions.json", "r", encoding="utf-8") as f:
        pdefs = json.load(f)
    f1_preds = np.load(EVID_DIR / "test_predictions_fold_1.npz")
    
    svg1 = generate_svg_rmse_curve(bench)
    svg2 = generate_svg_forest_plot(bench)
    svg3 = generate_svg_trace_plot(f1_preds, pdefs)
    svg4 = generate_svg_restrained_breakdown(bench)
    
    # Save SVGs to files
    (EVID_DIR / "figure1_rmse_curve.svg").write_text(svg1, encoding="utf-8")
    (EVID_DIR / "figure2_forest_plot.svg").write_text(svg2, encoding="utf-8")
    (EVID_DIR / "figure3_trace_plot.svg").write_text(svg3, encoding="utf-8")
    (EVID_DIR / "figure4_restrained_breakdown.svg").write_text(svg4, encoding="utf-8")
    
    # Compute summary metrics table
    methods = [
        ("m1_mean", "1. Training-Mean Baseline"),
        ("m2_random", "2. Random Selection (20 Seeds)"),
        ("m3_variance", "3. Highest Variance Selection"),
        ("m4_pca_qr", "4. PCA / Pivoted-QR Selection"),
        ("m5_info_imbalance", "5. Information Imbalance (DII)"),
        ("m6_pooled_greedy", "6. Pooled Greedy Selector"),
        ("m7_mean_transfer", "7. Matched Mean-Transfer Selector"),
        ("m8_robust_transfer", "8. Proposed Robust-Transfer Selector"),
        ("m9_graph_assisted", "9. Graph-Assisted Robust Selector")
    ]
    
    summary_rows = []
    for mid, mname in methods:
        row = {"method_id": mid, "name": mname, "k_results": {}}
        for k in [5, 10, 20]:
            f_rmses = []
            f_skills = []
            for fid in ["fold_1", "fold_2", "fold_3"]:
                m_rec = bench["folds"][fid]["methods"][mid]
                if mid == "m1_mean":
                    f_rmses.append(m_rec["k_eval"]["constant"]["rmse_nm"])
                    f_skills.append(m_rec["k_eval"]["constant"]["skill"])
                elif mid == "m2_random":
                    f_rmses.append(m_rec["k_eval"][str(k)]["mean_rmse_nm"])
                    f_skills.append(m_rec["k_eval"][str(k)]["skill"])
                else:
                    f_rmses.append(m_rec["k_eval"][str(k)]["rmse_nm"])
                    f_skills.append(m_rec["k_eval"][str(k)]["skill"])
            row["k_results"][str(k)] = {
                "fold_rmses": f_rmses,
                "mean_rmse_nm": float(np.mean(f_rmses)),
                "std_rmse_nm": float(np.std(f_rmses)),
                "fold_skills": f_skills,
                "mean_skill": float(np.mean(f_skills))
            }
        summary_rows.append(row)
        
    # Analysis of Selected Features Overlap at k=10
    selected_features_by_fold = {}
    for fid in ["fold_1", "fold_2", "fold_3"]:
        selected_features_by_fold[fid] = {
            "m8_robust": bench["folds"][fid]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["selected_features"],
            "m7_mean": bench["folds"][fid]["methods"]["m7_mean_transfer"]["k_eval"]["10"]["selected_features"],
            "m5_ii": bench["folds"][fid]["methods"]["m5_info_imbalance"]["k_eval"]["10"]["selected_features"],
        }
        
    rob_f1 = set(selected_features_by_fold["fold_1"]["m8_robust"])
    rob_f2 = set(selected_features_by_fold["fold_2"]["m8_robust"])
    rob_f3 = set(selected_features_by_fold["fold_3"]["m8_robust"])
    
    overlap_12 = len(rob_f1.intersection(rob_f2))
    overlap_13 = len(rob_f1.intersection(rob_f3))
    overlap_23 = len(rob_f2.intersection(rob_f3))
    overlap_all = len(rob_f1.intersection(rob_f2).intersection(rob_f3))
    
    # Save pilot summary metrics
    pilot_metrics = {
        "summary_table": summary_rows,
        "primary_k10_comparison": {
            "m8_robust_mean_rmse": float(np.mean([bench["folds"][f]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "m7_mean_transfer_rmse": float(np.mean([bench["folds"][f]["methods"]["m7_mean_transfer"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "m6_pooled_rmse": float(np.mean([bench["folds"][f]["methods"]["m6_pooled_greedy"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "m5_info_imbalance_rmse": float(np.mean([bench["folds"][f]["methods"]["m5_info_imbalance"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "m1_mean_baseline_rmse": float(np.mean([bench["folds"][f]["methods"]["m1_mean"]["k_eval"]["constant"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "m9_graph_rmse": float(np.mean([bench["folds"][f]["methods"]["m9_graph_assisted"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]]))
        },
        "robust_vs_comparator_diffs_k10": {
            "diff_vs_mean_transfer": float(np.mean([bench["folds"][f]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_nm"] - bench["folds"][f]["methods"]["m7_mean_transfer"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "diff_vs_pooled": float(np.mean([bench["folds"][f]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_nm"] - bench["folds"][f]["methods"]["m6_pooled_greedy"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "diff_vs_info_imbalance": float(np.mean([bench["folds"][f]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_nm"] - bench["folds"][f]["methods"]["m5_info_imbalance"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "diff_vs_mean_baseline": float(np.mean([bench["folds"][f]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_nm"] - bench["folds"][f]["methods"]["m1_mean"]["k_eval"]["constant"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "graph_gain_vs_robust": float(np.mean([bench["folds"][f]["methods"]["m9_graph_assisted"]["k_eval"]["10"]["rmse_nm"] - bench["folds"][f]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]]))
        },
        "negative_control_k10": {
            "temporal_shift_mean_rmse": float(np.mean([bench["folds"][f]["negative_control_temporal_shift_k10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "temporal_shift_mean_skill": float(np.mean([bench["folds"][f]["negative_control_temporal_shift_k10"]["skill"] for f in ["fold_1", "fold_2", "fold_3"]]))
        },
        "sensitivity_20pct_k10": {
            "mean_rmse": float(np.mean([bench["folds"][f]["sensitivity_20pct_discard_k10"]["rmse_nm"] for f in ["fold_1", "fold_2", "fold_3"]])),
            "mean_skill": float(np.mean([bench["folds"][f]["sensitivity_20pct_discard_k10"]["skill"] for f in ["fold_1", "fold_2", "fold_3"]]))
        },
        "feature_overlap_k10": {
            "overlap_fold1_fold2": overlap_12,
            "overlap_fold1_fold3": overlap_13,
            "overlap_fold2_fold3": overlap_23,
            "overlap_all_three_folds": overlap_all
        }
    }
    
    with open(EVID_DIR / "pilot_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(pilot_metrics, f, indent=2)
        
    print(f"[OK] Figures and summary metrics successfully generated.")
    print(f"     Metrics saved: {EVID_DIR / 'pilot_summary_metrics.json'}")

if __name__ == "__main__":
    main()
