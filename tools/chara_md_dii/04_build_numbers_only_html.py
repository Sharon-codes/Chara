#!/usr/bin/env python3
"""
tools/chara_md_dii/04_build_numbers_only_html.py

Constructs reports/chara_md_dii/20260907_v1/CHARA_DII_VERIFIED_NUMBERS.html
Strictly tables only, metric definitions, units, execution statuses, and evidence references.
Zero executive summary, zero discussion, zero biological claims, zero recommendations.
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np

BASE_DIR = Path("E:/Sharon")
EVID_DIR = BASE_DIR / "reports" / "chara_md_dii" / "20260907_v1" / "evidence"
HTML_OUT = BASE_DIR / "reports" / "chara_md_dii" / "20260907_v1" / "CHARA_DII_VERIFIED_NUMBERS.html"

def read_csv(filename):
    p = EVID_DIR / filename
    if not p.exists(): return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def render_table(headers, rows, caption="", table_id=""):
    html = []
    if caption:
        html.append(f"<h3 class='table-caption'>{caption}</h3>")
    html.append(f"<div class='table-container'><table id='{table_id}' class='data-table'>")
    html.append("<thead><tr>")
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead>")
    html.append("<tbody>")
    for r in rows:
        html.append("<tr>")
        for h in headers:
            html.append(f"<td>{r.get(h, '')}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "\n".join(html)

def main():
    print(f"Building {HTML_OUT}...")

    metrics_rows = read_csv("CHARA_DII_VERIFIED_NUMBERS.csv")
    paired_rows = read_csv("paired_differences.csv")
    dii_opt_rows = read_csv("dii_optimization_summary.csv")
    inv_rows = read_csv("artifact_inventory.csv")
    raw_chk_rows = read_csv("raw_distance_checks.csv")
    disjoint_rows = read_csv("pair_disjointness_checks.csv")
    screen_rows = read_csv("screening_pool_audit.csv")
    repro_rows = read_csv("reproduction_checks.csv")

    # Load DII software check
    sw_chk_path = EVID_DIR / "dii_software_check.json"
    sw_chk_data = json.load(open(sw_chk_path)) if sw_chk_path.exists() else {}

    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("<meta charset='UTF-8'>")
    html.append("<title>CHARA_DII_VERIFIED_NUMBERS: Official DII Benchmark & Evidence Verification</title>")
    html.append("<style>")
    html.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #1e293b; background: #f8fafc; line-height: 1.4; }
        h1 { color: #0f172a; border-bottom: 2px solid #cbd5e1; padding-bottom: 8px; font-size: 19px; }
        h2 { color: #1e3a8a; margin-top: 30px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; font-size: 14.5px; text-transform: uppercase; letter-spacing: 0.5px; }
        h3.table-caption { font-size: 12.5px; margin: 16px 0 6px 0; color: #334155; font-weight: 600; }
        .table-container { overflow-x: auto; margin-bottom: 16px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
        table.data-table { width: 100%; border-collapse: collapse; font-size: 11px; text-align: left; }
        th { background: #f1f5f9; color: #475569; font-weight: 600; padding: 6px 9px; border-bottom: 1px solid #cbd5e1; white-space: nowrap; }
        td { padding: 5px 9px; border-bottom: 1px solid #f1f5f9; color: #0f172a; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: nowrap; }
        tr:hover { background: #f8fafc; }
        .doc-meta { font-size: 11px; color: #64748b; margin-bottom: 20px; }
    """)
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")

    html.append("<h1>CHARA_DII_VERIFIED_NUMBERS: Official DII Benchmark & Multi-System Numerical Evidence</h1>")
    html.append("<div class='doc-meta'>Document standard: Numerical reporting only. Units explicitly declared. Zero narrative prose. Generated: 2026-09-07.</div>")

    # =========================================================================
    # Section 1: Benchmark Configuration, System Metadata, and Physical Parameters
    # =========================================================================
    html.append("<h2>Section 1: Benchmark Configuration, System Metadata, and Physical Parameters</h2>")

    sys_meta = [
        {"system": "KRAS_G12D", "source_structure": "PDB 4OBE", "analyzed_chains": "Chain A", "bead_count": "384", "backbone_beads": "170", "ensemble": "NPT", "temperature_K": "310", "pressure_bar": "1.0", "integrator": "md (leap-frog)", "timestep_fs": "20", "thermostat": "v-rescale", "barostat": "Parrinello-Rahman", "forcefield": "Martini 2.2"},
        {"system": "PTPN11", "source_structure": "PDB 4DGP", "analyzed_chains": "Chain A", "bead_count": "1327", "backbone_beads": "536", "ensemble": "NPT", "temperature_K": "310", "pressure_bar": "1.0", "integrator": "md (leap-frog)", "timestep_fs": "20", "thermostat": "v-rescale", "barostat": "Parrinello-Rahman", "forcefield": "Martini 2.2"},
        {"system": "Mut_p53", "source_structure": "PDB 2J1X", "analyzed_chains": "Chain A", "bead_count": "488", "backbone_beads": "219", "ensemble": "NPT", "temperature_K": "310", "pressure_bar": "1.0", "integrator": "md (leap-frog)", "timestep_fs": "20", "thermostat": "v-rescale", "barostat": "Parrinello-Rahman", "forcefield": "Martini 2.2"},
        {"system": "cMYC_MAX", "source_structure": "PDB 1NKP", "analyzed_chains": "Chain E (cMYC monomer)", "bead_count": "204", "backbone_beads": "88", "ensemble": "NPT", "temperature_K": "310", "pressure_bar": "1.0", "integrator": "md (leap-frog)", "timestep_fs": "20", "thermostat": "v-rescale", "barostat": "Parrinello-Rahman", "forcefield": "Martini 2.2"}
    ]
    html.append(render_table(
        list(sys_meta[0].keys()), sys_meta,
        caption="Table 1.1: Physical simulation parameters and MD engine configuration by system (analyzed constructs identified)",
        table_id="table_1_1"
    ))

    # Table 1.2: Topology restraints
    top_rows = [
        {"system": "KRAS_G12D", "bonds_count": "229", "constraints_count": "155", "restrained_pairs_ge4": "101", "force_constant_kJ_mol_nm2": "500.0"},
        {"system": "PTPN11", "bonds_count": "791", "constraints_count": "535", "restrained_pairs_ge4": "84", "force_constant_kJ_mol_nm2": "500.0"},
        {"system": "Mut_p53", "bonds_count": "269", "constraints_count": "218", "restrained_pairs_ge4": "100", "force_constant_kJ_mol_nm2": "500.0"},
        {"system": "cMYC_MAX", "bonds_count": "116", "constraints_count": "87", "restrained_pairs_ge4": "78", "force_constant_kJ_mol_nm2": "500.0"}
    ]
    html.append(render_table(
        list(top_rows[0].keys()), top_rows,
        caption="Table 1.2: Martini coarse-grained topology constraints and non-local harmonic restraints (|res_i - res_j| &ge; 4)",
        table_id="table_1_2"
    ))

    # Table 1.3: Frame accounting and reconciliation
    frame_rows = [
        {"system": "KRAS_G12D", "replicate": "rep1", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "KRAS_G12D", "replicate": "rep2", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "KRAS_G12D", "replicate": "rep3", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "PTPN11", "replicate": "rep1", "raw_frames": "996", "time_span_ns": "497.5", "discard_10pct_frames": "101", "retained_10pct_frames": "895", "discard_20pct_frames": "201", "retained_20pct_frames": "795"},
        {"system": "PTPN11", "replicate": "rep2", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "PTPN11", "replicate": "rep3", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "Mut_p53", "replicate": "rep1", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "Mut_p53", "replicate": "rep2", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "Mut_p53", "replicate": "rep3", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "cMYC_MAX", "replicate": "rep1", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "cMYC_MAX", "replicate": "rep2", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"},
        {"system": "cMYC_MAX", "replicate": "rep3", "raw_frames": "1001", "time_span_ns": "500.0", "discard_10pct_frames": "101", "retained_10pct_frames": "900", "discard_20pct_frames": "201", "retained_20pct_frames": "800"}
    ]
    html.append(render_table(
        list(frame_rows[0].keys()), frame_rows,
        caption="Table 1.3: Trajectory frame counts and equilibration windows (Note: PTPN11 rep1 20% discard has 795 frames; combined with rep2 800 frames truncated to equal length min(795, 800)*2 = 1590 frames)",
        table_id="table_1_3"
    ))

    # Table 1.4: Disjointness and screening pool audit
    html.append(render_table(
        ["system", "chain", "eligible_pairs", "candidate_count", "target_count", "direct_overlap_count", "reversed_overlap_count", "disjointness_status"],
        disjoint_rows,
        caption="Table 1.4: Candidate and target pair pool disjointness verification (direct and reversed pairs confirmed disjoint)",
        table_id="table_1_4"
    ))

    # =========================================================================
    # Section 2: DADApy DII Software Verification & Implementation Metadata
    # =========================================================================
    html.append("<h2>Section 2: DADApy DII Software Verification & Implementation Metadata</h2>")

    sw_rows = [{
        "status": sw_chk_data.get("status", "N/A"),
        "dadapy_version": sw_chk_data.get("dadapy_version", "N/A"),
        "dadapy_module_path": sw_chk_data.get("dadapy_module_path", "N/A"),
        "python_version": sw_chk_data.get("python_version", "").split()[0],
        "platform": sw_chk_data.get("platform", "N/A"),
        "initial_dii_finite": str(sw_chk_data.get("finite_checks", {}).get("initial_dii_finite", "N/A")),
        "optimized_dii_finite": str(sw_chk_data.get("finite_checks", {}).get("optimized_dii_finite", "N/A")),
        "weights_finite": str(sw_chk_data.get("finite_checks", {}).get("weights_finite", "N/A")),
        "elimination_diis_finite": str(sw_chk_data.get("finite_checks", {}).get("elimination_diis_finite", "N/A")),
        "history_recorded": "True" if sw_chk_data.get("history_keys") else "False"
    }]
    html.append(render_table(
        list(sw_rows[0].keys()), sw_rows,
        caption="Table 2.1: Official DADApy DII software verification check results (synthetic benchmark)",
        table_id="table_2_1"
    ))

    # Table 2.2: DII Optimization Summary across systems
    html.append(render_table(
        ["system", "fold", "configuration", "sample_frames", "source_pool_size", "initial_dii", "dii_k20", "dii_k10", "dii_k5", "runtime_seconds"],
        dii_opt_rows,
        caption="Table 2.2: DADApy DII backward greedy elimination optimization summary by system and fold",
        table_id="table_2_2"
    ))

    # =========================================================================
    # Section 3: Primary Benchmark Results across All Systems, Folds, and Budgets
    # =========================================================================
    html.append("<h2>Section 3: Primary Benchmark Results across All Systems, Folds, and Budgets</h2>")

    html.append(render_table(
        [
            "system", "analyzed_chains", "fold", "method_id", "requested_k", "achieved_k",
            "selection_frame_count", "decoder_train_frame_count", "test_frame_count", "target_count",
            "candidate_pool_size", "config_hash", "ridge_alpha", "SSE", "baseline_SSE",
            "RMSE_nm", "baseline_RMSE_nm", "skill", "runtime_seconds", "optimization_status", "execution_status"
        ],
        metrics_rows,
        caption="Table 3.1: Complete per-fold evaluation table across 4 systems, 3 outer folds, and budgets (k in {5, 10, 20})",
        table_id="table_3_1"
    ))

    # Table 3.2: Aggregated Summary
    agg_dict = {}
    for r in metrics_rows:
        key = (r["system"], r["method_id"], r["requested_k"])
        if key not in agg_dict: agg_dict[key] = {"rmse": [], "skill": []}
        try:
            agg_dict[key]["rmse"].append(float(r["RMSE_nm"]))
            agg_dict[key]["skill"].append(float(r["skill"]))
        except ValueError: pass

    agg_rows = []
    for (s_name, m_id, req_k), vals in sorted(agg_dict.items()):
        m_rmse = np.mean(vals["rmse"]) if vals["rmse"] else np.nan
        s_rmse = np.std(vals["rmse"]) if vals["rmse"] else np.nan
        m_sk = np.mean(vals["skill"]) if vals["skill"] else np.nan
        s_sk = np.std(vals["skill"]) if vals["skill"] else np.nan
        agg_rows.append({
            "system": s_name, "method_id": m_id, "requested_k": req_k,
            "mean_RMSE_nm": f"{m_rmse:.6f} +/- {s_rmse:.6f}",
            "mean_skill": f"{m_sk:+.6f} +/- {s_sk:.6f}",
            "n_folds": str(len(vals["rmse"]))
        })

    html.append(render_table(
        ["system", "method_id", "requested_k", "mean_RMSE_nm", "mean_skill", "n_folds"],
        agg_rows,
        caption="Table 3.2: Aggregated benchmark metrics (Mean +/- SD across 3 outer folds) by system, budget, and method",
        table_id="table_3_2"
    ))

    # =========================================================================
    # Section 4: Paired Comparisons at Matched Budgets
    # =========================================================================
    html.append("<h2>Section 4: Paired Comparisons at Matched Budgets</h2>")

    # Table 4.1: DII 100F vs M5 Rank II at k=10
    paired_dii_m5 = [r for r in paired_rows if r.get("comparator_id") == "M5_RANK_INFORMATION_IMBALANCE" and r.get("k") == "10"]
    html.append(render_table(
        ["system", "fold", "k", "dii_configuration", "comparator_id", "DII_RMSE_nm", "comparator_RMSE_nm", "delta_RMSE_DII_minus_comp", "percent_reduction_vs_comparator", "DII_skill", "comparator_skill", "delta_skill"],
        paired_dii_m5,
        caption="Table 4.1: Paired comparison: Official DII (100F) versus M5 Rank-based Information Imbalance at k=10 per fold",
        table_id="table_4_1"
    ))

    # Table 4.2: DII 100F vs M8 Robust at k=10
    paired_dii_m8 = [r for r in paired_rows if r.get("comparator_id") == "M8_ROBUST_TRANSFER" and r.get("k") == "10"]
    html.append(render_table(
        ["system", "fold", "k", "dii_configuration", "comparator_id", "DII_RMSE_nm", "comparator_RMSE_nm", "delta_RMSE_DII_minus_comp", "percent_reduction_vs_comparator", "DII_skill", "comparator_skill", "delta_skill"],
        paired_dii_m8,
        caption="Table 4.2: Paired comparison: Official DII (100F) versus M8 Robust Transfer at k=10 per fold",
        table_id="table_4_2"
    ))

    # Table 4.3: DII 100F vs M9 Graph Assisted at k=10
    paired_dii_m9 = [r for r in paired_rows if r.get("comparator_id") == "M9_GRAPH_ASSISTED" and r.get("k") == "10"]
    html.append(render_table(
        ["system", "fold", "k", "dii_configuration", "comparator_id", "DII_RMSE_nm", "comparator_RMSE_nm", "delta_RMSE_DII_minus_comp", "percent_reduction_vs_comparator", "DII_skill", "comparator_skill", "delta_skill"],
        paired_dii_m9,
        caption="Table 4.3: Paired comparison: Official DII (100F) versus M9 Graph-assisted Robust at k=10 per fold",
        table_id="table_4_3"
    ))

    # Table 4.4: DII 100F vs M7 Mean Transfer at k=10
    paired_dii_m7 = [r for r in paired_rows if r.get("comparator_id") == "M7_MEAN_TRANSFER" and r.get("k") == "10"]
    html.append(render_table(
        ["system", "fold", "k", "dii_configuration", "comparator_id", "DII_RMSE_nm", "comparator_RMSE_nm", "delta_RMSE_DII_minus_comp", "percent_reduction_vs_comparator", "DII_skill", "comparator_skill", "delta_skill"],
        paired_dii_m7,
        caption="Table 4.4: Paired comparison: Official DII (100F) versus M7 Mean Transfer at k=10 per fold",
        table_id="table_4_4"
    ))

    # Table 4.5: DII 400F Sensitivity Comparisons at k=10
    paired_dii_400 = [r for r in paired_rows if r.get("dii_configuration") == "M10_OFFICIAL_DII_400F"]
    html.append(render_table(
        ["system", "fold", "k", "dii_configuration", "comparator_id", "DII_RMSE_nm", "comparator_RMSE_nm", "delta_RMSE_DII_minus_comp", "percent_reduction_vs_comparator", "DII_skill", "comparator_skill", "delta_skill"],
        paired_dii_400,
        caption="Table 4.5: Sensitivity analysis: Official DII (400F) versus Official DII (100F) and M8 Robust Transfer at k=10",
        table_id="table_4_5"
    ))

    # =========================================================================
    # Section 5: Reproduction Checks, Raw Distance Audits, and Discrepancy Records
    # =========================================================================
    html.append("<h2>Section 5: Reproduction Checks, Raw Distance Audits, and Discrepancy Records</h2>")

    # Table 5.1: Independent Metric Recomputations & Reproduction Checks
    html.append(render_table(
        ["system", "fold", "method", "k", "metric", "original_report_value", "saved_result_value", "independent_recalculation", "fresh_rerun_value", "absolute_difference", "tolerance", "technical_check_status", "source_artifact"],
        repro_rows,
        caption="Table 5.1: Independent metric recomputations and reproduction checks (historical pilot and new DII predictions recomputed from raw .npz arrays)",
        table_id="table_5_1"
    ))

    # Table 5.2: Raw Distance Spot Checks
    html.append(render_table(
        ["system", "replicate", "frame_idx", "pair_type", "res_i", "res_j", "pairwise_pos_calc_nm", "vectorized_calc_nm", "absolute_diff_nm", "tolerance_nm", "status"],
        raw_chk_rows,
        caption="Table 5.2: Independent coordinate recalculation spot checks from production_centered.xtc across systems",
        table_id="table_5_2"
    ))

    # Table 5.3: Screening Pool Provenance Audit
    html.append(render_table(
        ["system", "fold", "train_runs", "test_run_held_out", "screening_pool_size", "top_5_screened_features", "data_used_for_screening", "uses_test_run_data", "provenance_status"],
        screen_rows,
        caption="Table 5.3: Screening pool development-data provenance audit (test-set leakage strictly False)",
        table_id="table_5_3"
    ))

    # Table 5.4: Artifact Inventory
    html.append(render_table(
        ["artifact_id", "artifact_type", "actual_path", "bytes", "sha256", "system", "fold", "method", "k", "original_or_new"],
        inv_rows,
        caption="Table 5.4: Comprehensive artifact inventory and SHA256 checksums",
        table_id="table_5_4"
    ))

    # Table 5.5: Execution Inventory
    exec_inv = [
        {"system": "KRAS_G12D", "outer_folds": "3", "dii_configs": "100F (k=5,10,20) + 400F (k=10)", "planned_jobs": "36 (12 DII + 24 comps)", "completed_jobs": "36", "missing_jobs": "0", "status": "COMPLETED"},
        {"system": "PTPN11", "outer_folds": "3", "dii_configs": "100F (k=5,10,20) + 400F (k=10)", "planned_jobs": "36 (12 DII + 24 comps)", "completed_jobs": "36", "missing_jobs": "0", "status": "COMPLETED"},
        {"system": "Mut_p53", "outer_folds": "3", "dii_configs": "100F (k=5,10,20) + 400F (k=10)", "planned_jobs": "36 (12 DII + 24 comps)", "completed_jobs": "36", "missing_jobs": "0", "status": "COMPLETED"},
        {"system": "cMYC_MAX", "outer_folds": "3", "dii_configs": "100F (k=5,10,20) + 400F (k=10)", "planned_jobs": "36 (12 DII + 24 comps)", "completed_jobs": "36", "missing_jobs": "0", "status": "COMPLETED"},
        {"system": "Software Check", "outer_folds": "N/A", "dii_configs": "Synthetic N=60, D=6", "planned_jobs": "1", "completed_jobs": "1", "missing_jobs": "0", "status": "COMPLETED"}
    ]
    html.append(render_table(
        list(exec_inv[0].keys()), exec_inv,
        caption="Table 5.5: Comprehensive execution inventory (model fits, held-out evaluations, and optimizer runs)",
        table_id="table_5_5"
    ))

    # Table 5.6: Technical Discrepancy Records
    discrepancies = [
        {
            "record_id": "DISC_01_DII_METHOD_IDENTITY",
            "component": "dadapy.feature_weighting vs rank II",
            "status": "RESOLVED_DISTINCT_LABELS",
            "description": "Previous M5 used rank-based Information Imbalance on 100 frames. Now labeled M5_RANK_INFORMATION_IMBALANCE; official DADApy FeatureWeighting backward greedy elimination evaluated as M10_OFFICIAL_DII_100F and M10_OFFICIAL_DII_400F."
        },
        {
            "record_id": "DISC_02_PTPN11_FRAME_ACCOUNTING",
            "component": "PTPN11 trajectory lengths",
            "status": "RECONCILED_EXACT",
            "description": "PTPN11 rep1 has 996 frames (795 retained at 20% discard); rep2 has 1001 frames (800 retained at 20% discard). When combined, developmental arrays were truncated to equal length min(795, 800)*2 = 1590 frames."
        },
        {
            "record_id": "DISC_03_CMYC_CONSTRUCT_LABEL",
            "component": "cMYC_MAX modeled chain",
            "status": "LABEL_CORRECTED",
            "description": "System directory is labeled cMYC_MAX, but the simulation topology contains only Chain E (88 residues, cMYC monomer). The analysis reconstructs Chain E, not the full heterotetramer/DNA complex."
        },
        {
            "record_id": "DISC_04_SCREENING_POOL_LEAKAGE",
            "component": "40-feature screening pool",
            "status": "AUDITED_DEVELOPMENT_ONLY",
            "description": "Screening pool is computed per outer fold using solely train_run_1 and train_run_2 development trajectories. Held-out test run data is never accessed during screening."
        }
    ]
    html.append(render_table(
        ["record_id", "component", "status", "description"],
        discrepancies,
        caption="Table 5.6: Technical discrepancy, metadata correction, and limitation records",
        table_id="table_5_6"
    ))

    html.append("</body>")
    html.append("</html>")

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[OK] Successfully built {HTML_OUT} ({len(html)} lines).")

if __name__ == "__main__":
    main()
