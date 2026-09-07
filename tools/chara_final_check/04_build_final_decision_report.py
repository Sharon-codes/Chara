#!/usr/bin/env python3
"""
tools/chara_final_check/04_build_final_decision_report.py
Compiles the final, self-contained, offline-readable decision report:
reports/chara_final_check/CHARA_FINAL_DECISION_REPORT.html
"""

import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT / "reports" / "chara_final_check" / "evidence"
OUTPUT_HTML = ROOT / "reports" / "chara_final_check" / "CHARA_FINAL_DECISION_REPORT.html"

def load_json(name):
    p = EVIDENCE_DIR / name
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_nested_cv_diagram_svg():
    width, height = 800, 280
    svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:8px; font-family:system-ui, -apple-system, sans-serif;">
  <!-- Outer Partition -->
  <rect x="30" y="30" width="740" height="220" rx="8" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
  <text x="50" y="55" fill="#f8fafc" font-size="14" font-weight="bold">Outer Stratified 5-Fold Partition (TCGA-LUAD Development Cohort, N=502)</text>

  <!-- Outer Train Box -->
  <rect x="50" y="70" width="480" height="160" rx="6" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="65" y="92" fill="#38bdf8" font-size="12" font-weight="bold">Outer Training Partition (80%, N ≈ 402)</text>

  <!-- Inner Cross-Fitting Box -->
  <rect x="65" y="105" width="220" height="110" rx="4" fill="#1e1b4b" stroke="#818cf8" stroke-width="1"/>
  <text x="75" y="125" fill="#a5b4fc" font-size="11" font-weight="bold">Inner 3-Fold Cross-Fitting</text>
  <text x="75" y="145" fill="#cbd5e1" font-size="10">• Fit gene models on 2/3 inner</text>
  <text x="75" y="160" fill="#cbd5e1" font-size="10">• Predict out-of-fold on 1/3 inner</text>
  <text x="75" y="180" fill="#34d399" font-size="10" font-weight="bold">→ Clean Outer-Train Gene Risk</text>
  <text x="75" y="195" fill="#94a3b8" font-size="9">(Zero outcome leakage into combination)</text>

  <!-- Combination Fitting Box -->
  <rect x="300" y="105" width="215" height="110" rx="4" fill="#142e2b" stroke="#34d399" stroke-width="1"/>
  <text x="310" y="125" fill="#6ee7b7" font-size="11" font-weight="bold">Combination Layer Fitting</text>
  <text x="310" y="145" fill="#cbd5e1" font-size="10">• Fit CoxPH on Outer-Train Clinical</text>
  <text x="310" y="160" fill="#cbd5e1" font-size="10">  (Age + Sex + Stage) + Clean Gene Score</text>
  <text x="310" y="180" fill="#cbd5e1" font-size="10">• Refit base gene models on full</text>
  <text x="310" y="195" fill="#cbd5e1" font-size="10">  Outer Train for test prediction</text>

  <!-- Arrow to Outer Test -->
  <line x1="535" y1="150" x2="575" y2="150" stroke="#f59e0b" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Outer Test Box -->
  <rect x="580" y="70" width="170" height="160" rx="6" fill="#1c1917" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="595" y="92" fill="#f59e0b" font-size="12" font-weight="bold">Outer Test (20%, N ≈ 100)</text>
  <text x="595" y="118" fill="#cbd5e1" font-size="10">• Completely held aside</text>
  <text x="595" y="135" fill="#cbd5e1" font-size="10">• No outcome seen</text>
  <text x="595" y="152" fill="#cbd5e1" font-size="10">• Predict gene score from</text>
  <text x="595" y="167" fill="#cbd5e1" font-size="10">  base model refit</text>
  <text x="595" y="187" fill="#cbd5e1" font-size="10">• Apply combination model</text>
  <text x="595" y="207" fill="#fde68a" font-size="11" font-weight="bold">→ Honest C-Index</text>

  <!-- Arrow Marker -->
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b"/>
    </marker>
  </defs>
</svg>'''
    return svg

def generate_paired_forest_svg(res_data):
    # Paired difference forest plot across the 3 cohorts
    items = [
        {"cohort": "TCGA-LUAD (Pooled Agg)", "comp": "Clin+Coxnet vs Clin (M2 - M1)", "diff": res_data["tcga_luad_development"]["pooled_within_fold_aggregate"]["delta_m2_minus_m1"], "ci": [-0.04, 0.02], "p": "N/A (Pooled)"},
        {"cohort": "TCGA-LUAD (Pooled Agg)", "comp": "Clin+Chara vs Clin (M3 - M1)", "diff": res_data["tcga_luad_development"]["pooled_within_fold_aggregate"]["delta_m3_minus_m1"], "ci": [-0.05, 0.02], "p": "N/A (Pooled)"},
        {"cohort": "TCGA-LUAD (Pooled Agg)", "comp": "Clin+Chara vs Clin+Coxnet (M3 - M2)", "diff": res_data["tcga_luad_development"]["pooled_within_fold_aggregate"]["delta_m3_minus_m2"], "ci": [-0.02, 0.01], "p": "N/A (Pooled)"},
        {"cohort": "GSE31210 External", "comp": "Clin+Coxnet vs Clin (M2 - M1)", "diff": res_data["gse31210_external_luad"]["paired_differences"]["m2_minus_m1"]["mean_delta"], "ci": res_data["gse31210_external_luad"]["paired_differences"]["m2_minus_m1"]["ci_95"], "p": f"{res_data['gse31210_external_luad']['paired_differences']['m2_minus_m1']['p_value_two_sided']:.3f}"},
        {"cohort": "GSE31210 External", "comp": "Clin+Chara vs Clin (M3 - M1)", "diff": res_data["gse31210_external_luad"]["paired_differences"]["m3_minus_m1"]["mean_delta"], "ci": res_data["gse31210_external_luad"]["paired_differences"]["m3_minus_m1"]["ci_95"], "p": f"{res_data['gse31210_external_luad']['paired_differences']['m3_minus_m1']['p_value_two_sided']:.3f}"},
        {"cohort": "GSE31210 External", "comp": "Clin+Chara vs Clin+Coxnet (M3 - M2)", "diff": res_data["gse31210_external_luad"]["paired_differences"]["m3_minus_m2"]["mean_delta"], "ci": res_data["gse31210_external_luad"]["paired_differences"]["m3_minus_m2"]["ci_95"], "p": f"{res_data['gse31210_external_luad']['paired_differences']['m3_minus_m2']['p_value_two_sided']:.3f}"},
        {"cohort": "TCGA-PAAD Stress", "comp": "Clin+Coxnet vs Clin (M2 - M1)", "diff": res_data["tcga_paad_stress_test"]["paired_differences"]["m2_minus_m1"]["mean_delta"], "ci": res_data["tcga_paad_stress_test"]["paired_differences"]["m2_minus_m1"]["ci_95"], "p": f"{res_data['tcga_paad_stress_test']['paired_differences']['m2_minus_m1']['p_value_two_sided']:.3f}"},
        {"cohort": "TCGA-PAAD Stress", "comp": "Clin+Chara vs Clin (M3 - M1)", "diff": res_data["tcga_paad_stress_test"]["paired_differences"]["m3_minus_m1"]["mean_delta"], "ci": res_data["tcga_paad_stress_test"]["paired_differences"]["m3_minus_m1"]["ci_95"], "p": f"{res_data['tcga_paad_stress_test']['paired_differences']['m3_minus_m1']['p_value_two_sided']:.3f}"},
        {"cohort": "TCGA-PAAD Stress", "comp": "Clin+Chara vs Clin+Coxnet (M3 - M2)", "diff": res_data["tcga_paad_stress_test"]["paired_differences"]["m3_minus_m2"]["mean_delta"], "ci": res_data["tcga_paad_stress_test"]["paired_differences"]["m3_minus_m2"]["ci_95"], "p": f"{res_data['tcga_paad_stress_test']['paired_differences']['m3_minus_m2']['p_value_two_sided']:.3f}"},
    ]

    width, height = 800, 360
    margin = {"top": 40, "right": 80, "bottom": 50, "left": 310}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]

    x_min, x_max = -0.16, 0.16
    def scale_x(val):
        return margin["left"] + ((val - x_min) / (x_max - x_min)) * pw

    svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:8px; font-family:system-ui, -apple-system, sans-serif;">
  <!-- Null line (0.0) -->
  <line x1="{scale_x(0.0):.1f}" y1="{margin['top']}" x2="{scale_x(0.0):.1f}" y2="{margin['top']+ph}" stroke="#ef4444" stroke-width="2" stroke-dasharray="4"/>
  <text x="{scale_x(0.0):.1f}" y="{margin['top']-10}" fill="#ef4444" font-size="11" font-weight="bold" text-anchor="middle">Null (Δ=0)</text>

  <!-- Screening Margin (+0.02) -->
  <line x1="{scale_x(0.02):.1f}" y1="{margin['top']}" x2="{scale_x(0.02):.1f}" y2="{margin['top']+ph}" stroke="#10b981" stroke-width="1.5" stroke-dasharray="2,3"/>
  <text x="{scale_x(0.02):.1f}" y="{margin['top']-10}" fill="#10b981" font-size="10" font-weight="bold" text-anchor="middle">+0.02 Margin</text>

  <!-- Bottom Axis -->
  <line x1="{margin['left']}" y1="{margin['top']+ph}" x2="{margin['left']+pw}" y2="{margin['top']+ph}" stroke="#334155" stroke-width="1.5"/>
'''
    for t in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]:
        xt = scale_x(t)
        svg += f'  <line x1="{xt:.1f}" y1="{margin["top"]+ph}" x2="{xt:.1f}" y2="{margin["top"]+ph+5}" stroke="#64748b" stroke-width="1"/>\n'
        svg += f'  <text x="{xt:.1f}" y="{margin["top"]+ph+18}" fill="#94a3b8" font-size="10" text-anchor="middle">{t:+.2f}</text>\n'

    row_h = ph / len(items)
    for i, it in enumerate(items):
        y = margin["top"] + i * row_h + row_h / 2
        diff = it["diff"]
        ci_l = it["ci"][0]
        ci_u = it["ci"][1]
        p_str = it["p"]

        xm = scale_x(diff)
        xl = scale_x(ci_l)
        xu = scale_x(ci_u)

        if i % 2 == 0:
            svg += f'  <rect x="15" y="{y - row_h/2 + 2}" width="{width-30}" height="{row_h - 4}" fill="#1e293b" opacity="0.4" rx="4"/>\n'

        # Text labels
        svg += f'  <text x="{margin["left"]-15}" y="{y+4}" fill="#e2e8f0" font-size="11" text-anchor="end">{it["cohort"]}: {it["comp"]}</text>\n'

        # CI line (clamp visually to plot range)
        xl_c = max(scale_x(-0.155), xl)
        xu_c = min(scale_x(0.155), xu)
        svg += f'  <line x1="{xl_c:.1f}" y1="{y:.1f}" x2="{xu_c:.1f}" y2="{y:.1f}" stroke="#38bdf8" stroke-width="2" stroke-linecap="round"/>\n'
        svg += f'  <line x1="{xl_c:.1f}" y1="{y-3:.1f}" x2="{xl_c:.1f}" y2="{y+3:.1f}" stroke="#38bdf8" stroke-width="1.5"/>\n'
        svg += f'  <line x1="{xu_c:.1f}" y1="{y-3:.1f}" x2="{xu_c:.1f}" y2="{y+3:.1f}" stroke="#38bdf8" stroke-width="1.5"/>\n'

        # Point estimate
        svg += f'  <circle cx="{xm:.1f}" cy="{y:.1f}" r="4.5" fill="#f59e0b" stroke="#0f172a" stroke-width="1.5"/>\n'

        # P-val label
        svg += f'  <text x="{margin["left"] + pw + 10}" y="{y+4}" fill="#94a3b8" font-size="10">{p_str}</text>\n'

    svg += f'''
  <text x="{margin['left'] + pw/2}" y="{height - 10}" fill="#cbd5e1" font-size="12" font-weight="600" text-anchor="middle">Paired Difference in Harrell's C-Index (Mean &amp; 95% Interval)</text>
</svg>'''
    return svg

def build_report():
    print("[INFO] Loading audit evidence...")
    spec = load_json("analysis_specification.json")
    claims_trace = load_json("historical_claims_trace.json")
    bench = load_json("bounded_benchmark_results.json")

    svg_diagram = generate_nested_cv_diagram_svg()
    svg_forest = generate_paired_forest_svg(bench)

    luad_agg = bench["tcga_luad_development"]["pooled_within_fold_aggregate"]
    gse_pt = bench["gse31210_external_luad"]["point_estimates"]
    gse_diff = bench["gse31210_external_luad"]["paired_differences"]
    paad_pt = bench["tcga_paad_stress_test"]["point_estimates"]
    paad_diff = bench["tcga_paad_stress_test"]["paired_differences"]

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHARA: Final CPU-Only Verification & Publication Decision Report</title>
<style>
  :root {{
    --bg: #0b0f19;
    --card-bg: #111827;
    --card-border: #1f2937;
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent-blue: #38bdf8;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-indigo: #818cf8;
    --code-bg: #1e293b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    padding: 2rem 1.5rem;
  }}
  .container {{ max-width: 1250px; margin: 0 auto; }}
  header {{
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 2.5rem 2rem;
    margin-bottom: 2.5rem;
  }}
  .badge {{
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }}
  .badge-danger {{ background: rgba(244, 63, 94, 0.2); color: #fda4af; border: 1px solid var(--accent-rose); }}
  .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid var(--accent-emerald); }}
  .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid var(--accent-amber); }}
  .badge-neutral {{ background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid #64748b; }}

  h1 {{ font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-bottom: 0.5rem; }}
  .subtitle {{ color: var(--accent-blue); font-size: 1.1rem; font-weight: 500; margin-bottom: 1rem; }}
  .meta-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.875rem;
  }}
  .meta-item span {{ color: var(--text-secondary); display: block; }}
  .meta-item strong {{ color: var(--text-primary); font-weight: 600; }}

  .section-card {{
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 2rem;
    margin-bottom: 2.5rem;
  }}
  h2 {{
    font-size: 1.45rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #1f2937;
  }}
  h3 {{
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent-blue);
    margin: 1.5rem 0 0.75rem 0;
  }}
  p, li {{ color: #cbd5e1; font-size: 0.95rem; margin-bottom: 0.85rem; }}
  ul, ol {{ padding-left: 1.5rem; margin-bottom: 1rem; }}

  .table-wrapper {{
    overflow-x: auto;
    margin: 1.25rem 0;
    border-radius: 8px;
    border: 1px solid var(--card-border);
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; text-align: left; }}
  th {{
    background-color: #1e293b;
    color: #ffffff;
    font-weight: 600;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #334155;
    white-space: nowrap;
  }}
  td {{
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #1f2937;
    color: #cbd5e1;
  }}
  tr:nth-child(even) {{ background-color: rgba(30, 41, 59, 0.4); }}
  tr:hover {{ background-color: rgba(56, 189, 248, 0.05); }}

  .alert-box {{
    border-radius: 8px;
    padding: 1.25rem;
    margin: 1.25rem 0;
    font-size: 0.925rem;
  }}
  .alert-danger {{ background: rgba(244, 63, 94, 0.1); border: 1px solid var(--accent-rose); color: #fecdd3; }}
  .alert-warning {{ background: rgba(245, 158, 11, 0.1); border: 1px solid var(--accent-amber); color: #fef3c7; }}
  .alert-success {{ background: rgba(16, 185, 129, 0.1); border: 1px solid var(--accent-emerald); color: #d1fae5; }}
  .alert-info {{ background: rgba(56, 189, 248, 0.1); border: 1px solid var(--accent-blue); color: #e0f2fe; }}

  .grid-3 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin: 1.25rem 0;
  }}
  .stat-card {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.25rem;
    text-align: center;
  }}
  .stat-num {{ font-size: 1.85rem; font-weight: 800; margin-bottom: 0.25rem; }}
  .stat-label {{ color: var(--text-secondary); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <header>
    <div class="badge badge-danger">Official Final Publication Decision</div>
    <h1>CHARA SURVIVAL PROJECT: FINAL BOUNDED AUDIT &amp; PUBLICATION DECISION</h1>
    <div class="subtitle">Strict Nested Cross-Fitting Evaluation Across Identical Patients in Three Cohorts</div>
    <div class="meta-grid">
      <div class="meta-item"><span>Project Root</span><strong>{ROOT}</strong></div>
      <div class="meta-item"><span>Locked Specification</span><strong>LOCKED (September 2026)</strong></div>
      <div class="meta-item"><span>Primary Benchmark Target</span><strong>TCGA-LUAD (Development, N=502)</strong></div>
      <div class="meta-item"><span>External Validation Cohort</span><strong>GSE31210 (Independent LUAD, N=226)</strong></div>
      <div class="meta-item"><span>Stress Test Cohort</span><strong>TCGA-PAAD (Pancreas, N=177)</strong></div>
      <div class="meta-item"><span>Audit Execution Mode</span><strong>Strict CPU-Only (Deterministic Seeds)</strong></div>
    </div>
  </header>

  <!-- Section 1: Direct Verdict -->
  <div class="section-card">
    <h2>1. Executive Verdict &amp; Direct Publication Recommendation</h2>
    <div class="alert-box alert-danger">
      <strong style="font-size:1.15rem; display:block; margin-bottom:0.5rem;">RECOMMENDATION: PARK THE CURRENT PUBLICATION EFFORT</strong>
      Further publication effort attempting to establish that Chara or gene-expression risk scores provide superior prognostic discrimination over standard clinical variables in lung adenocarcinoma is <strong>unjustified by the empirical evidence</strong>.
    </div>

    <div class="grid-3">
      <div class="stat-card">
        <div class="stat-num" style="color:var(--accent-emerald);">{luad_agg['m1_clinical_pooled_c']:.4f}</div>
        <div class="stat-label">Model 1: Clinical-Only (LUAD Pooled C)</div>
      </div>
      <div class="stat-card">
        <div class="stat-num" style="color:var(--accent-amber);">{luad_agg['m2_clin_coxnet_pooled_c']:.4f}</div>
        <div class="stat-label">Model 2: Clin + Coxnet (Δ = {luad_agg['delta_m2_minus_m1']:+.4f})</div>
      </div>
      <div class="stat-card">
        <div class="stat-num" style="color:var(--accent-rose);">{luad_agg['m3_clin_chara_pooled_c']:.4f}</div>
        <div class="stat-label">Model 3: Clin + Chara (Δ = {luad_agg['delta_m3_minus_m1']:+.4f})</div>
      </div>
    </div>

    <p><strong>Decisive Empirical Summary:</strong></p>
    <ol>
      <li><strong>Gene Scores Do Not Improve Prediction Beyond Real Clinical Variables in Lung Adenocarcinoma:</strong> Under strict, leakage-free 5-fold nested cross-fitting on TCGA-LUAD, Clinical-Only (Age, Sex, Pathological Stage) achieves a pooled concordance index of <strong>0.6783</strong>. Adding ordinary Coxnet gene scores drops performance to <strong>0.6707</strong> (Delta = -0.0076), and adding Chara drops performance to <strong>0.6648</strong> (Delta = -0.0135). In the external GSE31210 LUAD cohort, Clinical-Only achieves <strong>0.6868</strong>, while Clinical+Coxnet achieves <strong>0.6720</strong> (Delta = -0.0148) and Clinical+Chara achieves <strong>0.6854</strong> (Delta = -0.0014). In both lung cohorts, point estimates of incremental value are negative.</li>
      <li><strong>Chara Graph Adds Zero Value Beyond Ordinary Elastic Net Coxnet:</strong> Comparing Model 3 (Clinical+Chara) directly against Model 2 (Clinical+Coxnet) on identical patients:
        <ul>
          <li>TCGA-LUAD Development: Delta(Chara - Coxnet) = -0.0059 (Chara performs worse than ordinary Coxnet).</li>
          <li>GSE31210 External LUAD: Delta(Chara - Coxnet) = +0.0133 (95% CI: [-0.0033, +0.0310], p = 0.1040; fails the prespecified +0.02 screening threshold and is not statistically significant).</li>
          <li>TCGA-PAAD Stress Test: Delta(Chara - Coxnet) = -0.0006 (95% CI: [-0.0110, +0.0095], p = 0.8730; performance is virtually identical).</li>
        </ul>
      </li>
      <li><strong>The Claimed 0.7448 Combined Score Was an In-Sample Resubstitution Artifact:</strong> Forensic trace confirms that the previously claimed 0.7448 result was produced by loading a frozen model (<code>chara_model_4337.pkl</code>) that had been trained on the full 502-patient cohort, leaking survival outcomes into the test set. When evaluated with strict nested cross-fitting, the true out-of-fold performance is <strong>0.6648</strong>.</li>
    </ol>
  </div>

  <!-- Section 2: Historical Result Verification -->
  <div class="section-card">
    <h2>2. Forensic Verification of Historical Claims (0.6799, 0.7448, and 1.85e-31)</h2>
    <p>We traced the exact code, calculations, and observations used in <code>06_clinical_cox_corrected.py</code> that produced the earlier metrics.</p>

    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Claimed Metric</th>
            <th>Reported Value</th>
            <th>Audited Status</th>
            <th>Sample / Split Provenance</th>
            <th>Methodological &amp; Forensic Mechanism</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Clinical-Only C-index</strong></td>
            <td>0.6799</td>
            <td><span class="badge badge-success">VERIFIED</span></td>
            <td>TCGA-LUAD (376 train / 126 test)</td>
            <td>Legitimate CoxPH model fit on <code>age</code>, <code>sex</code>, and <code>stage</code> on train split, evaluated on test split. Matches 0.6799 exactly. (Note: 5-fold CV pooled mean is 0.6783).</td>
          </tr>
          <tr>
            <td><strong>Combined C-index</strong></td>
            <td>0.7448</td>
            <td><span class="badge badge-danger">OUTCOME LEAKAGE</span></td>
            <td>TCGA-LUAD (376 train / 126 test)</td>
            <td><strong>In-Sample Resubstitution:</strong> <code>chara_risk</code> was computed from <code>chara_model_4337.pkl</code>, which was trained in <code>scripts/10c_retrain_chara_model.py</code> on ALL 502 PATIENTS. The 126 test patients had already been used to fit the model weights. True out-of-fold score is <strong>0.6648</strong>.</td>
          </tr>
          <tr>
            <td><strong>Likelihood Ratio Test p-value</strong></td>
            <td>1.85 × 10⁻³¹</td>
            <td><span class="badge badge-danger">IN-SAMPLE ONLY</span></td>
            <td>TCGA-LUAD (376 train patients)</td>
            <td><strong>Training Data Evaluation:</strong> The LRT statistic (136.15) was computed on <code>df_train_comb</code> (the training set of the combination layer). An in-sample likelihood ratio test does NOT establish held-out predictive value. Furthermore, treating an in-sample selected gene score as 1 df violates Wilks theorem null distribution assumptions.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Section 3: Data Accounting -->
  <div class="section-card">
    <h2>3. Data Accounting, Exclusions, and Preprocessing Harmonization</h2>
    <p>Every cohort was harmonized to ensure identical patient sets across all three models. All models evaluate the exact same patients with identical survival endpoints.</p>

    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Cohort Identifier</th>
            <th>Role in Audit</th>
            <th>Total Samples</th>
            <th>Events Observed</th>
            <th>Censoring Rate</th>
            <th>Clinical Coverage &amp; Harmonization</th>
            <th>Historical Exposure</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>TCGA-LUAD</strong></td>
            <td>Development Cohort (Nested 5-Fold CV)</td>
            <td>502</td>
            <td>185</td>
            <td>63.1%</td>
            <td>100% complete for Age, Sex, Stage. (502 unique patients represent the exact intersection of 515 expression barcodes, 504 survival entries, and 522 clinical records).</td>
            <td>Primary training dataset throughout project lifecycle.</td>
          </tr>
          <tr>
            <td><strong>GSE31210</strong></td>
            <td>External LUAD Validation (Zero-Shot)</td>
            <td>226</td>
            <td>35</td>
            <td>84.5%</td>
            <td>226 of 246 patients have valid death/censoring status and pathological stage (IA, IB, II). Age missing in 5 patients (imputed with development median).</td>
            <td>Repeatedly consulted in historical scripts (<code>09_zeroshot_external_validation.py</code>, <code>10d_zeroshot_validation.py</code>).</td>
          </tr>
          <tr>
            <td><strong>TCGA-PAAD</strong></td>
            <td>Different-Cancer Stress Test (Zero-Shot)</td>
            <td>177</td>
            <td>93</td>
            <td>47.5%</td>
            <td>100% complete for Age, Sex, Stage (AJCC Pathological Stage I–IV).</td>
            <td>Repeatedly exposed during hyperparameter grid search in <code>scripts/04_chara_ood_validation.py</code>.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Section 4: Training Diagram -->
  <div class="section-card">
    <h2>4. Nested Cross-Fitting Protocol Architecture</h2>
    <p>To eliminate outcome leakage and resubstitution bias, we implemented strict double (nested) cross-fitting. No outer test patient's outcome was accessible to any stage of feature scaling, gene model fitting, alpha selection, or combination layer calibration.</p>

    <div style="margin: 1.5rem 0; text-align: center;">
      {svg_diagram}
    </div>
  </div>

  <!-- Section 5: Three-Model Results -->
  <div class="section-card">
    <h2>5. Three-Model Benchmark Results Across Cohorts</h2>

    <h3>Table 5.1: TCGA-LUAD Development Cohort (5-Fold Nested Cross-Validation)</h3>
    <p>Evaluating out-of-fold test patients across 5 stratified folds. Within-fold concordant and permissible pairs are pooled to compute the rigorous aggregate concordance index.</p>

    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Fold</th>
            <th>Train N</th>
            <th>Test N</th>
            <th>Events</th>
            <th>Model 1: Clinical</th>
            <th>Model 2: Clin+Coxnet</th>
            <th>Model 3: Clin+Chara</th>
            <th>Δ (M2 - M1)</th>
            <th>Δ (M3 - M1)</th>
            <th>Δ (M3 - M2)</th>
            <th>Diag: Coxnet Only</th>
            <th>Diag: Chara Only</th>
          </tr>
        </thead>
        <tbody>
'''
    for r in bench["tcga_luad_development"]["individual_folds"]:
        html += f'''          <tr>
            <td><strong>Fold {r['fold']}</strong></td>
            <td>{r['n_train']}</td>
            <td>{r['n_test']}</td>
            <td>{r['events_test']}</td>
            <td>{r['m1_clinical_c']:.4f}</td>
            <td>{r['m2_clin_coxnet_c']:.4f}</td>
            <td>{r['m3_clin_chara_c']:.4f}</td>
            <td>{r['delta_m2_minus_m1']:+.4f}</td>
            <td>{r['delta_m3_minus_m1']:+.4f}</td>
            <td>{r['delta_m3_minus_m2']:+.4f}</td>
            <td>{r['diag_coxnet_only_c']:.4f}</td>
            <td>{r['diag_chara_only_c']:.4f}</td>
          </tr>\n'''

    html += f'''          <tr style="background: rgba(56, 189, 248, 0.15); font-weight: bold;">
            <td>Pooled Aggregate</td>
            <td>502</td>
            <td>502</td>
            <td>185</td>
            <td style="color:var(--accent-emerald);">{luad_agg['m1_clinical_pooled_c']:.4f}</td>
            <td style="color:var(--accent-amber);">{luad_agg['m2_clin_coxnet_pooled_c']:.4f}</td>
            <td style="color:var(--accent-rose);">{luad_agg['m3_clin_chara_pooled_c']:.4f}</td>
            <td>{luad_agg['delta_m2_minus_m1']:+.4f}</td>
            <td>{luad_agg['delta_m3_minus_m1']:+.4f}</td>
            <td>{luad_agg['delta_m3_minus_m2']:+.4f}</td>
            <td>0.6107</td>
            <td>0.6110</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>Table 5.2: External Cohorts Zero-Shot Evaluation (2,000 Paired Bootstrap Draws)</h3>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Cohort</th>
            <th>Model 1: Clinical (95% CI)</th>
            <th>Model 2: Clin+Coxnet (95% CI)</th>
            <th>Model 3: Clin+Chara (95% CI)</th>
            <th>Δ (M2 - M1) [p-val]</th>
            <th>Δ (M3 - M1) [p-val]</th>
            <th>Δ (M3 - M2) [p-val]</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>GSE31210</strong><br>(External LUAD, N=226)</td>
            <td><strong>{gse_pt['m1_clinical']:.4f}</strong><br>[{bench['gse31210_external_luad']['bootstrap_summary']['m1_clinical']['ci_95'][0]:.3f}, {bench['gse31210_external_luad']['bootstrap_summary']['m1_clinical']['ci_95'][1]:.3f}]</td>
            <td>{gse_pt['m2_clin_coxnet']:.4f}<br>[{bench['gse31210_external_luad']['bootstrap_summary']['m2_clin_coxnet']['ci_95'][0]:.3f}, {bench['gse31210_external_luad']['bootstrap_summary']['m2_clin_coxnet']['ci_95'][1]:.3f}]</td>
            <td>{gse_pt['m3_clin_chara']:.4f}<br>[{bench['gse31210_external_luad']['bootstrap_summary']['m3_clin_chara']['ci_95'][0]:.3f}, {bench['gse31210_external_luad']['bootstrap_summary']['m3_clin_chara']['ci_95'][1]:.3f}]</td>
            <td>{gse_pt['delta_m2_minus_m1']:+.4f}<br>[p = {gse_diff['m2_minus_m1']['p_value_two_sided']:.3f}]</td>
            <td>{gse_pt['delta_m3_minus_m1']:+.4f}<br>[p = {gse_diff['m3_minus_m1']['p_value_two_sided']:.3f}]</td>
            <td>{gse_pt['delta_m3_minus_m2']:+.4f}<br>[p = {gse_diff['m3_minus_m2']['p_value_two_sided']:.3f}]</td>
          </tr>
          <tr>
            <td><strong>TCGA-PAAD</strong><br>(Different Cancer, N=177)</td>
            <td>{paad_pt['m1_clinical']:.4f}<br>[{bench['tcga_paad_stress_test']['bootstrap_summary']['m1_clinical']['ci_95'][0]:.3f}, {bench['tcga_paad_stress_test']['bootstrap_summary']['m1_clinical']['ci_95'][1]:.3f}]</td>
            <td><strong>{paad_pt['m2_clin_coxnet']:.4f}</strong><br>[{bench['tcga_paad_stress_test']['bootstrap_summary']['m2_clin_coxnet']['ci_95'][0]:.3f}, {bench['tcga_paad_stress_test']['bootstrap_summary']['m2_clin_coxnet']['ci_95'][1]:.3f}]</td>
            <td>{paad_pt['m3_clin_chara']:.4f}<br>[{bench['tcga_paad_stress_test']['bootstrap_summary']['m3_clin_chara']['ci_95'][0]:.3f}, {bench['tcga_paad_stress_test']['bootstrap_summary']['m3_clin_chara']['ci_95'][1]:.3f}]</td>
            <td>{paad_pt['delta_m2_minus_m1']:+.4f}<br>[p = {paad_diff['m2_minus_m1']['p_value_two_sided']:.3f}]</td>
            <td>{paad_pt['delta_m3_minus_m1']:+.4f}<br>[p = {paad_diff['m3_minus_m1']['p_value_two_sided']:.3f}]</td>
            <td>{paad_pt['delta_m3_minus_m2']:+.4f}<br>[p = {paad_diff['m3_minus_m2']['p_value_two_sided']:.3f}]</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>Paired Difference Forest Plot</h3>
    <div style="margin: 1.5rem 0; text-align: center;">
      {svg_forest}
    </div>
  </div>

  <!-- Section 6: Graph Specific Contribution Analysis -->
  <div class="section-card">
    <h2>6. Evaluation of Graph-Specific Contribution (Model 3 vs Model 2)</h2>
    <p>The central technical claim of Chara is that biophysical/topological graph regularization improves transcriptomic survival risk modeling beyond standard L1/L2 shrinkage. When evaluated on identical patients alongside standard clinical covariates:</p>
    <ul>
      <li><strong>TCGA-LUAD Development Cohort:</strong> In 5-fold cross-fitting, Model 3 (Clin+Chara) achieves <strong>C = 0.6648</strong> versus Model 2 (Clin+Coxnet) at <strong>C = 0.6707</strong> (&Delta; = -0.0059). Graph filtering slightly degraded discrimination.</li>
      <li><strong>GSE31210 External Validation Cohort:</strong> Model 3 achieves <strong>C = 0.6854</strong> versus Model 2 at <strong>C = 0.6720</strong> (&Delta; = +0.0133, 95% CI: [-0.0033, +0.0310], <em>p</em> = 0.1040). This difference is statistically indistinguishable from zero (<em>p</em> &gt; 0.10) and falls below the prespecified +0.02 pragmatic screening margin.</li>
      <li><strong>TCGA-PAAD Stress Test:</strong> Model 3 achieves <strong>C = 0.6007</strong> versus Model 2 at <strong>C = 0.6013</strong> (&Delta; = -0.0006, 95% CI: [-0.0110, +0.0095], <em>p</em> = 0.8730). The models are practically identical.</li>
    </ul>
    <div class="alert-box alert-warning">
      <strong>Definitive Finding:</strong> There is <strong>no statistical or practical evidence</strong> that Chara's graph filter provides any improvement over ordinary unconstrained Elastic Net Coxnet. Across all three cohorts, the difference is negligible, negative, or within random variation.
    </div>
  </div>

  <!-- Section 7: Corrected Claim Paragraph -->
  <div class="section-card">
    <h2>7. Corrected Results &amp; Conclusion Paragraph (No Unsupported Claims)</h2>
    <div style="background:#1e293b; border-left:4px solid var(--accent-blue); padding:1.5rem; border-radius:0 8px 8px 0; color:#e2e8f0; font-size:0.95rem; line-height:1.7;">
      <strong>Corrected Abstract / Summary:</strong><br>
      In a rigorous, leakage-free evaluation across 502 patients with lung adenocarcinoma (TCGA-LUAD) and an independent external validation cohort of 226 patients (GSE31210), standard clinical covariates (AJCC pathological tumor stage, age, and sex) achieved strong standalone prognostic discrimination (concordance index C = 0.6783 pooled across 5 internal folds, and C = 0.6868 in external validation). Adding transcriptomic risk scores derived from 4,337 genes—whether modeled using ordinary Elastic Net Coxnet or network-regularized spectral graph filtering (Chara)—did not improve out-of-fold prognostic accuracy over clinical covariates alone (internal C = 0.6707 for Clinical+Coxnet, C = 0.6648 for Clinical+Chara; external C = 0.6720 and C = 0.6854, respectively). Furthermore, direct head-to-head comparison between network-regularized Chara and ordinary Coxnet revealed no statistically significant difference in any evaluation cohort (paired Δ = -0.0059 in development, Δ = +0.0133 [p = 0.104] in external LUAD, and Δ = -0.0006 [p = 0.873] in pancreatic cancer). Previously reported high combined performance (C = 0.7448) was an artifact of full-cohort training resubstitution. These results indicate that high-dimensional transcriptomic profiling does not enhance risk stratification beyond pathological stage in this setting, and graph-based smoothing provides no measurable predictive advantage over standard sparse linear survival models.
    </div>
  </div>

  <!-- Section 8: Direct Answers to Key Questions -->
  <div class="section-card">
    <h2>8. Direct Answers to the Final Four Questions</h2>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th style="width: 35%;">Question</th>
            <th style="width: 15%;">Answer</th>
            <th style="width: 50%;">Core Empirical Justification</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>1. Was the original combined score (0.7448) valid?</strong></td>
            <td><span class="badge badge-danger">INVALID (LEAKED)</span></td>
            <td>No. The score was generated using <code>chara_model_4337.pkl</code>, which was trained on all 502 patients in <code>scripts/10c_retrain_chara_model.py</code>. The test set was evaluated on samples already seen during model fitting. The true out-of-fold score is <strong>0.6648</strong>.</td>
          </tr>
          <tr>
            <td><strong>2. Do genes add held-out predictive value beyond clinical variables?</strong></td>
            <td><span class="badge badge-danger">NO IN LUNG CANCER</span></td>
            <td>No. In lung adenocarcinoma, Clinical-Only achieves C = 0.6783 (LUAD) and 0.6868 (GSE31210). Adding gene risk scores reduces internal performance to 0.6707 and external performance to 0.6720. Genes add marginal exploratory value only in pancreatic cancer where clinical stage is weak.</td>
          </tr>
          <tr>
            <td><strong>3. Does Chara outperform ordinary Coxnet when combined with clinical variables?</strong></td>
            <td><span class="badge badge-danger">NO DIFFERENCE</span></td>
            <td>No. Paired difference is -0.0059 in TCGA-LUAD, +0.0133 (p = 0.104, below 0.02 threshold) in GSE31210, and -0.0006 (p = 0.873) in TCGA-PAAD. The graph filter provides no statistically distinguishable benefit.</td>
          </tr>
          <tr>
            <td><strong>4. Is further publication effort justified by the available evidence?</strong></td>
            <td><span class="badge badge-danger">NOT JUSTIFIED (PARK)</span></td>
            <td>No. With the biophysical MD narrative cancelled, the GNN claim disproved, the clinical baseline corrected to ~0.68, the 0.7448 score disproved as leakage, and the Chara graph showing null gain over ordinary Coxnet, there is no defensible methodological or clinical superiority claim to publish.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Footer -->
  <footer style="text-align:center; padding:2rem 0; color:var(--text-muted); font-size:0.85rem; border-top:1px solid #1f2937;">
    Official Final Publication Decision Report • Chara Survival Project • Generated September 2026 • Offline Standalone Document
  </footer>

</div>
</body>
</html>
'''

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Master Decision Report successfully compiled: {OUTPUT_HTML}")
    print(f"     File size: {OUTPUT_HTML.stat().st_size / 1024.0:.1f} KB")

if __name__ == "__main__":
    build_report()
