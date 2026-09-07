#!/usr/bin/env python3
"""
tools/chara_final_check/02_save_analysis_spec.py
Saves the locked, timestamped analysis specification for the final bounded
CPU-only evaluation before new comparison scores are inspected.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT / "reports" / "chara_final_check" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    spec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_PRE_EXECUTION_SPECIFICATION",
        "description": "Rigorous bounded evaluation specification locked prior to inspecting final benchmark scores.",
        "primary_research_questions": [
            "Do gene-expression risk scores improve prediction beyond real clinical variables when every evaluation patient's outcome is excluded from all preceding fitting and selection?",
            "If they do, does Chara add anything beyond ordinary Elastic Net Coxnet?"
        ],
        "models_evaluated": {
            "Model_1_Clinical_Only": {
                "description": "Cox Proportional Hazards model using actual Age, Sex, and Pathological Stage.",
                "covariates": ["age", "sex", "stage"],
                "encoding": {
                    "age": "Continuous (standardized with training mean and standard deviation)",
                    "sex": "Binary (Male=1.0, Female=0.0)",
                    "stage": "Ordinal integer: Stage I=1.0, Stage II=2.0, Stage III=3.0, Stage IV=4.0"
                },
                "penalizer": 0.01
            },
            "Model_2_Clinical_plus_Ordinary_Coxnet": {
                "description": "Clinical predictors plus unconstrained Elastic Net Coxnet gene risk score.",
                "gene_model": "CoxnetSurvivalAnalysis(l1_ratio=0.5, 25-point alpha path, alpha_min_ratio=0.01)",
                "graph_filter": "None (Identity matrix)",
                "training_protocol": "Nested 3-fold cross-fitting within outer training partition; base model refit on full outer train."
            },
            "Model_3_Clinical_plus_Chara": {
                "description": "Clinical predictors plus Chara network-regularized risk score.",
                "gene_model": "CoxnetSurvivalAnalysis(l1_ratio=0.5, 25-point alpha path, alpha_min_ratio=0.01)",
                "graph_filter": "Heat-kernel filter W = U * exp(-0.1 * Lambda) * U^T from Laplacian_Chara_4337.csv",
                "training_protocol": "Nested 3-fold cross-fitting within outer training partition; base model refit on full outer train.",
                "inference_consistency": "Transformed test features X_test @ W"
            }
        },
        "required_comparisons": [
            "Model 2 minus Model 1 (Clinical+Coxnet vs Clinical-Only)",
            "Model 3 minus Model 1 (Clinical+Chara vs Clinical-Only)",
            "Model 3 minus Model 2 (Clinical+Chara vs Clinical+Coxnet)"
        ],
        "screening_margin": {
            "c_index_delta_threshold": 0.02,
            "interpretation": "Pragmatic screening margin for continuation decision, not an established clinical threshold."
        },
        "cohorts": {
            "TCGA-LUAD": {
                "role": "Development Cohort",
                "sample_count": 502,
                "event_count": 185,
                "censored_count": 317,
                "validation_strategy": "5-fold stratified cross-validation (seed=4337)"
            },
            "GSE31210": {
                "role": "External LUAD Validation Cohort",
                "sample_count": 226,
                "event_count": 35,
                "censored_count": 191,
                "validation_strategy": "Zero-shot application of source-trained pipeline with 2,000 paired bootstrap resamples"
            },
            "TCGA-PAAD": {
                "role": "Out-of-Distribution / Different-Cancer Stress Test",
                "sample_count": 177,
                "event_count": 93,
                "censored_count": 84,
                "validation_strategy": "Zero-shot application of source-trained pipeline with 2,000 paired bootstrap resamples"
            }
        },
        "gene_universe": {
            "source_file": "intersecting_genes_4337.txt",
            "gene_count": 4337
        },
        "multiplicity_correction": "Holm-Bonferroni across the 3 comparisons",
        "cpu_budget": {
            "max_outer_folds": 5,
            "inner_folds": 3,
            "bootstrap_draws": 2000
        }
    }

    out_file = EVIDENCE_DIR / "analysis_specification.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"[OK] Locked analysis specification saved to: {out_file}")

if __name__ == "__main__":
    main()
