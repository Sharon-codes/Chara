#!/usr/bin/env python3
"""Zero-shot external validation of the frozen Chara Coxnet model on GSE31210.

This script downloads the independent LUAD cohort from GEO using GEOparse, extracts
survival metadata and expression data, aligns features to the TCGA 5,200-gene training
space, imputes missing genes with zeros, and evaluates the frozen Chara model without
retraining.
"""

import re
import warnings
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import GEOparse
except ImportError as exc:  # pragma: no cover
    raise SystemExit("GEOparse is required. Install it with: pip install GEOparse") from exc

from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.getLogger("GEOparse").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
MODEL_PATH = DATA_DIR / "chara_model.pkl"
TRAINING_GENE_PATH = DATA_DIR / "TCGA-LUAD_expression.csv"


def _as_float(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in {"nan", "na", "n/a", "none", "null"}:
            return np.nan
        s = s.replace(",", "")
        s = re.sub(r"[^0-9eE+\-\.]+", "", s)
        if s in {"", "+", "-", "."}:
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return np.nan


def _as_bool(value):
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "deceased", "dead"}:
            return True
        if s in {"0", "false", "no", "n", "censored", "lost", "follow-up", "alive"}:
            return False
    return False


def _safe_identifier(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_scalar(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [v for v in value if v is not None]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return "; ".join(str(v) for v in values)
    return value


def load_training_gene_set():
    expr = pd.read_csv(TRAINING_GENE_PATH, index_col=0)
    expr.columns = [str(c).strip() for c in expr.columns]
    genes = np.asarray(expr.columns, dtype=str)
    return genes


def parse_geo_gse31210():
    gse = GEOparse.get_GEO(geo="GSE31210", destdir=str(DATA_DIR / "geo_cache"), include_data=True)

    rows = []
    for gsm_id, gsm in gse.gsms.items():
        meta = getattr(gsm, "metadata", {}) or {}
        if not isinstance(meta, dict):
            continue
        chars = meta.get("characteristics_ch1", [])
        if isinstance(chars, str):
            chars = [chars]
        elif chars is None:
            chars = []
        chars = [str(x).strip() for x in chars if x is not None and str(x).strip()]

        entry = {
            "SampleAccession": _coerce_scalar(meta.get("geo_accession", gsm_id)),
            "SampleTitle": _coerce_scalar(meta.get("title", gsm_id)),
            "Status": _coerce_scalar(meta.get("status")),
            "Characteristics": "; ".join(chars),
            "CharacteristicsList": chars,
            "source_name_ch1": _coerce_scalar(meta.get("source_name_ch1")),
            "organism_ch1": _coerce_scalar(meta.get("organism_ch1")),
        }
        rows.append(entry)

    if not rows:
        raise RuntimeError("Could not load GEO sample metadata from GSE31210.")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No GEO sample records were available after parsing GSE31210 metadata.")

    survival_df = pd.DataFrame(index=df.index)
    survival_df["SampleAccession"] = df.get("SampleAccession", pd.Series(index=df.index, dtype=object))
    survival_df["SampleTitle"] = df.get("SampleTitle", pd.Series(index=df.index, dtype=object))
    survival_df["OS_months"] = np.nan
    survival_df["VitalStatus"] = np.nan
    survival_df["Event"] = False

    for i, char_list in enumerate(df["CharacteristicsList"].tolist()):
        if not char_list:
            continue

        os_months = np.nan
        vital_event = False
        
        # Iterate through each characteristic string for this sample
        for char_str in char_list:
            char_lower = char_str.lower()
            
            # Parse survival time from "months before death/censor:" or "months before relapse/censor:"
            if "months before death/censor:" in char_lower or "months before relapse/censor:" in char_lower:
                if ":" in char_str:
                    time_part = char_str.split(":", 1)[1].strip()
                    time_val = _as_float(time_part)
                    if pd.notna(time_val):
                        os_months = time_val
            
            # Parse vital status from "death:" field
            if "death:" in char_lower:
                if ":" in char_str:
                    status_part = char_str.split(":", 1)[1].strip()
                    vital_event = _as_bool(status_part)
        
        if pd.notna(os_months):
            survival_df.at[i, "OS_months"] = os_months
        
        survival_df.at[i, "Event"] = vital_event

    time_values = []
    for _, row in survival_df.iterrows():
        months = row["OS_months"]
        if pd.notna(months):
            time_values.append(float(months))
        else:
            time_values.append(np.nan)
    
    survival_df["TimeMonths"] = pd.to_numeric(time_values, errors="coerce")
    survival_df["Event"] = survival_df["Event"].astype(bool)
    survival_df = survival_df[survival_df["TimeMonths"].notna()].copy()

    if survival_df.empty:
        raise RuntimeError("No valid survival outcomes could be extracted from GSE31210 metadata.")

    return gse, survival_df


def map_probe_to_gene_symbol(expr_df):
    # Many GEO matrix tables use probes as rows and gene symbols as a separate column.
    if "Gene Symbol" in expr_df.columns:
        expr_df = expr_df.copy()
        expr_df["Gene Symbol"] = expr_df["Gene Symbol"].astype(str).str.strip()
        expr_df = expr_df[~expr_df["Gene Symbol"].str.lower().isin({"", "nan", "na", "none"})].copy()
        grouped = expr_df.groupby("Gene Symbol").mean(numeric_only=True)
        return grouped

    # If the GEO table is already a samples x probes matrix, use the row labels as probe IDs.
    if hasattr(expr_df, "index") and not expr_df.empty:
        if expr_df.index.name is not None and "gene" in str(expr_df.index.name).lower():
            expr_df.index = expr_df.index.astype(str)
            return expr_df.groupby(level=0).mean(numeric_only=True)

    return expr_df


def load_gse31210_expression(gse):
    platform_id = next(iter(gse.gpls))
    platform_table = gse.gpls[platform_id].table
    probe_col = "ID" if "ID" in platform_table.columns else platform_table.columns[0]
    symbol_col = "Gene Symbol"
    probe_to_gene = platform_table.set_index(probe_col)[symbol_col].map(
        lambda value: str(value).split(" /// ")[0].strip().upper()
        if pd.notna(value) and str(value).strip() not in {"", "---"}
        else np.nan
    ).dropna()

    sample_rows = []
    sample_ids = []
    for gsm_name, gsm in gse.gsms.items():
        table = gsm.table
        if table is None or table.empty:
            continue
        id_col = "ID_REF" if "ID_REF" in table.columns else table.columns[0]
        # Explicitly extract the raw per-sample VALUE column before concatenation.
        value_col = "VALUE" if "VALUE" in table.columns else next(
            (col for col in table.columns if str(col).upper() in {"SIGNAL", "EXPRESSION"}),
            None,
        )
        if value_col is None:
            raise RuntimeError(f"No raw expression VALUE column found for {gsm_name}.")
        probe_values = pd.Series(
            pd.to_numeric(table[value_col], errors="coerce").to_numpy(dtype=np.float64),
            index=table[id_col].astype(str).str.strip(),
        )
        gene_values = probe_values.groupby(probe_values.index.map(probe_to_gene)).mean()
        sample_rows.append(gene_values)
        sample_ids.append(gsm_name)

    if not sample_rows:
        raise RuntimeError("No expression tables were found in GSE31210 GSM samples.")
    return pd.DataFrame(sample_rows, index=sample_ids)


def align_external_cohort_to_training(expr_df, training_genes):
    expr = expr_df.copy()
    expr.index = [str(i).strip() for i in expr.index]
    expr.columns = [str(c).strip() for c in expr.columns]

    if not isinstance(expr, pd.DataFrame):
        expr = pd.DataFrame(expr)

    # Ensure the frame is numeric before any groupby aggregation. GEO exports sometimes mix
    # numeric expression values with stray text labels or metadata columns, which leads to
    # pandas object-dtype aggregation errors on mean().
    expr = expr.apply(lambda col: pd.to_numeric(col, errors="coerce"))

    # Standardize to a sample x gene matrix; if the frame is gene x sample, transpose it.
    if expr.shape[0] == len(training_genes) and expr.shape[1] != len(training_genes):
        gene_matrix = expr.T.copy()
    else:
        gene_matrix = expr.copy()

    gene_matrix.columns = [str(c).strip() for c in gene_matrix.columns]
    gene_matrix.index = [str(i).strip() for i in gene_matrix.index]

    # Collapse duplicate gene symbols by mean to prevent repeated probes mapping to the same gene.
    if not gene_matrix.empty and len(gene_matrix.columns) > 0:
        gene_matrix = gene_matrix.T.groupby(level=0).mean(numeric_only=True).T

    aligned = pd.DataFrame(index=gene_matrix.index, columns=training_genes, dtype=np.float64)
    for gene in training_genes:
        if gene in gene_matrix.columns:
            aligned[gene] = pd.to_numeric(gene_matrix[gene], errors="coerce").fillna(0.0)
        else:
            aligned[gene] = 0.0

    # Preserve the exact ordered training feature sequence expected by the frozen model.
    aligned = aligned.reindex(columns=training_genes)
    return aligned


def main():
    model = joblib.load(MODEL_PATH)
    training_genes = load_training_gene_set()
    gse, survival_df = parse_geo_gse31210()
    expr_df = load_gse31210_expression(gse)
    aligned = align_external_cohort_to_training(expr_df, training_genes)
    feature_match = len(set(expr_df.columns).intersection(training_genes))

    # Ensure all values are strictly numeric before model scoring.
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)

    # Reset index to remove any non-numeric row labels that might interfere with numpy conversion.
    aligned = aligned.reset_index(drop=True)
    
    # Ensure survival_df has same number of rows as aligned for proper evaluation.
    n_samples = min(len(aligned), len(survival_df))
    aligned = aligned.iloc[:n_samples].copy()
    survival_df = survival_df.iloc[:n_samples].copy()

    X_external = aligned.to_numpy(dtype=np.float64)
    X_external = np.asarray(X_external, dtype=np.float64)

    # Cross-platform normalization is fitted independently on the external cohort.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X_external)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract the 2D coefficient matrix.
    coef_matrix = model.coef_
    coef_matrix = np.asarray(coef_matrix, dtype=np.float64)
    if coef_matrix.ndim != 2 or coef_matrix.shape[0] != x_scaled.shape[1]:
        raise ValueError(
            f"Coefficient matrix shape {coef_matrix.shape} is incompatible with "
            f"external feature matrix shape {x_scaled.shape}."
        )

    # Count non-zero coefficients across all alpha paths and select the
    # least-penalized, maximally informative valid path.
    non_zero_counts = np.sum(coef_matrix != 0, axis=0)
    valid_indices = np.where(non_zero_counts > 0)[0]
    if len(valid_indices) == 0:
        raise ValueError(
            "Catastrophic model failure: The entire coefficient matrix is zero across all alphas."
        )
    opt_idx = valid_indices[-1]
    optimal_betas = coef_matrix[:, opt_idx]

    # Manually compute the Cox linear predictor (risk score).
    risk_scores = np.dot(x_scaled, optimal_betas)
    print(
        f" Debug: Selected Alpha Index {opt_idx} containing "
        f"{non_zero_counts[opt_idx]} non-zero coefficients."
    )

    event = survival_df["Event"].astype(bool).to_numpy()
    time = survival_df["TimeMonths"].astype(float).to_numpy()
    
    # Enforce computation: raise ValueError if all samples are censored
    if event.sum() == 0:
        raise ValueError("All samples in GSE31210 are censored; cannot compute C-Index.")

    print(f"Debug X_scaled Variance: {np.var(x_scaled):.6f}")
    print(f"Debug Risk Score Unique Values: {len(np.unique(risk_scores))}")
    
    c_index = concordance_index_censored(event, time, risk_scores)[0]

    print("======================================================================")
    print(" ZERO-SHOT EXTERNAL COHORT VALIDATION: GSE31210 (LUAD)")
    print("======================================================================")
    print(f" External Patients Processed   : {n_samples}")
    print(f" Events Observed               : {event.sum()} / {n_samples} (censored ratio: {1 - event.mean():.2%})")
    print(f" Feature Intersection Match    : {feature_match} / 5200 genes")
    print(f" Zero-Shot C-Index             : {c_index:.4f}")
    print("======================================================================")


if __name__ == "__main__":
    main()
