"""Cross-platform expression preprocessing and feature alignment."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def align_expression(expression: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Aligns arbitrary expression dataframe columns to target biomarker feature signature."""
    frame = expression.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.T.groupby(level=0).mean().T
    return frame.reindex(columns=features, fill_value=0.0).fillna(0.0).astype(np.float64)

def scale_external_expression(expression: pd.DataFrame, features: list[str]):
    """
    Standardizes expression matrix. Supports both cohort batches (N > 1) and single-patient samples (N = 1).
    """
    aligned = align_expression(expression, features)
    arr = aligned.to_numpy(dtype=np.float64)
    
    if len(arr) == 1:
        # Single patient inference: standardize across gene distribution
        mean = np.mean(arr, axis=1, keepdims=True)
        std = np.std(arr, axis=1, keepdims=True) + 1e-8
        scaled = (arr - mean) / std
        scaler = StandardScaler()
    else:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(arr)
        
    return scaled, aligned, scaler
