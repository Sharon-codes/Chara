"""Cross-platform expression preprocessing."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def align_expression(expression: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = expression.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.T.groupby(level=0).mean().T
    return frame.reindex(columns=features, fill_value=0.0).fillna(0.0).astype(np.float64)

def scale_external_expression(expression: pd.DataFrame, features: list[str]):
    aligned = align_expression(expression, features)
    scaler = StandardScaler()
    return scaler.fit_transform(aligned.to_numpy(dtype=np.float64)), aligned, scaler
