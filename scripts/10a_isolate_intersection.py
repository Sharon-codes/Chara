#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("validation09", ROOT / "scripts" / "09_zeroshot_external_validation.py")
validation09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation09)

def main():
    training = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    training.columns = training.columns.astype(str).str.strip()
    gse, _ = validation09.parse_geo_gse31210()
    external = validation09.load_gse31210_expression(gse)
    external = external.apply(pd.to_numeric, errors="coerce")
    external = external.replace([np.inf, -np.inf], np.nan)
    external = external.loc[:, external.var(axis=0, skipna=True).fillna(0.0) > 0.0]
    genes = [g for g in training.columns if g in set(external.columns)]
    if not genes:
        raise ValueError("No non-zero-variance HGNC intersection was found.")
    pd.Series(genes, dtype=str).to_csv(ROOT / "intersecting_genes_4337.txt", index=False, header=False)
    print(f"Saved {len(genes)} strictly verified genes to intersecting_genes_4337.txt")

if __name__ == "__main__":
    main()
