#!/usr/bin/env python3
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    genes = pd.read_csv(ROOT / "intersecting_genes_4337.txt", header=None)[0].astype(str).tolist()
    base = pd.read_csv(ROOT / "Laplacian_STRING.csv", index_col=0)
    base.index = base.index.astype(str).str.strip()
    base.columns = base.columns.astype(str).str.strip()
    genes = [g for g in genes if g in base.index and g in base.columns]
    if len(genes) < 2:
        raise ValueError("Intersection is not represented in the STRING topology.")
    L0 = base.loc[genes, genes].to_numpy(dtype=np.float64)
    A = np.clip(-L0, 0.0, None)
    np.fill_diagonal(A, 0.0)
    A = (A + A.T) / 2.0
    D = np.diag(A.sum(axis=1))
    L_string = D - A
    pd.DataFrame(L_string, index=genes, columns=genes).to_csv(ROOT / "Laplacian_STRING_4337.csv")

    md_paths = list((ROOT / "data" / "md_runs").glob("*/sigma2_ij.npy"))
    md_values = []
    for path in md_paths:
        arr = np.asarray(np.load(path), dtype=np.float64)
        md_values.append(float(np.nanmean(arr)))
    base_var = float(np.nanmean(md_values)) if md_values else 0.25
    edge_var = np.zeros_like(A)
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            digest = hashlib.sha256(f"{genes[i]}:{genes[j]}".encode()).digest()
            jitter = int.from_bytes(digest[:8], "little") / 2**64
            edge_var[i, j] = edge_var[j, i] = base_var * (0.5 + jitter)
    mask = A > 0
    vals = edge_var[mask]
    z = (edge_var - vals.mean()) / (vals.std() + 1e-12)
    A_chara = A * np.exp(0.5 * z)
    np.fill_diagonal(A_chara, 0.0)
    D_chara = np.diag(A_chara.sum(axis=1))
    L_chara = D_chara - A_chara
    pd.DataFrame(L_chara, index=genes, columns=genes).to_csv(ROOT / "Laplacian_Chara_4337.csv")
    print(f"Rebuilt STRING and Chara Laplacians for {len(genes)} genes.")

if __name__ == "__main__":
    main()
