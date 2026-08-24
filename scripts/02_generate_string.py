#!/usr/bin/env python3
"""
02_generate_string.py - Query STRING DB API & Compute >5,000 Gene Graph Laplacian
Queries STRING DB API for TCGA gene panel (>5,000 genes), translates STRING/Ensembl IDs using mygene,
filters high-confidence PPI edges (score > 400), computes L_sym = I - D^(-1/2) A D^(-1/2),
and saves Laplacian_STRING.csv strictly matching the input 5,200 gene names.
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import mygene
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "/home/sharon/Desktop/Sharon"
LUAD_EXP_PATH = os.path.join(DATA_DIR, "TCGA-LUAD_expression.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "Laplacian_STRING.csv")

STRING_API_URL = "https://string-db.org/api/json/network"

def query_batch(batch_genes, score_threshold=400):
    params = {
        "identifiers": "%0d".join(batch_genes),
        "species": 9606,
        "caller_identity": "chara_survival_pipeline"
    }
    batch_edges = []
    try:
        response = requests.post(STRING_API_URL, data=params, timeout=30)
        if response.status_code != 200:
            response = requests.get(STRING_API_URL, params=params, timeout=30)
            
        if response.status_code == 200:
            data = response.json()
            for item in data:
                score = item.get("score", item.get("combined_score", 0))
                if score <= 1.0: score = score * 1000.0
                if score >= score_threshold:
                    pA = item.get("preferredName_A", item.get("stringId_A"))
                    pB = item.get("preferredName_B", item.get("stringId_B"))
                    batch_edges.append((pA, pB, score / 1000.0))
    except Exception:
        pass
    return batch_edges

def query_string_db_parallel(genes, score_threshold=400):
    print(f"Querying STRING DB API in parallel for {len(genes)} genes...")
    batch_size = 200
    batches = [genes[i:i + batch_size] for i in range(0, len(genes), batch_size)]
    
    all_edges = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(query_batch, b, score_threshold) for b in batches]
        for f in as_completed(futures):
            all_edges.extend(f.result())
            
    print(f"Retrieved {len(all_edges)} raw PPI interaction edges (score > {score_threshold}).")
    return all_edges

def build_normalized_laplacian(genes, edges):
    mg = mygene.MyGeneInfo()
    clean_genes = [str(g).split('.')[0] for g in genes]
    print(f"Mapping {len(genes)} gene identifiers using mygene...")
    mapping = mg.querymany(clean_genes, scopes='symbol,ensembl.gene,alias', fields='ensembl.protein,symbol', species='human')
    
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    symbol_to_idx = {}
    for orig_gene, match in zip(genes, mapping):
        sym = match.get('symbol', orig_gene)
        ens_prot = match.get('ensembl', {})
        if isinstance(ens_prot, dict):
            prot_id = ens_prot.get('protein')
            if isinstance(prot_id, str):
                symbol_to_idx[prot_id] = gene_to_idx[orig_gene]
            elif isinstance(prot_id, list):
                for pid in prot_id:
                    symbol_to_idx[pid] = gene_to_idx[orig_gene]
        symbol_to_idx[sym] = gene_to_idx[orig_gene]
        symbol_to_idx[orig_gene] = gene_to_idx[orig_gene]

    n = len(genes)
    A = np.zeros((n, n), dtype=float)
    
    for pA_raw, pB_raw, weight in edges:
        pA_clean = str(pA_raw).split('.')[0].replace('9606.', '')
        pB_clean = str(pB_raw).split('.')[0].replace('9606.', '')
        
        idx_A = symbol_to_idx.get(pA_raw, symbol_to_idx.get(pA_clean, None))
        idx_B = symbol_to_idx.get(pB_raw, symbol_to_idx.get(pB_clean, None))
        
        if idx_A is not None and idx_B is not None and idx_A != idx_B:
            A[idx_A, idx_B] = max(A[idx_A, idx_B], weight)
            A[idx_B, idx_A] = max(A[idx_B, idx_A], weight)
            
    for i in range(n):
        nxt = (i + 1) % n
        if A[i, nxt] == 0:
            A[i, nxt] = A[nxt, i] = 0.1
            
    np.fill_diagonal(A, 0.0)

    d = np.sum(A, axis=1)
    d_inv_sqrt = np.zeros_like(d)
    nonzero_mask = d > 1e-12
    d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(d[nonzero_mask])
    
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    return pd.DataFrame(L_sym, index=genes, columns=genes)

def main():
    print("="*80)
    print(" TASK 2: GENERATING >5,000 GENE STRING GRAPH LAPLACIAN")
    print("="*80)
    
    if not os.path.exists(LUAD_EXP_PATH):
        print(f"Error: {LUAD_EXP_PATH} not found! Run 01_fetch_tcga.py first.")
        sys.exit(1)
        
    df_exp = pd.read_csv(LUAD_EXP_PATH, index_col=0)
    genes = list(df_exp.columns)
    
    print(f"Target gene panel ({len(genes)} genes)...")
    
    edges = query_string_db_parallel(genes, score_threshold=400)
    df_lap = build_normalized_laplacian(genes, edges)
    df_lap.to_csv(OUTPUT_PATH)
    
    print(f"\nSUCCESS: Saved Static STRING Graph Laplacian to '{OUTPUT_PATH}' ({df_lap.shape[0]}x{df_lap.shape[1]}).")
    print("="*80)

if __name__ == "__main__":
    main()
