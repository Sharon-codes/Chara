#!/usr/bin/env python3
"""
01_fetch_tcga.py - Full Transcriptome ETL Pipeline for TCGA LUAD & PAAD Datasets (>5,000 Genes)
Downloads 20,531 gene expression profiles, strips decimal version numbers from Ensembl/Gene IDs,
aligns patient sample barcodes, filters >5,000 high-variance common genes, and outputs 4 CSV files.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import io

OUTPUT_DIR = "/home/sharon/Desktop/Sharon"

URLS = {
    "LUAD_surv": "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/luad_tcga/data_clinical_patient.txt",
    "PAAD_surv": "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/paad_tcga/data_clinical_patient.txt",
    "LUAD_exp":  "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/luad_tcga/data_mrna_seq_v2_rsem.txt",
    "PAAD_exp":  "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/paad_tcga/data_mrna_seq_v2_rsem.txt"
}

def fetch_survival_table(url):
    print(f"Fetching clinical survival dataset: {url}")
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
    lines = [l for l in r.text.split('\n') if not l.startswith('#')]
    df = pd.read_csv(io.StringIO('\n'.join(lines)), sep='\t', low_memory=False)
    
    col_pat = 'PATIENT_ID' if 'PATIENT_ID' in df.columns else df.columns[0]
    col_status = 'OS_STATUS' if 'OS_STATUS' in df.columns else [c for c in df.columns if 'VITAL' in c or 'STATUS' in c][0]
    col_time = 'OS_MONTHS' if 'OS_MONTHS' in df.columns else [c for c in df.columns if 'TIME' in c or 'DAYS' in c][0]
    
    df_sub = df[[col_pat, col_status, col_time]].copy()
    df_sub.columns = ['sample_id', 'Event_Raw', 'Time_Raw']
    
    def parse_evt(v):
        s = str(v).upper()
        if '1' in s or 'DECEASED' in s or 'DEAD' in s: return 1
        if '0' in s or 'LIVING' in s or 'ALIVE' in s: return 0
        return np.nan

    df_sub['Event'] = df_sub['Event_Raw'].apply(parse_evt)
    df_sub['Time'] = pd.to_numeric(df_sub['Time_Raw'], errors='coerce')
    
    if df_sub['Time'].max() > 200:
        df_sub['Time'] = df_sub['Time'] / 30.4375

    df_sub = df_sub.dropna(subset=['sample_id', 'Event', 'Time'])
    df_sub = df_sub[df_sub['Time'] > 0.0]
    df_sub['sample_id'] = df_sub['sample_id'].astype(str).str.strip().str[:12]
    df_sub = df_sub.set_index('sample_id')[['Event', 'Time']]
    return df_sub[~df_sub.index.duplicated(keep='first')]

def fetch_full_expression_matrix(url):
    print(f"Fetching full transcriptomic matrix: {url}")
    r = requests.get(url, stream=True, timeout=120)
    lines = []
    for l_bytes in r.iter_lines():
        if not l_bytes: continue
        s = l_bytes.decode('utf-8')
        if not s.startswith('#'):
            lines.append(s)
            
    df_exp = pd.read_csv(io.StringIO('\n'.join(lines)), sep='\t', low_memory=False)
    gene_col = df_exp.columns[0]
    
    # Task 1: Strip decimal version numbers from Ensembl/Gene IDs
    df_exp[gene_col] = df_exp[gene_col].astype(str).str.split('.').str[0]
    df_exp = df_exp.dropna(subset=[gene_col])
    
    # Drop non-sample columns
    drop_cols = [c for c in ['Entrez_Gene_Id', 'GENE_ID'] if c in df_exp.columns]
    df_exp = df_exp.drop(columns=drop_cols, errors='ignore')
    
    # Group duplicate genes by mean
    df_exp = df_exp.groupby(gene_col).mean()
    
    # Transpose so rows = Samples, columns = Genes
    df_trans = df_exp.T
    
    # Strip decimal versions from column names if any
    df_trans.columns = [str(c).split('.')[0] for c in df_trans.columns]
    
    # Format sample IDs to 12-char patient barcode
    df_trans.index = [str(s).replace('.', '-').strip()[:12] for s in df_trans.index]
    df_trans = df_trans[~df_trans.index.duplicated(keep='first')]
    
    # Log2 transformation log2(RSEM + 1)
    df_trans = np.log2(df_trans.astype(float) + 1.0)
    return df_trans

def main():
    print("="*80)
    print(" TASK 1: TCGA FULL TRANSCRIPTOME ETL PIPELINE (>5,000 GENES)")
    print(" Source: cBioPortal Full Repository (TCGA-LUAD & TCGA-PAAD)")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Clinical Survival Datasets
    df_luad_surv = fetch_survival_table(URLS["LUAD_surv"])
    df_paad_surv = fetch_survival_table(URLS["PAAD_surv"])
    
    # 2. Full Expression Datasets
    df_luad_exp = fetch_full_expression_matrix(URLS["LUAD_exp"])
    df_paad_exp = fetch_full_expression_matrix(URLS["PAAD_exp"])
    
    # 3. Align Samples Between Expression & Survival
    luad_common_samples = df_luad_exp.index.intersection(df_luad_surv.index)
    paad_common_samples = df_paad_exp.index.intersection(df_paad_surv.index)
    
    df_luad_exp_clean = df_luad_exp.loc[luad_common_samples]
    df_luad_surv_clean = df_luad_surv.loc[luad_common_samples]
    
    df_paad_exp_clean = df_paad_exp.loc[paad_common_samples]
    df_paad_surv_clean = df_paad_surv.loc[paad_common_samples]
    
    # 4. Find Common Genes across LUAD and PAAD
    common_gene_pool = df_luad_exp_clean.columns.intersection(df_paad_exp_clean.columns)
    print(f"Total overlapping genes between LUAD & PAAD: {len(common_gene_pool)}")
    
    # Filter for top 5,200 highest variance genes to ensure >5,000 common genes after Graph Laplacian alignment
    var_luad = df_luad_exp_clean[common_gene_pool].var(axis=0)
    top_5200_genes = var_luad.nlargest(5200).index
    
    df_luad_exp_final = df_luad_exp_clean[top_5200_genes]
    df_paad_exp_final = df_paad_exp_clean[top_5200_genes]
    
    # Verify >5,000 genes
    if df_luad_exp_final.shape[1] <= 5000:
        raise ValueError(f"FATAL ERROR: Intersection yielded {df_luad_exp_final.shape[1]} genes, required >5,000.")

    # 5. Save Output CSV Files
    p_luad_exp = os.path.join(OUTPUT_DIR, "TCGA-LUAD_expression.csv")
    p_luad_surv = os.path.join(OUTPUT_DIR, "TCGA-LUAD_survival.csv")
    p_paad_exp = os.path.join(OUTPUT_DIR, "TCGA-PAAD_expression.csv")
    p_paad_surv = os.path.join(OUTPUT_DIR, "TCGA-PAAD_survival.csv")
    
    df_luad_exp_final.to_csv(p_luad_exp)
    df_luad_surv_clean.to_csv(p_luad_surv)
    df_paad_exp_final.to_csv(p_paad_exp)
    df_paad_surv_clean.to_csv(p_paad_surv)
    
    print("\nSUCCESS: Downloaded, cleaned, aligned, and saved 4 TCGA data files:")
    print(f"  1. {p_luad_exp} ({df_luad_exp_final.shape[0]} samples x {df_luad_exp_final.shape[1]} genes)")
    print(f"  2. {p_luad_surv} ({df_luad_surv_clean.shape[0]} samples)")
    print(f"  3. {p_paad_exp} ({df_paad_exp_final.shape[0]} samples x {df_paad_exp_final.shape[1]} genes)")
    print(f"  4. {p_paad_surv} ({df_paad_surv_clean.shape[0]} samples)")
    print("="*80)

if __name__ == "__main__":
    main()
