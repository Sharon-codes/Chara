---
language:
- en
license: mit
library_name: scikit-survival
tags:
- survival-analysis
- computational-biology
- oncology
- graph-laplacian
- precision-medicine
- transcriptomics
- bioinformatics
- coxnet
- cancer-prognostics
pipeline_tag: tabular-regression
datasets:
- TCGA-PAAD
- ICGC-PACA
- GSE15471
- GSE28735
- GSE57495
- GSE31210
metrics:
- c-index
- brier-score
- time-dependent-auc
---

# 🧬 Chara: Thermodynamic Graph Laplacian Survival Inference

<p align="center">
  <img src="https://raw.githubusercontent.com/Sharon-codes/Chara/main/assets/iit-mandi-logo.png" alt="IIT Mandi Logo" width="120" />
</p>

<p align="center">
  <strong>Computational & Physical Genomics Laboratory (CPG Lab)</strong><br>
  <em>Indian Institute of Technology Mandi (IIT Mandi), Himachal Pradesh, India</em><br>
  <strong>Creator & Lead Developer:</strong> Sharon Melhi · <strong>PI:</strong> Dr. Kharerin Hungyo
</p>

---

## 🌟 Overview

**Chara** is a biophysically grounded survival inference model designed to solve the long-standing **cross-platform transcriptomic domain shift** in oncology (e.g. Illumina RNA-Seq vs. Affymetrix Microarrays).

By constructing a **Thermodynamic Graph Laplacian ($L_{\text{Chara}}$)** informed by coarse-grained molecular dynamics (MARTINI 3) residue fluctuations ($\sigma^2_{ij}$) and applying heat diffusion smoothing:
$$H_t = \exp(-t L)$$
Chara dissipates machine-specific noise while preserving true biological cancer survival signals—enabling a frozen **4,337-gene Cox Proportional Hazards signature (58 active regularized biomarkers)** to execute on completely unseen hospital cohorts with **zero data leakage**.

- 📦 **PyPI Package:** [`chara-survival`](https://pypi.org/project/chara-survival/) (`pip install chara-survival`)
- 🌐 **Interactive Web Portal:** [https://chara-frontend.vercel.app](https://chara-frontend.vercel.app)
- 💻 **GitHub Repository:** [https://github.com/Sharon-codes/Chara](https://github.com/Sharon-codes/Chara)

---

## 🔬 Benchmark Performance

Across 6 independent multi-center international cohorts ($n = 889$ total patients), Chara demonstrates substantial performance gains over conventional survival machine learning architectures:

| Validation Cohort | Platform | Sample Size ($n$) | Unadjusted Cox | ComBat Harmonization | **Chara Laplacian (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ICGC-PACA (AU)** | RNA-Seq (HiSeq) | $n = 269$ | 0.531 | 0.682 | **0.784 (+0.253)** |
| **GSE15471** | Affymetrix HG-U133+2.0 | $n = 78$ | 0.508 | 0.641 | **0.762 (+0.254)** |
| **GSE28735** | GeneChip Human 1.0 ST | $n = 90$ | 0.522 | 0.665 | **0.771 (+0.249)** |
| **GSE57495** | Agilent Human Genome | $n = 63$ | 0.495 | 0.628 | **0.758 (+0.263)** |
| **GSE31210** | Zero-Shot Microarray | $n = 226$ | 0.512 | 0.639 | **0.731 (+0.219)** |

---

## 🚀 Quickstart & Usage in Python

### 1. Installation
```bash
pip install chara-survival huggingface_hub joblib scikit-survival
```

### 2. Load Model Directly from Hugging Face Hub
```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# Download the model file from Hugging Face
model_path = hf_hub_download(
    repo_id="SharonMelhi/chara-survival", 
    filename="chara_model_4337.pkl"
)

# Load the trained model bundle
bundle = joblib.load(model_path)
model = bundle["model"]
alpha_idx = bundle["alpha_index"]
genes = bundle["genes"]

print(f"Loaded Chara model with {len(genes)} intersecting gene features.")
print(f"Active regularized biomarkers: {len(bundle['non_zero_genes'])}")
```

### 3. Predict Patient Survival Curves
```python
import numpy as np

# Load unadjusted patient transcriptomics (e.g. from RNA-seq or Microarray)
# X_aligned: (n_patients, 4337)
# baseline_s0: Breslow baseline survival estimator S0(t)

def predict_patient_survival(X_smoothed, model, alpha_idx, baseline_s0):
    # Compute Coxnet log-hazard risk
    risk_scores = X_smoothed @ model.coef_[:, alpha_idx]
    
    # Evaluate Kaplan-Meier survival curves S_i(t) = [S0(t)]^exp(r_i)
    hazard_multipliers = np.exp(risk_scores * 0.85)
    curves = np.array([baseline_s0 ** h for h in hazard_multipliers])
    return risk_scores, curves
```

---

## 🧬 Key Biomarkers & Hazard Drivers

The regularized signature isolates 58 key prognostic genes:
- **Top Oncogenic Hazard Drivers ($\beta > 0$):** `CCL20` (+0.0642), `DKK1` (+0.0610), `IGF2BP1` (+0.0351), `BARX1` (+0.0328), `SPRR1B` (+0.0324).
- **Top Favorable / Protective Biomarkers ($\beta < 0$):** `MS4A1` (-0.0708, CD20 B-cell marker), `FAIM2` (-0.0524), `FAM133A` (-0.0470), `SLC5A5` (-0.0290).

---

## 📜 Citation & Attribution

```bibtex
@software{Melhi_Chara_Survival_2026,
  author = {Melhi, Sharon and Hungyo, Kharerin},
  title = {Chara: Thermodynamic Graph Laplacian Survival Inference for Transcriptomic Oncology},
  url = {https://github.com/Sharon-codes/Chara},
  year = {2026},
  publisher = {Computational and Physical Genomics Laboratory, Indian Institute of Technology Mandi}
}
```
