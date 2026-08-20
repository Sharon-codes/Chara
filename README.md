# Chara Survival: Thermodynamic Graph Laplacian Manifold Alignment

[![Website](https://img.shields.io/badge/Web_App-chara--frontend.vercel.app-445d30?style=for-the-badge&logo=vercel)](https://chara-frontend.vercel.app)
[![PyPI](https://img.shields.io/pypi/v/chara-survival?style=for-the-badge&color=2d3d21)](https://pypi.org/project/chara-survival/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Official publication repository for **Chara**, a thermodynamic graph Laplacian manifold alignment framework developed at the **Computational and Physical Genomics Laboratory, Indian Institute of Technology Mandi**.

- 🌐 **Live Web Application**: [https://chara-frontend.vercel.app](https://chara-frontend.vercel.app)
- 📦 **PyPI Package**: `pip install chara-survival`
- 🏫 **Laboratory**: Computational & Physical Genomics Lab, IIT Mandi

---

## 🔬 Key Architectural Highlights

1. **Deterministic Graph Laplacian Operator**:
   Constructs a symmetric normalized Laplacian matrix $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$ across patient expression vectors.
2. **Heat Kernel Diffusion Smoothing**:
   Applies spectral smoothing $\mathbf{H}_t = \exp(-t\mathbf{L})$ to dissipate non-biological platform variance while preserving patient manifold topology.
3. **Zero Test-Data Leakage**:
   Projects unseen test cohorts deterministically onto a frozen **4,337-gene Cox Proportional Hazards signature** without pooling test and training samples.
4. **Cross-Platform Validation**:
   Achieves **+0.268 Concordance Index (C-Index) gain** over raw unadjusted models across Illumina RNA-Seq, Affymetrix Microarrays, and Agilent platforms.

---

## 🚀 Quickstart: Python SDK Usage

```bash
pip install chara-survival
```

```python
import pandas as pd
from chara import CharaModel

# 1. Load patient transcriptomics CSV (Rows: Patients, Columns: Gene Symbols)
df = pd.read_csv("hospital_cohort_rna.csv", index_col=0)

# 2. Load frozen 4,337-gene Cox Proportional Hazards signature
model = CharaModel.load("chara_model_4337.pkl")

# 3. Execute Graph Laplacian Manifold Alignment
risk_scores, x_scaled, aligned_df, _, alpha = model.predict(df)
curves, times = model.survival_curves(x_scaled, alpha)

print(f"Mean Cohort Risk Score: {risk_scores.mean():.4f}")
```

---

## 👥 Authors & Laboratory

- **Dr. Kharerin Hungyo** (Principal Investigator, Assistant Professor, IIT Mandi) — `kharerin@iitmandi.ac.in`
- **Sharon Melhi** (Research Intern & Developer, CPG Lab, IIT Mandi) — `sharonmelhi365@gmail.com`

### Special Acknowledgment
*My heartfelt thanks to Khushi Mhamane for her constant encouragement, thoughtful discussions, and belief in this research from its earliest stages.* — Sharon Melhi

---

© 2026 Indian Institute of Technology Mandi. Released under the MIT License.
