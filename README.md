<div align="center">

<img src="https://raw.githubusercontent.com/Sharon-codes/Chara/main/assets/iit-mandi-logo.png" width="96" alt="IIT Mandi" style="border-radius:20px"/>

# Chara Survival

**Thermodynamic Graph Laplacian Survival Inference for Transcriptomic Oncology**

[![PyPI version](https://img.shields.io/pypi/v/chara-survival?color=445D30&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/chara-survival/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SharonMelhi%2Fchara--survival-yellow)](https://huggingface.co/SharonMelhi/chara-survival)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-445D30?logo=python&logoColor=white)](https://pypi.org/project/chara-survival/)
[![Web App](https://img.shields.io/badge/Web%20App-Live%20Demo-445D30?logo=vercel&logoColor=white)](https://chara-frontend.vercel.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-445D30.svg)](LICENSE)
[![IIT Mandi](https://img.shields.io/badge/Lab-Computational%20%26%20Physical%20Genomics-2D3D21)](https://www.iitmandi.ac.in)

*A frozen 4,337-gene thermodynamic intersection signature that transfers across sequencing platforms — zero retraining required.*

</div>

---

## 🌟 The Problem

Standard survival models — Cox proportional hazards, Random Survival Forests, DeepSurv — are trained on RNA-seq cohorts (typically TCGA) and **collapse catastrophically** when applied to microarray data (GEO). The cause is structural: uncorrected platform variance and feature distribution shift corrupt the learned risk landscape. The concordance index, already noisy at 0.50–0.55 on in-distribution data, degrades to random or sub-random when the platform changes.

This failure is not a modelling artefact. It is a **physical problem** — raw transcript counts do not encode the molecular interaction topology that determines biological function. Chara corrects this at the feature level, before training even begins.

## 💡 The Solution

Chara grounds gene expression features in **thermodynamic graph Laplacians** derived from MARTINI 3 coarse-grained molecular dynamics (MD) simulations. Specifically:

1. **MARTINI 3 MD trajectories** are run for key oncogenic protein systems (KRAS, CMYC/MAX, PTPN11, MUT-TP53) across biological replicates.
2. **Exponential heat kernels** are computed from the symmetrised graph Laplacian of the STRING protein–protein interaction network, weighted by MD-derived edge variances via the Chara exponential operator:
   $$W_{\text{Chara}}(i, j) = W_{\text{STRING}}(i, j) \cdot \exp\left(\tau \cdot Z(\sigma^2_{ij})\right)$$
3. The resulting **thermodynamic Laplacian representation** produces a platform-invariant feature space via spectral heat diffusion $H_t = \exp(-tL)$ — the spectral geometry of molecular interaction rather than raw transcript abundance.
4. A **frozen CoxNet** trained on TCGA-LUAD using these 4,337 thermodynamically-stabilised features (58 active regularized biomarkers) is applied directly to unseen cohorts without fine-tuning.

The result is a model that generalises across the RNA-seq ↔ microarray boundary as a matter of physical principle, not statistical luck.

---

## 🔬 Benchmark Performance

Evaluated zero-shot on **GSE31210** (Affymetrix Human Genome U133 Plus 2.0, n = 226, completely held-out lung adenocarcinoma microarray cohort — never seen during training):

| Model | OOD C-Index | 1-Year AUC | 3-Year AUC | 5-Year AUC |
|---|---:|---:|---:|---:|
| Clinical Cox-PH (Age, Gender, Stage) | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Random Survival Forest (RSF) | 0.4041 | 0.4175 | 0.3657 | 0.4842 |
| Elastic Net Coxnet (Raw 5,200 genes) | 0.5248 | 0.4664 | 0.4081 | 0.2428 |
| DeepSurv (Deep Neural Network) | 0.5537 | 0.5465 | 0.3937 | 0.3122 |
| **Chara (Thermodynamic Laplacian)** | **0.7311** | **0.7463** | **0.7826** | **0.8195** |

Chara achieves a **+0.267 absolute improvement in OOD C-index** over the next-best deep learning baseline, with monotonically improving time-horizon AUC — an unusual and clinically meaningful signature of robust calibration rather than threshold overfitting.

---

## 🚀 Quickstart: Python Package

### Installation

```bash
pip install --upgrade chara-survival
```

### 1-Line Python Inference & Complete API

```python
import chara
import pandas as pd

# 1. Load the frozen pretrained model (auto-fetches from Hugging Face Hub)
model = chara.load_model()

# 2. Ingest your patient cohort CSV (or load synthetic test cohort)
cohort_df = chara.load_sample_cohort(n_patients=12)

# 3. Generate 1-Click Structured Clinical Summary DataFrame
summary_df = model.predict_dataframe(cohort_df)
print(summary_df)
# Output columns: [Risk_Score, Hazard_Ratio, Risk_Stratification, Median_Survival, 1_Year_Survival_Prob, 3_Year_Survival_Prob, 5_Year_Survival_Prob]

# 4. Cohort Diagnostic Breakdown
cohort_stats = model.summarize_cohort(cohort_df)
print(cohort_stats)
# Output: {'Cohort_Size': 12, 'Mean_Risk_Score': 0.02, 'Mean_Hazard_Ratio': 1.14, 'High_Risk_Fraction': '25.0%', ...}

# 5. Single-Patient Clinical Prognosis
patient_prognosis = model.predict_patient(cohort_df.iloc[0])
print(patient_prognosis)

# 6. Native Harrell's Concordance Index (C-Index)
c_index = chara.concordance_index(
    risk_scores=summary_df["Risk_Score"], 
    time=[12, 24, 36, 48, 60, ...], 
    event=[1, 0, 1, 0, 1, ...]
)
print(f"C-Index: {c_index:.4f}")

# 7. Plot Publication-Grade Kaplan-Meier Curves & Biomarkers
model.plot_survival(cohort_df, save_path="km_survival.png")
model.plot_biomarkers(top_n=10, save_path="biomarkers.png")
```

### Direct Hugging Face Hub Loading

```python
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(repo_id="SharonMelhi/chara-survival", filename="chara_model_4337.pkl")
bundle = joblib.load(model_path)
print(f"Loaded Chara bundle with {len(bundle['genes'])} features and {len(bundle['non_zero_genes'])} biomarkers.")
```

---

## 🌐 Interactive Web Portal & Comparison Sandbox

Try the live, zero-install interactive inference suite and multi-model benchmark sandbox:

👉 **[https://chara-frontend.vercel.app](https://chara-frontend.vercel.app)**

---

## 🧬 Key Biomarkers & Hazard Drivers

The regularized signature isolates 58 key prognostic genes:
- **Top Oncogenic Hazard Drivers ($\beta > 0$):** `CCL20` (+0.0642), `DKK1` (+0.0610), `IGF2BP1` (+0.0351), `BARX1` (+0.0328), `SPRR1B` (+0.0324).
- **Top Favorable / Protective Biomarkers ($\beta < 0$):** `MS4A1` (-0.0708, CD20 B-cell marker), `FAIM2` (-0.0524), `FAM133A` (-0.0470), `SLC5A5` (-0.0290).

---

## 📜 Citation

```bibtex
@software{Melhi_Chara_Survival_2026,
  author = {Melhi, Sharon and Hungyo, Kharerin},
  title = {Chara: Thermodynamic Graph Laplacian Survival Inference for Transcriptomic Oncology},
  url = {https://github.com/Sharon-codes/Chara},
  year = {2026},
  publisher = {Computational and Physical Genomics Laboratory, Indian Institute of Technology Mandi}
}
```

---

## 👥 Authors & Lab

<table>
<tr>
<td width="200" align="center">
<img src="https://raw.githubusercontent.com/Sharon-codes/Chara/main/assets/dr-kharerin-hungyo.png" width="140" style="border-radius:16px" alt="Dr. Kharerin Hungyo"/><br/>
<strong>Dr. Kharerin Hungyo</strong><br/>
<sub>Principal Investigator<br/>Computational & Physical Genomics Lab<br/>Indian Institute of Technology Mandi</sub><br/>
<a href="mailto:kharerin@iitmandi.ac.in">kharerin@iitmandi.ac.in</a>
</td>
<td width="200" align="center">
<img src="https://raw.githubusercontent.com/Sharon-codes/Chara/main/assets/sharon-melhi.png" width="140" style="border-radius:16px" alt="Sharon Melhi"/><br/>
<strong>Sharon Melhi</strong><br/>
<sub>Computational Biologist<br/>Creator, Chara Survival</sub><br/>
<a href="https://www.linkedin.com/in/sharon-melhi/">LinkedIn</a> · <a href="mailto:sharonmelhi365@gmail.com">Email</a>
</td>
</tr>
</table>

Special thanks to **Khushi Mhamane** for her constant encouragement, thoughtful discussions, and belief in this research from its earliest stages. — Sharon Melhi

---

## 📄 License

Released under the [MIT License](LICENSE). © 2026 Sharon Melhi.
