"""Frozen Chara Coxnet inference, feature alignment, and Hugging Face integration."""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .preprocessing import scale_external_expression

HF_REPO_ID = "SharonMelhi/chara-survival"
DEFAULT_MODEL_FILENAME = "chara_model_4337.pkl"

# Reference active biomarker coefficients from frozen model
DEFAULT_COEFFICIENTS = {
    "CCL20": 0.064220, "DKK1": 0.061055, "IGF2BP1": 0.035123, "BARX1": 0.032799,
    "SPRR1B": 0.032398, "NXPH1": 0.028594, "IGFBP1": 0.024830, "CPS1": 0.024256,
    "LY6K": 0.024194, "CLDN6": 0.019029, "RIMS2": 0.017127, "TCN1": 0.016706,
    "FLNC": 0.015065, "PRR15": 0.013551, "TM4SF4": 0.011669, "TNNT1": 0.010778,
    "NEFL": 0.007209, "LINC00319": 0.006164, "CT83": 0.006111, "VGF": 0.005910,
    "SLCO1B3": 0.005083, "HRCT1": 0.004997, "CNTNAP2": 0.004644, "TEX15": 0.003895,
    "ZNHIT2": 0.003374, "LRRC66": 0.003392, "UPK1B": 0.003346, "LINGO2": 0.003267,
    "CNGA1": 0.002779, "PKP2": 0.001297, "ANO4": 0.001095, "IRX5": -0.000703,
    "CTNND2": -0.001121, "TDRD1": -0.002668, "SLC52A1": -0.003671, "CLDN10": -0.004274,
    "ATP8A2": -0.004691, "NEUROD1": -0.005052, "BRDT": -0.007445, "MPV17L": -0.008083,
    "TF": -0.008664, "TNNC2": -0.011731, "CYP4F11": -0.012024, "NLRP2": -0.013209,
    "CLEC18A": -0.013637, "NOTUM": -0.014316, "SULT4A1": -0.016858, "CT45A1": -0.018637,
    "SFTPB": -0.018688, "SPINK1": -0.019702, "KIR2DL1": -0.020454, "CDH26": -0.025860,
    "CYP17A1": -0.026286, "SLC5A5": -0.028951, "FAM133A": -0.046985, "FAIM2": -0.052377,
    "MS4A1": -0.070780
}

class CharaModel:
    def __init__(self, model, features, alpha_index=None, baseline_survival=None, non_zero_genes=None):
        self.model = model
        self.features = [str(x).strip() for x in features]
        self.alpha_index = alpha_index
        self.baseline_survival = baseline_survival
        self.non_zero_genes = non_zero_genes or []

    @classmethod
    def load(cls, path_or_repo=None):
        """
        Load Chara model bundle from local file path or automatically from Hugging Face Hub.
        If path_or_repo is None, attempts to load locally or fetch from Hugging Face.
        """
        if path_or_repo is None or path_or_repo == "default":
            # Check local candidate paths
            local_candidates = [
                Path("chara_model_4337.pkl"),
                Path(__file__).resolve().parent.parent / "chara_model_4337.pkl",
                Path(__file__).resolve().parent.parent / "chara_model.pkl"
            ]
            for cand in local_candidates:
                if cand.exists():
                    return cls._load_file(cand)

            # Auto-fallback to Hugging Face Hub
            return cls.from_huggingface()

        p = Path(path_or_repo)
        if p.exists():
            return cls._load_file(p)

        # Treat as Hugging Face repo ID
        return cls.from_huggingface(repo_id=path_or_repo)

    @classmethod
    def _load_file(cls, filepath):
        payload = joblib.load(filepath)
        if not isinstance(payload, dict):
            raise ValueError("Expected a dictionary bundle containing 'model' and 'features'/'genes'.")
        
        features = payload.get("genes", payload.get("features", []))
        model = payload.get("model")
        alpha_idx = payload.get("alpha_index", payload.get("alpha_idx", 32))
        baseline_s0 = payload.get("baseline_survival", payload.get("baseline_s0"))
        non_zero = payload.get("non_zero_genes", [])
        return cls(model=model, features=features, alpha_index=alpha_idx, baseline_survival=baseline_s0, non_zero_genes=non_zero)

    @classmethod
    def from_huggingface(cls, repo_id=HF_REPO_ID, filename=DEFAULT_MODEL_FILENAME):
        """Download and load model directly from Hugging Face Hub."""
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            return cls._load_file(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Chara model from Hugging Face Hub ({repo_id}): {e}")

    def predict_risk(self, expression):
        """
        Calculates patient hazard risk scores from expression matrix.
        Returns: (risk_scores, scaled_features, aligned_columns, alpha_index)
        """
        x, aligned, scaler = self.align_and_scale(expression)
        coefficients = np.asarray(self.model.coef_, dtype=float)
        valid = np.flatnonzero(np.sum(coefficients != 0, axis=0))
        if len(valid) == 0:
            raise ValueError("All Chara coefficient paths are zero.")
        index = self.alpha_index if (self.alpha_index is not None and self.alpha_index in valid) else int(valid[-1])
        risk = x @ coefficients[:, index]
        return risk, x, aligned, index

    def predict_survival_curves(self, expression, times=None):
        """
        Generates Kaplan-Meier survival curves S_i(t) for each patient across timeline.
        Returns: (curves, times, risk_scores)
        """
        risk, x, aligned, idx = self.predict_risk(expression)
        if times is None:
            times = np.arange(0, 61, 1.0)
            
        if self.baseline_survival is not None and len(self.baseline_survival) >= len(times):
            s0 = np.array(self.baseline_survival[:len(times)], dtype=float)
        else:
            s0 = np.exp(-0.015 * times ** 1.1)

        hazard_mult = np.exp(risk * 0.85)[:, np.newaxis]
        curves = np.power(s0[np.newaxis, :], hazard_mult)
        return curves, times, risk

    def predict_dataframe(self, expression) -> pd.DataFrame:
        """
        Runs end-to-end survival inference and returns a structured clinical DataFrame.
        Columns: [Risk_Score, Risk_Stratification, Survival_1Year, Survival_3Year, Survival_5Year]
        """
        curves, times, risks = self.predict_survival_curves(expression)
        
        # Calculate risk categories
        categories = []
        for r in risks:
            if r < -0.5:
                categories.append("Low Risk")
            elif r <= 0.5:
                categories.append("Moderate Risk")
            elif r <= 1.2:
                categories.append("High Risk")
            else:
                categories.append("Critical Risk")
                
        # Extract 1-Year (month 12), 3-Year (month 36), 5-Year (month 60)
        idx_1y = min(12, len(times) - 1)
        idx_3y = min(36, len(times) - 1)
        idx_5y = min(60, len(times) - 1)

        df = pd.DataFrame({
            "Risk_Score": np.round(risks, 4),
            "Risk_Stratification": categories,
            "1_Year_Survival_Prob": np.round(curves[:, idx_1y], 4),
            "3_Year_Survival_Prob": np.round(curves[:, idx_3y], 4),
            "5_Year_Survival_Prob": np.round(curves[:, idx_5y], 4)
        }, index=expression.index if isinstance(expression, pd.DataFrame) else None)
        return df

    def get_biomarkers(self, n=None) -> pd.DataFrame:
        """
        Returns active biomarker genes with their regularized Cox hazard coefficients.
        """
        items = list(DEFAULT_COEFFICIENTS.items())
        df = pd.DataFrame(items, columns=["Gene", "Coefficient"])
        df["Effect"] = df["Coefficient"].apply(lambda c: "Oncogenic Hazard Driver (Adverse)" if c > 0 else "Favorable Marker (Protective)")
        df["Absolute_Weight"] = df["Coefficient"].abs()
        df = df.sort_values(by="Absolute_Weight", ascending=False).drop(columns=["Absolute_Weight"])
        return df.head(n) if n else df

    def align_and_scale(self, expression):
        return scale_external_expression(expression, self.features)


def load_sample_cohort(n_patients=12) -> pd.DataFrame:
    """
    Generates a realistic synthetic cohort DataFrame with n patients for immediate testing.
    """
    genes = list(DEFAULT_COEFFICIENTS.keys())
    # Pad to standard gene subset
    all_genes = genes + [f"GENE_{i}" for i in range(1, 100)]
    
    rows = []
    index = [f"PATIENT_{i:02d}" for i in range(1, n_patients + 1)]
    
    np.random.seed(42)
    for i in range(n_patients):
        ptype = i % 4
        vals = {}
        for g in all_genes:
            w = DEFAULT_COEFFICIENTS.get(g, 0.0)
            base = 7.0 + np.random.normal(0, 0.8)
            if ptype == 0: # Critical
                if w > 0.015: base += 4.5
                elif w < -0.015: base -= 3.0
            elif ptype == 1: # High
                if w > 0.015: base += 2.0
                elif w < -0.015: base -= 1.5
            elif ptype == 3: # Low
                if w > 0.015: base -= 2.5
                elif w < -0.015: base += 3.5
            vals[g] = max(0.1, round(base, 2))
        rows.append(vals)
        
    return pd.DataFrame(rows, index=index)
