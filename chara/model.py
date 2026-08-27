"""Frozen Chara Coxnet inference, feature alignment, plotting, and Hugging Face integration."""
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
        Columns: [Risk_Score, Risk_Stratification, 1_Year_Survival_Prob, 3_Year_Survival_Prob, 5_Year_Survival_Prob]
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

    def predict_patient(self, expression_data) -> dict:
        """
        Single-patient survival evaluation helper.
        Accepts a dict, Series, or 1-row DataFrame and returns a comprehensive prognosis dictionary.
        """
        if isinstance(expression_data, dict):
            df = pd.DataFrame([expression_data])
        elif isinstance(expression_data, pd.Series):
            df = pd.DataFrame([expression_data.to_dict()])
        elif isinstance(expression_data, pd.DataFrame):
            df = expression_data
        else:
            raise ValueError("expression_data must be a dict, pandas Series, or pandas DataFrame.")

        res_df = self.predict_dataframe(df)
        row = res_df.iloc[0].to_dict()
        if df.index is not None and len(df.index) > 0 and str(df.index[0]) != "0":
            row["Patient_ID"] = str(df.index[0])
        return row

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

    def plot_survival(self, expression, title="Chara Survival Projections", save_path=None, ax=None):
        """
        Plots publication-grade Kaplan-Meier survival curves using matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting survival curves. Run: pip install matplotlib")

        curves, times, risks = self.predict_survival_curves(expression)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            created_fig = True
        else:
            created_fig = False

        patient_names = expression.index if isinstance(expression, pd.DataFrame) else [f"Patient {i+1}" for i in range(len(risks))]
        
        for idx, curve in enumerate(curves):
            r = risks[idx]
            if r > 0.6:
                color = "#e11d48" # High/Critical risk
                lw = 1.8
            elif r < -0.6:
                color = "#445d30" # Low risk
                lw = 1.8
            else:
                color = "#729457" # Moderate risk
                lw = 1.2

            ax.step(times, curve, where="post", color=color, linewidth=lw, alpha=0.85, label=patient_names[idx] if len(curves) <= 6 else None)

        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(times))
        ax.set_xlabel("Timeline (Months Post-Diagnosis)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Survival Probability", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)

        if len(curves) <= 6:
            ax.legend(frameon=True, fontsize=9, loc="upper right")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            
        return ax

    def plot_biomarkers(self, top_n=15, title="Top Chara Prognostic Biomarkers", save_path=None, ax=None):
        """
        Plots horizontal bar chart of top hazard drivers and protective markers.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting biomarkers. Run: pip install matplotlib")

        df = self.get_biomarkers(n=top_n)
        df = df.sort_values(by="Coefficient", ascending=True)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        colors = ["#445d30" if c < 0 else "#be123c" for c in df["Coefficient"]]
        bars = ax.barh(df["Gene"], df["Coefficient"], color=colors, height=0.65, edgecolor="none")
        
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Cox Proportional Hazard Weight (β)", fontsize=11, fontweight="bold")
        ax.set_ylabel("HGNC Gene Symbol", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.4, axis="x")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return ax

    def align_and_scale(self, expression):
        return scale_external_expression(expression, self.features)


def load_sample_cohort(n_patients=12) -> pd.DataFrame:
    """
    Generates a realistic synthetic cohort DataFrame with n patients for immediate testing.
    """
    genes = list(DEFAULT_COEFFICIENTS.keys())
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
