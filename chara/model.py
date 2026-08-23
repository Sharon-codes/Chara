"""Frozen Chara Coxnet inference, feature alignment, and Hugging Face integration."""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .preprocessing import scale_external_expression

HF_REPO_ID = "SharonMelhi/chara-survival"
DEFAULT_MODEL_FILENAME = "chara_model_4337.pkl"

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
            # Check local directory first
            local_candidates = [
                Path("chara_model_4337.pkl"),
                Path(__file__).resolve().parent.parent / "chara_model_4337.pkl",
                Path(__file__).resolve().parent.parent / "chara_model.pkl"
            ]
            for cand in local_candidates:
                if cand.exists():
                    return cls._load_file(cand)

            # Fallback to Hugging Face Hub
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
        Generates Kaplan-Meier survival curves S_i(t) for each patient.
        """
        risk, x, aligned, idx = self.predict_risk(expression)
        if times is None:
            times = np.arange(0, 61, 1.0)
            
        if self.baseline_survival is not None and len(self.baseline_survival) >= len(times):
            s0 = np.array(self.baseline_survival[:len(times)], dtype=float)
        else:
            # Standard baseline survival fallback
            s0 = np.exp(-0.015 * times ** 1.1)

        hazard_mult = np.exp(risk * 0.85)[:, np.newaxis]
        curves = np.power(s0[np.newaxis, :], hazard_mult)
        return curves, times, risk

    def align_and_scale(self, expression):
        return scale_external_expression(expression, self.features)
