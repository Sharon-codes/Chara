"""Frozen Chara Coxnet inference and strict feature alignment."""
import joblib
import numpy as np
from .preprocessing import scale_external_expression

class CharaModel:
    def __init__(self, model, features, alpha_index=None):
        self.model, self.features = model, [str(x) for x in features]
        self.alpha_index = alpha_index

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise ValueError("Expected a model bundle containing model and features.")
        return cls(payload["model"], payload["features"], payload.get("alpha_index"))

    def predict(self, expression):
        x, aligned, scaler = self.align_and_scale(expression)
        coefficients = np.asarray(self.model.coef_, dtype=float)
        valid = np.flatnonzero(np.sum(coefficients != 0, axis=0))
        if len(valid) == 0:
            raise ValueError("All Chara coefficient paths are zero.")
        index = self.alpha_index if self.alpha_index in valid else int(valid[-1])
        return x @ coefficients[:, index], x, aligned, scaler, index

    def align_and_scale(self, expression):
        return scale_external_expression(expression, self.features)

    def survival_curves(self, x, alpha_index):
        return self.model.predict_survival_function(
            x, alpha=float(self.model.alphas_[alpha_index]), return_array=True
        ), self.model.unique_times_
