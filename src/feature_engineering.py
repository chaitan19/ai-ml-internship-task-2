"""Stateless derived features shared by training and future unseen records."""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add meaningful career-stage and company-tenure bins without using target."""
    def fit(self, X: pd.DataFrame, y=None):
        self._validate(X)
        self.feature_names_in_ = list(X.columns)
        return self

    @staticmethod
    def _validate(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HRFeatureEngineer requires a pandas DataFrame to preserve feature names.")
        if "Attrition" in X.columns:
            raise ValueError("Target leakage prevention: pass features without the Attrition target.")
        needed = {"Age", "YearsAtCompany"}
        absent = sorted(needed - set(X.columns))
        if absent:
            raise ValueError(f"Cannot create HR derived features; missing columns: {absent}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        self._validate(X)
        result = X.copy()
        age = pd.to_numeric(result["Age"], errors="coerce")
        tenure = pd.to_numeric(result["YearsAtCompany"], errors="coerce")
        career = pd.cut(
            age, bins=[-float("inf"), 29, 44, float("inf")],
            labels=["Early (<30)", "Mid (30-44)", "Later (45+)"], right=True,
        ).astype(object)
        tenure_band = pd.cut(
            tenure, bins=[-float("inf"), 1, 5, 10, float("inf")],
            labels=["0-1 years", "2-5 years", "6-10 years", "11+ years"], right=True,
        ).astype(object)
        result["CareerStage"] = career.mask(pd.isna(career), np.nan)
        result["CompanyTenureBand"] = tenure_band.mask(pd.isna(tenure_band), np.nan)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        names = list(self.feature_names_in_ if input_features is None else input_features)
        for derived in ("CareerStage", "CompanyTenureBand"):
            if derived not in names:
                names.append(derived)
        return pd.Index(names, dtype=object).to_numpy()
