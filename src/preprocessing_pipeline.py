"""Leakage-safe, serializable preprocessing pipeline for Task 2 and Task 3."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from src.feature_engineering import HRFeatureEngineer
from src.hr_validation import TARGET

ID_COLUMNS = {"EmployeeNumber"}

class IQRClipper(BaseEstimator, TransformerMixin):
    """Fit IQR fences on training data and cap only continuous numeric columns.

    Low-cardinality integer ratings/codes are left untouched. Fences are learned
    solely by the training-fold pipeline and reused unchanged on test/new rows.
    """
    def __init__(self, factor: float = 1.5, min_unique: int = 11):
        self.factor = factor
        self.min_unique = min_unique

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("IQRClipper expects a 2D numeric array.")
        self.n_features_in_ = values.shape[1]
        self.lower_bounds_ = np.full(self.n_features_in_, -np.inf)
        self.upper_bounds_ = np.full(self.n_features_in_, np.inf)
        self.clipped_features_ = []
        for j in range(self.n_features_in_):
            column = values[:, j]
            finite = column[np.isfinite(column)]
            if finite.size and np.unique(finite).size >= self.min_unique:
                q1, q3 = np.quantile(finite, [0.25, 0.75])
                spread = q3 - q1
                self.lower_bounds_[j] = q1 - self.factor * spread
                self.upper_bounds_[j] = q3 + self.factor * spread
                self.clipped_features_.append(j)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} numeric columns; received shape {values.shape}.")
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "lower_bounds_")
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)

def identify_feature_types(frame: pd.DataFrame) -> dict:
    """Identify numeric/categorical columns and features excluded by policy."""
    if TARGET in frame.columns:
        raise ValueError("Attrition is the target and must be removed before preprocessing.")
    dropped = [c for c in frame.columns if c in ID_COLUMNS]
    dropped += [c for c in frame.columns if c not in dropped and frame[c].nunique(dropna=False) <= 1]
    candidates = [c for c in frame.columns if c not in dropped]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in candidates if c not in numeric]
    if not numeric and not categorical:
        raise ValueError("No usable feature columns remain after excluding IDs and constants.")
    return {"numeric": numeric, "categorical": categorical, "dropped": dropped}

def build_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """Construct an unfitted preprocessing pipeline from training schema only."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame with named columns.")
    if TARGET in X_train.columns:
        raise ValueError("Do not pass Attrition into build_preprocessor; keep the target separate.")
    feature_engineering = HRFeatureEngineer()
    training_features = feature_engineering.fit_transform(X_train)
    types = identify_feature_types(training_features)
    transformers = []
    if types["numeric"]:
        numerical = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("iqr_clip", IQRClipper()),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numerical, types["numeric"]))
    if types["categorical"]:
        categorical = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical, types["categorical"]))
    columns = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    return Pipeline(steps=[("feature_engineering", feature_engineering), ("preprocessor", columns)])

def save_preprocessor(preprocessor: Pipeline, path) -> Path:
    """Serialize a fitted pipeline; fail fast if it has not been fitted."""
    if not hasattr(preprocessor, "named_steps") or "preprocessor" not in preprocessor.named_steps:
        raise TypeError("Expected the Task 2 preprocessing Pipeline.")
    if not hasattr(preprocessor.named_steps["preprocessor"], "transformers_"):
        raise ValueError("Refusing to save an unfitted preprocessing pipeline.")
    import joblib
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, destination)
    return destination

def load_preprocessor(path) -> Pipeline:
    """Load a previously fitted Task 2 pipeline from Joblib."""
    import joblib
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Saved preprocessor not found at {source}.")
    fitted = joblib.load(source)
    if not isinstance(fitted, Pipeline) or "preprocessor" not in fitted.named_steps:
        raise TypeError(f"File at {source} is not a compatible Task 2 preprocessing pipeline.")
    return fitted
