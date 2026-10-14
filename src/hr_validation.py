"""Schema validation, data-quality profiling, and conservative HR data cleaning."""
from __future__ import annotations
import pandas as pd
import numpy as np

TARGET = "Attrition"
REQUIRED_COLUMNS = {"Attrition", "Age", "YearsAtCompany", "JobLevel", "MonthlyIncome"}

def validate_hr_schema(frame: pd.DataFrame) -> None:
    """Raise actionable errors for an incompatible HR table or target values."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for employee attrition data.")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"HR dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("HR dataset contains no employee rows.")

def profile_data(frame: pd.DataFrame) -> dict:
    """Return a JSON-compatible snapshot of data quality before cleaning."""
    return {
        "rows": int(frame.shape[0]), "columns": int(frame.shape[1]),
        "column_names": frame.columns.astype(str).tolist(),
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
        "missing_by_column": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "target_counts": ({str(k): int(v) for k, v in frame[TARGET].value_counts(dropna=False).items()}
                          if TARGET in frame else {}),
    }

def clean_hr_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize strings/types and remove exact duplicates without fitting imputers.

    Missing feature values are intentionally left for the train-fitted sklearn
    imputer. Rows with a missing target cannot be used for supervised Task 3 and
    are dropped; unknown target labels raise a clear error.
    """
    validate_hr_schema(frame)
    before = profile_data(frame)
    cleaned = frame.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        values = cleaned[col].astype("string").str.strip().replace("", pd.NA)
        cleaned[col] = values.astype(object).mask(values.isna(), np.nan)
    target_text = cleaned[TARGET].astype("string").str.strip().str.casefold()
    target_map = {"yes": "Yes", "no": "No"}
    target_missing = int(target_text.isna().sum())
    cleaned = cleaned.loc[target_text.notna()].copy()
    target_text = target_text.loc[cleaned.index]
    invalid = sorted(set(target_text.dropna().unique()) - set(target_map))
    if invalid:
        raise ValueError(f"Unexpected Attrition labels {invalid}; expected Yes or No.")
    cleaned[TARGET] = target_text.map(target_map).astype("category")
    # Convert numeric strings only when every observed non-empty value parses.
    for col in cleaned.columns:
        if col == TARGET:
            continue
        if cleaned[col].dtype == "object" or isinstance(cleaned[col].dtype, pd.StringDtype):
            parsed = pd.to_numeric(cleaned[col], errors="coerce")
            observed = cleaned[col].notna()
            if observed.any() and parsed.loc[observed].notna().all():
                cleaned[col] = parsed
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    after = profile_data(cleaned)
    metadata = {
        "rows_before": before["rows"], "columns_before": before["columns"],
        "missing_cells_before": before["missing_cells"],
        "duplicate_rows_before": before["duplicate_rows"],
        "duplicate_rows_removed": duplicate_count,
        "rows_with_missing_target_removed": target_missing,
        "rows_after": int(len(cleaned)), "missing_cells_after": after["missing_cells"],
        "categorical_values_trimmed": True,
        "numeric_string_columns_converted": [c for c in cleaned.columns if c != TARGET and pd.api.types.is_numeric_dtype(cleaned[c])],
        "imputation": "No full-data imputation; numerical medians and categorical modes are learned from training rows inside the pipeline.",
        "target_handling": "Attrition retained separately as Yes/No and is never passed into feature transformations.",
    }
    return cleaned, metadata

def load_employee_data(path) -> pd.DataFrame:
    """Load the UTF-8 CSV with an informative missing-file error."""
    from pathlib import Path
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Employee attrition CSV not found at {source}. See README.md for dataset recovery instructions.")
    frame = pd.read_csv(source)
    validate_hr_schema(frame)
    return frame
