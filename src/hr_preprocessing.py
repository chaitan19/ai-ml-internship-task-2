"""End-to-end Task 2 data preparation, diagnostics, split, and artifact builder."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from src.feature_engineering import HRFeatureEngineer
from src.hr_validation import TARGET, clean_hr_data, load_employee_data, profile_data
from src.preprocessing_pipeline import (build_preprocessor, identify_feature_types,
                                        save_preprocessor)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "employee_attrition.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_DIR = PROJECT_ROOT / "images"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20

def compare_scalers(training_values) -> dict:
    """Fit StandardScaler and MinMaxScaler on training values and summarize results."""
    values = np.asarray(training_values, dtype=float).reshape(-1, 1)
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("Scaler comparison requires non-empty finite training values.")
    result = {}
    for name, transformer in (("StandardScaler", StandardScaler()), ("MinMaxScaler", MinMaxScaler())):
        transformed = transformer.fit_transform(values).ravel()
        result[name] = {"values": transformed, "mean": float(transformed.mean()),
                        "std": float(transformed.std()), "min": float(transformed.min()),
                        "max": float(transformed.max()), "fitted_rows": int(len(transformed))}
    return result

def split_employee_data(frame: pd.DataFrame, test_size: float = TEST_SIZE,
                        random_state: int = RANDOM_STATE):
    """Stratified split with target separate and schema validation."""
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    X = frame.drop(columns=[TARGET]).copy()
    y = frame[TARGET].astype(str).copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def iqr_outlier_table(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Count IQR-flagged observations per numeric feature for diagnostics."""
    numeric = frame.select_dtypes(include="number")
    if columns is not None:
        numeric = numeric[[c for c in columns if c in numeric.columns]]
    rows = []
    for col in numeric.columns:
        values = numeric[col].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([.25, .75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"feature": col, "q1": float(q1), "q3": float(q3),
                     "lower_fence": float(lower), "upper_fence": float(upper),
                     "outlier_count": int(((values < lower) | (values > upper)).sum()),
                     "outlier_pct": float(100 * ((values < lower) | (values > upper)).mean())})
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False).reset_index(drop=True)

def _save_charts(frame: pd.DataFrame, train_X: pd.DataFrame, iqr_table: pd.DataFrame) -> dict:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    files = {}
    counts = frame[TARGET].value_counts().reindex(["No", "Yes"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#4f81bd", "#d97941"], legend=False, ax=ax)
    ax.set(title="Employee Attrition Target Distribution", xlabel="Attrition", ylabel="Employees")
    for container in ax.containers: ax.bar_label(container, fmt="%d")
    fig.tight_layout(); files["attrition_target_distribution"] = IMAGE_DIR / "attrition_target_distribution.png"
    fig.savefig(files["attrition_target_distribution"], dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(data=frame, x=TARGET, y="MonthlyIncome", hue=TARGET, palette=["#4f81bd", "#d97941"], legend=False, ax=ax)
    ax.set(title="Monthly Income Distribution and IQR Flags", xlabel="Attrition", ylabel="Monthly income (dataset units)")
    fig.tight_layout(); files["monthly_income_outliers"] = IMAGE_DIR / "monthly_income_outliers.png"
    fig.savefig(files["monthly_income_outliers"], dpi=160); plt.close(fig)

    top = iqr_table.head(12).sort_values("outlier_count")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top, x="outlier_count", y="feature", color="#6b9f78", ax=ax)
    ax.set(title="IQR-Flagged Values in Numeric Features (Full Data Diagnostic)", xlabel="Count outside 1.5 × IQR fences", ylabel="Numeric feature")
    fig.tight_layout(); files["iqr_outlier_counts"] = IMAGE_DIR / "iqr_outlier_counts.png"
    fig.savefig(files["iqr_outlier_counts"], dpi=160); plt.close(fig)

    comparisons = compare_scalers(pd.to_numeric(train_X["MonthlyIncome"], errors="coerce").dropna())
    z = comparisons["StandardScaler"]["values"]
    mm = comparisons["MinMaxScaler"]["values"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.histplot(z, bins=24, kde=True, ax=axes[0], color="#4f81bd")
    axes[0].set(title="StandardScaler (train fit)", xlabel="Transformed MonthlyIncome", ylabel="Employees")
    sns.histplot(mm, bins=24, kde=True, ax=axes[1], color="#d97941")
    axes[1].set(title="MinMaxScaler (train fit)", xlabel="Transformed MonthlyIncome", ylabel="Employees")
    fig.suptitle("Scaling Comparison Using Training Data Only"); fig.tight_layout()
    files["scaler_comparison"] = IMAGE_DIR / "scaler_comparison_monthly_income.png"
    fig.savefig(files["scaler_comparison"], dpi=160); plt.close(fig)

    engineer = HRFeatureEngineer().fit(train_X)
    with_features = engineer.transform(train_X).assign(Attrition=frame.loc[train_X.index, TARGET].astype(str))
    rate = with_features.groupby("CompanyTenureBand", observed=False)[TARGET].apply(lambda values: values.eq("Yes").mean()).reset_index(name="attrition_rate")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=rate, x="CompanyTenureBand", y="attrition_rate", hue="CompanyTenureBand", legend=False, palette="dark:#8c6bb1", ax=ax)
    ax.set(title="Observed Attrition Share by Engineered Company Tenure Band", xlabel="Company tenure band", ylabel="Share marked Yes", ylim=(0, max(.05, float(rate.attrition_rate.max()) * 1.2)))
    fig.tight_layout(); files["tenure_band_attrition"] = IMAGE_DIR / "attrition_by_tenure_band.png"
    fig.savefig(files["tenure_band_attrition"], dpi=160); plt.close(fig)
    return {name: str(path.relative_to(PROJECT_ROOT)) for name, path in files.items()}

def run_preprocessing(raw_path=RAW_DATA_PATH) -> dict:
    """Run the complete workflow and persist all Task 2 reproducibility artifacts."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_employee_data(raw_path)
    before = profile_data(raw)
    clean, cleaning = clean_hr_data(raw)
    features_only = clean.drop(columns=[TARGET])
    # Save reusable derived columns for human inspection; they are also recreated
    # inside the fitted pipeline for raw unseen input records.
    engineered_preview = HRFeatureEngineer().fit_transform(features_only)
    exported = engineered_preview.copy()
    exported[TARGET] = clean[TARGET].astype(str)
    clean_path = PROCESSED_DIR / "employee_attrition_clean.csv"
    exported.to_csv(clean_path, index=False)

    X_train, X_test, y_train, y_test = split_employee_data(exported)
    engineered_train = HRFeatureEngineer().fit_transform(X_train)
    feature_types = identify_feature_types(engineered_train)
    pipeline = build_preprocessor(X_train)
    train_matrix = pipeline.fit_transform(X_train)
    test_matrix = pipeline.transform(X_test)
    model_path = save_preprocessor(pipeline, MODEL_DIR / "preprocessor.pkl")
    outlier_table = iqr_outlier_table(clean)
    outlier_table.to_csv(PROCESSED_DIR / "iqr_outlier_summary.csv", index=False)
    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    scaler_comparison = {}
    comparison_details = compare_scalers(X_train["MonthlyIncome"])
    for name, details in comparison_details.items():
        scaler_comparison[name] = {key: details[key] for key in ("mean", "std", "min", "max", "fitted_rows")}
    travel = X_train["BusinessTravel"].fillna(X_train["BusinessTravel"].mode().iloc[0]).astype(str)
    le = LabelEncoder().fit(travel)
    label_codes = le.transform(travel)
    ohe_demo = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(travel.to_frame())
    label_demo = {"feature": "BusinessTravel (same training feature compared with both encoders)",
                  "label_encoder_mapping": {str(label): int(code) for code, label in enumerate(le.classes_)},
                  "onehot_categories": [str(x) for x in ohe_demo.categories_[0]],
                  "onehot_output_columns": [str(x) for x in ohe_demo.get_feature_names_out(["BusinessTravel"])],
                  "label_codes_first_five": [int(x) for x in label_codes[:5]],
                  "onehot_first_five": ohe_demo.transform(travel.to_frame())[:5].astype(int).tolist(),
                  "note": "BusinessTravel is nominal, so its integer LabelEncoder codes create a false order. The saved pipeline uses OneHotEncoder(handle_unknown='ignore'). Attrition remains a separate target."}
    # Save the split manifest so index membership and stratification can be audited.
    manifest = {
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "stratify": TARGET,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "train_indices": [int(x) for x in X_train.index], "test_indices": [int(x) for x in X_test.index],
        "train_target_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_target_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "transformed_train_shape": list(train_matrix.shape), "transformed_test_shape": list(test_matrix.shape),
        "transformed_feature_names": transformed_names,
        "pipeline_fitted_on": "X_train only; test and unseen records are transform-only.",
    }
    (PROCESSED_DIR / "train_test_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    charts = _save_charts(clean, X_train, outlier_table)
    summary = {
        "dataset_source": "IBM HR Analytics Employee Attrition & Performance (Kaggle dataset mirrored on GitHub)",
        "raw_profile": before, "cleaning": cleaning,
        "clean_profile": profile_data(clean), "target": TARGET,
        "target_percentages": {str(k): float(v) for k, v in (clean[TARGET].astype(str).value_counts(normalize=True) * 100).items()},
        "feature_types_after_engineering": feature_types,
        "engineered_features": {
            "CareerStage": "Age bands: Early (<30), Mid (30-44), Later (45+).",
            "CompanyTenureBand": "YearsAtCompany bins: 0-1, 2-5, 6-10, 11+ years.",
        },
        "scaler_comparison_monthly_income_train_only": scaler_comparison,
        "label_encoding_demonstration": label_demo,
        "iqr_outlier_rows": outlier_table.to_dict(orient="records"),
        "pipeline": {
            "numeric_features": feature_types["numeric"], "categorical_features": feature_types["categorical"],
            "dropped_id_and_constant_features": feature_types["dropped"],
            "numeric_steps": ["median imputation", "train-fitted IQR capping for continuous columns", "StandardScaler"],
            "categorical_steps": ["most-frequent imputation", "OneHotEncoder(handle_unknown='ignore')"],
            "output_feature_count": int(train_matrix.shape[1]),
            "serialization": str(model_path.relative_to(PROJECT_ROOT)),
        },
        "split": manifest, "charts": charts,
        "limitations": [
            "The IBM HR dataset is a fictional, static sample, not representative employee census data.",
            "Attrition is retained as a separate target; this task prepares data and does not train a predictor.",
            "IQR flags are screening signals; extreme values are capped only inside the train-fitted numeric pipeline, not deleted from the clean source table.",
            "Standard scaling is the pipeline default; scaler choice should be revisited for each Task 3 model and validation design.",
        ],
    }
    # Keep the transform-only dataset schema and fitted behavior together.
    (REPORT_DIR / "preprocessing_results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"raw_shape": raw.shape, "clean_shape": clean.shape,
                      "missing": before["missing_cells"], "duplicates_removed": cleaning["duplicate_rows_removed"],
                      "train_rows": len(X_train), "test_rows": len(X_test),
                      "transformed_train_shape": train_matrix.shape,
                      "model": str(model_path), "charts": list(charts.values())}, indent=2))
    return summary

if __name__ == "__main__":
    run_preprocessing()
