from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.feature_engineering import HRFeatureEngineer
from src.hr_preprocessing import split_employee_data
from src.hr_validation import clean_hr_data, load_employee_data
from src.preprocessing_pipeline import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]


def test_real_dataset_load_clean_and_quality():
    raw = load_employee_data(ROOT / "data/raw/employee_attrition.csv")
    clean, log = clean_hr_data(raw)
    assert raw.shape == (1470, 35)
    assert clean.shape == raw.shape
    assert log["duplicate_rows_removed"] == 0
    assert clean["Attrition"].astype(str).isin(["Yes", "No"]).all()


def test_cleaning_drops_duplicates_and_leaves_features_for_pipeline_imputation():
    frame = pd.DataFrame({"Attrition": [" yes ", "No", "No"], "Age": [20, 30, 30],
                          "YearsAtCompany": [1, 2, 2], "JobLevel": [1, 2, 2],
                          "MonthlyIncome": [np.nan, 2000, 2000], "Department": [" R&D ", "Sales", "Sales"]})
    clean, log = clean_hr_data(frame)
    assert len(clean) == 2 and log["duplicate_rows_removed"] == 1
    assert clean["MonthlyIncome"].isna().sum() == 1
    assert clean.loc[0, "Department"] == "R&D"
    assert clean["Attrition"].astype(str).tolist() == ["Yes", "No"]


def test_engineering_and_target_leakage_guard():
    X = pd.DataFrame({"Age": [25, 39, 51], "YearsAtCompany": [1, 5, 12]})
    transformed = HRFeatureEngineer().fit_transform(X)
    assert transformed["CareerStage"].tolist() == ["Early (<30)", "Mid (30-44)", "Later (45+)"]
    assert transformed["CompanyTenureBand"].tolist() == ["0-1 years", "2-5 years", "11+ years"]
    with pytest.raises(ValueError, match="Target leakage"):
        HRFeatureEngineer().fit(X.assign(Attrition="No"))


def test_split_is_reproducible_and_stratified():
    frame = load_employee_data(ROOT / "data/raw/employee_attrition.csv")
    first = split_employee_data(frame)
    second = split_employee_data(frame)
    assert first[0].index.tolist() == second[0].index.tolist()
    assert first[1].index.tolist() == second[1].index.tolist()
    assert "Attrition" not in first[0].columns
    assert first[2].value_counts().to_dict() == {"No": 986, "Yes": 190}
    assert first[3].value_counts().to_dict() == {"No": 247, "Yes": 47}


def test_scaling_encoding_unknown_categories_and_joblib_roundtrip(tmp_path):
    clean, _ = clean_hr_data(load_employee_data(ROOT / "data/raw/employee_attrition.csv"))
    X_train, X_test, y_train, y_test = split_employee_data(clean)
    pipe = build_preprocessor(X_train).fit(X_train)
    transformed = pipe.transform(X_test)
    assert transformed.shape == (294, 58)
    assert np.isfinite(transformed).all()
    unknown = X_test.iloc[[0]].copy()
    unknown.loc[:, "BusinessTravel"] = "Unseen"
    unknown.loc[:, "JobRole"] = "New Role"
    assert np.isfinite(pipe.transform(unknown)).all()
    path = tmp_path / "preprocessor.pkl"
    joblib.dump(pipe, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(transformed, restored.transform(X_test))


def test_generated_task2_artifacts_exist_and_reload_equal():
    processed = ROOT / "data/processed"
    model = joblib.load(ROOT / "models/preprocessor.pkl")
    clean = pd.read_csv(processed / "employee_attrition_clean.csv")
    manifest = json.loads((processed / "train_test_split_manifest.json").read_text(encoding="utf-8"))
    assert clean.shape == (1470, 37)
    assert manifest["train_rows"] == 1176 and manifest["test_rows"] == 294
    assert model.transform(clean.drop(columns="Attrition").iloc[:3]).shape == (3, 58)
    for image in ["attrition_target_distribution.png", "monthly_income_outliers.png",
                  "iqr_outlier_counts.png", "scaler_comparison_monthly_income.png",
                  "attrition_by_tenure_band.png"]:
        assert (ROOT / "images" / image).stat().st_size > 1000
