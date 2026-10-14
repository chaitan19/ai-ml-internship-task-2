# AI/ML Internship - Task 2

Data preprocessing, feature engineering, and a reusable scikit-learn pipeline for the IBM HR Analytics Employee Attrition sample. This repository contains Task 2 deliverables only. It prepares data for Task 3 and does **not** train an attrition prediction model.

## Results at a glance

- Source data: 1,470 rows and 35 columns; 0 missing cells and 0 exact duplicates were found in this source copy.
- Cleaning removed 0 rows; text was trimmed and `Attrition` labels normalized. The clean export adds two derived columns (`CareerStage`, `CompanyTenureBand`) and keeps `Attrition` as a separate target.
- Reproducible split: 1,176 training and 294 test rows, 80/20, stratified by `Attrition`, `random_state=42`.
- The saved preprocessor uses train-fitted median imputation, IQR clipping of continuous numeric features, StandardScaler, most-frequent categorical imputation, and OneHotEncoder with unknown-category handling.
- The trained preprocessing artifact has 58 transformed feature columns. There is no predictive estimator in the artifact.

### Split assumption

The assignment's practical section lists training, validation, and test sets as concepts, while its Step 6 and final checklist direct a training/test split. This implementation follows the explicit train/test instructions and defers a separate validation split to Task 3 model selection. The test partition is held out and is never used to fit preprocessing.

## Dataset and attribution

The included `data/raw/employee_attrition.csv` is the IBM HR Analytics Employee Attrition & Performance sample listed on [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). The file was retrieved from a public [GitHub mirror](https://github.com/IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance/blob/main/WA_Fn-UseC_-HR-Employee-Attrition.csv) of that sample. The Kaggle listing identifies its license as **Database: Open Database; Contents: Database Contents**. The project includes attribution and keeps the raw file separate from the processed export. The data are a fictional, static sample and do not represent actual IBM employees or a representative workforce. If redistributing the dataset, follow the terms shown on the source listing and any applicable database/content license conditions.

## Requirements

- Windows 10/11 (PowerShell or Command Prompt)
- Python 3.12 or later; the current project was run with Python 3.13.7
- Git (optional, for version control)
- VS Code with Python and Jupyter extensions is recommended

## Setup (PowerShell)

Open PowerShell in the repository root (`AI-ML-Internship-Task-2`):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, invoke the project interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Command Prompt activation: `.venv\Scripts\activate.bat`.

## Run the workflow

From the repository root with `.venv` active:

```powershell
python -m src.hr_preprocessing
python -m src.create_task2_notebook
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_preprocessing.ipynb
python -m src.generate_preprocessing_report
python -m pytest -q
```

The first command reads the raw CSV, validates and cleans it, creates the fixed stratified split, fits the reusable preprocessing pipeline on training features only, transforms test features with that fitted pipeline, and regenerates the cleaned CSV, manifest, IQR summary, charts, fitted Joblib object, and actual results JSON. The notebook can also be opened in VS Code and run top to bottom using the `.venv` kernel. The report command rebuilds the Markdown and PDF report from the generated results and chart images.

## Reuse the preprocessor

`models/preprocessor.pkl` is a fitted scikit-learn `Pipeline` that accepts compatible feature DataFrames without `Attrition`. It adds the derived fields internally. The fitted one-hot encoder ignores unseen category values. For example:

```python
import joblib
import pandas as pd

preprocessor = joblib.load("models/preprocessor.pkl")
new_hr_features = pd.read_csv("new_hr_features.csv")  # omit Attrition
transformed = preprocessor.transform(new_hr_features)
print(transformed.shape)
```

Keep the feature names and scikit-learn/joblib versions compatible with the environment that created the pickle. For Task 3, place this preprocessing pipeline inside the model-validation pipeline so each validation fold fits its transformations only on that fold's training data.

## Contents

```text
data/
  raw/employee_attrition.csv
  processed/employee_attrition_clean.csv
  processed/iqr_outlier_summary.csv
  processed/train_test_split_manifest.json
images/                         five generated Task 2 charts
models/preprocessor.pkl         fitted feature-preprocessing pipeline
notebooks/02_data_preprocessing.ipynb
reports/Preprocessing_Report.md
reports/Preprocessing_Report.pdf
reports/preprocessing_results.json
src/                            validation, cleaning, features, pipeline, report
tests/test_task2_preprocessing.py
```

## Method decisions and limitations

- Missing feature values are left for imputers fitted inside the training pipeline (numeric median; categorical most frequent). The included source has no missing cells. Rows without a usable target label would be removed; invalid target values raise an error.
- Exact duplicate rows are removed; none occurred in this copy. `EmployeeNumber` is excluded as an identifier.
- `CareerStage` bins age and `CompanyTenureBand` bins years at the company. Bins are deterministic and do not use the target.
- LabelEncoder and OneHotEncoder are compared on the same nominal `BusinessTravel` feature. LabelEncoder codes imply a false order, so the reusable pipeline uses OneHotEncoder with `handle_unknown="ignore"`.
- StandardScaler and MinMaxScaler are compared on training-set `MonthlyIncome`. The pipeline uses StandardScaler as a general starting point; model choice may justify another strategy.
- IQR counts are diagnostic. The clean export retains records; the pipeline fits IQR fences on training data and caps continuous values only. IQR flags do not establish that observations are errors.
- The sample is fictional and static. Descriptive group differences and correlations are not causal evidence and should not be generalized to a real workforce.

## Assignment checklist

See [`reports/acceptance_checklist.md`](reports/acceptance_checklist.md) for the requirement-by-requirement evidence and the screenshot status. Actual chart PNGs are in `images/`; the notebook contains executed cell outputs. Separate screenshots of the notebook in a desktop application are not included in this repository copy.

## GitHub

This directory is an independent Git repository. Review `git status` and the commit history before connecting it to the new remote. Do not put access tokens, passwords, `.env` files, or a virtual environment in Git. Add the new repository URL as `origin` only after verifying that it is the Task 2 repository.
