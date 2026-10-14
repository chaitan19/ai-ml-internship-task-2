# Task 2 acceptance checklist

Evidence is based on the included source, generated artifacts, executed notebook, and tests in this repository.

| Requirement | Status | Evidence |
|---|---|---|
| IBM HR Attrition dataset loaded | PASS | Included 1,470 × 35 source CSV; schema validation in `src/hr_validation.py`. |
| Data inspection and types | PASS | Notebook inspects dimensions, columns, dtypes, rows, target counts, and quality. |
| Missing and duplicate detection | PASS | Real source: 0 missing cells, 0 duplicate rows; recorded in the generated report. |
| Cleaning and type handling | PASS | Trimming, target normalization, numeric-string conversion, duplicate removal, and target validation implemented. |
| Missing-value strategy | PASS | Numeric median and categorical mode imputers are fitted inside the pipeline; no full-data imputation. |
| Feature engineering | PASS | `CareerStage` and `CompanyTenureBand` created from source columns and recreated in the pipeline. |
| Label versus One-Hot Encoding | PASS | Demonstrated on the same `BusinessTravel` feature; pipeline uses OneHotEncoder with unknown-category support. |
| StandardScaler versus MinMaxScaler | PASS | Training-only comparison; StandardScaler used by the reusable pipeline. |
| Outlier detection and handling | PASS | Box plot, IQR counts, diagnostic summary, and train-fitted IQR capping for continuous numeric columns. |
| Reproducible train-test split | PASS | 80/20, `random_state=42`, stratified target; 1,176 train and 294 test rows in manifest. |
| Validation-set discussion | PASS | The PDF lists validation as a concept but Step 6 and the final checklist specify train/test. The README documents following the explicit split instruction and deferring validation to Task 3. |
| Data leakage prevention | PASS | `Attrition` is kept outside X; fit on training data, transform held-out test data. |
| Pipeline and ColumnTransformer | PASS | Fitted scikit-learn Pipeline/ColumnTransformer saved in `models/preprocessor.pkl`. |
| Joblib reload and unseen data | PASS | Covered in notebook and tests; unknown categorical values transform successfully. |
| Required notebook, code, clean dataset, model, report, README | PASS | All listed artifacts are present; notebook executed all 11 code cells without error. |
| Required visualizations | PASS | Five actual generated Task 2 PNG charts in `images/` and embedded in the report. |
| Assignment screenshots | PARTIAL | Chart exports and executed notebook outputs are included. Separate desktop application screenshots are not included. |
| GitHub repository and remote push | PASS | Separate Task 2-only repository pushed to `https://github.com/chaitan19/ai-ml-internship-task-2` on `main`; remote commit verified. |
