"""Build an evidence-based Task 2 Markdown and PDF report from run artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RESULTS_PATH = REPORTS / "preprocessing_results.json"
MD_PATH = REPORTS / "Preprocessing_Report.md"
PDF_PATH = REPORTS / "Preprocessing_Report.pdf"


def build_report() -> tuple[Path, Path]:
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError("Run `python -m src.hr_preprocessing` before generating the report.")
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    cleaning, raw, clean = data["cleaning"], data["raw_profile"], data["clean_profile"]
    split, pipe = data["split"], data["pipeline"]
    yes = data["target_percentages"].get("Yes", 0.0)
    no = data["target_percentages"].get("No", 0.0)
    counts = data["iqr_outlier_rows"]
    top_outliers = sorted(counts, key=lambda row: row["outlier_count"], reverse=True)[:8]
    md = f'''# Data Preprocessing, Feature Engineering & Machine Learning Pipeline

## Project overview
This Task 2 project prepares the IBM HR Analytics Employee Attrition & Performance sample for later modeling. It validates and cleans records, performs feature engineering, demonstrates encoding and scaling, diagnoses outliers, and builds a reusable scikit-learn preprocessor. **No predictive model is trained.** `Attrition` remains a separate target and is excluded from every feature transformation.

## Dataset and source
The source CSV contains {raw['rows']:,} employee records and {raw['columns']} columns. The dataset is a fictional IBM HR sample distributed through [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), with a repository mirror used to obtain the CSV. It is a fixed learning dataset, not a representative real-world workforce census.

## Data-quality assessment and cleaning
Before cleaning: {raw['rows']:,} rows × {raw['columns']} columns; {raw['missing_cells']} missing cells; {raw['duplicate_rows']} exact duplicate rows. The cleaning routine trims text, canonicalizes target values, converts numeric strings only when all observed values parse, removes {cleaning['duplicate_rows_removed']} exact duplicate rows, and drops {cleaning['rows_with_missing_target_removed']} rows with missing target labels. Result: {cleaning['rows_after']:,} rows × {clean['columns']} columns and {clean['missing_cells']} missing cells. No feature imputation was necessary on this source. The reusable pipeline nevertheless includes numeric median and categorical most-frequent imputers, learned on training data only.

## Target and split
The target balance is {no:.2f}% `No` and {yes:.2f}% `Yes`. The reproducible split uses {split['train_rows']:,} training rows and {split['test_rows']:,} test rows (80/20, random state {split['random_state']}, stratified by `Attrition`). Class counts: train {split['train_target_counts']}; test {split['test_target_counts']}.

## Feature engineering, encoding, and scaling
Two deterministic features are added: `CareerStage` (age bands) and `CompanyTenureBand` (company-tenure bands). They use only row values and can be recreated for future records. `EmployeeNumber` is excluded as an identifier. The pipeline identifies {len(pipe['numeric_features'])} numeric and {len(pipe['categorical_features'])} categorical inputs after feature engineering, excluding {pipe['dropped_id_and_constant_features']}.

The encoding exercise compares LabelEncoder and OneHotEncoder on the same nominal `BusinessTravel` training feature. Label codes impose an artificial order, so the production preprocessor uses OneHotEncoder with `handle_unknown='ignore'`. The target is kept separate. StandardScaler and MinMaxScaler are compared on training-set `MonthlyIncome`; the saved numeric pipeline uses median imputation, train-fitted IQR capping for continuous values, and StandardScaler. Categorical inputs use most-frequent imputation and one-hot encoding.

## Outlier assessment
The 1.5×IQR diagnostic flags values for review; it does not establish that a value is erroneous. The table below is computed from the real source values. The cleaned CSV retains all eligible rows. In the fitted reusable transformation, IQR fences are learned from training rows and cap continuous columns; low-cardinality ratings/codes are not capped.

| Numeric feature | IQR-flagged rows | Percentage |
|---|---:|---:|
''' + "\n".join(f"| {r['feature']} | {r['outlier_count']} | {r['outlier_pct']:.2f}% |" for r in top_outliers) + f'''

## Reusable, leakage-safe preprocessing
The `Pipeline` and `ColumnTransformer` are fitted using `X_train` only. The held-out `X_test` is transformed using the fitted object without refitting. The transformed training matrix has shape {split['transformed_train_shape']}; the test matrix has shape {split['transformed_test_shape']}. Unknown categorical labels are safely ignored as all-zero indicators within their encoded field. The fitted artifact is `models/preprocessor.pkl`, serialized with Joblib. Reload equivalence and unseen-category transformation are verified in the notebook and tests.

## Findings and interpretation
- The observed class imbalance is substantial ({yes:.2f}% attrition marked Yes), so Task 3 evaluation should use suitable stratified validation and metrics that do not rely on accuracy alone.
- The plot of observed attrition shares across tenure bands is descriptive. Differences in this sample do not establish that tenure causes attrition.
- Numeric IQR flags are screening indicators and may reflect valid extreme values; records were not removed based solely on an outlier rule.
- Scaling changes feature ranges, not the underlying employee records. StandardScaler is a reusable starting point; the final choice depends on the Task 3 estimator and validation.

## Visualizations
'''
    chart_captions = [
        ("attrition_target_distribution.png", "Figure 1. Counts of the Attrition target classes."),
        ("monthly_income_outliers.png", "Figure 2. Monthly income distributions by target class; points beyond whiskers are potential outliers."),
        ("iqr_outlier_counts.png", "Figure 3. Numeric features with the highest 1.5×IQR flag counts."),
        ("scaler_comparison_monthly_income.png", "Figure 4. Training-only StandardScaler and MinMaxScaler transformations for MonthlyIncome."),
        ("attrition_by_tenure_band.png", "Figure 5. Observed attrition share by engineered company-tenure band."),
    ]
    md += "\n".join(f"\n### {caption}\n\n![{caption}](../images/{filename})\n" for filename, caption in chart_captions)
    md += '''
## Limitations and conclusion
This dataset is fictional, cross-sectional, and static; it may not represent actual employees, contemporary workplaces, or other organizations. Associations are descriptive and do not establish causation. The IQR rule is a conventional screening choice, and the clipping policy should be evaluated with the later model. The artifact prepares features only; no predictive model is fitted. In Task 3, fit the preprocessor within each cross-validation/model-selection fold to keep validation data from influencing learned transformations.

## Reproducibility and references
- Raw CSV: `data/raw/employee_attrition.csv`; cleaned export: `data/processed/employee_attrition_clean.csv`.
- Split membership and class counts: `data/processed/train_test_split_manifest.json`.
- Fitted pipeline: `models/preprocessor.pkl`.
- Source: [IBM HR Analytics Employee Attrition & Performance dataset listing](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). Data are presented as a fictional IBM sample; this analysis makes no claims about IBM employees.
- Software: pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib.
'''
    MD_PATH.write_text(md, encoding="utf-8")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#17365D"), spaceAfter=12))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.5,
                              leading=11, textColor=colors.HexColor("#455468"), spaceAfter=10))
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter, rightMargin=0.65*inch,
                            leftMargin=0.65*inch, topMargin=0.62*inch, bottomMargin=0.62*inch,
                            title="Task 2 Data Preprocessing Report", author="AI/ML Internship Project")
    story = [Paragraph("Data Preprocessing, Feature Engineering &amp; ML Pipeline", styles["ReportTitle"]),
             Paragraph("AI/ML Internship — Task 2 | Reproducible employee attrition data preparation", styles["Heading2"]),
             Paragraph("Project overview", styles["Heading1"]),
             Paragraph("This report documents real, computed preprocessing results for the IBM HR Analytics Employee Attrition sample. The work prepares a reusable pipeline for a later task; it does not train a classifier. Attrition stays separate from transformed features.", styles["BodyText"]), Spacer(1, 10),
             Paragraph("Dataset and data quality", styles["Heading1"]),
             Paragraph(f"Source sample: {raw['rows']:,} rows × {raw['columns']} fields. Missing cells: {raw['missing_cells']}; exact duplicate rows: {raw['duplicate_rows']}. After cleaning: {cleaning['rows_after']:,} rows, {clean['missing_cells']} missing cells, {cleaning['duplicate_rows_removed']} duplicates removed, and {cleaning['rows_with_missing_target_removed']} missing-target rows removed. Text was trimmed and target labels normalized. Feature imputation remains in the train-fitted pipeline.", styles["BodyText"]), Spacer(1, 8),
             Paragraph(f"Target: No {no:.2f}% / Yes {yes:.2f}%. Split: {split['train_rows']} train / {split['test_rows']} test, stratified, random_state={split['random_state']}. No transformation is fitted on the test partition.", styles["BodyText"]),
             Paragraph("Feature engineering and pipeline", styles["Heading1"]),
             Paragraph("CareerStage and CompanyTenureBand are derived from Age and YearsAtCompany. EmployeeNumber is omitted as an identifier. The fitted Pipeline uses median numeric imputation, training-only IQR capping for continuous variables, StandardScaler, categorical mode imputation, and OneHotEncoder(handle_unknown='ignore'). Encoders, imputers, scaling, and IQR fences are learned from X_train only. The cleaned table retains all rows; outlier flags do not cause row deletion.", styles["BodyText"]),
             Paragraph("Outlier summary (1.5 × IQR)", styles["Heading2"])]
    table_data = [["Feature", "Flagged", "Share"]] + [[r["feature"], str(r["outlier_count"]), f"{r['outlier_pct']:.1f}%"] for r in top_outliers]
    table = Table(table_data, colWidths=[3.4*inch, 1.0*inch, 1.0*inch], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
                               ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#B7C9D6")),
                               ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EFF4F8")]),
                               ("FONTSIZE", (0,0), (-1,-1), 8), ("PADDING", (0,0), (-1,-1), 5)]))
    story += [table, Spacer(1, 12), Paragraph("Observed findings and caveats", styles["Heading1"]),
              Paragraph(f"The sample has {yes:.2f}% records labelled Yes and {no:.2f}% labelled No, indicating class imbalance. The tenure-band chart reports descriptive shares only. IQR flags identify unusual values, not errors; rows are retained. The dataset is a fictional, static sample and cannot support causal or population-wide conclusions. Correlation or group differences do not establish causation.", styles["BodyText"]),
              PageBreak(), Paragraph("Charts", styles["Heading1"])]
    for filename, caption in chart_captions:
        path = ROOT / "images" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required report chart is missing: {path}")
        story.append(KeepTogether([Image(str(path), width=6.9*inch, height=3.55*inch, kind="proportional"), Paragraph(escape(caption), styles["Caption"])]))
    story += [PageBreak(), Paragraph("Reproducibility, interpretation, and conclusion", styles["Heading1"]),
              Paragraph(f"The stratified split uses random_state={split['random_state']} and test_size={split['test_size']}. The train matrix shape is {split['transformed_train_shape']} and test matrix shape is {split['transformed_test_shape']}. LabelEncoder and OneHotEncoder are compared on the same nominal BusinessTravel feature; the reusable pipeline uses OneHotEncoder because integer label codes would imply a false order. StandardScaler and MinMaxScaler are compared using training rows. The persisted Joblib transformer is reusable for compatible records and ignores categories unseen during fit.", styles["BodyText"]), Spacer(1, 8),
              Paragraph("The output is a preprocessing artifact, not a trained model. In the next task, fit this preprocessing inside the model-validation pipeline so that each validation fold learns imputation, capping, encoding, and scaling only from its training fold. Revisit clipping/scaling choices for the selected estimator.", styles["BodyText"]),
              Paragraph("Dataset attribution and references", styles["Heading1"]),
              Paragraph("IBM HR Analytics Employee Attrition &amp; Performance dataset listing: Kaggle (fictional IBM sample), https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset. A public GitHub mirror provided the CSV used for this reproducible project. Analysis libraries: pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and Joblib.", styles["BodyText"])]
    doc.build(story)
    return MD_PATH, PDF_PATH


if __name__ == "__main__":
    for output in build_report():
        print(output)
