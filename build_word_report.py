"""
Generate the Word report for the Employee Attrition project.

Run from the project root:
    python build_word_report.py
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from sklearn.metrics import classification_report

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_attrition_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "YourName_Employee_Attrition_Report.docx"


def load_dataset(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    # Match the AgeGroup binning used during training in the notebook.
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["<25", "25-35", "35-45", "45-55", "55+"],
        right=False,
    )
    return df


def build_model_results(pipe, df: pd.DataFrame):
    df = add_age_group(df)
    drop = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
        "Attrition",
    ]
    df_features = df.copy()
    df_features = df_features.drop(columns=[c for c in drop if c in df_features.columns])
    missing = [c for c in pipe.feature_names_in_ if c not in df_features.columns]
    if missing:
        raise KeyError(f"Saved model expects columns that are not in the current dataset: {missing}")
    X = df_features[pipe.feature_names_in_].copy()

    y_true = df["Attrition"].map({"No": 0, "Yes": 1}).astype(int)
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]

    report = classification_report(
        y_true,
        y_pred,
        target_names=["No (stayed)", "Yes (left)"],
        zero_division=0,
        output_dict=True,
    )

    df_out = df.copy()
    df_out["Predicted_Attrition"] = np.where(y_pred == 1, "Yes", "No")
    df_out["Predicted_Prob_Leave"] = np.round(y_prob, 4)

    return df_out, report, y_true, y_pred, y_prob


def make_word_report(path: pathlib.Path, content_sections: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()

    doc.add_paragraph()
    doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "Employee Attrition Analysis\n& Prediction Using AI & Machine Learning"
    )
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run("AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026")
    r2.font.size = Pt(14)
    r2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()
    doc.add_paragraph()
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run("Author: [YourName]\nDate: October 10, 2026\nStatus: Executed\n")

    doc.add_page_break()

    for heading, body in content_sections.items():
        doc.add_heading(heading, level=1)
        if body:
            doc.add_paragraph(body)

    doc.save(str(path))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = load_dataset()
    pipe = joblib.load(MODEL_PATH)
    df_pred, report, y_true, y_pred, y_prob = build_model_results(pipe, df)

    n_total = len(df_pred)
    n_leavers = int((df_pred["Attrition"] == "Yes").sum())
    n_predicted_leavers = int((df_pred["Predicted_Attrition"] == "Yes").sum())
    match_rate = float((df_pred["Predicted_Attrition"] == df_pred["Attrition"]).mean())

    dept = (
        df_pred.groupby("Department")
        .agg(
            employees=("Attrition", "size"),
            actual_leavers=("Attrition", lambda s: (s == "Yes").sum()),
            predicted_leavers=("Predicted_Attrition", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    dept["actual_rate"] = dept["actual_leavers"] / dept["employees"]
    dept["predicted_rate"] = dept["predicted_leavers"] / dept["employees"]
    dept = dept.sort_values("employees", ascending=False)

    role = (
        df_pred.groupby("JobRole")
        .agg(
            employees=("Attrition", "size"),
            actual_leavers=("Attrition", lambda s: (s == "Yes").sum()),
            predicted_leavers=("Predicted_Attrition", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    role = role.sort_values("actual_leavers", ascending=False)
    role["actual_rate"] = role["actual_leavers"] / role["employees"]
    role["predicted_rate"] = role["predicted_leavers"] / role["employees"]

    overtime = (
        df_pred.groupby("OverTime")
        .agg(
            employees=("Attrition", "size"),
            actual_leavers=("Attrition", lambda s: (s == "Yes").sum()),
            predicted_leavers=("Predicted_Attrition", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    overtime["actual_rate"] = overtime["actual_leavers"] / overtime["employees"]
    overtime["predicted_rate"] = overtime["predicted_leavers"] / overtime["employees"]

    sections = {}

    sections["1. Executive Summary"] = (
        f"This report summarizes an employee attrition analysis and prediction project "
        f"built on a fictional HR dataset of {n_total} employee records.\n\n"
        f"Key findings:\n"
        f"- Actual attrition rate: {n_leavers} out of {n_total} employees "
        f"({n_leavers / n_total * 100:.2f}%).\n"
        f"- Best model chosen in the notebook: RandomForest.\n"
        f"- On the executed data, the saved model predicted {n_predicted_leavers} employees "
        f"as likely to leave.\n"
        f"- Prediction match rate on the executed data: {match_rate * 100:.2f}%.\n\n"
        f"This report is educational. It is based on patterns in historical, fictional data "
        f"and must not be used as the sole basis for decisions about real people."
    )

    sections["2. Dataset Overview"] = (
        f"The dataset contains {n_total} employee records and 35 columns, including "
        f"demographic, compensation, role, satisfaction, and tenure fields.\n\n"
        f"Target variable: Attrition with Yes/No labels.\n\n"
        f"Class distribution:\n"
        f"- Stayed (No): {n_total - n_leavers}\n"
        f"- Left (Yes): {n_leavers}\n\n"
        f"The dataset is imbalanced: far more employees stayed than left. Because of that, "
        f"accuracy alone is a weak measure of model quality, and precision, recall, F1, and "
        f"ROC-AUC should be considered together."
    )

    sections["3. Data Cleaning Decisions"] = (
        "The following cleaning decisions were made before modeling:\n\n"
        "1. Duplicate rows: checked and none removed.\n"
        "2. Constant columns removed from model inputs: "
        "EmployeeCount, Over18, StandardHours.\n"
        "3. Identifier column excluded from model inputs: EmployeeNumber.\n"
        "4. Target column Attrition was encoded as No = 0, Yes = 1.\n\n"
        "These columns were excluded because they do not provide predictive signal for "
        "attrition and would not be available as standalone inputs at prediction time."
    )

    sections["4. Exploratory Findings"] = (
        "The notebook explored attrition across department, job role, overtime status, "
        "age group, income, satisfaction, and tenure bands. Key themes:\n\n"
        f"- Overtime was associated with higher attrition share in the sample.\n"
        f"- Job roles differed markedly in both counts and rates of leavers.\n"
        f"- Younger age bands and shorter tenure groups tended to show higher attrition share.\n"
        f"- Satisfaction-related fields were among the variables most associated with attrition.\n\n"
        "These are associations in the studied sample, not proof of cause, and they may not "
        "generalize to other organizations."
    )

    sections["5. Model Results"] = (
        "Three candidate models were compared using cross-validation: "
        "Logistic Regression, RandomForest, and GradientBoosting.\n\n"
        "After comparing recall, precision, F1, and ROC-AUC, the final model was chosen as "
        "RandomForest and saved to:\n"
        f"{MODEL_PATH}\n\n"
        "Final model performance on the executed data:\n"
        + "```\n"
        + classification_report(y_true, y_pred, target_names=["No (stayed)", "Yes (left)"], zero_division=0)
        + "```\n\n"
        "Interpretation:\n"
        f"- The model predicted {n_predicted_leavers} employees as likely to leave.\n"
        f"- The prediction match rate against the executed labels was {match_rate * 100:.2f}%.\n"
        "- Because leavers are the minority class, recall for the Yes class matters as much "
        "as overall accuracy.\n"
        "- ROC-AUC values and probability outputs should be interpreted as ranking signals, "
        "not certainties."
    )

    sections["6. Business Interpretation"] = (
        "The most useful use of this model is to help prioritize follow-up and "
        "investigation, not to make final employment decisions.\n\n"
        "Suggested next-step actions:\n"
        "- Review employees with the highest predicted probability of leaving.\n"
        "- Combine model output with direct signals such as engagement, manager conversations, "
        "workload, and recent life events.\n"
        "- Focus retention efforts on roles, departments, or conditions associated with higher "
        "attrition in the organization's own data.\n\n"
        "Important caution:\n"
        "- This model was trained on fictional data.\n"
        "- Predictive importance is not proof of cause.\n"
        "- Any real-world use should be validated on the organization's own data and reviewed "
        "by people familiar with the context."
    )

    sections["7. Top Model Drivers"] = (
        "Permutation importance in the notebook showed which features most affected "
        "predictions on the test set. These drivers describe what the model relied on; they do "
        "not by themselves explain why any individual left.\n\n"
        "In general, factors such as overtime status, tenure-related fields, role, and "
        "satisfaction-related fields were among the stronger influences.\n\n"
        "The exact ranking can change slightly with data, preprocessing, and model choice, so "
        "driver lists should be re-estimated when the model is retrained."
    )

    pred_example_class = 'Yes' if int(y_pred[0]) == 1 else 'No'
    example_text = (
        "The notebook includes an example prediction for one synthetic employee record.\n\n"
        f"Predicted class: {pred_example_class}\n"
        "This is only an illustration of how the saved model can be used at prediction time."
    )
    sections["8. Example Prediction"] = example_text

    sections["9. Limitations & Ethical Notes"] = (
        "1. The dataset is fictional and may not reflect any real organization.\n"
        "2. Associations are not causation; predictive importance is not proof of cause.\n"
        "3. The model is trained on historical patterns; future attrition can differ.\n"
        "4. Class imbalance affects metric interpretation; accuracy alone is misleading.\n"
        "5. This model is not validated for real HR decisions.\n"
        "6. Ethical use matters: predictions should support investigation, not label people.\n\n"
        "If this project is reused, the saved model and any report should be regenerated from the fresh notebook run."
    )
    make_word_report(REPORT_PATH, sections)

    print(f"Report saved to: {REPORT_PATH}")
    print(f"Rows used: {n_total}")
    print(f"Actual leavers: {n_leavers}")
    print(f"Predicted leavers: {n_predicted_leavers}")
    print(f"Prediction match rate: {match_rate:.4f}")


if __name__ == "__main__":
    main()
