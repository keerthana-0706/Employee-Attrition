"""
Regenerate key chart images from the executed notebook outputs.

Run from the project root:
    python regen_charts.py
"""

import pathlib

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_attrition_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CHART_DIR = PROJECT_ROOT / "reports" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["<25", "25-35", "35-45", "45-55", "55+"],
        right=False,
    )
    return df


def predict_for_dataset(pipe, df: pd.DataFrame):
    df = add_age_group(df)
    drop = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
        "Attrition",
    ]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X[pipe.feature_names_in_]

    y_true = df["Attrition"].map({"No": 0, "Yes": 1}).astype(int)
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    return X, y_true, y_pred, y_prob


def make_confusion_chart(y_true, y_pred, fname: pathlib.Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def make_roc_chart(df: pd.DataFrame, pipe, fname: pathlib.Path, title: str):
    df = add_age_group(df)
    drop = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
        "Attrition",
    ]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X[pipe.feature_names_in_]
    y_true = df["Attrition"].map({"No": 0, "Yes": 1}).astype(int)
    y_prob = pipe.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"Model (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def make_attrition_by_dept_chart(df: pd.DataFrame, fname: pathlib.Path):
    dept = (
        df.groupby("Department")
        .agg(
            employees=("Attrition", "size"),
            leavers=("Attrition", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    dept["leaver_rate"] = dept["leavers"] / dept["employees"]
    dept = dept.sort_values("employees", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dept, x="Department", y="leaver_rate", errorbar=None, ax=ax)
    ax.set_title("Actual attrition rate by department")
    ax.set_ylabel("Leaver rate")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def make_attrition_by_role_chart(df: pd.DataFrame, fname: pathlib.Path):
    role = (
        df.groupby("JobRole")
        .agg(
            employees=("Attrition", "size"),
            leavers=("Attrition", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    role["leaver_rate"] = role["leavers"] / role["employees"]
    role = role.sort_values("employees", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=role, x="leaver_rate", y="JobRole", errorbar=None, ax=ax)
    ax.set_title("Actual attrition rate by job role")
    ax.set_xlabel("Leaver rate")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def make_attrition_by_overtime_chart(df: pd.DataFrame, fname: pathlib.Path):
    ot = (
        df.groupby("OverTime")
        .agg(
            employees=("Attrition", "size"),
            leavers=("Attrition", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    ot["leaver_rate"] = ot["leavers"] / ot["employees"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=ot, x="OverTime", y="leaver_rate", errorbar=None, ax=ax)
    ax.set_title("Actual attrition rate by overtime status")
    ax.set_ylabel("Leaver rate")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def make_top_probabilities_chart(df: pd.DataFrame, pipe, fname: pathlib.Path):
    X, y_true, y_pred, y_prob = predict_for_dataset(pipe, df)
    plot_df = df.copy()
    plot_df["Predicted_Prob_Leave"] = np.round(y_prob, 4)
    plot_df = plot_df.sort_values("Predicted_Prob_Leave", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=plot_df, x="Predicted_Prob_Leave", y="JobRole", errorbar=None, ax=ax)
    ax.set_title("Top 20 employees by predicted probability of leaving")
    ax.set_xlabel("Predicted probability of leaving")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = load_dataset()
    pipe = joblib.load(MODEL_PATH)

    _, y_true, y_pred, _ = predict_for_dataset(pipe, df)

    paths = []
    paths.append(make_confusion_chart(y_true, y_pred, CHART_DIR / "confusion_matrix.png", "Predicted vs actual attrition (RandomForest)"))
    paths.append(make_roc_chart(df, pipe, CHART_DIR / "roc_curve.png", "ROC curve (RandomForest)"))
    paths.append(make_attrition_by_dept_chart(df, CHART_DIR / "attrition_by_department.png"))
    paths.append(make_attrition_by_role_chart(df, CHART_DIR / "attrition_by_job_role.png"))
    paths.append(make_attrition_by_overtime_chart(df, CHART_DIR / "attrition_by_overtime.png"))
    paths.append(make_top_probabilities_chart(df, pipe, CHART_DIR / "top_20_predicted_leavers.png"))

    print(f"Charts saved to: {CHART_DIR}")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
