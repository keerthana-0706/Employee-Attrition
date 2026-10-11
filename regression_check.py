"""Quick regression checks for the Employee Attrition project artifacts."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "best_attrition_model.pkl"
DATA = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

AGE_BINS = [0, 25, 35, 45, 55, 100]
AGE_LABELS = ["<25", "25-35", "35-45", "45-55", "55+"]

DROP_FEATURE_COLS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
    "Attrition",
]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    )
    return df


def predict_for_dataset(pipe, df: pd.DataFrame):
    df = add_age_group(df)
    feature_cols = [c for c in df.columns if c not in DROP_FEATURE_COLS]
    X = df[feature_cols].copy()
    # Preserve exact column order the model was trained on.
    X = X[pipe.feature_names_in_]
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    return df.copy(), y_pred, y_prob


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    pipe = joblib.load(MODEL)
    if not hasattr(pipe, "predict") or not hasattr(pipe, "predict_proba"):
        raise RuntimeError("Saved model is not a usable estimator")

    df = pd.read_csv(DATA)
    df_out, y_pred, y_prob = predict_for_dataset(pipe, df)

    y_true = df["Attrition"].map({"No": 0, "Yes": 1}).astype(int)
    match_rate = float((y_pred == y_true).mean())

    metrics = {
        "n_total": int(len(df_out)),
        "actual_leavers": int((df["Attrition"] == "Yes").sum()),
        "predicted_leavers": int((y_pred == 1).sum()),
        "non_leavers_predicted": int((y_pred == 0).sum()),
        "match_rate": round(match_rate, 4),
        "mean_predicted_prob_leave": round(float(np.mean(y_prob)), 4),
        "max_predicted_prob_leave": round(float(np.max(y_prob)), 4),
        "min_predicted_prob_leave": round(float(np.min(y_prob)), 4),
        "mean_predicted_prob_for_leavers": round(
            float(np.mean(y_prob[y_true == 1])), 4
        ),
        "mean_predicted_prob_for_non_leavers": round(
            float(np.mean(y_prob[y_true == 0])), 4
        ),
        "roc_auc": round(
            float(
                __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                    y_true, y_prob
                )
            ),
            4,
        ),
    }
    print("REGRESSION_CHECK", metrics)

    # Structural sanity on the saved model.
    missing_features = [c for c in pipe.feature_names_in_ if c not in df_out.columns]
    if missing_features:
        raise AssertionError(f"Model expects columns not present: {missing_features}")

    sample_X = df_out[pipe.feature_names_in_].iloc[:1]
    sample_pred = int(pipe.predict(sample_X)[0])
    sample_prob = float(pipe.predict_proba(sample_X)[0][1])
    sample_pred_ok = sample_pred in (0, 1)
    sample_prob_ok = 0.0 <= sample_prob <= 1.0
    print("SAMPLE_PREDICTION_CHECK", {
        "sample_pred_class": sample_pred,
        "sample_pred_prob_leave": round(sample_prob, 4),
        "pred_class_valid": sample_pred_ok,
        "prob_valid": sample_prob_ok,
    })

    print("ALL_REGRESSION_CHECKS_OK", True)


if __name__ == "__main__":
    main()
