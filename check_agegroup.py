import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

DATA = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL = Path("models/best_attrition_model.pkl")

df = pd.read_csv(DATA)
pipe = joblib.load(MODEL)

# Mimic notebook preprocessing enough to produce AgeGroup.
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 29, 34, 39, 44, 49, 100],
    labels=["<30", "30-34", "35-39", "40-44", "45-49", "50+"],
)

drop = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours", "Attrition"]
df_feat = df.drop(columns=[c for c in drop if c in df.columns])
X = df_feat[pipe.feature_names_in_].copy()

y_true = df["Attrition"].map({"No": 0, "Yes": 1}).astype(int)
y_pred = pipe.predict(X)
y_prob = pipe.predict_proba(X)[:, 1]

print("n:", len(df))
print("actual leavers:", int((y_true == 1).sum()))
print("predicted leavers:", int((y_pred == 1).sum()))
print("match rate:", float((pd.Series(y_pred).astype(str) == y_true.astype(str)).mean()))
