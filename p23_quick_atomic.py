import pathlib, json, sys, warnings

NB_PATH = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

def find_cell(nb, header):
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and header in "".join(c.get("source", [])):
            return i, c
    return None, None

# Cell 20 = STEP 9 permutation importance
i, c = find_cell(nb, "# STEP 9: Model interpretation - permutation importance")
if c is None:
    print("could not find permutation cell"); sys.exit(2)

c["source"] = [
    "# ============================================================\n",
    "# STEP 9: Model interpretation - permutation importance\n",
    "# ============================================================\n",
    "perm = permutation_importance(\n",
    "    chosen_pipe, X_test, y_test,\n",
    "    n_repeats=10, random_state=RANDOM_STATE, scoring='f1'\n",
    ")\n",
    "\n",
    "ohe = chosen_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']\n",
    "cat_names = ohe.get_feature_names_out(categorical_features)\n",
    "feature_names = numeric_features + list(cat_names)\n",
    "\n",
    "perm_df = pd.DataFrame({\n",
    "    'feature': feature_names,\n",
    "    'mean_importance': perm.importances_mean,\n",
    "    'std_importance': perm.importances_std,\n",
    "}).sort_values('mean_importance', ascending=False)\n",
    "\n",
    "print('Top 20 predictive features (permutation importance, F1 scoring):')\n",
    "print(perm_df.head(20).to_string())\n",
    "\n",
    "top_n = min(15, len(perm_df))\n",
    "plt.figure(figsize=(9, max(4, top_n * 0.35)))\n",
    "sns.barplot(data=perm_df.head(top_n), x='mean_importance', y='feature', errorbar=None)\n",
    "plt.title(f'Top {top_n} features by permutation importance ({chosen_model_name})')\n",
    "plt.xlabel('Mean decrease in F1 when feature is shuffled')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print('\\nBusiness-readable interpretation:')\n",
    "print('  The most influential features in this model are the ones whose values most')\n",
    "print('  affect predictions on the test set. That does NOT mean they cause attrition.')\n",
    "print('  It means the model relies on them. Use this to guide further investigation,')\n",
    "print('  not to make individual employment decisions.')\n",
]
c["outputs"] = []
c["execution_count"] = None
print("patched permutation cell")

# Cell 23 = STEP 11 example prediction
i, c = find_cell(nb, "# STEP 11: Example prediction on a valid sample record")
if c is None:
    print("could not find example prediction cell"); sys.exit(2)

c["source"] = [
    "# ============================================================\n",
    "# STEP 11: Example prediction on a valid sample record\n",
    "# ============================================================\n",
    "# Build one realistic-looking sample using the same columns the model expects.\n",
    "# The pipeline handles encoding and scaling internally.\n",
    "\n",
    "sample = pd.DataFrame({\n",
    "    'Age': [32],\n",
    "    'BusinessTravel': ['Travel_Rarely'],\n",
    "    'DailyRate': [800],\n",
    "    'Department': ['Sales'],\n",
    "    'DistanceFromHome': [5],\n",
    "    'Education': [3],\n",
    "    'EducationField': ['Life Sciences'],\n",
    "    'EnvironmentSatisfaction': [3],\n",
    "    'Gender': ['Male'],\n",
    "    'HourlyRate': [60],\n",
    "    'JobInvolvement': [3],\n",
    "    'JobLevel': [2],\n",
    "    'JobRole': ['Sales Executive'],\n",
    "    'JobSatisfaction': [3],\n",
    "    'MonthlyIncome': [4500],\n",
    "    'MonthlyRate': [11000],\n",
    "    'NumCompaniesWorked': [2],\n",
    "    'Over18': ['Y'],\n",
    "    'OverTime': ['Yes'],\n",
    "    'PercentSalaryHike': [12],\n",
    "    'PerformanceRating': [3],\n",
    "    'RelationshipSatisfaction': [3],\n",
    "    'StandardHours': [80],\n",
    "    'StockOptionLevel': [0],\n",
    "    'TotalWorkingYears': [8],\n",
    "    'TrainingTimesLastYear': [3],\n",
    "    'WorkLifeBalance': [3],\n",
    "    'YearsAtCompany': [4],\n",
    "    'YearsInCurrentRole': [2],\n",
    "    'YearsSinceLastPromotion': [2],\n",
    "    'YearsWithCurrManager': [2],\n",
    "})\n",
    "\n",
    "# Align to the exact preprocessed feature set computed during training.\n",
    "proc_cols = list(preprocessor.get_feature_names_out())\n",
    "sample_proc = pd.DataFrame(0.0, index=range(len(sample)), columns=proc_cols)\n",
    "\n",
    "for col in numeric_features:\n",
    "    if col in sample.columns:\n",
    "        sample_proc[col] = sample[col].astype(float).values\n",
    "\n",
    "for col in categorical_features:\n",
    "    val = str(sample[col].iloc[0])\n",
    "    for proc_col in proc_cols:\n",
    "        if proc_col == col:\n",
    "            sample_proc[proc_col] = val\n",
    "        elif proc_col.startswith(col + '_'):\n",
    "            sample_proc[proc_col] = 1.0 if proc_col.split(col + '_', 1)[1] == val else 0.0\n",
    "\n",
    "pred_class = loaded.predict(sample_proc)[0]\n",
    "pred_label = 'Yes (likely to leave)' if pred_class == 1 else 'No (likely to stay)'\n",
    "pred_prob = loaded.predict_proba(sample_proc)[0]\n",
    "\n",
    "print('Sample input (single record):')\n",
    "print(sample.to_string())\n",
    "print('\\nPrediction class (0=No, 1=Yes):', int(pred_class))\n",
    "print('Predicted label                :', pred_label)\n",
    "print('Probability No / Yes           : {:.3f} / {:.3f}'.format(float(pred_prob[0]), float(pred_prob[1])))\n",
    "\n",
    "print('\\nIMPORTANT:')\n",
    "print('  This is an educational estimate from a model trained on fictional data.')\n",
    "print('  It is NOT a validated employment decision tool and must not be used as the')\n",
    "print('  sole basis for any decision about a real person.')\n",
]
c["outputs"] = []
c["execution_count"] = None
print("patched example prediction cell")

out = {
    "cells": nb["cells"],
    "metadata": {
        "kernelspec": {"display_name": "Python [attrition_env]", "language": "python", "name": "attrition_env"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

NB_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
json.loads(NB_PATH.read_text(encoding="utf-8"))
print("notebook patched+valid; cells:", len(out["cells"]))

# Now execute top to bottom.
print("\nStarting top-to-bottom execution...")
g = {"__name__": "__main__", "pathlib": pathlib, "warnings": warnings}
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve
g.update(np=np, pd=pd, plt=plt, sns=sns, joblib=joblib, roc_curve=roc_curve)

code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
errors = []
for i, c in enumerate(code_cells):
    src = "".join(c.get("source", []))
    print(f"\n===== EXECUTING CODE CELL {i} =====")
    try:
        exec(compile(src, f"<cell_{i}>", "exec"), g)
    except Exception as e:
        msg = f"Cell {i}: {type(e).__name__}: {e}"
        print(msg)
        errors.append(msg)

if errors:
    print("\nEXECUTION ERRORS:")
    for e in errors:
        print("-", e)
    sys.exit(2)

print("\nALL CODE CELLS EXECUTED WITHOUT RAISED EXCEPTIONS")
print("\nSaved model present:", (pathlib.Path("models/best_attrition_model.pkl")).exists())
