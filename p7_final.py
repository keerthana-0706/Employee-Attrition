import pathlib, json

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
text = p.read_text(encoding="utf-8")
text = text[:text.rfind("}")+1]
lines = text.splitlines()

# Fuse every source list line that ends with '\\n",' and is immediately
# followed by an orphan '",' line into one valid source line.
out = []
i = 0
n = len(lines)
fused = 0
while i < n:
    cur = lines[i]
    nxt = lines[i+1] if i+1 < n else ""
    if cur.endswith('\\n",') and nxt.strip() == '",':
        # fuse: drop the orphan '",' line
        out.append(cur)
        i += 2
        fused += 1
        continue
    out.append(cur)
    i += 1

text = "\n".join(out)
print("fused orphan lines:", fused)

def load_or_split(t):
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        import re
        m = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", str(e))
        if not m:
            raise
        ln = int(m.group(1))
        ls = t.splitlines()
        cur = ls[ln-1]
        print("split attempt on line", ln, ":", repr(cur[:120]))
        if '",' in cur and cur.endswith('\\n",'):
            # split on SECOND '",' occurrence to break merged source entries
            idx = cur.find('",', cur.find('",') + 2)
            if idx > 0:
                head = cur[:idx+2]
                tail = cur[idx+2:]
                ls[ln-1] = head
                ls.insert(ln, tail)
                return load_or_split("\n".join(ls))
        raise

nb = load_or_split(text)
print("json OK after cleanup; cells:", len(nb["cells"]))

# Now replace any cell whose source still contains the bad split marker by
# rebuilding that cell from its STEP header. We do this for cells that touch
# the example/sample and permutation sections if they still contain the
# fusing artifact.
def cell_has_bad_split(c):
    src = "".join(c.get("source", []))
    return "\\n\",    \"" in src or "\",    \"" in src

def find_cell_idx(nb, needle):
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and needle in "".join(c.get("source", [])):
            return i
    return None

to_replace = {}
for needle in [
    "STEP 9: Model interpretation - permutation importance",
    "STEP 10: Save the best pipeline for later use",
    "STEP 11: Example prediction on a valid sample record",
    "STEP 8c: ROC curve comparison",
]:
    j = find_cell_idx(nb, needle)
    if j is not None and cell_has_bad_split(nb["cells"][j]):
        to_replace[j] = needle
        print("marked for rewrite:", needle)

def rebuild_cell_source(needle):
    if needle == "STEP 8c: ROC curve comparison":
        return [
            "# ============================================================\n",
            "# STEP 8c: ROC curve comparison\n",
            "# ============================================================\n",
            "from sklearn.metrics import roc_curve\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(7, 6))\n",
            "\n",
            "for name, pipe in fitted_models.items():\n",
            "    y_prob = pipe.predict_proba(X_test)[:, 1]\n",
            "    fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
            "    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_prob):.3f})')\n",
            "\n",
            "ax.plot([0, 1], [0, 1], linestyle='--', color='gray')\n",
            "ax.set_title('ROC curves on the held-out test set')\n",
            "ax.set_xlabel('False positive rate')\n",
            "ax.set_ylabel('True positive rate')\n",
            "ax.legend(loc='lower right')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('Interpretation:')\n",
            "print('  ROC-AUC summarizes ranking quality across thresholds. Higher is better,')\n",
            "print('  but the best threshold for action depends on how costly missed leavers are')\n",
            "print('  relative to false alarms. That choice is a business decision, not a model one.')\n",
        ]
    if needle == "STEP 9: Model interpretation - permutation importance":
        return [
            "# ============================================================\n",
            "# STEP 9: Model interpretation - permutation importance\n",
            "# ============================================================\n",
            "# Permutation importance measures how much performance drops when a feature is\n",
            "# randomized. It is model-agnostic and works with the fitted pipeline.\n",
            "\n",
            "perm = permutation_importance(\n",
            "    chosen_pipe, X_test, y_test,\n",
            "    n_repeats=10, random_state=RANDOM_STATE, scoring='f1'\n",
            ")\n",
            "\n",
            "# Recover feature names after preprocessing.\n",
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
    if needle == "STEP 10: Save the best pipeline for later use":
        return [
            "# ============================================================\n",
            "# STEP 10: Save the best pipeline for later use\n",
            "# ============================================================\n",
            "# We save the full preprocessing + model pipeline so a later script or dashboard\n",
            "# can load it and make predictions consistently.\n",
            "\n",
            "MODEL_DIR = project_root() / 'models'\n",
            "MODEL_DIR.mkdir(parents=True, exist_ok=True)\n",
            "MODEL_PATH = MODEL_DIR / 'best_attrition_model.pkl'\n",
            "\n",
            "# Use the chosen pipeline from the test evaluation.\n",
            "best_pipeline = chosen_pipe\n",
            "\n",
            "joblib.dump(best_pipeline, MODEL_PATH)\n",
            "\n",
            "print('Saved pipeline to:', MODEL_PATH)\n",
            "print('File exists      :', MODEL_PATH.exists())\n",
            "print('Model name used  :', chosen_model_name)\n",
            "\n",
            "# Quick reload sanity check\n",
            "loaded = joblib.load(MODEL_PATH)\n",
            "print('Reloaded object type:', type(loaded).__name__)\n",
            "print('Has predict        :', hasattr(loaded, 'predict'))\n",
            "print('Has predict_proba  :', hasattr(loaded, 'predict_proba'))\n",
        ]
    if needle == "STEP 11: Example prediction on a valid sample record":
        return [
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
            "# Ensure column order matches the model input.\n",
            "sample = sample[X_model.columns]\n",
            "\n",
            "pred_class = loaded.predict(sample)[0]\n",
            "pred_label = 'Yes (likely to leave)' if pred_class == 1 else 'No (likely to stay)'\n",
            "pred_prob = loaded.predict_proba(sample)[0]\n",
            "\n",
            "print('Sample input (single record):')\n",
            "print(sample.T)\n",
            "print('\\nPrediction class (0=No, 1=Yes):', int(pred_class))\n",
            "print('Predicted label                :', pred_label)\n",
            "print('Probability No / Yes           : {:.3f} / {:.3f}'.format(float(pred_prob[0]), float(pred_prob[1])))\n",
            "\n",
            "print('\\nIMPORTANT:')\n",
            "print('  This is an educational estimate from a model trained on fictional data.')\n",
            "print('  It is NOT a validated employment decision tool and must not be used as the')\n",
            "print('  sole basis for any decision about a real person.')\n",
        ]
    raise SystemExit("unknown needle: " + needle)

for j, needle in to_replace.items():
    nb["cells"][j]["source"] = rebuild_cell_source(needle)
    nb["cells"][j]["outputs"] = []
    nb["cells"][j]["execution_count"] = None
    print("rebuilt cell", j, needle)

out_path = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
nb_full = {
    "cells": nb["cells"],
    "metadata": {
        "kernelspec": {"display_name": "Python [attrition_env]", "language": "python", "name": "attrition_env"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
out_path.write_text(json.dumps(nb_full, indent=1, ensure_ascii=False), encoding="utf-8")

json.loads(out_path.read_text(encoding="utf-8"))
print("FINAL notebook JSON OK; cells:", len(out_path.read_text(encoding="utf-8").splitlines()))
