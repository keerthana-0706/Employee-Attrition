import pathlib, json, re

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
text = p.read_text(encoding="utf-8")
text = text[: text.rfind("}") + 1]

lines = text.splitlines(keepends=True)

# We already know cell c24 (STEP 11) is corrupt. We'll replace the entire cell
# object with a clean one, plus also ensure cells c21 (STEP 9) and c19 (STEP 8c)
# are clean. We can detect them by searching for their STEP header inside the
# "source" content even if the source array is currently malformed.

def find_cell_by_header(lines, header):
    # Find the first source line (inside any cell) containing header
    for i, l in enumerate(lines):
        if header in l and '"source":' in "".join(lines[max(0,i-12):i+1]):
            # ascend to enclosing cell object
            id_line = None
            j = i
            while j >= 0:
                if '"id":' in lines[j]:
                    m = re.search(r'"id":\s*"([^"]+)"', lines[j])
                    if m:
                        id_line = j
                        cell_id = m.group(1)
                    break
                j -= 1
            if id_line is None:
                continue
            obj_start = None
            k = id_line
            while k >= 0:
                if lines[k].strip() == '{':
                    obj_start = k
                    break
                k -= 1
            if obj_start is None:
                continue
            depth = 0
            obj_end = None
            for m in range(obj_start, len(lines)):
                for ch in lines[m]:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            obj_end = m
                            break
                if obj_end is not None:
                    break
            return obj_start, obj_end, cell_id
    return None

targets = {
    "STEP 8c: ROC curve comparison": "c19",
    "STEP 9: Model interpretation - permutation importance": "c21",
    "STEP 11: Example prediction on a valid sample record": "c24",
}

# Build clean source text per cell id
clean_sources = {}

clean_sources["c19"] = """   "source": [
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
   ],
"""

clean_sources["c21"] = """   "source": [
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
   ],
"""

clean_sources["c24"] = """   "source": [
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
   ],
"""

replaced = []
for header, expected_id in targets.items():
    span = find_cell_by_header(lines, header)
    if span is None:
        print("cannot locate", header)
        continue
    obj_start, obj_end, cell_id = span
    print("found", header, "->", cell_id, "span", obj_start+1, obj_end+1)
    # Replace entire object with a clean minimal cell object
    # We parse the existing metadata/id to preserve notebook-relative ids only if clean;
    # here we just rebuild a clean cell object.
    new_obj_lines = [
        "  {\n",
        f'   "cell_type": "code",\n',
        f'   "execution_count": null,\n',
        f'   "id": "{cell_id}",\n',
        "   \"metadata\": {},\n",
    ]
    new_obj_lines.append(clean_sources[expected_id])
    new_obj_lines.append("  },\n")
    lines[obj_start:obj_end+1] = new_obj_lines
    replaced.append(expected_id)

new_text = "".join(lines)
try:
    nb = json.loads(new_text)
    print("json OK after cell replacements; cells:", len(nb["cells"]))
except json.JSONDecodeError as e:
    print("still invalid:", e)
    m = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", str(e))
    if m:
        ln = int(m.group(1))
        print("offending line:", ln, repr(lines[ln-1] if ln-1 < len(lines) else ""))
    raise SystemExit(2)

final = {
    "cells": nb["cells"],
    "metadata": {
        "kernelspec": {"display_name": "Python [attrition_env]", "language": "python", "name": "attrition_env"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

p.write_text(json.dumps(final, indent=1, ensure_ascii=False), encoding="utf-8")
json.loads(p.read_text(encoding="utf-8"))
print("FINAL notebook written+valid; cells:", len(final["cells"]))
print("replaced:", replaced)
