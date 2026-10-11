import pathlib, json, re

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
text = p.read_text(encoding="utf-8")
text = text[: text.rfind("}") + 1]

lines = text.splitlines(keepends=True)

# Split the top-level JSON into:
#  - preamble up to '"cells": ['
#  - each top-level cell object
#  - postamble from '  ]' after cells to end
preamble = []
postamble = []
cells_lines = []
state = "preamble"
brace_depth = 0
current = []
cell_start_seen = False

for i, l in enumerate(lines):
    if state == "preamble":
        preamble.append(l)
        if '"cells": [' in l:
            state = "cells"
            brace_depth = 0
            current = []
        continue
    if state == "cells":
        # track braces to find top-level object boundaries within the cells array
        for ch in l:
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
        current.append(l)
        if brace_depth == 0:
            cells_lines.append("".join(current))
            current = []
            state = "postamble"
            continue
        # after a top-level cell, if next char starts another object at depth 0
        # we need to detect the closing '],' before next object. We'll handle
        # by checking when brace_depth returns to 0 and the next non-empty line
        # does not start another object immediately.
        continue
    if state == "postamble":
        postamble.append(l)

print("preamble lines:", len(preamble))
print("cells found:", len(cells_lines))
print("postamble lines:", len(postamble))

def parse_cell(text_block):
    try:
        obj = json.loads(text_block)
        return obj
    except json.JSONDecodeError:
        # Try to fix by ensuring it is surrounded by { }
        candidate = text_block.strip()
        if not candidate.startswith('{'):
            candidate = '{' + candidate
        if not candidate.endswith('}'):
            candidate = candidate + '}'
        return json.loads(candidate)

cells = []
for blk in cells_lines:
    obj = parse_cell(blk)
    cells.append(obj)

print("parsed cells:", len(cells))

# Now fix any cell whose source contains an embedded real newline or other
# invalid control char by checking whether json.dumps of the cell fails.
def cell_source_ok(cell):
    try:
        json.dumps(cell)
        return True
    except (TypeError, ValueError):
        return False

fixed_count = 0
for cell in cells:
    if not cell_source_ok(cell):
        src = "".join(cell.get("source", []))
        print("cell", cell.get("metadata",{}).get("id"), "has problematic source")
        print("snippet:", repr(src[:120]))
        fixed_count += 1

print("cells with problematic source:", fixed_count)

# Rebuild problematic cells from STEP headers
def rebuild_cell(cell, notebook_cells):
    src = "".join(cell.get("source", []))
    if "STEP 8c: ROC curve comparison" in src:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": cell.get("metadata", {}).get("id", "c999")},
            "outputs": [],
            "source": [
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
        }
    if "STEP 9: Model interpretation - permutation importance" in src:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": cell.get("metadata", {}).get("id", "c999")},
            "outputs": [],
            "source": [
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
        }
    if "STEP 11: Example prediction on a valid sample record" in src:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": cell.get("metadata", {}).get("id", "c999")},
            "outputs": [],
            "source": [
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
        }
    # fallback: try to fix source by re-encoding
    return cell

fixed_cells = 0
for i, cell in enumerate(cells):
    if not cell_source_ok(cell):
        cells[i] = rebuild_cell(cell, cells)
        fixed_cells += 1
        print("rebuilt cell idx", i, cells[i].get("metadata",{}).get("id"))

print("rebuilt cells:", fixed_cells)

# Now assemble a clean notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python [attrition_env]", "language": "python", "name": "attrition_env"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

clean_text = json.dumps(nb, indent=1, ensure_ascii=False)
p.write_text(clean_text, encoding="utf-8")

# validate again
json.loads(p.read_text(encoding="utf-8"))
print("FINAL notebook written+valid; cells:", len(nb["cells"]))
