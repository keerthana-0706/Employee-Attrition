import pathlib, json, re

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
text = p.read_text(encoding="utf-8")
text = text[: text.rfind("}") + 1]

def try_load(t):
    try:
        return json.loads(t), None
    except json.JSONDecodeError as e:
        m = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", str(e))
        if not m:
            raise
        ln, col, char = map(int, m.groups())
        ls = t.splitlines()
        cur = ls[ln - 1]
        return None, (ln, col, char, cur, ls)

nb, err = try_load(text)
if nb is None:
    print("initially invalid; begin repair loop")
    ln, col, char, cur, ls = err
    print("bad line:", ln, "col:", col, "snippet:", repr(cur[:120]))
    # Repair heuristic: find a source-list line that contains an unescaped newline
    # because two source entries got merged, and split it at the boundary of the
    # first fully-formed entry (the part ending with '\n",').
    #
    # We operate on the whole text as lines, but JSON errors point to a char offset
    # inside a line; that char is an actual LF inside a JSON string => the line
    # contains at least one true newline embedded in the string content.
    #
    # Split strategy:
    #   For the offending line, find the earliest suffix '",' preceded by '\\n".
    #   If found, and there is additional content after it, split into two lines.
    repaired = 0
    max_passes = 200
    for _ in range(max_passes):
        nb, err = try_load(text)
        if nb is not None:
            break
        ln, col, char, cur, ls = err
        # If the offending line does not look mergeable, stop.
        if not cur.strip().startswith('"') or '",' not in cur:
            print("cannot repair this line; aborting repair loop")
            print("line:", repr(cur))
            raise SystemExit(2)
        # Find last '\n",' within the line; if there is trailing content after it,
        # that trailing content is likely the next source entry.
        idx = cur.rfind('",')
        # Prefer splitting at a boundary that leaves a valid source line:
        # look for ',\\n",' occurrence and split after it.
        split_at = None
        search_from = 0
        while True:
            candidate = cur.find('",', search_from)
            if candidate < 0:
                break
            before = cur[:candidate]
            if before.endswith('\\n")') or before.endswith('\\n",') or before.endswith('"\\n")'):
                split_at = candidate + 2
                break
            search_from = candidate + 2
        if split_at is None:
            # Fallback: split at the first '",' that leaves a prefix ending with '\\n"
            candidate = cur.find('",')
            if candidate > 0 and cur[:candidate].endswith('\\n"'):
                split_at = candidate + 2
            else:
                print("no clean split boundary found for line:", repr(cur))
                raise SystemExit(2)
        head = cur[:split_at]
        tail = cur[split_at:]
        if tail == "":
            # Nothing after; this means the line itself is complete but contains an
            # embedded newline earlier; in that case, trim the embedded newline.
            print("tail empty; trying embedded newline removal")
            # Remove a literal newline between quotes
            fixed = re.sub(r'\\n"\n"', '\\\\n\"', cur)
            if fixed != cur:
                ls[ln - 1] = fixed
                text = "\n".join(ls)
                repaired += 1
                continue
            print("embedded newline removal did not apply")
            raise SystemExit(2)
        ls[ln - 1] = head
        ls.insert(ln, tail)
        text = "\n".join(ls)
        repaired += 1
    else:
        print("repair loop exceeded max passes")
        raise SystemExit(2)

    print("repair passes:", repaired)
    nb, err = try_load(text)
    if nb is None:
        print("still invalid after repair; last error:", err)
        raise SystemExit(2)

print("json OK after repair; cells:", len(nb["cells"]))

# Now ensure cells 20 (ROC) and 23 (save/sample area) use safe source if needed.
def cell_text(c):
    return "".join(c.get("source", []))

def replace_cell(nb, needle, new_source_lines):
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and needle in cell_text(c):
            c["source"] = [line + "\n" for line in new_source_lines]
            c["outputs"] = []
            c["execution_count"] = None
            print("replaced:", needle)
            return True
    return False

roc_source = [
    "# ============================================================",
    "# STEP 8c: ROC curve comparison",
    "# ============================================================",
    "from sklearn.metrics import roc_curve",
    "",
    "fig, ax = plt.subplots(figsize=(7, 6))",
    "",
    "for name, pipe in fitted_models.items():",
    "    y_prob = pipe.predict_proba(X_test)[:, 1]",
    "    fpr, tpr, _ = roc_curve(y_test, y_prob)",
    "    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_prob):.3f})')",
    "",
    "ax.plot([0, 1], [0, 1], linestyle='--', color='gray')",
    "ax.set_title('ROC curves on the held-out test set')",
    "ax.set_xlabel('False positive rate')",
    "ax.set_ylabel('True positive rate')",
    "ax.legend(loc='lower right')",
    "plt.tight_layout()",
    "plt.show()",
    "",
    "print('Interpretation:')",
    "print('  ROC-AUC summarizes ranking quality across thresholds. Higher is better,')",
    "print('  but the best threshold for action depends on how costly missed leavers are')",
    "print('  relative to false alarms. That choice is a business decision, not a model one.')",
]

perm_source = [
    "# ============================================================",
    "# STEP 9: Model interpretation - permutation importance",
    "# ============================================================",
    "# Permutation importance measures how much performance drops when a feature is",
    "# randomized. It is model-agnostic and works with the fitted pipeline.",
    "",
    "perm = permutation_importance(",
    "    chosen_pipe, X_test, y_test,",
    "    n_repeats=10, random_state=RANDOM_STATE, scoring='f1'",
    ")",
    "",
    "# Recover feature names after preprocessing.",
    "ohe = chosen_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']",
    "cat_names = ohe.get_feature_names_out(categorical_features)",
    "feature_names = numeric_features + list(cat_names)",
    "",
    "perm_df = pd.DataFrame({",
    "    'feature': feature_names,",
    "    'mean_importance': perm.importances_mean,",
    "    'std_importance': perm.importances_std,",
    "}).sort_values('mean_importance', ascending=False)",
    "",
    "print('Top 20 predictive features (permutation importance, F1 scoring):')",
    "print(perm_df.head(20).to_string())",
    "",
    "top_n = min(15, len(perm_df))",
    "plt.figure(figsize=(9, max(4, top_n * 0.35)))",
    "sns.barplot(data=perm_df.head(top_n), x='mean_importance', y='feature', errorbar=None)",
    "plt.title(f'Top {top_n} features by permutation importance ({chosen_model_name})')",
    "plt.xlabel('Mean decrease in F1 when feature is shuffled')",
    "plt.tight_layout()",
    "plt.show()",
    "",
    "print('\\nBusiness-readable interpretation:')",
    "print('  The most influential features in this model are the ones whose values most')",
    "print('  affect predictions on the test set. That does NOT mean they cause attrition.')",
    "print('  It means the model relies on them. Use this to guide further investigation,')",
    "print('  not to make individual employment decisions.')",
]

save_source = [
    "# ============================================================",
    "# STEP 10: Save the best pipeline for later use",
    "# ============================================================",
    "# We save the full preprocessing + model pipeline so a later script or dashboard",
    "# can load it and make predictions consistently.",
    "",
    "MODEL_DIR = project_root() / 'models'",
    "MODEL_DIR.mkdir(parents=True, exist_ok=True)",
    "MODEL_PATH = MODEL_DIR / 'best_attrition_model.pkl'",
    "",
    "# Use the chosen pipeline from the test evaluation.",
    "best_pipeline = chosen_pipe",
    "",
    "joblib.dump(best_pipeline, MODEL_PATH)",
    "",
    "print('Saved pipeline to:', MODEL_PATH)",
    "print('File exists      :', MODEL_PATH.exists())",
    "print('Model name used  :', chosen_model_name)",
    "",
    "# Quick reload sanity check",
    "loaded = joblib.load(MODEL_PATH)",
    "print('Reloaded object type:', type(loaded).__name__)",
    "print('Has predict        :', hasattr(loaded, 'predict'))",
    "print('Has predict_proba  :', hasattr(loaded, 'predict_proba'))",
]

sample_source = [
    "# ============================================================",
    "# STEP 11: Example prediction on a valid sample record",
    "# ============================================================",
    "# Build one realistic-looking sample using the same columns the model expects.",
    "# The pipeline handles encoding and scaling internally.",
    "",
    "sample = pd.DataFrame({",
    "    'Age': [32],",
    "    'BusinessTravel': ['Travel_Rarely'],",
    "    'DailyRate': [800],",
    "    'Department': ['Sales'],",
    "    'DistanceFromHome': [5],",
    "    'Education': [3],",
    "    'EducationField': ['Life Sciences'],",
    "    'EnvironmentSatisfaction': [3],",
    "    'Gender': ['Male'],",
    "    'HourlyRate': [60],",
    "    'JobInvolvement': [3],",
    "    'JobLevel': [2],",
    "    'JobRole': ['Sales Executive'],",
    "    'JobSatisfaction': [3],",
    "    'MonthlyIncome': [4500],",
    "    'MonthlyRate': [11000],",
    "    'NumCompaniesWorked': [2],",
    "    'Over18': ['Y'],",
    "    'OverTime': ['Yes'],",
    "    'PercentSalaryHike': [12],",
    "    'PerformanceRating': [3],",
    "    'RelationshipSatisfaction': [3],",
    "    'StandardHours': [80],",
    "    'StockOptionLevel': [0],",
    "    'TotalWorkingYears': [8],",
    "    'TrainingTimesLastYear': [3],",
    "    'WorkLifeBalance': [3],",
    "    'YearsAtCompany': [4],",
    "    'YearsInCurrentRole': [2],",
    "    'YearsSinceLastPromotion': [2],",
    "    'YearsWithCurrManager': [2],",
    "})",
    "",
    "# Ensure column order matches the model input.",
    "sample = sample[X_model.columns]",
    "",
    "pred_class = loaded.predict(sample)[0]",
    "pred_label = 'Yes (likely to leave)' if pred_class == 1 else 'No (likely to stay)'",
    "pred_prob = loaded.predict_proba(sample)[0]",
    "",
    "print('Sample input (single record):')",
    "print(sample.T)",
    "print('\\nPrediction class (0=No, 1=Yes):', int(pred_class))",
    "print('Predicted label                :', pred_label)",
    "print('Probability No / Yes           : {:.3f} / {:.3f}'.format(float(pred_prob[0]), float(pred_prob[1])))",
    "",
    "print('\\nIMPORTANT:')",
    "print('  This is an educational estimate from a model trained on fictional data.')",
    "print('  It is NOT a validated employment decision tool and must not be used as the')",
    "print('  sole basis for any decision about a real person.')",
]

for needle, new_src in [
    ("STEP 8c: ROC curve comparison", roc_source),
    ("STEP 9: Model interpretation - permutation importance", perm_source),
    ("STEP 10: Save the best pipeline for later use", save_source),
    ("STEP 11: Example prediction on a valid sample record", sample_source),
]:
    replace_cell(nb, needle, new_src)

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
print("FINAL notebook written and valid; cells:", len(final["cells"]))
