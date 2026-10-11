import pathlib, json

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
text = p.read_text(encoding="utf-8")

# 1) Trim any trailing garbage after final '}'
idx = text.rfind("}")
if idx >= 0:
    text = text[:idx+1]

# 2) Validate; if still bad, surgically fix the known splitting artifact in the
#    permutation importance cell by rewriting the whole notebook from state3+
#    with corrected sources for cells 20 and 23.
try:
    nb = json.loads(text)
    print("json OK before surgery; cells:", len(nb["cells"]))
except json.JSONDecodeError as e:
    print("still invalid; doing surgery on source lines only")
    import re
    m = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", str(e))
    if not m:
        raise
    char_idx = int(m.group(3))
    print("bad char index:", char_idx)
    # Show context
    print("context:", repr(text[char_idx-60:char_idx+60]))
    raise

# Locate cells by STEP header text
def find_cell(nb, needle):
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and needle in "".join(c.get("source", [])):
            return i, c
    return None, None

idx_perm, cell_perm = find_cell(nb, "STEP 9: Model interpretation - permutation importance")
idx_save, cell_save = find_cell(nb, "STEP 10: Save the best pipeline for later use")
idx_sample, cell_sample = find_cell(nb, "STEP 11: Example prediction on a valid sample record")

print("perm cell idx:", idx_perm, "save cell idx:", idx_save, "sample cell idx:", idx_sample)

new_perm_source = [
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

new_save_source = [
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

new_sample_source = [
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

if idx_perm is not None:
    nb["cells"][idx_perm]["source"] = new_perm_source
    print("rewrote perm cell")
if idx_save is not None:
    nb["cells"][idx_save]["source"] = new_save_source
    print("rewrote save cell")
if idx_sample is not None:
    nb["cells"][idx_sample]["source"] = new_sample_source
    print("rewrote sample cell")

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("rewrote notebook; cells:", len(nb["cells"]))

json.loads(p.read_text(encoding="utf-8"))
print("json OK")
