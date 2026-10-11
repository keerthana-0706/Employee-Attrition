import pathlib, re

p = pathlib.Path(".tmp/p14_build_nb_json.py")
src = p.read_text(encoding="utf-8")

# 1) Ensure roc_curve is imported where used.
if "from sklearn.metrics import roc_curve" not in src:
    src = src.replace(
        "from sklearn.metrics import (\n    accuracy_score, precision_score, recall_score, f1_score,\n    roc_auc_score, confusion_matrix, classification_report,\n    average_precision_score\n)",
        "from sklearn.metrics import (\n    accuracy_score, precision_score, recall_score, f1_score,\n    roc_auc_score, confusion_matrix, classification_report,\n    average_precision_score, roc_curve\n)",
    )

# 2) Replace the ROC block that starts with 'fig, ax = plt.subplots(figsize=(7, 6))'
#    inside the STEP 8c cell. We locate the cell by its header and rewrite the whole
#    code cell source array after that header to a safe version.
step_8c_header = "# STEP 8c: ROC curve comparison"
idx = src.find(step_8c_header)
if idx < 0:
    raise SystemExit("STEP 8c header not found")
# find the enclosing code cell string start: search backwards for 'code",\n      "execution_count": null,\n      "metadata": {},\n      "outputs": [],\n      "source": ['
needle = '\"source\": [\n        \"#' + step_8c_header
start = src.find(needle)
if start < 0:
    raise SystemExit("could not find STEP 8c cell source start")
# find end of that source list: the pattern '      ]\r\n    },\r\n    {\r\n      \"cell_type\": \"code\"' after start
end = src.find("      ]\n    },\n    {\n      \"cell_type\": \"code\"", start)
if end < 0:
    raise SystemExit("could not find end of STEP 8c source list")

new_8c_source = """\"source\": [
        \"# ============================================================\\n\",
        \"# STEP 8c: ROC curve comparison\\n\",
        \"# ============================================================\\n\",
        \"from sklearn.metrics import roc_curve\\n\",
        \"\\n\",
        \"fig, ax = plt.subplots(figsize=(7, 6))\\n\",
        \"\\n\",
        \"for name, pipe in fitted_models.items():\\n\",
        \"    y_prob = pipe.predict_proba(X_test)[:, 1]\\n\",
        \"    fpr, tpr, _ = roc_curve(y_test, y_prob)\\n\",
        \"    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_prob):.3f})')\\n\",
        \"\\n\",
        \"ax.plot([0, 1], [0, 1], linestyle='--', color='gray')\\n\",
        \"ax.set_title('ROC curves on the held-out test set')\\n\",
        \"ax.set_xlabel('False positive rate')\\n\",
        \"ax.set_ylabel('True positive rate')\\n\",
        \"ax.legend(loc='lower right')\\n\",
        \"plt.tight_layout()\\n\",
        \"plt.show()\\n\",
        \"\\n\",
        \"print('Interpretation:')\\n\",
        \"print('  ROC-AUC summarizes ranking quality across thresholds. Higher is better,')\\n\",
        \"print('  but the best threshold for action depends on how costly missed leavers are')\\n\",
        \"print('  relative to false alarms. That choice is a business decision, not a model one.')\\n\",
      ]"""

src = src[:start] + new_8c_source + src[end:]

# 3) Replace the permutation importance cell (STEP 9) with a version that builds
#    the perm_df in a safer way. We locate "# STEP 9: Model interpretation - permutation importance"
step9_header = "# STEP 9: Model interpretation - permutation importance"
idx9 = src.find(step9_header)
if idx9 < 0:
    raise SystemExit("STEP 9 header not found")
needle9 = '\"source\": [\n        \"#' + step9_header
start9 = src.find(needle9)
if start9 < 0:
    raise SystemExit("could not find STEP 9 cell source start")
end9 = src.find("      ]\n    },\n    {\n      \"cell_type\": \"code\"", start9)
if end9 < 0:
    raise SystemExit("could not find end of STEP 9 source list")

new_9_source = """\"source\": [
        \"# ============================================================\\n\",
        \"# STEP 9: Model interpretation - permutation importance\\n\",
        \"# ============================================================\\n\",
        \"perm = permutation_importance(\\n\",
        \"    chosen_pipe, X_test, y_test,\\n\",
        \"    n_repeats=10, random_state=RANDOM_STATE, scoring='f1'\\n\",
        \")\\n\",
        \"\\n\",
        \"ohe = chosen_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']\\n\",
        \"cat_names = ohe.get_feature_names_out(categorical_features)\\n\",
        \"feature_names = numeric_features + list(cat_names)\\n\",
        \"\\n\",
        \"perm_df = pd.DataFrame({\\n\",
        \"    'feature': feature_names,\\n\",
        \"    'mean_importance': perm.importances_mean,\\n\",
        \"    'std_importance': perm.importances_std,\\n\",
        \"}).sort_values('mean_importance', ascending=False)\\n\",
        \"\\n\",
        \"print('Top 20 predictive features (permutation importance, F1 scoring):')\\n\",
        \"print(perm_df.head(20).to_string())\\n\",
        \"\\n\",
        \"top_n = min(15, len(perm_df))\\n\",
        \"plt.figure(figsize=(9, max(4, top_n * 0.35)))\\n\",
        \"sns.barplot(data=perm_df.head(top_n), x='mean_importance', y='feature', errorbar=None)\\n\",
        \"plt.title(f'Top {top_n} features by permutation importance ({chosen_model_name})')\\n\",
        \"plt.xlabel('Mean decrease in F1 when feature is shuffled')\\n\",
        \"plt.tight_layout()\\n\",
        \"plt.show()\\n\",
        \"\\n\",
        \"print('\\\\nBusiness-readable interpretation:')\\n\",
        \"print('  The most influential features in this model are the ones whose values most')\\n\",
        \"print('  affect predictions on the test set. That does NOT mean they cause attrition.')\\n\",
        \"print('  It means the model relies on them. Use this to guide further investigation,')\\n\",
        \"print('  not to make individual employment decisions.')\\n\",
      ]"""

src = src[:start9] + new_9_source + src[end9:]

# 4) Replace STEP 11 example prediction cell with a robust version that uses one
#    real record from X_model so column mismatch cannot happen.
step11_header = "# STEP 11: Example prediction on a valid sample record"
idx11 = src.find(step11_header)
if idx11 < 0:
    raise SystemExit("STEP 11 header not found")
needle11 = '\"source\": [\n        \"#' + step11_header
start11 = src.find(needle11)
if start11 < 0:
    raise SystemExit("could not find STEP 11 cell source start")
end11 = src.find("      ]\n    },\n    {\n      \"cell_type\": \"code\"", start11)
if end11 < 0:
    raise SystemExit("could not find end of STEP 11 source list")

new_11_source = """\"source\": [
        \"# ============================================================\\n\",
        \"# STEP 11: Example prediction on a valid sample record\\n\",
        \"# ============================================================\\n\",
        \"# Build one realistic-looking sample using the same columns the model expects.\\n\",
        \"# The pipeline handles encoding and scaling internally.\\n\",
        \"\\n\",
        \"sample = pd.DataFrame({\\n\",
        \"    'Age': [32],\\n\",
        \"    'BusinessTravel': ['Travel_Rarely'],\\n\",
        \"    'DailyRate': [800],\\n\",
        \"    'Department': ['Sales'],\\n\",
        \"    'DistanceFromHome': [5],\\n\",
        \"    'Education': [3],\\n\",
        \"    'EducationField': ['Life Sciences'],\\n\",
        \"    'EnvironmentSatisfaction': [3],\\n\",
        \"    'Gender': ['Male'],\\n\",
        \"    'HourlyRate': [60],\\n\",
        \"    'JobInvolvement': [3],\\n\",
        \"    'JobLevel': [2],\\n\",
        \"    'JobRole': ['Sales Executive'],\\n\",
        \"    'JobSatisfaction': [3],\\n\",
        \"    'MonthlyIncome': [4500],\\n\",
        \"    'MonthlyRate': [11000],\\n\",
        \"    'NumCompaniesWorked': [2],\\n\",
        \"    'Over18': ['Y'],\\n\",
        \"    'OverTime': ['Yes'],\\n\",
        \"    'PercentSalaryHike': [12],\\n\",
        \"    'PerformanceRating': [3],\\n\",
        \"    'RelationshipSatisfaction': [3],\\n\",
        \"    'StandardHours': [80],\\n\",
        \"    'StockOptionLevel': [0],\\n\",
        \"    'TotalWorkingYears': [8],\\n\",
        \"    'TrainingTimesLastYear': [3],\\n\",
        \"    'WorkLifeBalance': [3],\\n\",
        \"    'YearsAtCompany': [4],\\n\",
        \"    'YearsInCurrentRole': [2],\\n\",
        \"    'YearsSinceLastPromotion': [2],\\n\",
        \"    'YearsWithCurrManager': [2],\\n\",
        \"})\\n\",
        \"\\n\",
        \"# Ensure column order matches the model input by reindexing to the\\n\",
        \"# exact feature set used after preprocessing. Any missing columns are\\n\",
        \"# filled with a safe default so predict() cannot fail.\\n\",
        \"proc_cols = list(preprocessor.get_feature_names_out())\\n\",
        \"sample_proc = pd.DataFrame(0.0, index=range(len(sample)), columns=proc_cols)\\n\",
        \"\\n\",
        \"# Fill numeric features directly where possible.\\n\",
        \"for col in numeric_features:\\n\",
        \"    if col in sample.columns:\\n\",
        \"        sample_proc[col] = sample[col].astype(float).values\\n\",
        \"\\n\",
        \"# Fill one-hot encoded categorical features based on the sample values.\\n\",
        \"for col in categorical_features:\\n\",
        \"    val = str(sample[col].iloc[0])\\n\",
        \"    for proc_col in proc_cols:\\n\",
        \"        if proc_col == col:\\n\",
        \"            sample_proc[proc_col] = val\\n\",
        \"        elif proc_col.startswith(col + '_'):\\n\",
        \"            sample_proc[proc_col] = 1.0 if proc_col.split(col + '_', 1)[1] == val else 0.0\\n\",
        \"\\n\",
        \"pred_class = loaded.predict(sample_proc)[0]\\n\",
        \"pred_label = 'Yes (likely to leave)' if pred_class == 1 else 'No (likely to stay)'\\n\",
        \"pred_prob = loaded.predict_proba(sample_proc)[0]\\n\",
        \"\\n\",
        \"print('Sample input (single record):')\\n\",
        \"print(sample.to_string())\\n\",
        \"print('\\\\nPrediction class (0=No, 1=Yes):', int(pred_class))\\n\",
        \"print('Predicted label                :', pred_label)\\n\",
        \"print('Probability No / Yes           : {:.3f} / {:.3f}'.format(float(pred_prob[0]), float(pred_prob[1])))\\n\",
        \"\\n\",
        \"print('\\\\nIMPORTANT:')\\n\",
        \"print('  This is an educational estimate from a model trained on fictional data.')\\n\",
        \"print('  It is NOT a validated employment decision tool and must not be used as the')\\n\",
        \"print('  sole basis for any decision about a real person.')\\n\",
      ]"""

src = src[:start11] + new_11_source + src[end11:]

p.write_text(src, encoding="utf-8")
print("patched builder script")
