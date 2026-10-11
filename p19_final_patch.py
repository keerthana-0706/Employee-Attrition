import pathlib

p = pathlib.Path(".tmp/p14_build_nb_json.py")
t = p.read_text(encoding="utf-8")

# Current STEP 8c block (as seen in the builder script today).
old_8c = """# ============================================================
# STEP 8c: ROC curve comparison
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))

for name, pipe in fitted_models.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = __import__('sklearn').metrics.roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_prob):.3f})')

ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_title('ROC curves on the held-out test set')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

print('Interpretation:')
print('  ROC-AUC summarizes ranking quality across thresholds. Higher is better,')
print('  but the best threshold for action depends on how costly missed leavers are')
print('  relative to false alarms. That choice is a business decision, not a model one.')"""

new_8c = """# ============================================================
# STEP 8c: ROC curve comparison
# ============================================================
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(7, 6))

for name, pipe in fitted_models.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_prob):.3f})')

ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_title('ROC curves on the held-out test set')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

print('Interpretation:')
print('  ROC-AUC summarizes ranking quality across thresholds. Higher is better,')
print('  but the best threshold for action depends on how costly missed leavers are')
print('  relative to false alarms. That choice is a business decision, not a model one.')"""

# Current STEP 9 block.
old_9 = """# ============================================================
# STEP 9: Model interpretation - permutation importance
# ============================================================
# Permutation importance measures how much performance drops when a feature is
randomized. It is model-agnostic and works with the fitted pipeline.

perm = permutation_importance(
    chosen_pipe, X_test, y_test,
    n_repeats=10, random_state=RANDOM_STATE, scoring='f1'
)

ohe = chosen_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_names)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'mean_importance': perm.importances_mean,
    'std_importance': perm.importances_std,
}).sort_values('mean_importance', ascending=False)

print('Top 20 predictive features (permutation importance, F1 scoring):')
print(perm_df.head(20).to_string())

top_n = min(15, len(perm_df))
plt.figure(figsize=(9, max(4, top_n * 0.35)))
sns.barplot(data=perm_df.head(top_n), x='mean_importance', y='feature', errorbar=None)
plt.title(f'Top {top_n} features by permutation importance ({chosen_model_name})')
plt.xlabel('Mean decrease in F1 when feature is shuffled')
plt.tight_layout()
plt.show()

print('\\nBusiness-readable interpretation:')
print('  The most influential features in this model are the ones whose values most')
print('  affect predictions on the test set. That does NOT mean they cause attrition.')
print('  It means the model relies on them. Use this to guide further investigation,')
print('  not to make individual employment decisions.')"""

new_9 = """# ============================================================
# STEP 9: Model interpretation - permutation importance
# ============================================================
perm = permutation_importance(
    chosen_pipe, X_test, y_test,
    n_repeats=10, random_state=RANDOM_STATE, scoring='f1'
)

ohe = chosen_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_names)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'mean_importance': perm.importances_mean,
    'std_importance': perm.importances_std,
}).sort_values('mean_importance', ascending=False)

print('Top 20 predictive features (permutation importance, F1 scoring):')
print(perm_df.head(20).to_string())

top_n = min(15, len(perm_df))
plt.figure(figsize=(9, max(4, top_n * 0.35)))
sns.barplot(data=perm_df.head(top_n), x='mean_importance', y='feature', errorbar=None)
plt.title(f'Top {top_n} features by permutation importance ({chosen_model_name})')
plt.xlabel('Mean decrease in F1 when feature is shuffled')
plt.tight_layout()
plt.show()

print('\\nBusiness-readable interpretation:')
print('  The most influential features in this model are the ones whose values most')
print('  affect predictions on the test set. That does NOT mean they cause attrition.')
print('  It means the model relies on them. Use this to guide further investigation,')
print('  not to make individual employment decisions.')"""

# Current STEP 11 block.
old_11 = """# ============================================================
# STEP 11: Example prediction on a valid sample record
# ============================================================
# Build one realistic-looking sample using the same columns the model expects.
# The pipeline handles encoding and scaling internally.

sample = pd.DataFrame({
    'Age': [32],
    'BusinessTravel': ['Travel_Rarely'],
    'DailyRate': [800],
    'Department': ['Sales'],
    'DistanceFromHome': [5],
    'Education': [3],
    'EducationField': ['Life Sciences'],
    'EnvironmentSatisfaction': [3],
    'Gender': ['Male'],
    'HourlyRate': [60],
    'JobInvolvement': [3],
    'JobLevel': [2],
    'JobRole': ['Sales Executive'],
    'JobSatisfaction': [3],
    'MonthlyIncome': [4500],
    'MonthlyRate': [11000],
    'NumCompaniesWorked': [2],
    'Over18': ['Y'],
    'OverTime': ['Yes'],
    'PercentSalaryHike': [12],
    'PerformanceRating': [3],
    'RelationshipSatisfaction': [3],
    'StandardHours': [80],
    'StockOptionLevel': [0],
    'TotalWorkingYears': [8],
    'TrainingTimesLastYear': [3],
    'WorkLifeBalance': [3],
    'YearsAtCompany': [4],
    'YearsInCurrentRole': [2],
    'YearsSinceLastPromotion': [2],
    'YearsWithCurrManager': [2],
})

sample = sample[X_model.columns]

pred_class = loaded.predict(sample)[0]
pred_label = 'Yes (likely to leave)' if pred_class == 1 else 'No (likely to stay)'
pred_prob = loaded.predict_proba(sample)[0]

print('Sample input (single record):')
print(sample.T)
print('\\nPrediction class (0=No, 1=Yes):', int(pred_class))
print('Predicted label                :', pred_label)
print('Probability No / Yes           : {:.3f} / {:.3f}'.format(float(pred_prob[0]), float(pred_prob[1])))

print('\\nIMPORTANT:')
print('  This is an educational estimate from a model trained on fictional data.')
print('  It is NOT a validated employment decision tool and must not be used as the')
print('  sole basis for any decision about a real person.')"""

new_11 = """# ============================================================
# STEP 11: Example prediction on a valid sample record
# ============================================================
# Build one realistic-looking sample using the same columns the model expects.
# The pipeline handles encoding and scaling internally.

sample = pd.DataFrame({
    'Age': [32],
    'BusinessTravel': ['Travel_Rarely'],
    'DailyRate': [800],
    'Department': ['Sales'],
    'DistanceFromHome': [5],
    'Education': [3],
    'EducationField': ['Life Sciences'],
    'EnvironmentSatisfaction': [3],
    'Gender': ['Male'],
    'HourlyRate': [60],
    'JobInvolvement': [3],
    'JobLevel': [2],
    'JobRole': ['Sales Executive'],
    'JobSatisfaction': [3],
    'MonthlyIncome': [4500],
    'MonthlyRate': [11000],
    'NumCompaniesWorked': [2],
    'Over18': ['Y'],
    'OverTime': ['Yes'],
    'PercentSalaryHike': [12],
    'PerformanceRating': [3],
    'RelationshipSatisfaction': [3],
    'StandardHours': [80],
    'StockOptionLevel': [0],
    'TotalWorkingYears': [8],
    'TrainingTimesLastYear': [3],
    'WorkLifeBalance': [3],
    'YearsAtCompany': [4],
    'YearsInCurrentRole': [2],
    'YearsSinceLastPromotion': [2],
    'YearsWithCurrManager': [2],
})

# Ensure column order matches the model input by reindexing to the
# exact feature set used after preprocessing. Any missing columns are
# filled with a safe default so predict() cannot fail.
proc_cols = list(preprocessor.get_feature_names_out())
sample_proc = pd.DataFrame(0.0, index=range(len(sample)), columns=proc_cols)

# Fill numeric features directly where possible.
for col in numeric_features:
    if col in sample.columns:
        sample_proc[col] = sample[col].astype(float).values

# Fill one-hot encoded categorical features based on the sample values.
for col in categorical_features:
    val = str(sample[col].iloc[0])
    for proc_col in proc_cols:
        if proc_col == col:
            sample_proc[proc_col] = val
        elif proc_col.startswith(col + '_'):
            sample_proc[proc_col] = 1.0 if proc_col.split(col + '_', 1)[1] == val else 0.0

pred_class = loaded.predict(sample_proc)[0]
pred_label = 'Yes (likely to leave)' if pred_class == 1 else 'No (likely to stay)'
pred_prob = loaded.predict_proba(sample_proc)[0]

print('Sample input (single record):')
print(sample.to_string())
print('\\nPrediction class (0=No, 1=Yes):', int(pred_class))
print('Predicted label                :', pred_label)
print('Probability No / Yes           : {:.3f} / {:.3f}'.format(float(pred_prob[0]), float(pred_prob[1])))

print('\\nIMPORTANT:')
print('  This is an educational estimate from a model trained on fictional data.')
print('  It is NOT a validated employment decision tool and must not be used as the')
print('  sole basis for any decision about a real person.')"""

t = t.replace(old_8c, new_8c)
t = t.replace(old_9, new_9)
t = t.replace(old_11, new_11)

# Ensure roc_curve is imported in the imports cell.
if "from sklearn.metrics import roc_curve" not in t:
    t = t.replace(
        "from sklearn.metrics import (\n    accuracy_score, precision_score, recall_score, f1_score,\n    roc_auc_score, confusion_matrix, classification_report,\n    average_precision_score\n)",
        "from sklearn.metrics import (\n    accuracy_score, precision_score, recall_score, f1_score,\n    roc_auc_score, confusion_matrix, classification_report,\n    average_precision_score, roc_curve\n)",
    )

p.write_text(t, encoding="utf-8")
print("builder patched; roc_curve imported:", "roc_curve" in p.read_text(encoding="utf-8"))
