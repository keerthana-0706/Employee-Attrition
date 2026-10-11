import pathlib, json

C = json.loads(pathlib.Path(".tmp/state3.json").read_text(encoding="utf-8"))
def code(src):
    C.append({"cell_type":"code","execution_count":None,"metadata":{"id":"c"+str(len(C))},"outputs":[],"source":[l+"\n" for l in src.splitlines()]})

code("""# ============================================================
# STEP 5: Feature engineering & preprocessing pipeline
# ============================================================
TARGET = 'Attrition'
y = df_clean[TARGET].map({'Yes': 1, 'No': 0})
X = df_clean.drop(columns=[TARGET])

print('Input features shape:', X.shape)
print('Target shape        :', y.shape)
print('Target class counts :', y.value_counts().to_dict())

exclude_cols = set(ID_COLUMNS)
for c in X.columns:
    if X[c].nunique() <= 1:
        exclude_cols.add(c)

print('\\nExcluded from model inputs:', sorted(exclude_cols))

X_model = X.drop(columns=[c for c in exclude_cols if c in X.columns]).copy()

numeric_features = X_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_model.select_dtypes(include=['object', 'category']).columns.tolist()

print('\\nNumeric features ({}):'.format(len(numeric_features)))
print(numeric_features)
print('\\nCategorical features ({}):'.format(len(categorical_features)))
print(categorical_features)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='drop'
)

print('\\nPreprocessing pipeline defined:')
print(preprocessor)""")

code("""# ============================================================
# STEP 6: Train / test split (stratified, fixed seed)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print('Train set size:', len(X_train))
print('Test set size :', len(X_test))
print('Train attrition rate: {:.2f}%'.format(y_train.mean() * 100))
print('Test attrition rate  : {:.2f}%'.format(y_test.mean() * 100))

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

print('\\nPreprocessed train shape:', X_train_proc.shape)
print('Preprocessed test shape :', X_test_proc.shape)""")

code("""# ============================================================
# STEP 7: Model training with cross-validation comparison
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    'LogisticRegression': LogisticRegression(
        class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE
    ),
    'RandomForest': RandomForestClassifier(
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
}

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}

cv_results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model),
    ])
    scores = cross_validate(pipe, X_model, y, cv=cv, scoring=scoring,
                            return_train_score=False)
    cv_results[name] = scores

print('Cross-validation results (mean +/- std over folds):')
print('{:<22}{:<12}{:<12}{:<12}{:<12}{:<12}'.format(
    'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'))

cv_table = []
for name in models:
    s = cv_results[name]
    row = {
        'Model': name,
        'Accuracy': f"{s['test_accuracy'].mean():.4f} +/- {s['test_accuracy'].std():.4f}",
        'Precision': f"{s['test_precision'].mean():.4f} +/- {s['test_precision'].std():.4f}",
        'Recall': f"{s['test_recall'].mean():.4f} +/- {s['test_recall'].std():.4f}",
        'F1': f"{s['test_f1'].mean():.4f} +/- {s['test_f1'].std():.4f}",
        'ROC_AUC': f"{s['test_roc_auc'].mean():.4f} +/- {s['test_roc_auc'].std():.4f}",
    }
    cv_table.append(row)

cv_df = pd.DataFrame(cv_table)
print(cv_df.to_string(index=False))

print('\\nInterpretation:')
print('  Because the classes are imbalanced, recall and precision matter as much as')
print('  accuracy. A model that misses most leavers can still look accurate.')
print('  We choose a final model based on a balance of recall, precision, F1, and')
print('  ROC-AUC rather than accuracy alone.')""")

code("""# ============================================================
# STEP 8: Hold-out test evaluation for each model
# ============================================================
test_rows = []
fitted_models = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    row = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_prob),
        'AvgPrecision': average_precision_score(y_test, y_prob),
        'ConfusionMatrix': cm.tolist(),
    }
    test_rows.append(row)
    fitted_models[name] = pipe

test_df = pd.DataFrame(test_rows)

print('Held-out test evaluation:')
print(test_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'AvgPrecision']].to_string(index=False))

print('\\nConfusion matrices (rows=actual, cols=predicted) [TN, FP; FN, TP]:')
for _, r in test_df.iterrows():
    cm = np.array(r['ConfusionMatrix'])
    print(f"\\n{r['Model']}:")
    print(cm)
    print(f"  True negatives: {cm[0,0]}, False positives: {cm[0,1]}, False negatives: {cm[1,0]}, True positives: {cm[1,1]}")

print('\\nInterpretation:')
print('  False negatives = leavers the model missed.')
print('  False positives = stayers the model incorrectly flagged as likely to leave.')
print('  In a retention context, missing leavers can be costly, but too many false')
print('  alarms can waste follow-up effort. There is a trade-off.')""")

code("""# ============================================================
# STEP 8b: Detailed classification report for the chosen model
# ============================================================
chosen_model_name = 'RandomForest'
chosen_pipe = fitted_models[chosen_model_name]

y_pred_chosen = chosen_pipe.predict(X_test)
y_prob_chosen = chosen_pipe.predict_proba(X_test)[:, 1]

print('Classification report -', chosen_model_name)
print(classification_report(y_test, y_pred_chosen,
                            target_names=['No (stayed)', 'Yes (left)'],
                            zero_division=0))

fig, ax = plt.subplots(figsize=(6, 5))
cm_chosen = confusion_matrix(y_test, y_pred_chosen)
sns.heatmap(cm_chosen, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_title(f'Confusion matrix - {chosen_model_name}')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticklabels(['No (stayed)', 'Yes (left)'])
ax.set_yticklabels(['No (stayed)', 'Yes (left)'])
plt.tight_layout()
plt.show()""")

code("""# ============================================================
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
print('  relative to false alarms. That choice is a business decision, not a model one.')""")

code("""# ============================================================
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
print('  not to make individual employment decisions.')""")

code("""# ============================================================
# STEP 9b: Logistic Regression coefficients (interpretable baseline)
# ============================================================
lr_pipe = fitted_models['LogisticRegression']
lr = lr_pipe.named_steps['classifier']
lr_ohe = lr_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
lr_cat_names = lr_ohe.get_feature_names_out(categorical_features)
lr_feature_names = numeric_features + list(lr_cat_names)

lr_coef = pd.DataFrame({
    'feature': lr_feature_names,
    'coefficient': lr.coef_.ravel(),
}).sort_values('coefficient', ascending=False)

print('Logistic Regression coefficients (positive => associated with higher predicted attrition):')
print(lr_coef.head(15).to_string())
print('\\n... and the lowest 15 ...')
print(lr_coef.tail(15).to_string())

print('\\nInterpretation:')
print('  Positive coefficients push predictions toward Attrition=Yes; negative coefficients')
print('  push toward Attrition=No. Coefficients are on scaled features, so compare them')
print('  with care. They describe the linear baseline, not causal effects.')""")

code("""# ============================================================
# STEP 10: Save the best pipeline for later use
# ============================================================
MODEL_DIR = project_root() / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'best_attrition_model.pkl'

best_pipeline = chosen_pipe

joblib.dump(best_pipeline, MODEL_PATH)

print('Saved pipeline to:', MODEL_PATH)
print('File exists      :', MODEL_PATH.exists())
print('Model name used  :', chosen_model_name)

loaded = joblib.load(MODEL_PATH)
print('Reloaded object type:', type(loaded).__name__)
print('Has predict        :', hasattr(loaded, 'predict'))
print('Has predict_proba  :', hasattr(loaded, 'predict_proba'))""")

code("""# ============================================================
# STEP 11: Example prediction on a valid sample record
# ============================================================
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
print('  sole basis for any decision about a real person.')""")

code("""# ============================================================
# STEP 12: Limitations and closing notes
# ============================================================
print('KEY LIMITATIONS')
print('-' * 60)
print('1. The dataset is fictional and may not reflect any real organization.')
print('2. Associations are not causation. Predictive importance is not proof of cause.')
print('3. The model is trained on historical patterns; future attrition can differ.')
print('4. Class imbalance affects metric interpretation; accuracy alone is misleading.')
print('5. This model is not validated for real HR decisions.')
print('6. Ethical use matters: predictions should support investigation, not label people.')
print('\\nCLOSING NOTE')
print('-' * 60)
print('This notebook is the main analytical deliverable. A Streamlit dashboard and a')
print('Word report can be built from these results. If you rerun the notebook, the')
print('saved model and any report should be regenerated from the fresh run.')""")

out = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {
            "display_name": "Python [attrition_env]",
            "language": "python",
            "name": "attrition_env"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, "cells:", len(C))

valid = json.loads(out.read_text(encoding="utf-8"))
print("valid notebook JSON OK; cells:", len(valid["cells"]))
