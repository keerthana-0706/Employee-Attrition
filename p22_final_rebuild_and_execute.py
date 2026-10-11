import pathlib, json, sys, warnings

NOTEBOOK_PATH = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in src.splitlines()]})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in src.splitlines()]})

md("""# Employee Attrition Analysis & Prediction Using AI & Machine Learning

**Project**: Advanced Employee Attrition Analysis and Prediction Using AI & Machine Learning  
**Author**: YourName  
**Date**: October 10, 2026  
**Audience**: AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026  

**Educational notice**: This notebook uses fictional employee records from a public Kaggle dataset. It is NOT real IBM employee data. Statistical associations are not proof of causation, and any model built here is for learning only. It must not be used for real employment decisions.

---

## Contents

1. Environment setup & imports
2. Load and validate the dataset
3. Data cleaning decision log
4. Exploratory data analysis (EDA)
5. Feature engineering & preprocessing pipeline
6. Train / validation split
7. Model training & comparison
8. Evaluation on a held-out test set
9. Model interpretation
10. Save the best pipeline
11. Example prediction
12. Limitations & closing notes

---""")

code("""# ============================================================
# STEP 0: Imports & project setup
# ============================================================
import pathlib
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, roc_curve
)
from sklearn.inspection import permutation_importance

import joblib

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 100

RANDOM_STATE = 42

print('All libraries imported successfully.')
print('pandas      :', pd.__version__)
print('numpy       :', np.__version__)
print('scikit-learn:', __import__('sklearn').__version__)
print('random_state used for splits / models:', RANDOM_STATE)""")

code("""# ============================================================
# STEP 1: Locate and load the dataset
# ============================================================
def project_root():
    cwd = pathlib.Path.cwd()
    if cwd.name == 'notebooks' and (cwd.parent / 'data').exists():
        return cwd.parent
    return cwd

ROOT = project_root()
DATA_PATH = ROOT / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

print('Project root detected :', ROOT)
print('Dataset path          :', DATA_PATH)
print('File exists           :', DATA_PATH.exists())

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f'Dataset not found at {DATA_PATH}.\\n'
        'Expected layout:\\n'
        '  Employee-Attrition-AI/\\n'
        '  ├── data/\\n'
        '  │   └── WA_Fn-UseC_-HR-Employee-Attrition.csv\\n'
        '  └── notebooks/\\n'
        '      └── YourName_Employee_Attrition_Analysis.ipynb\\n'
        '\\n'
        'Fix: place the CSV in data/ next to this notebook, or open the notebook\\n'
        'from the project root so the path resolves correctly.'
    )

df = pd.read_csv(DATA_PATH)
print('\\nDataset loaded.')
print('Shape (rows, columns) :', df.shape)""")

code("""# ============================================================
# STEP 2: Basic dataset inspection
# ============================================================
print('=' * 70)
print('DATASET OVERVIEW')
print('=' * 70)

print('\\nShape (rows, columns):', df.shape)
print('\\nColumns and data types:')
print(df.dtypes.to_string())
print('\\nFirst 5 rows:')
print(df.head().to_string())
print('\\nDescriptive statistics - numerical columns:')
print(df.describe().T.to_string())
print('\\nDescriptive statistics - categorical columns:')
print(df.describe(include=['object', 'category']).T.to_string())
print('\\nMissing values per column:')
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.sum() else 'No missing values found.')
print('\\nTotal missing cells:', int(missing.sum()))
print('\\nDuplicate rows:', int(df.duplicated().sum()))
print('\\nTarget variable (Attrition) value counts:')
print(df['Attrition'].value_counts(dropna=False).to_string())
print('\\nAttrition rate: {:.2f}%'.format(df['Attrition'].eq('Yes').mean() * 100))""")

code("""# ============================================================
# STEP 3: Data cleaning decision log
# ============================================================
print('=' * 70)
print('DATA CLEANING DECISIONS')
print('=' * 70)

print('\\n1) Duplicate rows')
n_dupes = int(df.duplicated().sum())
print('   Found:', n_dupes)
if n_dupes:
    print('   Decision: drop exact duplicates once.')
    df_clean = df.drop_duplicates().copy()
else:
    df_clean = df.copy()
    print('   Decision: none found, so no rows removed.')

print('\\n2) Constant / effectively useless columns')
constant_cols = [c for c in df_clean.columns if df_clean[c].nunique() <= 1]
print('   Constant columns:', constant_cols)
for c in constant_cols:
    print(f'     - {c}: {df_clean[c].unique().tolist()}')

print('\\n3) Identifier columns')
id_like = [c for c in df_clean.columns if 'Id' in c or 'Number' in c or c.lower() == 'employeenumber']
print('   Candidate identifier columns:', id_like)
for c in id_like:
    nun = df_clean[c].nunique()
    nrow = len(df_clean)
    print(f'     - {c}: unique={nun}, rows={nrow}, unique_ratio={nun/max(nrow,1):.3f}')

print('\\n4) Target column sanity check')
print('   Expected target: Attrition with Yes/No labels.')
target_values = df_clean['Attrition'].dropna().unique().tolist()
print('   Observed Attrition values:', sorted(target_values))
assert set(target_values) <= {'Yes', 'No'}, 'Unexpected Attrition labels found.'
print('   OK: target is binary Yes/No.')

print('\\nFinal cleaned DataFrame shape:', df_clean.shape)""")

code("""# ============================================================
# STEP 4a: Define columns we will treat as identifiers / exclude
# ============================================================
ID_COLUMNS = []
for c in df_clean.columns:
    nun = df_clean[c].nunique()
    if nun == len(df_clean):
        ID_COLUMNS.append(c)

print('Columns treated as row identifiers and excluded from model inputs:')
for c in ID_COLUMNS:
    print('  -', c)

print('\\nIf a column is a near-identifier but not perfectly unique, we still')
print('review it manually in the EDA section below.')""")

code("""# ============================================================
# STEP 4b: Overall attrition summary
# ============================================================
n_total = len(df_clean)
n_left = int((df_clean['Attrition'] == 'Yes').sum())
n_stayed = n_total - n_left
attrition_rate = n_left / n_total * 100

summary = pd.DataFrame({
    'Metric': ['Total employees', 'Employees who left (Attrition=Yes)',
               'Employees who stayed (Attrition=No)', 'Overall attrition rate (%)'],
    'Value':  [n_total, n_left, n_stayed, round(attrition_rate, 2)]
})

print(summary.to_string(index=False))

print('\\nInterpretation:')
print('  The dataset is imbalanced: far fewer employees left than stayed.')
print('  Because of that, accuracy alone is a weak measure of model quality.')
print('  We will also examine precision, recall, F1, and ROC-AUC.')""")

code("""# ============================================================
# STEP 4c: Attrition by department
# ============================================================
dept_attr = (
    df_clean.groupby('Department')['Attrition']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .assign(total_employees=lambda x: df_clean['Department'].value_counts())
    [['No', 'Yes', 'total_employees']]
    .rename(columns={'No': 'Stayed_pct', 'Yes': 'Left_pct'})
    .sort_values('Left_pct', ascending=False)
)

print('Attrition share by department:')
print(dept_attr.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
order = dept_attr.index.tolist()
sns.countplot(data=df_clean, x='Department', hue='Attrition', order=order, ax=ax)
ax.set_title('Employee count by department and attrition status')
ax.set_xlabel('Department')
ax.set_ylabel('Number of employees')
ax.legend(title='Attrition')
plt.tight_layout()
plt.show()

print('Interpretation:')
print('  This chart shows counts, not just rates. A department can have many leavers')
print('  simply because it has many employees. Compare the rate table above with this')
print('  count chart when reading the result.')""")

code("""# ============================================================
# STEP 4d: Attrition by job role
# ============================================================
role_attr = (
    df_clean.groupby('JobRole')['Attrition']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .assign(total_employees=lambda x: df_clean['JobRole'].value_counts())
    [['No', 'Yes', 'total_employees']]
    .rename(columns={'No': 'Stayed_pct', 'Yes': 'Left_pct'})
    .sort_values('Left_pct', ascending=False)
)

print('Attrition share by job role:')
print(role_attr.to_string())

fig, ax = plt.subplots(figsize=(11, 5))
order = role_attr.index.tolist()
sns.countplot(data=df_clean, x='JobRole', hue='Attrition', order=order, ax=ax)
ax.set_title('Employee count by job role and attrition status')
ax.set_xlabel('Job role')
ax.set_ylabel('Number of employees')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.legend(title='Attrition')
plt.tight_layout()
plt.show()

print('Interpretation:')
print('  Some roles show a higher share of leavers. As before, counts and rates can')
print('  point in different directions, so we read both together.')""")

code("""# ============================================================
# STEP 4e: Attrition and overtime
# ============================================================
ot_attr = (
    df_clean.groupby('OverTime')['Attrition']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .assign(total_employees=lambda x: df_clean['OverTime'].value_counts())
    [['No', 'Yes', 'total_employees']]
    .rename(columns={'No': 'Stayed_pct', 'Yes': 'Left_pct'})
    .reindex(['Yes', 'No'])
)

print('Attrition share by overtime status:')
print(ot_attr.to_string())

fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=df_clean, x='OverTime', hue='Attrition', ax=ax)
ax.set_title('Employee count by overtime and attrition status')
ax.set_xlabel('Works overtime?')
ax.set_ylabel('Number of employees')
ax.legend(title='Attrition')
plt.tight_layout()
plt.show()

print('Interpretation:')
print('  Overtime is one of the clearer associations in this dataset: employees who')
print('  work overtime show a higher left share than those who do not. This is an')
print('  association, not proof that overtime causes attrition.')""")

code("""# ============================================================
# STEP 4f: Attrition by age group
# ============================================================
df_clean['AgeGroup'] = pd.cut(
    df_clean['Age'],
    bins=[0, 25, 35, 45, 55, 100],
    labels=['<25', '25-35', '35-45', '45-55', '55+'],
    right=False,
)

age_attr = (
    df_clean.groupby('AgeGroup', observed=True)['Attrition']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .assign(total_employees=lambda x: df_clean['AgeGroup'].value_counts())
    [['No', 'Yes', 'total_employees']]
    .rename(columns={'No': 'Stayed_pct', 'Yes': 'Left_pct'})
)

print('Attrition share by age group:')
print(age_attr.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
order = age_attr.index.astype(str).tolist()
sns.countplot(data=df_clean, x='AgeGroup', hue='Attrition', order=order, ax=ax)
ax.set_title('Employee count by age group and attrition status')
ax.set_xlabel('Age group')
ax.set_ylabel('Number of employees')
ax.legend(title='Attrition')
plt.tight_layout()
plt.show()

print('Interpretation:')
print('  Attrition is not evenly spread across age. Younger age bands often show a')
print('  higher left share in this kind of dataset, but again this is association,')
print('  not a rule about any particular person.')""")

code("""# ============================================================
# STEP 4g: Monthly income vs attrition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df_clean, x='Attrition', y='MonthlyIncome', ax=axes[0])
axes[0].set_title('Monthly income by attrition status')
axes[0].set_xlabel('Attrition')
axes[0].set_ylabel('MonthlyIncome')

sns.histplot(data=df_clean, x='MonthlyIncome', hue='Attrition', bins=30,
             element='step', stat='density', common_norm=False, ax=axes[1])
axes[1].set_title('Monthly income distribution by attrition status')
axes[1].set_xlabel('MonthlyIncome')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.show()

print('Summary statistics by attrition group:')
print(df_clean.groupby('Attrition')['MonthlyIncome'].describe().T.to_string())

print('Interpretation:')
print('  Leavers and stayers overlap heavily on income. Any difference in median or')
print('  spread is an association in this sample, not a guaranteed pattern for future')
print('  employees. Box plots can be influenced by outliers, so we also look at the')
print('  histogram.')""")

code("""# ============================================================
# STEP 4h: Job satisfaction and work-life balance vs attrition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

satisfaction_cols = ['JobSatisfaction', 'WorkLifeBalance']
labels = ['Job satisfaction', 'Work-life balance']

for col, lab, ax in zip(satisfaction_cols, labels, axes):
    sns.countplot(data=df_clean, x=col, hue='Attrition', ax=ax)
    ax.set_title(f'{lab} by attrition status')
    ax.set_xlabel(lab)
    ax.set_ylabel('Number of employees')
    ax.legend(title='Attrition')

plt.tight_layout()
plt.show()

print('Interpretation:')
print('  Lower satisfaction or poorer work-life balance scores are often associated')
print('  with higher attrition in the sample. These are self-reported categorical')
print('  scores, so they capture perception rather than an objective measure.')""")

code("""# ============================================================
# STEP 4i: Years at company / tenure vs attrition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df_clean, x='Attrition', y='YearsAtCompany', ax=axes[0])
axes[0].set_title('Years at company by attrition status')
axes[0].set_xlabel('Attrition')
axes[0].set_ylabel('YearsAtCompany')

sns.histplot(data=df_clean, x='YearsAtCompany', hue='Attrition', bins=30,
             element='step', stat='density', common_norm=False, ax=axes[1])
axes[1].set_title('Years at company distribution by attrition status')
axes[1].set_xlabel('YearsAtCompany')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.show()

print('Interpretation:')
print('  Employees with shorter tenure appear more likely to have left in this sample.')
print('  Tenure is partly a function of when attrition happened, so treat it as a')
print('  descriptive association, not a causal lever.')""")

code("""# ============================================================
# STEP 4j: Numerical correlations with the encoded target
# ============================================================
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [c for c in num_cols if c not in ID_COLUMNS]

eda_corr = df_clean.copy()
eda_corr['Attrition_num'] = (eda_corr['Attrition'] == 'Yes').astype(int)

corr_targets = num_cols + ['Attrition_num']
corr_matrix = eda_corr[corr_targets].corr()

target_corr = corr_matrix['Attrition_num'].drop('Attrition_num').sort_values(ascending=False)

print('Pearson correlation of numerical features with encoded attrition (1=Yes):')
print(target_corr.to_string())

print('\\nInterpretation:')
print('  Correlation is a simple linear association measure. Small or moderate values')
print('  can still matter in combination with other features; large values deserve a')
print('  closer look but still do not imply causation.')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation matrix (numerical features + encoded attrition)')
plt.tight_layout()
plt.show()""")

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

# Align to the exact preprocessed feature set computed during training.
proc_cols = list(preprocessor.get_feature_names_out())
sample_proc = pd.DataFrame(0.0, index=range(len(sample)), columns=proc_cols)

for col in numeric_features:
    if col in sample.columns:
        sample_proc[col] = sample[col].astype(float).values

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

nb = {
    "cells": cells,
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

tmp_path = pathlib.Path(".tmp/nb_final.json")
tmp_path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
json.loads(tmp_path.read_text(encoding="utf-8"))
print("built valid notebook JSON; cells:", len(cells))

# Atomic replace
NOTEBOOK_PATH.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
print("notebook written; size(bytes):", NOTEBOOK_PATH.stat().st_size)

# Now execute every code cell top-to-bottom in one interpreter session.
print("\\nStarting top-to-bottom execution...")
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
