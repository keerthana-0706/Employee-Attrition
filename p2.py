import pathlib, json

C = json.loads(pathlib.Path(".tmp/state1.json").read_text(encoding="utf-8"))
def code(src):
    C.append({"cell_type":"code","execution_count":None,"metadata":{"id":"c"+str(len(C))},"outputs":[],"source":[l+"\n" for l in src.splitlines()]})

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
    average_precision_score
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

pathlib.Path(".tmp/state2.json").write_text(json.dumps(C), encoding="utf-8")
print("p2 done; cells:", len(C))
