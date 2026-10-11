"""
src/attrition_analysis.py

Reusable utility module for the Employee Attrition Analysis project.

This module contains the reusable code used in the Jupyter notebook,
kept separate so the notebook itself stays readable for beginners.
"""

import logging
import pathlib

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths (relative to the project root, so the project can be moved)
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

DATA_PATH = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = MODEL_DIR / "best_attrition_model.pkl"
REPORT_PATH = REPORT_DIR / "YourName_Employee_Attrition_Report.docx"

# Target column name (as it appears in the raw CSV)
TARGET_COL = "Attrition"

# Columns that are constant / non-informative and should be dropped
DROP_COLUMNS = [
    "EmployeeCount",      # always 1 for every record
    "EmployeeNumber",     # a unique ID, carries no predictive signal
    "Over18",             # always 'Y' for every record
    "StandardHours",      # always 80 for every record
]

# Columns that should NEVER be used as input features
# (the target column and any columns that are a direct derivative of it)
FORBIDDEN_COLS = [TARGET_COL]

# Columns to drop from the validation rows before calling predict()
# These are constant columns plus the target, neither of which is a feature.
DROP_FROM_VALIDATION = DROP_COLUMNS + FORBIDDEN_COLS

logger = logging.getLogger(__name__)


def load_dataset(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the IBM HR Analytics employee attrition CSV.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file. Defaults to data/WA_Fn-UseC_-HR-Employee-Attrition.csv

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the expected CSV is not found at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Place the file WA_Fn-UseC_-HR-Employee-Attrition.csv in the data/ folder\n"
            "next to this project, or adjust the DATA_PATH above."
        )

    df = pd.read_csv(path)
    logger.info("Dataset loaded from %s -> shape: %s", path, df.shape)
    return df


def prepare_features(df: pd.DataFrame, random_state: int = 42):
    """
    Split the DataFrame into features (X) and target (y), and
    encode the target from Yes/No to 1/0.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X : pd.DataFrame
        Input features (dropping constant and forbidden columns).
    y : pd.Series
        Target encoded as 1 = Attrition, 0 = No Attrition.
    """
    # Target in original Yes/No form
    y_raw = df[TARGET_COL].copy()

    # Encode Yes -> 1, No -> 0
    y = y_raw.map({"Yes": 1, "No": 0})
    y = y.astype(int)

    # Features = everything except constant/non-informative and the target
    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLUMNS and c not in FORBIDDEN_COLS
    ]

    X = df[feature_cols].copy()

    logger.info("Features: %d columns; target: %s", len(feature_cols), TARGET_COL)
    return X, y


def train_val_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/validation split that respects class imbalance.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    test_size : float
        Proportion of data for the validation set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,           # keep class ratio in both splits
        random_state=random_state,
    )
    return X_train, X_val, y_train, y_val


def write_docx_report(report_path: pathlib.Path, content_sections: dict):
    """
    Write the Word report using python-docx.

    Parameters
    ----------
    report_path : pathlib.Path
        Where to save the .docx file.
    content_sections : dict
        Mapping of section titles to their body text (strings).
    """
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    report_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # ---------- Title page ----------
    doc.add_paragraph()
    doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Employee Attrition Analysis\n& Prediction Using AI & Machine Learning")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run("AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026")
    r2.font.size = Pt(14)
    r2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()
    doc.add_paragraph()
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run(f"Author: [YourName]\nDate: October 10, 2026\nStatus: Executed\n")

    doc.add_page_break()

    # ---------- Body sections ----------
    for heading, body in content_sections.items():
        h = doc.add_heading(heading, level=1)
        if body:
            doc.add_paragraph(body)

    doc.save(str(report_path))
    logger.info("Report saved to %s", report_path)
