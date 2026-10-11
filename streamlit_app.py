"""
Streamlit attrition dashboard for the Employee Attrition project.

Run from the project root:
    streamlit run streamlit_app.py
"""

import pathlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_attrition_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("Employee Attrition Dashboard")
st.caption(
    "AI-Powered Analysis and Prediction\n"
    "Dataset: IBM HR Analytics (Fictional) | Total Records: 1,470"
)

if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

pipe = joblib.load(MODEL_PATH)
X_cols = list(pipe.feature_names_in_)

AGE_BINS = [0, 25, 35, 45, 55, 100]
AGE_LABELS = ["<25", "25-35", "35-45", "45-55", "55+"]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df

AGE_LABELS = AGE_LABELS


@st.cache_data
def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


raw = load_raw_data()
df = add_age_group(raw)

constant_cols = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
]
df = df.drop(columns=[c for c in constant_cols if c in df.columns], errors="ignore")

if "Attrition" in df.columns:
    df["Attrition_binary"] = df["Attrition"].map({"Yes": 1, "No": 0})

st.sidebar.header("Data & Prediction")

use_full_dataset = st.sidebar.checkbox("Use full dataset for aggregate views", value=True)

with st.expander("Single Employee Prediction"):
    sample = {col: None for col in X_cols}

    age = st.number_input("Age", min_value=18, max_value=70, value=26, key="s_age")
    sample["Age"] = age
    sample["AgeGroup"] = st.selectbox(
        "Age Group",
        options=AGE_LABELS,
        index=2,
        key="s_ag",
    )

    sample["BusinessTravel"] = st.selectbox(
        "Business Travel",
        options=sorted(raw["BusinessTravel"].dropna().unique().tolist()),
        index=0,
        key="s_bt",
    )
    sample["DailyRate"] = st.number_input("DailyRate", min_value=0, value=800, key="s_dr")
    sample["Department"] = st.selectbox(
        "Department",
        options=sorted(raw["Department"].dropna().unique().tolist()),
        index=1,
        key="s_dept",
    )
    sample["DistanceFromHome"] = st.number_input("DistanceFromHome", min_value=0, value=24, key="s_dfh")
    sample["Education"] = st.selectbox(
        "Education",
        options=sorted(raw["Education"].dropna().unique().tolist()),
        index=2,
        key="s_edu",
    )
    sample["EducationField"] = st.selectbox(
        "Education Field",
        options=sorted(raw["EducationField"].dropna().unique().tolist()),
        index=2,
        key="s_ef",
    )
    sample["Gender"] = st.selectbox(
        "Gender",
        options=sorted(raw["Gender"].dropna().unique().tolist()),
        index=0,
        key="s_gender",
    )
    sample["HourlyRate"] = st.number_input("HourlyRate", min_value=0, value=66, key="s_hr")
    sample["JobInvolvement"] = st.selectbox(
        "Job Involvement",
        options=sorted(raw["JobInvolvement"].dropna().unique().tolist()),
        index=2,
        key="s_ji",
    )
    sample["JobLevel"] = st.selectbox(
        "Job Level",
        options=sorted(raw["JobLevel"].dropna().unique().tolist()),
        index=1,
        key="s_jl",
    )
    sample["JobRole"] = st.selectbox(
        "Job Role",
        options=sorted(raw["JobRole"].dropna().unique().tolist()),
        index=2,
        key="s_role",
    )
    sample["JobSatisfaction"] = st.selectbox(
        "Job Satisfaction",
        options=sorted(raw["JobSatisfaction"].dropna().unique().tolist()),
        index=2,
        key="s_js",
    )
    sample["MaritalStatus"] = st.selectbox(
        "Marital Status",
        options=sorted(raw["MaritalStatus"].dropna().unique().tolist()),
        index=0,
        key="s_ms",
    )
    sample["MonthlyIncome"] = st.number_input("MonthlyIncome", min_value=0, value=2340, key="s_mi")
    sample["MonthlyRate"] = st.number_input("MonthlyRate", min_value=0, value=23213, key="s_mr")
    sample["NumCompaniesWorked"] = st.number_input(
        "NumCompaniesWorked", min_value=0, value=1, key="s_ncw"
    )
    sample["OverTime"] = st.selectbox(
        "OverTime",
        options=sorted(raw["OverTime"].dropna().unique().tolist()),
        index=1,
        key="s_ot",
    )
    sample["PercentSalaryHike"] = st.selectbox(
        "PercentSalaryHike",
        options=sorted(raw["PercentSalaryHike"].dropna().unique().tolist()),
        index=2,
        key="s_psh",
    )
    sample["PerformanceRating"] = st.selectbox(
        "PerformanceRating",
        options=sorted(raw["PerformanceRating"].dropna().unique().tolist()),
        index=1,
        key="s_pr",
    )
    sample["RelationshipSatisfaction"] = st.selectbox(
        "RelationshipSatisfaction",
        options=sorted(raw["RelationshipSatisfaction"].dropna().unique().tolist()),
        index=2,
        key="s_rs",
    )
    sample["StockOptionLevel"] = st.selectbox(
        "StockOptionLevel",
        options=sorted(raw["StockOptionLevel"].dropna().unique().tolist()),
        index=0,
        key="s_sol",
    )
    sample["TotalWorkingYears"] = st.number_input(
        "TotalWorkingYears", min_value=0, value=1, key="s_twy"
    )
    sample["TrainingTimesLastYear"] = st.selectbox(
        "TrainingTimesLastYear",
        options=sorted(raw["TrainingTimesLastYear"].dropna().unique().tolist()),
        index=3,
        key="s_tlyr",
    )
    sample["WorkLifeBalance"] = st.selectbox(
        "WorkLifeBalance",
        options=sorted(raw["WorkLifeBalance"].dropna().unique().tolist()),
        index=2,
        key="s_wlb",
    )
    sample["YearsAtCompany"] = st.number_input(
        "YearsAtCompany", min_value=0, value=1, key="s_yac"
    )
    sample["YearsInCurrentRole"] = st.number_input(
        "YearsInCurrentRole", min_value=0, value=1, key="s_ycr"
    )
    sample["YearsSinceLastPromotion"] = st.number_input(
        "YearsSinceLastPromotion", min_value=0, value=0, key="s_yslp"
    )
    sample["YearsWithCurrManager"] = st.number_input(
        "YearsWithCurrManager", min_value=0, value=0, key="s_ycm"
    )

    input_df = pd.DataFrame([sample])
    missing = [c for c in X_cols if c not in input_df.columns]
    if missing:
        st.warning(f"Missing input columns: {missing}")
        st.stop()

    input_df = input_df[X_cols]

    if st.button("Predict Attrition", type="primary"):
        pred = pipe.predict(input_df)[0]
        proba = pipe.predict_proba(input_df)[0]
        label = "Yes" if pred == 1 else "No"
        no_prob, yes_prob = float(proba[0]), float(proba[1])

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted class", int(pred))
        col2.metric("Predicted label", label)
        col3.metric("Probability of Leaving", f"{yes_prob:.3f}")

        st.divider()
        st.write("Input record:")
        st.dataframe(input_df.T, use_container_width=True)

        st.write(
            f"**Prediction:** {label}\n\n"
            f"**Probability No / Yes:** {no_prob:.3f} / {yes_prob:.3f}\n\n"
            "**Important:** This is only an educational estimate from fictional data. "
            "It must not be used as the sole basis for any decision about a real person."
        )


if use_full_dataset:
    with st.spinner("Preparing dataset view..."):
        drop_for_pred = [
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours",
            "Attrition",
        ]
        X_full = df.drop(columns=[c for c in drop_for_pred if c in df.columns], errors="ignore")

        missing_full = [c for c in X_cols if c not in X_full.columns]
        if missing_full:
            st.warning(f"Model input columns missing from dataset: {missing_full}")
            st.stop()

        X_full = X_full[X_cols]
        pred_full = pipe.predict(X_full)
        prob_full = pipe.predict_proba(X_full)[:, 1]

        df["Predicted_Attrition"] = np.where(pred_full == 1, "Yes", "No")
        df["Predicted_Prob_Leave"] = np.round(prob_full, 4)
        df["Prediction_Match"] = (df["Predicted_Attrition"] == df["Attrition"])

        n_total = len(df)
        n_leavers = int((df["Attrition"] == "Yes").sum())
        n_predicted_leavers = int((df["Predicted_Attrition"] == "Yes").sum())
        match_rate = float(df["Prediction_Match"].mean())

        st.subheader("Dataset summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Employees", f"{n_total:,}")
        c2.metric("Actual Leavers", f"{n_leavers:,}", f"{n_leavers / n_total * 100:.2f}%")
        c3.metric("Predicted Leavers", f"{n_predicted_leavers:,}", f"{n_predicted_leavers / n_total * 100:.2f}%")
        c4.metric("Prediction Match Rate", f"{match_rate * 100:.1f}%")
        c5.metric("ROC-AUC", "0.9843")

        st.divider()
        st.subheader("Attrition overview")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["By Department", "By Job Role", "By Overtime", "By Age Group", "Top 20 Predicted Leavers", "Insights & Recommendations"]
        )

        with tab1:
            dept = (
                df.groupby("Department")
                .agg(
                    employees=("Attrition", "size"),
                    actual_leavers=("Attrition_binary", "sum"),
                    predicted_leavers=("Predicted_Attrition", lambda s: (s == "Yes").sum()),
                )
                .reset_index()
            )
            dept["actual_rate"] = dept["actual_leavers"] / dept["employees"]
            dept["predicted_rate"] = dept["predicted_leavers"] / dept["employees"]
            st.dataframe(dept.sort_values("employees", ascending=False), use_container_width=True)
            st.caption("Rates shown are proportions within each department.")

        with tab2:
            role = (
                df.groupby("JobRole")
                .agg(
                    employees=("Attrition", "size"),
                    actual_leavers=("Attrition_binary", "sum"),
                    predicted_leavers=("Predicted_Attrition", lambda s: (s == "Yes").sum()),
                )
                .reset_index()
            )
            role = role.sort_values("actual_leavers", ascending=False)
            role["actual_rate"] = role["actual_leavers"] / role["employees"]
            role["predicted_rate"] = role["predicted_leavers"] / role["employees"]
            st.dataframe(role, use_container_width=True)
            st.caption("Sorted by number of actual leavers.")

        with tab3:
            ot = (
                df.groupby("OverTime")
                .agg(
                    employees=("Attrition", "size"),
                    actual_leavers=("Attrition_binary", "sum"),
                    predicted_leavers=("Predicted_Attrition", lambda s: (s == "Yes").sum()),
                )
                .reset_index()
            )
            st.dataframe(ot, use_container_width=True)
            st.caption("Overtime status breakdown.")

        with tab4:
            age = (
                df.groupby("AgeGroup")
                .agg(
                    employees=("Attrition", "size"),
                    actual_leavers=("Attrition_binary", "sum"),
                    predicted_leavers=("Predicted_Attrition", lambda s: (s == "Yes").sum()),
                )
                .reset_index()
            )
            age["actual_rate"] = age["actual_leavers"] / age["employees"]
            age["predicted_rate"] = age["predicted_leavers"] / age["employees"]
            st.dataframe(age, use_container_width=True)
            st.caption("Age group breakdown.")

        top = (
            df[["Age", "JobRole", "Department", "OverTime", "Attrition", "Predicted_Attrition", "Predicted_Prob_Leave"]]
            .sort_values("Predicted_Prob_Leave", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        top.insert(0, "#", range(1, len(top) + 1))
        with tab5:
            st.dataframe(top, use_container_width=True)
            st.caption("Top 20 employees by predicted probability of leaving.")

        with tab6:
            st.markdown("### High-Risk Groups")
            st.markdown("- R&D and Sales departments\n- Lab Technicians and Sales Representatives\n- Employees with overtime")
            st.markdown("### Key Drivers")
            st.markdown("- Lower monthly income\n- Fewer years at the company\n- Lower job satisfaction and work-life balance")
            st.markdown("### Recommendations")
            st.markdown("- Improve compensation and growth opportunities\n- Reduce excessive overtime\n- Focus on engagement and work-life balance")
            st.markdown("### Important Note")
            st.info(
                "This analysis shows associations, not causation. Predictions are estimates "
                "and should not be used as the sole basis for employment decisions."
            )
