import pathlib, json

C = []
def md(src):
    C.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.splitlines()]})
def code(src):
    C.append({"cell_type":"code","execution_count":None,"metadata":{"id":"c"+str(len(C))},"outputs":[],"source":[l+"\n" for l in src.splitlines()]})

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
print("p1 done; cells:", len(C))
pathlib.Path(".tmp/state1.json").write_text(json.dumps(C), encoding="utf-8")
