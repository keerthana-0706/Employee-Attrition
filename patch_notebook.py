import pathlib, json, re

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
text = p.read_text(encoding="utf-8")

# Fix structural lines missing a trailing '",'
lines = text.splitlines()
fixed = 0
for i in range(len(lines)):
    s = lines[i].rstrip("\r")
    if s.startswith('    "') and not s.endswith('",') and i+1 < len(lines) and lines[i+1].lstrip().startswith('"'):
        lines[i] = s + '",'
        fixed += 1
text = "\n".join(lines)
print("structural repairs:", fixed)

def attempt_load(t):
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        m = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", str(e))
        if not m:
            raise
        ln, col, char = map(int, m.groups())
        ls = t.splitlines()
        cur = ls[ln-1]
        print(f"FAIL line {ln} col {col}: {cur[:120]!r}")
        # If the failure is an unescaped control char inside a source string
        # because two entries got merged, split on the first '",' that ends an entry.
        if '",' in cur and cur.count('"') >= 4:
            idx = cur.find('",')
            head = cur[:idx+2]
            if head.endswith('\\n",'):
                tail = cur[idx+2:]
                ls[ln-1] = head
                ls.insert(ln, tail)
                return attempt_load("\n".join(ls))
        raise

nb = attempt_load(text)
p.write_text(text, encoding="utf-8")
print("json OK; cells", len(nb["cells"]))

# Now rebuild the example-prediction cell source from scratch so we know it is clean.
sample_src = r'''# ============================================================
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

# Ensure column order matches the model input.
sample = sample[X_model.columns]

pred_class = loaded.predict(sample)[0]
pred_label = 'Yes (likely to leave)' if pred_class == 1 else 'No (likely to stay)'
pred_prob = loaded.predict_proba(sample)[0]

print('Sample input (single record):')
print(sample.T)
print('\nPrediction class (0=No, 1=Yes):', int(pred_class))
print('Predicted label                :', pred_label)
print('Probability No / Yes           : {:.3f} / {:.3f}'.format(float(pred_prob[0]), float(pred_prob[1])))

print('\nIMPORTANT:')
print('  This is an educational estimate from a model trained on fictional data.')
print('  It is NOT a validated employment decision tool and must not be used as the')
print('  sole basis for any decision about a real person.')
'''

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell_24",
    "metadata": {},
    "outputs": [],
    "source": sample_src.splitlines(keepends=True),
}

# Replace existing cell with id cell_24
for i, c in enumerate(nb["cells"]):
    if c.get("metadata", {}).get("id") == "cell_24":
        nb["cells"][i] = new_cell
        print("replaced example-prediction cell")
        break
else:
    raise SystemExit("cell_24 not found")

nbformat_version = nb.get("nbformat", 4)
nbformat_minor = nb.get("nbformat_minor", 5)

out = {
    "nbformat": nbformat_version,
    "nbformat_minor": nbformat_minor,
    "metadata": nb.get("metadata", {}),
    "cells": nb["cells"],
}

p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

# Validate the rewritten file
valid = json.loads(p.read_text(encoding="utf-8"))
print("rewritten notebook valid; cells", len(valid["cells"]))
print("cell_24 source length:", len(valid["cells"][-2]["source"]) if valid["cells"][-2]["metadata"].get("id")=="cell_24" else None)
