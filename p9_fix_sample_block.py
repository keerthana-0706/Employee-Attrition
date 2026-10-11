import pathlib, json, re

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
text = p.read_text(encoding="utf-8")
text = text[: text.rfind("}") + 1]

lines = text.splitlines()

start = None
end = None
for i, l in enumerate(lines):
    if l.strip().startswith('"sample = pd.DataFrame({'):
        start = i
    if start is not None and l.strip().startswith('"# Ensure column order matches the model input.'):
        end = i
        break

print("start line:", start + 1, "end line:", end + 1)
print("context before:", repr(lines[start-1]))
print("context after:", repr(lines[end]))
print("bad block snippet:")
print(repr("\n".join(lines[start:end])))

# Build the corrected block.
fixed_block = [
    '    "sample = pd.DataFrame({\\n",',
    '    "    \'Age\': [32],\\n",',
    '    "    \'BusinessTravel\': [\'Travel_Rarely\'],\\n",',
    '    "    \'DailyRate\': [800],\\n",',
    '    "    \'Department\': [\'Sales\'],\\n",',
    '    "    \'DistanceFromHome\': [5],\\n",',
    '    "    \'Education\': [3],\\n",',
    '    "    \'EducationField\': [\'Life Sciences\'],\\n",',
    '    "    \'EnvironmentSatisfaction\': [3],\\n",',
    '    "    \'Gender\': [\'Male\'],\\n",',
    '    "    \'HourlyRate\': [60],\\n",',
    '    "    \'JobInvolvement\': [3],\\n",',
    '    "    \'JobLevel\': [2],\\n",',
    '    "    \'JobRole\': [\'Sales Executive\'],\\n",',
    '    "    \'JobSatisfaction\': [3],\\n",',
    '    "    \'MonthlyIncome\': [4500],\\n",',
    '    "    \'MonthlyRate\': [11000],\\n",',
    '    "    \'NumCompaniesWorked\': [2],\\n",',
    '    "    \'Over18\': [\'Y\'],\\n",',
    '    "    \'OverTime\': [\'Yes\'],\\n",',
    '    "    \'PercentSalaryHike\': [12],\\n",',
    '    "    \'PerformanceRating\': [3],\\n",',
    '    "    \'RelationshipSatisfaction\': [3],\\n",',
    '    "    \'StandardHours\': [80],\\n",',
    '    "    \'StockOptionLevel\': [0],\\n",',
    '    "    \'TotalWorkingYears\': [8],\\n",',
    '    "    \'TrainingTimesLastYear\': [3],\\n",',
    '    "    \'WorkLifeBalance\': [3],\\n",',
    '    "    \'YearsAtCompany\': [4],\\n",',
    '    "    \'YearsInCurrentRole\': [2],\\n",',
    '    "    \'YearsSinceLastPromotion\': [2],\\n",',
    '    "    \'YearsWithCurrManager\': [2],\\n",',
    '    "})\\n",',
    '    "\\n",',
]

assert start is not None and end is not None
new_lines = lines[:start] + fixed_block + lines[end:]
new_text = "\n".join(new_lines)

# Validate
try:
    nb = json.loads(new_text)
    print("json OK after sample block replacement; cells:", len(nb["cells"]))
except json.JSONDecodeError as e:
    print("still invalid:", e)
    m = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", str(e))
    if m:
        ln = int(m.group(1))
        print("offending line:", ln, repr(new_lines[ln-1] if ln-1 < len(new_lines) else ""))
    raise SystemExit(2)

p.write_text(new_text, encoding="utf-8")
print("rewrote notebook file")
