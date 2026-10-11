import pathlib, json, re

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
text = p.read_text(encoding="utf-8")

# Trim trailing garbage if any (there shouldn't be, but safe)
text = text[: text.rfind("}") + 1]

lines = text.splitlines(keepends=True)

# Find cell object boundaries by scanning for '"source": [' ... '   ]' pairs
# associated with the id c24.
def find_cell_span(lines, cell_id):
    # locate '    "id": "<cell_id>"'
    for i, l in enumerate(lines):
        if f'"id": "{cell_id}"' in l:
            id_line = i
            break
    else:
        return None
    # find the enclosing object start: the previous '  {' at lower indent
    obj_start = None
    j = id_line
    while j >= 0:
        if lines[j].strip() == '{':
            obj_start = j
            break
        j -= 1
    # find the matching object end: count braces from obj_start
    depth = 0
    obj_end = None
    for k in range(obj_start, len(lines)):
        for ch in lines[k]:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    obj_end = k
                    break
        if obj_end is not None:
            break
    return obj_start, obj_end

span = find_cell_span(lines, "c24")
print("c24 span:", span)
if span is None:
    raise SystemExit("c24 cell object not found")

obj_start, obj_end = span
print("object start line:", obj_start + 1, "end line:", obj_end + 1)

# Find the "source": [ ... ] within this object and replace it.
src_start = None
src_end = None
for k in range(obj_start, obj_end + 1):
    if lines[k].strip().startswith('"source": ['):
        src_start = k
        # find closing '   ]' for source array
        depth = 0
        for m in range(k, obj_end + 1):
            for ch in lines[m]:
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        src_end = m
                        break
            if src_end is not None:
                break
        break

print("source start line:", src_start + 1, "source end line:", src_end + 1)

clean_source = """   "source": [
    "# ============================================================\n",
    "# STEP 11: Example prediction on a valid sample record\n",
    "# ============================================================\n",
    "# Build one realistic-looking sample using the same columns the model expects.\n",
    "# The pipeline handles encoding and scaling internally.\n",
    "\n",
    "sample = pd.DataFrame({\n",
    "    'Age': [32],\n",
    "    'BusinessTravel': ['Travel_Rarely'],\n",
    "    'DailyRate': [800],\n",
    "    'Department': ['Sales'],\n",
    "    'DistanceFromHome': [5],\n",
    "    'Education': [3],\n",
    "    'EducationField': ['Life Sciences'],\n",
    "    'EnvironmentSatisfaction': [3],\n",
    "    'Gender': ['Male'],\n",
    "    'HourlyRate': [60],\n",
    "    'JobInvolvement': [3],\n",
    "    'JobLevel': [2],\n",
    "    'JobRole': ['Sales Executive'],\n",
    "    'JobSatisfaction': [3],\n",
    "    'MonthlyIncome': [4500],\n",
    "    'MonthlyRate': [11000],\n",
    "    'NumCompaniesWorked': [2],\n",
    "    'Over18': ['Y'],\n",
    "    'OverTime': ['Yes'],\n",
    "    'PercentSalaryHike': [12],\n",
    "    'PerformanceRating': [3],\n",
    "    'RelationshipSatisfaction': [3],\n",
    "    'StandardHours': [80],\n",
    "    'StockOptionLevel': [0],\n",
    "    'TotalWorkingYears': [8],\n",
    "    'TrainingTimesLastYear': [3],\n",
    "    'WorkLifeBalance': [3],\n",
    "    'YearsAtCompany': [4],\n",
    "    'YearsInCurrentRole': [2],\n",
    "    'YearsSinceLastPromotion': [2],\n",
    "    'YearsWithCurrManager': [2],\n",
    "})\n",
    "\n",
    "# Ensure column order matches the model input.\n",
    "sample = sample[X_model.columns]\n",
    "\n",
    "pred_class = loaded.predict(sample)[0]\n",
    "pred_label = 'Yes (likely to leave)' if pred_class == 1 else 'No (likely to stay)'\n",
    "pred_prob = loaded.predict_proba(sample)[0]\n",
    "\n",
    "print('Sample input (single record):')\n",
    "print(sample.T)\n",
    "print('\\nPrediction class (0=No, 1=Yes):', int(pred_class))\n",
    "print('Predicted label                :', pred_label)\n",
    "print('Probability No / Yes           : {:.3f} / {:.3f}'.format(float(pred_prob[0]), float(pred_prob[1])))\n",
    "\n",
    "print('\\nIMPORTANT:')\n",
    "print('  This is an educational estimate from a model trained on fictional data.')\n",
    "print('  It is NOT a validated employment decision tool and must not be used as the')\n",
    "print('  sole basis for any decision about a real person.')\n",
   ],
"""

# Remove trailing newline from clean_source to splice cleanly
clean_source_lines = clean_source.splitlines(keepends=True)

new_lines = lines[:src_start] + clean_source_lines + lines[src_end+1:]
new_text = "".join(new_lines)

try:
    nb = json.loads(new_text)
    print("json OK after object replace; cells:", len(nb["cells"]))
except json.JSONDecodeError as e:
    print("still invalid:", e)
    m = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", str(e))
    if m:
        ln = int(m.group(1))
        print("offending line:", ln, repr(new_lines[ln-1] if ln-1 < len(new_lines) else ""))
    raise SystemExit(2)

p.write_text(new_text, encoding="utf-8")
print("rewrote notebook file")
