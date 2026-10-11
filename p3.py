import pathlib, json

C = json.loads(pathlib.Path(".tmp/state2.json").read_text(encoding="utf-8"))
def code(src):
    C.append({"cell_type":"code","execution_count":None,"metadata":{"id":"c"+str(len(C))},"outputs":[],"source":[l+"\n" for l in src.splitlines()]})
def md(src):
    C.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.splitlines()]})

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

pathlib.Path(".tmp/state3.json").write_text(json.dumps(C), encoding="utf-8")
print("p3 done; cells:", len(C))
