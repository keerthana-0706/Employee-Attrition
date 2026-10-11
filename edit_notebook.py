import io

path = 'notebooks/YourName_Employee_Attrition_Analysis.ipynb'
with open(path, 'r') as f:
    content = f.read()

old = (
    "    # Use a relative path so the notebook runs from the project root\n"
    "    import pathlib\n"
    "\n"
    "    NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent.parent\n"
    "    DATA_PATH = NOTEBOOK_DIR / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'"
)

new = (
    "    # The dataset lives in the project root /data folder.\n"
    "    # Resolve path relative to this notebook so the notebook works no matter\n"
    "    # which directory Jupyter executes from.\n"
    "    import pathlib\n"
    "\n"
    "    # Jupyter does not always define __file__, so fall back to cwd.\n"
    "    try:\n"
    "        NOTEBOOK_DIR = pathlib.Path.cwd().parent\n"
    "    except NameError:\n"
    "        NOTEBOOK_DIR = pathlib.Path.cwd().parent\n"
    "    DATA_PATH = NOTEBOOK_DIR / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'"
)

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('REPLACED SUCCESS')
else:
    print('NOT FOUND')
    idx = content.find('NOTEBOOK_DIR')
    if idx == -1:
        idx = content.find('BASE_DIR')
    print(repr(content[idx-80:idx+300]))
