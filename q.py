import pathlib, json, nbformat, warnings, io, uuid
from nbclient import NotebookClient

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
nb = nbformat.read(p, as_version=4)
for c in nb.cells:
    c.setdefault("metadata",{})
    if c.cell_type=="code":
        if "id" not in c:
            c["id"]=uuid.uuid4().hex
        c["outputs"]=[]
        if c.get("execution_count") not in (None,):
            if isinstance(c["execution_count"],str):
                try: c["execution_count"]=int(c["execution_count"])
                except: c["execution_count"]=None
warnings.filterwarnings("ignore")
out=io.StringIO()
client=NotebookClient(nb, timeout=120, kernel_name="python3")
client.execute()
nbformat.write(nb,p)
print("OK cells:", len(nb.cells))
print(out.getvalue()[-6000:])
