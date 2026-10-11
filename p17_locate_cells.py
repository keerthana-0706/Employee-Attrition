import pathlib
p = pathlib.Path(".tmp/p14_build_nb_json.py")
text = p.read_text(encoding="utf-8")

for header in [
    "# STEP 8c: ROC curve comparison",
    "# STEP 9: Model interpretation - permutation importance",
    "# STEP 11: Example prediction on a valid sample record",
]:
    idx = text.find(header)
    print("HEADER:", header)
    print("IDX:", idx)
    if idx >= 0:
        print("CONTEXT:")
        print(repr(text[idx-120:idx+40]))
    print("="*80)
