import pathlib, json

p = pathlib.Path("notebooks/YourName_Employee_Attrition_Analysis.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

new_cell_20_source = [
    "# ============================================================\n",
    "# STEP 8c: ROC curve comparison\n",
    "# ============================================================\n",
    "from sklearn.metrics import roc_curve\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7, 6))\n",
    "\n",
    "for name, pipe in fitted_models.items():\n",
    "    y_prob = pipe.predict_proba(X_test)[:, 1]\n",
    "    fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
    "    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_prob):.3f})')\n",
    "\n",
    "ax.plot([0, 1], [0, 1], linestyle='--', color='gray')\n",
    "ax.set_title('ROC curves on the held-out test set')\n",
    "ax.set_xlabel('False positive rate')\n",
    "ax.set_ylabel('True positive rate')\n",
    "ax.legend(loc='lower right')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print('Interpretation:')\n",
    "print('  ROC-AUC summarizes ranking quality across thresholds. Higher is better,')\n",
    "print('  but the best threshold for action depends on how costly missed leavers are')\n",
    "print('  relative to false alarms. That choice is a business decision, not a model one.')\n",
]

for c in nb["cells"]:
    if c.get("cell_type") == "code" and "STEP 8c: ROC curve comparison" in "".join(c["source"]):
        c["source"] = new_cell_20_source
        print("patched ROC cell")
        break
else:
    raise SystemExit("ROC cell not found")

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("rewrote notebook; cells:", len(nb["cells"]))

# validate
json.loads(p.read_text(encoding="utf-8"))
print("json OK")
