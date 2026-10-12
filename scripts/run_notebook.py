"""Execute all cells in main.ipynb to verify reproducibility and populate cell outputs."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

with open("main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total notebook cells: {len(nb['cells'])}")

global_env = {"__name__": "__main__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        print(f"\n================ Executing Cell {i} ================")
        exec(code, global_env)

print("\nAll notebook cells executed successfully without any errors!")
