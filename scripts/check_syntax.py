"""Quick AST syntax check for code cells in notebooks."""
import json, ast, sys

for nb_path in [
    "content/module0_foundations/nb0b_phase_space.ipynb",
    "content/module0_foundations/nb0c_open_systems.ipynb",
]:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    errors = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        clean = "\n".join(l for l in src.splitlines() if not l.strip().startswith("%"))
        try:
            ast.parse(clean)
        except SyntaxError as e:
            errors.append((i, str(e), src[:200]))
    n = len(nb["cells"])
    if errors:
        print(f"ERRORS in {nb_path}:")
        for i, msg, snippet in errors:
            print(f"  cell {i}: {msg}")
            print(f"  snippet:\n{snippet}\n")
    else:
        print(f"OK  {nb_path}  ({n} cells)")
