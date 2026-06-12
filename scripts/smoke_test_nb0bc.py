"""
Smoke-test: extract the first 3 cells (params + imports + helpers) from nb0b and nb0c
and exec them to confirm all imports and function definitions succeed.
"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

for nb_path, name in [
    ("content/module0_foundations/nb0b_phase_space.ipynb", "nb0b"),
    ("content/module0_foundations/nb0c_open_systems.ipynb", "nb0c"),
]:
    print(f"\n{'='*60}")
    print(f"Smoke testing {name}: params + imports + helpers")
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {}
    errors = []
    for i, cell in enumerate(code_cells[:3]):
        src = "".join(cell["source"])
        # Skip magic lines
        clean = "\n".join(l for l in src.splitlines() if not l.strip().startswith("%"))
        try:
            exec(clean, ns)
            print(f"  cell {i} OK")
        except Exception as e:
            print(f"  cell {i} FAILED: {e}")
            errors.append((i, e))

    if errors:
        print(f"  !! {len(errors)} errors in {name}")
    else:
        # Quick functional check for nb0b helpers
        if name == "nb0b":
            import numpy as np
            ns["N"] = ns["N"]
            s = ns["coherent_state"](1.0, N=ns["N"])
            q = ns["mandel_Q"](s)
            nv = ns["wigner_neg_volume"](s)
            print(f"  mandel_Q(coherent)={q:.4f}  (expect ~0)")
            print(f"  wigner_neg_volume(coherent)={nv:.4f}  (expect ~0)")
        if name == "nb0c":
            import numpy as np
            c_ops = ns["build_c_ops"](ns["N"], ns["kappa"], ns["n_bar_bath"], ns["gamma_phi"])
            print(f"  build_c_ops returned {len(c_ops)} operators")

print("\nDone.")
