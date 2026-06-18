# CLAUDE.md — bosonic-gates

Instructions for AI and human contributors to this tutorial repository.

## Environment Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate bosonic-gates

# Install the package in editable mode with dev extras
pip install -e ".[dev]"

# Register the Jupyter kernel
python -m ipykernel install --user --name bosonic-gates --display-name "Bosonic Gates"
```

Python executable (if calling directly): `python`

## Project Architecture

```
bosonic-gates/
├── src/bosonic_gates/     # Installable library — physics code goes here
│   ├── states.py          # Bosonic state preparation
│   ├── hamiltonians/      # Transmon, harmonic oscillator
│   ├── snail/             # SNAIL potential and circuit parameters
│   ├── driven_kerr/       # Lindblad / Redfield / Floquet-Markov
│   ├── gates/             # SNAP gates, ECD gates, optimal control
│   └── error_budget/      # Per-channel error budget framework
├── notebooks/             # Jupyter tutorial notebooks + companion PDFs
│   ├── module0_foundations/
│   ├── module1_circuits/
│   ├── module2_snail/
│   ├── module3_dynamics/
│   ├── module4_gates/
│   ├── module5_optimal_control/
│   └── module6_error_budget/
├── tests/                 # pytest test suite (library only, not notebooks)
└── build/                 # build_all_pdfs.sh, check_notebooks.sh
```

**Key rule:** `tests/` covers the library in `src/`. Notebooks are smoke-tested separately with `pytest-nbmake`.

## Library Extension Rules

When adding a new physics function to `src/bosonic_gates/`:

1. **Where it goes:** any function used by ≥2 notebooks belongs in the library. Single-notebook helpers stay inside the notebook.
2. **Function signatures:** follow the existing pattern — positional physics parameters, keyword-only optional parameters (`N`, `tlist`, `options`). Example: `def my_op(state, param, *, N=50, options=None)`.
3. **Docstrings:** NumPy-style with an `Example` block using `>>>` notation.
4. **No matplotlib in library modules.** Visualization is always in a separate function or in `visualization.py`.
5. **Gate functions** in `gates/` must accept `qt.Qobj` (not `BosonicState`) and return `qt.Qobj`.
6. **Tests:** every new public function in `src/` needs a corresponding test in `tests/`.

## Notebook Conventions

Each notebook in `notebooks/` follows these conventions:

1. **Parameters cell at the top** (tagged `parameters`). All adjustable quantities go here. Binder users only need to touch this cell.
2. **`# BINDER_FAST` variant:** expensive notebooks include a `# BINDER_FAST` comment block in the parameters cell to reduce N and n_steps for cloud execution.
3. **Theory cells:** plain Markdown with `$...$` and `$$...$$` LaTeX. Use `\hat{a}`, `\hbar`, physics package notation (`\bra{}`, `\ket{}`).
4. **Cell tags:** add `# hide` as the first line of import / rcParams cells that should not appear in the PDF output.
5. **No file output from notebooks.** Figures go into the notebook output cell only. Do not call `plt.savefig` or write any files.
6. **Exercises:** include exercise cells with `# YOUR CODE HERE` and solution cells tagged `solution`. Solutions are hidden in the PDF by the nbconvert template.

### Notebook template (first 3 cells)

```python
# parameters
# BINDER_FAST: use N=6, n_steps=64 for fast cloud execution
N = 20          # Hilbert space truncation
n_steps = 200   # time slices for optimal control
```

```python
# hide
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
%matplotlib widget
from bosonic_gates import ...
```

```markdown
## Module X: Title
**Learning objectives:** ...
```

## PDF Build

Auto-generate PDFs from all notebooks:

```bash
bash build/build_all_pdfs.sh
```

This runs `jupyter nbconvert --to pdf --execute` on every notebook using the shared template in `build/nbconvert_template.tplx`.

The CI workflow `pdf-build.yml` does this automatically on every push to `main` and uploads PDFs as GitHub Actions artifacts.

## Testing

```bash
# Unit tests (fast, ~seconds)
pytest tests/

# Notebook smoke tests (slower, ~minutes per notebook)
pytest --nbmake notebooks/ --nbmake-timeout=600

# Run only fast unit tests with verbose output
pytest tests/ -v
```

## Banned Symbols (Private Content Guard)

CI checks that no file under `src/` or `notebooks/` contains any of these strings:

- `qubitdyne`
- `bosonsampling`
- `ATS_analytics`

This ensures the private research code boundary stays clean. **Do not import or reference these modules.**

## Adding a New Module

1. Create `src/bosonic_gates/mymodule/` with `__init__.py`.
2. Expose public functions in `__init__.py`.
3. Add tests in `tests/test_mymodule.py`.
4. If the module adds a new tutorial topic, create a notebook stub in `notebooks/moduleN_name/`.
5. Update `src/bosonic_gates/__init__.py` if the module exports top-level symbols.

## Key Physics Conventions

- **All frequencies in rad/s** in `driven_kerr/` (multiply GHz by 2π).
- **Hilbert space dimension N** is the Fock truncation; always check convergence by doubling N.
- **Tensor ordering:** qubit ⊗ cavity throughout (qubit is the left factor).
- **Gate fidelity:** Nielsen formula, F = |Tr(U†U_target)|² / d², where d = Hilbert space dimension.
- **Optimal control sign:** GRAPE *maximizes* fidelity; functions return negative fidelity for scipy `minimize`.
