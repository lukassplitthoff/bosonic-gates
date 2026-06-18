# Contributing to bosonic-gates

Thank you for your interest in contributing. This document covers the practical steps for adding to the library or notebooks.

## Getting started

```bash
git clone https://github.com/lukassplitthoff/bosonic-gates.git
cd bosonic-gates
conda env create -f environment.yml
conda activate bosonic-gates
pip install -e ".[dev]"
```

## What belongs where

| Location | Rule |
|----------|------|
| `src/bosonic_gates/` | Physics functions used by ≥2 notebooks |
| `notebooks/moduleN_*/` | Tutorial notebooks and companion content |
| `tests/` | Unit tests for the library (not notebooks) |

Single-notebook helpers should stay inside the notebook. Move to the library once a second notebook needs the same function.

## Adding a library function

1. Put it in the appropriate submodule (or create `src/bosonic_gates/mymodule/` with `__init__.py`).
2. Follow the existing signature pattern: positional physics parameters, keyword-only optional parameters (`N`, `tlist`, `options`).
3. Write a NumPy-style docstring with an `Example` block.
4. No `matplotlib` in library modules — visualization lives in the notebook or in `visualization.py`.
5. Add a test in `tests/test_mymodule.py`.
6. Expose public symbols in the submodule's `__init__.py`; add top-level exports to `src/bosonic_gates/__init__.py` if warranted.

## Adding or editing a notebook

- Keep a `parameters` cell at the top (tagged `parameters`). All user-adjustable quantities go there.
- Expensive notebooks must include a `# BINDER_FAST` block to reduce `N` and `n_steps` for cloud execution.
- Do not call `plt.savefig` or write any files from a notebook cell.
- Add `# hide` as the first line of cells (imports, rcParams) that should not appear in the PDF.
- Exercises use `# YOUR CODE HERE`; solution cells are tagged `solution`.

## Running the tests

```bash
# Library unit tests
pytest tests/ -v

# Notebook smoke tests (slow)
pytest --nbmake notebooks/ --nbmake-timeout=600
```

All tests must pass before opening a pull request.

## Physics conventions

- Frequencies in `driven_kerr/` are in **rad/s** (multiply GHz values by 2π).
- Hilbert space truncation is `N`; verify convergence by doubling it.
- Tensor ordering is **qubit ⊗ cavity** throughout.
- Gate fidelity uses the Nielsen formula: `F = |Tr(U†U_target)|² / d²`.

## Pull request checklist

- [ ] New library functions have tests.
- [ ] Notebooks execute cleanly (`pytest --nbmake`).
- [ ] No `plt.savefig` or file writes in notebook cells.
- [ ] No references to private modules (`qubitdyne`, `bosonsampling`, `ATS_analytics`).

## Questions

Open an issue or reach out via the repository discussions.
