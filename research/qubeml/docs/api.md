# QubeML API Documentation

QubeML provides a small set of **utility modules** under the `src` package that are used by the
notebooks for quantum computing and materials informatics. This page summarizes the practical
API surface you are expected to use.

> Note: In code we typically import directly from `src.*` inside this repository. When/if the
> package is renamed for distribution (e.g. to `qubeml`), the imports will be updated, but the
> module structure and functions will remain the same.

---

## 1. Quantum Utilities (`src.quantum_utils`)

Helpers for building and analyzing small quantum states, used heavily in the quantum
computing notebooks and unit tests.

Typical import pattern:

```python
from src import quantum_utils as qu
# or
from src.quantum_utils import create_bell_state, create_ghz_state, compute_fidelity
```

Core capabilities (non-exhaustive):

- **State preparation**
  - Create Bell pairs and GHZ states for a given number of qubits.
  - Initialize computational basis states and simple superpositions.
- **Operators and evolution**
  - Apply single-qubit Pauli operators (X, Y, Z) to multi-qubit states.
  - Apply simple noise channels and depolarizing noise to test robustness.
- **Measurement & analysis**
  - Sample measurement outcomes from a state in the computational basis.
  - Compute state fidelity between two state vectors or density matrices.
  - Estimate entanglement entropy for bipartite partitions.

These utilities are designed to be **lightweight and NumPy-based**, so they run quickly in
standard Python environments and in CI.

---

## 2. Materials Utilities (`src.materials_utils`)

Utilities for basic **materials informatics workflows**, especially when working with
crystal structures, composition-based features, and tabular datasets.

Typical import pattern:

```python
from src import materials_utils as mu
# or
from src.materials_utils import (
    featurize_composition,
    train_test_split_dataframe,
)
```

Core capabilities (non-exhaustive):

- **Feature engineering**
  - Construct simple composition-based features for materials (e.g., averages over element
    properties, counts of species, etc.).
- **Dataset handling**
  - Helper functions for splitting pandas DataFrames into train/validation/test sets with
    reproducible random seeds.
  - Basic normalization / scaling wrappers suitable for small demos.
- **Interop with notebooks**
  - Small helpers used by the PyTorch / scikit-learn notebooks for cleaning inputs and
    wiring up datasets.

These utilities intentionally stay **minimal** so that the notebooks can show the core
ideas without hiding too much behind a complex API.

---

## 3. Plotting Utilities (`src.plotting_utils`)

Convenience plotting helpers used throughout the notebooks to keep them readable and
consistent.

Typical import pattern:

```python
from src import plotting_utils as pu
# or
from src.plotting_utils import plot_convergence, plot_materials_scatter
```

Core capabilities (non-exhaustive):

- **Quantum experiments**
  - Plot probability distributions of measurement outcomes.
  - Visualize convergence curves for variational algorithms.
- **Materials informatics**
  - Scatter plots and histograms for materials properties (e.g., band gaps).
  - Parity plots comparing predictions vs. ground truth.

All plotting utilities are built on top of Matplotlib (and optionally Seaborn) and are
intended for quick, publication-quality figures in the example notebooks.

---

## 4. Testing & Stability

The unit tests in `tests/test_quantum_utils.py` exercise the core quantum utilities
(e.g., Bell states, GHZ states, fidelity, entanglement entropy, simple noise models).
All tests currently pass in the standardized workspace environment.

If you add new utilities, prefer to:

1. Add them to one of the existing modules (`quantum_utils`, `materials_utils`,
   `plotting_utils`) rather than creating many small files.
2. Add a minimal usage example in an appropriate notebook.
3. Add at least one unit test that checks basic correctness and shapes.
