from __future__ import annotations

import numpy as np
import pytest

import Librex.QAP_backend as qb


def test_as_qap_matrices_from_AB_valid():
    A = [[1, 2], [3, 4]]
    B = [[4, 3], [2, 1]]
    AA, BB = qb._as_qap_matrices({"A": A, "B": B})
    assert isinstance(AA, np.ndarray) and isinstance(BB, np.ndarray)
    assert AA.shape == (2, 2) and BB.shape == (2, 2)


def test_as_qap_matrices_from_AB_mismatch_raises():
    A = [[1, 2], [3, 4]]
    B = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(ValueError):
        qb._as_qap_matrices({"A": A, "B": B})


def test_as_qap_matrices_from_fit_defaults_B_identity():
    fit = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    A, B = qb._as_qap_matrices({"fit": fit})
    assert A.shape == (3, 3) and B.shape == (3, 3)
    assert np.allclose(B, np.eye(3))

