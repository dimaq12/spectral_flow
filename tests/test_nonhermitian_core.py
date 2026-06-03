"""Non-Hermitian spectral geometry core contracts."""
import numpy as np

from sft.core import OperatorFamily, BiorthogonalState


def _pt_family(gamma: float = 0.3) -> OperatorFamily:
    A0 = np.array([[1j * gamma, 1.0], [1.0, -1j * gamma]], dtype=np.complex128)
    basis = [
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
    ]
    return OperatorFamily(A0, basis, hermitian=False)


def test_nonhermitian_biorthogonal_state_normalized():
    fam = _pt_family()
    state = fam.biorthogonal_state
    assert isinstance(state, BiorthogonalState)
    gram = state.left_vecs.conj().T @ state.right_vecs
    np.testing.assert_allclose(gram, np.eye(fam.N), atol=1e-8)


def test_nonhermitian_kernel_is_complex_and_matches_finite_difference():
    fam = _pt_family()
    assert np.iscomplexobj(fam.W)
    eps = 1e-6
    lam0 = fam.lam0
    lam1 = fam.spectrum(np.array([eps, 0.0]))
    fd = (lam1 - lam0) / eps
    np.testing.assert_allclose(fd, fam.W[:, 0], atol=1e-5)


def test_nonhermitian_inverse_uses_real_parameter_step():
    fam = _pt_family()
    target = fam.spectrum(np.array([0.02, 0.0]))
    result = fam.inverse(target, steps=4, alpha=0.5)
    assert result.k.shape == (fam.M,)
    assert np.all(np.isfinite(result.k))
    assert result.method == "linear"
    assert result.residual_history.shape[0] >= 1
