"""Sparse and partial-spectrum graph contracts."""
import numpy as np
from scipy import sparse

import sft
from sft.core import OperatorFamily, edge_laplacian_basis


def _path_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def test_sparse_operatorfamily_partial_spectrum_matches_dense_prefix():
    n = 40
    edges = _path_edges(n)
    adj = sft.graph_gen.path_graph(n)
    L_dense = np.diag(adj.sum(axis=1)) - adj
    L_sparse = sparse.csc_matrix(L_dense)
    fam = OperatorFamily(L_sparse, edge_laplacian_basis(n, edges), k_eigs=6)

    partial = fam.spectrum(np.zeros(fam.M), n_eigs=6)
    full = np.linalg.eigvalsh(L_dense)[:6]
    np.testing.assert_allclose(partial, full, atol=1e-8)


def test_sparse_graph_laplacian_keeps_sparse_mode_for_sparse_input():
    n = 30
    adj = sparse.csr_matrix(sft.graph_gen.path_graph(n))
    fam = sft.families.graph_laplacian(adj)
    assert fam._sparse_mode
    assert fam.basis_kind == "edge_laplacian"
    assert fam.W.shape == (n, n - 1)


def test_graph_adapter_accepts_sparse_adjacency():
    adj = sparse.csr_matrix(sft.graph_gen.path_graph(12))
    adapter = sft.graph(adj)
    assert adapter.n_edges == 11
    assert adapter.kernel.shape == (12, 11)


def test_disconnected_graph_kernel_shape():
    adj = np.zeros((6, 6))
    adj[0, 1] = adj[1, 0] = 1.0
    adj[3, 4] = adj[4, 3] = 1.0
    fam = sft.families.graph_laplacian(adj)
    assert fam.M == 2
    assert fam.W.shape == (6, 2)


def test_partial_inverse_uses_selected_modes_shape_contract():
    n = 30
    adj = sparse.csr_matrix(sft.graph_gen.path_graph(n))
    fam = sft.families.graph_laplacian(adj)
    target = fam.spectrum(np.zeros(fam.M), n_eigs=4)
    result = fam.inverse(target, steps=1, n_eigs=4)
    assert result.k.shape == (fam.M,)
    assert result.converged
    assert fam.W.shape == (4, fam.M)
