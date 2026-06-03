"""Performance guardrails for operator algebra."""
import numpy as np

import sft


def test_cost_model_estimates_family_memory_and_eigensolve_mode():
    fam = sft.families.random(20, 4, seed=21)
    cost = sft.CostModel.estimate(fam)
    assert cost.N == 20
    assert cost.M == 4
    assert cost.memory_mb > 0
    assert cost.eigensolve == "dense"


def test_compose_records_materialization_warning_for_structured_backend():
    adj = sft.graph_gen.path_graph(12)
    fam = sft.families.graph_laplacian(adj)
    C = np.eye(fam.M, 2)
    projected = sft.algebra.compose(fam, C)
    assert projected.cost_before[0].basis_kind == "edge_laplacian"
    assert projected.cost_before[0].materializes_basis is True
    assert projected.cost_after.N == fam.N


def test_tensor_cost_marks_growth():
    a = sft.families.random(6, 2, seed=22)
    b = sft.families.random(5, 2, seed=23)
    ts = sft.algebra.tensor_sum(a, b)
    assert ts.cost_after.N == 30
    assert ts.cost_after.M == 4
    assert ts.cost_after.memory_mb > a.cost().memory_mb
