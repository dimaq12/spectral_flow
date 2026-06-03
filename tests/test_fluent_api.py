"""Expressive operator API contracts."""
import numpy as np

import sft


def test_family_fluent_verbs_match_existing_methods():
    fam = sft.families.random(12, 4, seed=4)
    k0 = np.array([0.01, -0.02, 0.0, 0.03])
    k1 = k0 + np.array([0.001, 0.0, -0.001, 0.0])
    fam.refresh(k0)
    np.testing.assert_allclose(fam.at(k1), fam.predict_at(k1))
    np.testing.assert_allclose(fam.shift(k1 - k0), fam.predict(k1 - k0))
    target = np.sort(fam.lam0 + np.linspace(-0.01, 0.01, fam.N))
    assert fam.toward(target, steps=2).k.shape == (fam.M,)


def test_adapter_fluent_parity():
    adj = sft.graph_gen.path_graph(8)
    adapter = sft.graph(adj)
    k0 = np.zeros(adapter.family.M)
    adapter.refresh(k0)
    np.testing.assert_allclose(adapter.at(k0), adapter.reference_spectrum)
    assert adapter.toward(adapter.reference_spectrum, steps=1).k.shape == (adapter.family.M,)


def test_operator_dunders_direct_sum_prediction_and_composition():
    a = sft.families.random(8, 3, seed=5)
    b = sft.families.random(5, 2, seed=6)
    ab = a + b
    assert ab.N == a.N + b.N
    assert ab.M == a.M + b.M

    dk = np.zeros(a.M)
    np.testing.assert_allclose(a @ dk, a.predict(dk))

    C = np.eye(a.M, 2)
    projected = a @ C
    assert projected.M == 2
    np.testing.assert_allclose(projected.W, a.W @ C)


def test_blueprint_dataclass_builds_like_dict_pipeline():
    data = np.arange(10.0)
    bp = sft.OperatorBlueprint.from_task("sort", data)
    fam = bp.build(data)
    assert fam.N == 10
    assert fam.M > 0


def test_task_pipe_alias_matches_compose_tasks():
    data = np.linspace(1, 0, 16)
    np.testing.assert_allclose(
        sft.pipe("sort", "filter", data),
        sft.tasks.compose_tasks("sort", "filter", data),
    )
