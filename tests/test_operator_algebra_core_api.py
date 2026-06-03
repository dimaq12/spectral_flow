"""Core operator-algebra API contracts."""
import numpy as np

import sft


def test_task_builder_produces_operator_spec_without_solving():
    X = np.array([[0.0], [0.1], [2.0], [2.1]])
    spec = sft.task("cluster points").on(X)
    assert isinstance(spec, sft.OperatorSpec)
    assert spec.genus.name == "GRAPH"
    assert spec.invariant.name == "connectivity_laplacian"
    assert spec.basis_type == "diagonal"


def test_operator_spec_plans_and_builds_family():
    x = np.sin(np.linspace(0, 1, 16))
    spec = sft.operator_algebra.OperatorSpec.from_task("bandpass filter", x)
    blueprint = spec.plan()
    assert blueprint.invariant == "frequency_band_energy"
    assert blueprint.basis_type == "toeplitz"
    family = blueprint.build(x)
    assert family.N == 16
    assert family.M > 0


def test_fluent_operator_algebra_api_laws_and_transforms():
    a = sft.families.random(8, 3, seed=11)
    b = sft.families.random(5, 2, seed=12)
    combo = a.oplus(b)
    assert combo.N == 13
    assert combo.cost().N == 13

    C = np.eye(a.M, 2)
    projected = a.then(sft.algebra.compose(C))
    assert projected.M == 2
    np.testing.assert_allclose(projected.W, a.W @ C)

    report = a.laws().verify()
    assert report.status == "PASS"
    assert "ALG-DIRECT-SUM-001" in {row["claim_id"] for row in report.results}


def test_core_boundaries_mark_applied_packages_outside_core():
    boundaries = sft.operator_algebra.CORE_BOUNDARIES
    assert "theory-core" in boundaries
    assert "applied-package" in boundaries
    assert "text graphs" in boundaries["applied-package"]
