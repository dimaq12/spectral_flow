"""Backwards-compatible OperatorBlueprint v2 contracts."""
import numpy as np

import sft


def test_blueprint_v2_keeps_legacy_fields_and_adds_core_metadata():
    data = np.linspace(0.0, 1.0, 32)
    bp = sft.OperatorBlueprint.from_task("compress signal", data)
    d = bp.to_dict()
    assert d["genus"] == bp.genus
    assert d["N"] == 32
    assert d["M"] > 0
    assert d["basis_type"] == "toeplitz"
    assert d["invariant"] in {"frequency_band_energy", "spectral_response"}
    assert d["layer"] == "theory-core"
    assert d["cost"]["memory_mb"] >= 0


def test_blueprint_verify_routes_to_alg_claim():
    bp = sft.OperatorBlueprint.from_task("bandpass filter", np.linspace(0, 1, 16))
    result = bp.verify()
    assert result["claim_id"] == "ALG-BLUEPRINT-005"
    assert result["status"] == "PASS"
    assert result["details"]["invariant"] == "frequency_band_energy"


def test_legacy_construct_dict_still_works():
    data = np.arange(8.0)
    blueprint = sft.plan_operator("sort", data)
    fam = sft.construct(blueprint, data)
    assert fam.N == 8
    assert fam.M == blueprint["M"]


def test_task_old_route_and_solve_behavior_remains_available():
    data = np.array([3.0, 1.0, 2.0])
    np.testing.assert_array_equal(sft.task("sort", data), np.array([1.0, 2.0, 3.0]))
