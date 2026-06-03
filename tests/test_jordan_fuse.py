"""Jordan-fused operator algebra contracts."""
import numpy as np
import pytest

import sft


def ep2():
    return sft.physics.exceptional_point_2x2().family()


def puiseux_exponent(family, axis: int) -> float:
    radii = np.array([1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5])
    values = []
    for radius in radii:
        k = np.zeros(family.M)
        k[axis] = radius
        values.append(float(np.max(np.abs(family.eigensystem(k)[0]))))
    return float(np.polyfit(np.log(radii), np.log(values), 1)[0])


def test_plain_tensor_sum_is_reducible_not_ep4():
    plain = sft.algebra.tensor_sum(ep2(), ep2())
    fp = sft.algebra.jordan_fingerprint(plain.A0)
    assert fp.geometric_multiplicity == 2
    assert fp.nilpotent_index == 3
    assert fp.is_nilpotent
    assert not fp.is_single_chain
    assert not sft.algebra.is_single_jordan_chain(plain.A0, order=4)


def test_jordan_fuse_synthesizes_single_ep4_chain():
    fused = sft.algebra.jordan_fuse(ep2(), ep2())
    fp = sft.algebra.jordan_fingerprint(fused.A0)
    assert fused.N == 4
    assert fused.M == 6
    assert fused.algebra_operation == "jordan_fuse"
    assert fused.jordan_coupling == (1, 2)
    assert fp.geometric_multiplicity == 1
    assert fp.nilpotent_index == 4
    assert fp.rank_sequence == (3, 2, 1, 0)
    assert fp.nullity_sequence == (1, 2, 3, 4)
    assert fp.is_single_chain
    assert fused.jordan_fingerprint.is_single_chain


def test_jordan_fuse_top_level_api_and_puiseux_scaling():
    fused = sft.jordan_fuse(ep2(), ep2())
    assert isinstance(sft.jordan_fingerprint(fused.A0), sft.JordanFingerprint)
    exponent = puiseux_exponent(fused, axis=fused.M - 2)
    assert abs(exponent - 0.25) < 0.08


def test_jordan_fuse_can_preserve_tensor_parameter_count_without_closure():
    fused = sft.algebra.jordan_fuse(ep2(), ep2(), add_closure=False)
    assert fused.M == 4
    assert sft.algebra.jordan_fingerprint(fused.A0).is_single_chain


def test_jordan_fuse_explicit_and_search_couplings():
    explicit = sft.algebra.jordan_fuse(ep2(), ep2(), coupling=(2, 1))
    searched = sft.algebra.jordan_fuse(ep2(), ep2(), coupling="search")
    assert sft.algebra.jordan_fingerprint(explicit.A0).is_single_chain
    assert sft.algebra.jordan_fingerprint(searched.A0).is_single_chain


def test_jordan_fuse_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="indices"):
        sft.algebra.jordan_fuse(ep2(), ep2(), coupling=(9, 0))
    with pytest.raises(ValueError, match="finite"):
        sft.algebra.jordan_fuse(ep2(), ep2(), strength=np.inf)
    with pytest.raises(ValueError, match="coupling is required"):
        sft.algebra.jordan_fuse(sft.families.random(3, 2), sft.families.random(2, 2))


def test_multi_jordan_fuse_synthesizes_ep8_and_ep16():
    ep4 = sft.algebra.jordan_fuse(ep2(), ep2(), add_closure=False)
    ep8 = sft.algebra.multi_jordan_fuse(ep4, ep2(), add_closure=False)
    ep16 = sft.algebra.multi_jordan_fuse(ep4, ep4, add_closure=False)

    assert ep8.jordan_couplings == ((6, 1),)
    assert ep8.jordan_fingerprint.is_single_chain
    assert ep8.jordan_fingerprint.nilpotent_index == 8
    assert ep8.jordan_fingerprint.rank_sequence == tuple(range(7, -1, -1))

    assert ep16.jordan_couplings == ((3, 4), (7, 8), (11, 12))
    assert ep16.jordan_fingerprint.is_single_chain
    assert ep16.jordan_fingerprint.nilpotent_index == 16
    assert ep16.jordan_fingerprint.rank_sequence == tuple(range(15, -1, -1))


def test_multi_jordan_fuse_synthesizes_ep32_fast_path():
    ep4 = sft.algebra.jordan_fuse(ep2(), ep2(), add_closure=False)
    ep16 = sft.algebra.multi_jordan_fuse(ep4, ep4, add_closure=False)
    ep32 = sft.algebra.multi_jordan_fuse(ep16, ep2(), add_closure=False)
    assert ep32.jordan_couplings == ((30, 1),)
    assert ep32.jordan_fingerprint.geometric_multiplicity == 1
    assert ep32.jordan_fingerprint.nilpotent_index == 32
    assert ep32.jordan_fingerprint.rank_sequence == tuple(range(31, -1, -1))


def test_plain_ep4_tensor_ep4_is_reducible_negative_control():
    ep4 = sft.algebra.jordan_fuse(ep2(), ep2(), add_closure=False)
    plain = sft.algebra.tensor_sum(ep4, ep4)
    fp = sft.algebra.jordan_fingerprint(plain.A0)
    assert fp.geometric_multiplicity == 4
    assert fp.nilpotent_index == 7
    assert not fp.is_single_chain


def test_multi_jordan_fuse_explicit_couplings_and_exports():
    ep4 = sft.algebra.jordan_fuse(ep2(), ep2(), add_closure=False)
    fused = sft.multi_jordan_fuse(
        ep4, ep4, couplings=[(3, 4), (7, 8), (11, 12)], add_closure=False
    )
    assert sft.jordan_fuse_chain is sft.algebra.multi_jordan_fuse
    assert fused.algebra_operation == "multi_jordan_fuse"
    assert fused.jordan_fingerprint.is_single_chain


def test_multi_jordan_fuse_rejects_insufficient_bridge_budget():
    ep4 = sft.algebra.jordan_fuse(ep2(), ep2(), add_closure=False)
    with pytest.raises(ValueError, match="did not produce"):
        sft.algebra.multi_jordan_fuse(ep4, ep4, max_bridges=1, add_closure=False)
