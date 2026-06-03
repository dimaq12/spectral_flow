"""README-style smoke snippets that should remain copy-paste safe."""
import numpy as np

import sft


def test_version_and_main_exports():
    assert sft.__version__ >= "0.1.3"
    for name in ("OperatorFamily", "OperatorBlueprint", "solve", "pipe"):
        assert hasattr(sft, name)


def test_quick_start_smoke():
    fam = sft.families.random(N=20, M=6, seed=42)
    rng = np.random.default_rng(0)
    dk = 0.01 * rng.standard_normal(fam.M)
    assert fam.predict(dk).shape == (fam.N,)
    target = np.sort(fam.lam0 + np.linspace(-0.02, 0.02, fam.N))
    result = sft.solve(fam, target, steps=3)
    assert result.k.shape == (fam.M,)


def test_topology_readme_smoke():
    fam2 = sft.families.avoided_crossing_2x2(Delta=0.3)
    loop = [
        np.array([0.4 * np.cos(t), 0.4 * np.sin(t)])
        for t in np.linspace(0, 2 * np.pi, 20)
    ]
    tracked, swaps = sft.topology.monodromy(fam2, loop)
    assert tracked.shape == (20, 2)
    assert isinstance(swaps, list)


def test_graph_operator_readme_smoke():
    adj = sft.graph_gen.path_graph(10)
    row, col = np.triu(adj, 1).nonzero()
    edges = list(zip(row.tolist(), col.tolist()))
    gop = sft.graphop.GraphOperator(edges)
    assert gop.is_bridge(0, 1)


def test_codec_and_arnoldi_smoke():
    fam = sft.families.random(12, 4, seed=1)
    codec = sft.InstantSpectralCodec(fam)
    dk = np.linspace(-0.1, 0.1, fam.M)
    _, err = codec.roundtrip(dk)
    assert err < 1.0

    A = np.diag([1.0, 2.0, 3.0])
    V, H = sft.arnoldi.arnoldi_iteration(lambda x: A @ x, np.ones(3), m=3)
    assert V.shape[0] == 3
    assert H.shape[0] >= 1


def test_spectral_geometry_lab_smoke():
    ep = sft.physics.exceptional_point_2x2().family()
    loop = [
        np.array([0.25 * np.cos(t), 0.25 * np.sin(t)])
        for t in np.linspace(0, 2 * np.pi, 20)
    ]
    summary = sft.topology.complex_monodromy(ep, loop)
    assert abs(abs(summary["windings"][(0, 1)]) - 0.5) < 0.2

    x = np.linspace(0.0, 1.0, 34)[1:-1]
    sch = sft.physics.schrodinger_1d(x, np.zeros_like(x), max_potential_params=4).family()
    assert sch.spectrum(np.zeros(sch.M), n_eigs=4).shape == (4,)
