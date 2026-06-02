"""Tests for sft audit fixes: new modules and fixed adapters."""
import numpy as np
import pytest
from sft.families import random, graph_laplacian, avoided_crossing_2x2
from sft.core import graph_response_kernel
from sft.adapters import VideoAdapter, VoxelAdapter, TextAdapter
from sft.tasks import classify_task, OperatorGenus, cdf_rank_sort, dct_matrix, filter_via_dct, route_and_solve
from sft.constructor import plan_operator, construct, synthesize
from sft.homotopy import regularised_pinv, track_homotopy, trust_region_corrector
from sft.graphop import GraphOperator
from sft.invariants import all_invariants, svd_kurtosis, zeta_fingerprint
from sft.compress import compress_spectral, dct_codec
from sft.streaming import StreamingCDF, StreamingOrderOnline
from sft.inversion import bottleneck_inverse, fixed_point_inverse


class TestGraphLaplacianFix:
    def test_uses_unweighted_laplacian(self):
        N = 5
        adj = np.zeros((N, N))
        for i in range(N - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        fam = graph_laplacian(adj)
        D = np.diag(np.sum(adj, axis=1))
        L0 = D - adj
        np.testing.assert_array_almost_equal(fam.A0, L0)

    def test_edge_count(self):
        adj = np.array([[0,1,0],[1,0,1],[0,1,0]])
        fam = graph_laplacian(adj)
        assert fam.M == 2


class TestGraphResponseKernel:
    def test_shape(self):
        N = 5
        adj = np.zeros((N, N))
        for i in range(N - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        lam, V, W = graph_response_kernel(adj)
        assert W.shape == (N, N - 1)
        assert np.all(W >= -1e-10)

    def test_nonnegative(self):
        N = 10
        rng = np.random.default_rng(0)
        adj = np.triu(rng.standard_normal((N, N)), 1)
        adj = (adj > 0).astype(float)
        adj = adj + adj.T
        _, _, W = graph_response_kernel(adj)
        assert np.all(W >= -1e-10)


class TestVideoAdapterVec:
    def test_sliding_window_used(self):
        rng = np.random.default_rng(0)
        frames = rng.standard_normal((12, 20, 20))
        adapter = VideoAdapter(frames, patch_t=3, patch_xy=4, n_regions=4)
        assert adapter.n_patches > 0
        assert adapter.kernel.shape[1] == 4


class TestVoxelAdapterVec:
    def test_sliding_window_used(self):
        rng = np.random.default_rng(0)
        vol = rng.standard_normal((12, 12, 12))
        adapter = VoxelAdapter(vol, patch_size=3, n_zones=4)
        assert adapter.n_patches > 0


class TestTextAdapterRich:
    def test_multiple_params(self):
        texts = ["the cat sat on the mat", "the dog sat on the log",
                 "the cat and the dog are friends"]
        adapter = TextAdapter(texts, max_words=10, window=3)
        assert adapter.kernel.shape[1] > 1


class TestTasks:
    def test_classify_sort(self):
        assert classify_task("sort these numbers") == OperatorGenus.MONO

    def test_classify_filter(self):
        assert classify_task("filter the signal via DCT") == OperatorGenus.QUAD

    def test_classify_cluster(self):
        assert classify_task("cluster the data into groups") == OperatorGenus.GRAPH

    def test_classify_compress(self):
        assert classify_task("compress the image spectrally") == OperatorGenus.COMPRESS

    def test_classify_inverse(self):
        assert classify_task("inverse design a spectrum") == OperatorGenus.CONTROL

    def test_classify_topology(self):
        assert classify_task("compute berry holonomy") == OperatorGenus.FLOW

    def test_cdf_rank_sort(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        result = cdf_rank_sort(x, n_bins=20)
        assert len(result) == 100
        assert np.all(np.diff(result) >= -1e-10)

    def test_dct_matrix(self):
        C = dct_matrix(8)
        assert C.shape == (8, 8)
        np.testing.assert_array_almost_equal(C @ C.T, np.eye(8), decimal=10)

    def test_filter_via_dct(self):
        t = np.linspace(0, 1, 64)
        sig = np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.default_rng(1).standard_normal(64)
        filtered = filter_via_dct(sig, keep_low=5)
        assert filtered.shape == sig.shape

    def test_route_and_solve_sort(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(20)
        result = route_and_solve("sort these values", data)
        assert len(result) == 20


class TestConstructor:
    def test_synthesize(self):
        fam = synthesize("sort these numbers", np.eye(10))
        assert fam.N == 10
        assert fam.M > 0

    def test_plan_operator(self):
        bp = plan_operator("filter with DCT", np.eye(16))
        assert bp["genus"] == OperatorGenus.QUAD

    def test_classify_returns_genus(self):
        assert classify_task("sort this array") == OperatorGenus.MONO
        assert classify_task("filter via DCT") == OperatorGenus.QUAD
        assert classify_task("cluster my data") == OperatorGenus.GRAPH


class TestHomotopy:
    def test_regularised_pinv(self):
        W = np.random.default_rng(0).standard_normal((20, 10))
        Wp = regularised_pinv(W, reg=1e-4)
        assert Wp.shape == (10, 20)

    def test_track_homotopy(self):
        fam = random(10, 5, seed=42)
        target = np.sort(fam.lam0 + np.linspace(-0.05, 0.05, fam.N))
        k, err, ok = track_homotopy(target, n_steps=15, family=fam)
        assert err < 0.5

    def test_trust_region(self):
        W = np.eye(5, 3)
        x = np.ones(3)
        t = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        dk = trust_region_corrector(x, t, W, radius=1.0)
        assert dk.shape == (3,)


class TestGraphOperator:
    def test_bridges(self):
        edges = [(0,1),(1,2),(2,0),(2,3)]
        gop = GraphOperator(edges)
        assert gop.is_bridge(2, 3)
        assert not gop.is_bridge(0, 1)

    def test_articulations(self):
        edges = [(0,1),(1,2),(2,0),(2,3)]
        gop = GraphOperator(edges)
        assert gop.is_articulation(2)

    def test_coreness(self):
        edges = [(0,1),(1,2),(2,3),(3,4),(0,4)]
        gop = GraphOperator(edges)
        assert np.all(gop.coreness >= 0)

    def test_k_core(self):
        edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)]
        gop = GraphOperator(edges)
        k3 = gop.k_core(2)
        assert len(k3) >= 3


class TestLockpick:
    def test_all_invariants(self):
        fam = random(20, 10, seed=42)
        result = all_invariants(fam)
        assert "svd_kurtosis" in result
        assert result["svd_kurtosis"] >= 1.0

    def test_k1(self):
        W = np.random.default_rng(1).standard_normal((10, 5))
        k1 = svd_kurtosis(W)
        assert k1 >= 1.0

    def test_k5(self):
        W = np.random.default_rng(2).standard_normal((10, 3))
        fp = zeta_fingerprint(W)
        assert len(fp) == 10


class TestCompress:
    def test_compress_spectral(self):
        t = np.linspace(0, 4 * np.pi, 100)
        sig = np.sin(t * 3) * np.exp(-t / 10) + 0.02 * np.random.default_rng(0).standard_normal(100)
        compressed = compress_spectral(sig, K=5)
        assert len(compressed) == 100

    def test_dct_codec(self):
        sig = np.sin(np.linspace(0, 1, 32) * 10)
        reconstructed = dct_codec(sig, keep_frac=0.3)
        assert len(reconstructed) == 32


class TestStreaming:
    def test_streaming_cdf(self):
        cdf = StreamingCDF(capacity=100)
        rng = np.random.default_rng(0)
        for _ in range(500):
            cdf.add(rng.standard_normal())
        p = cdf.cdf(0.0)
        assert 0.0 <= p <= 1.0
        q = cdf.quantile(0.5)
        assert abs(q) < 10.0

    def test_streaming_order(self):
        order = StreamingOrderOnline(capacity=200)
        order.insert(3.0)
        order.insert(1.0)
        order.insert(5.0)
        assert order.rank(3.0) == pytest.approx(1.0 / 3.0, abs=0.1)


class TestHacks:
    def test_bottleneck_inverse(self):
        fam = random(10, 5, seed=42)
        target = np.sort(fam.lam0 + np.linspace(-0.02, 0.02, fam.N))
        k = bottleneck_inverse(fam, target, bottleneck_dim=3, steps=10, alpha=0.3)
        assert len(k) == 5

    def test_fixed_point_inverse(self):
        fam = random(10, 5, seed=42)
        target = np.sort(fam.lam0 + np.linspace(-0.01, 0.01, fam.N))
        k = fixed_point_inverse(fam, target, linear_basis_indices=[0, 1, 2],
                             outer_steps=3, inner_steps=3, alpha=0.3)
        assert len(k) == 5
