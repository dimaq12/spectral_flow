"""Tests for sft.order, embed, cluster, carleman, transport, verify."""
import numpy as np
import pytest
from sft.order import UniversalRankOperator, DefectPrecomputedCDF, rank_defect_analysis, carleman_cdf
from sft.embed import GraphEmbedder, LogicalGraphEmbedder, build_ternary_laplacian
from sft.cluster import cluster_spectral_k, cluster_knn_spectral, choose_basis_auto
from sft.carleman import operator_family_gf2, complex_hf_check
from sft.transport import optimal_transport_map
from sft.verify import run_verification_suite
from sft.tasks import sort_via_fiedler, hash_injective_map, compose_tasks, diagnose_mismatch
from sft.topology import exceptional_point_locus_nd
from sft.families import random as randfam


class TestOrder:
    def test_universal_rank_operator(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(200)
        op = UniversalRankOperator(data, n_bins=30)
        ranks = op.rank(np.array([-1.0, 0.0, 1.0]))
        assert len(ranks) == 3
        assert np.all((0 <= ranks) & (ranks < 200))

    def test_universal_rank_quantile(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal(100)
        op = UniversalRankOperator(data)
        q = op.quantile(0.5)
        assert -3 < q < 3

    def test_defect_field(self):
        rng = np.random.default_rng(2)
        data = rng.standard_normal(50)
        op = UniversalRankOperator(data)
        d = op.defect_field()
        assert len(d) == 50

    def test_precomputed_cdf(self):
        rng = np.random.default_rng(3)
        data = rng.standard_normal(100)
        cdf = DefectPrecomputedCDF(data)
        r = cdf.rank(0.0)
        assert 0.0 <= r <= 1.0
        q = cdf.quantile(0.5)
        assert abs(q) < 5.0

    def test_precomputed_iqr(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cdf = DefectPrecomputedCDF(data)
        assert cdf.iqr() > 0

    def test_rank_defect_analysis(self):
        rng = np.random.default_rng(4)
        data = rng.standard_normal(200)
        res = rank_defect_analysis(data, bins_list=[10, 20, 40])
        assert "slope" in res

    def test_carleman_distribution(self):
        moments = np.array([0.0, 1.0, 0.0])
        cdf_arr = carleman_cdf(moments, n_points=50)
        assert cdf_arr.shape == (50, 2)
        assert np.all(np.diff(cdf_arr[:, 1]) >= -1e-10)


class TestEmbed:
    def test_graph_embedder(self):
        adj = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
        ge = GraphEmbedder(adj, K=3, R=2)
        v = ge.embed_node(0)
        assert len(v) == 5
        g = ge.embed_graph()
        assert len(g) > 0

    def test_logical_embedder(self):
        and_edges = [(0, 1), (1, 2)]
        not_edges = [(2, 3)]
        le = LogicalGraphEmbedder(4, and_edges, not_edges, K=3)
        v = le.embed_node(0)
        assert len(v) == 3

    def test_build_ternary_laplacian(self):
        L = build_ternary_laplacian(4, [(0, 1), (1, 2)], [(2, 3)])
        assert L.shape == (4, 4)
        assert L[2, 3] == -2.0
        assert L[0, 1] == -1.0


class TestCluster:
    def test_cluster_spectral_k(self):
        rng = np.random.default_rng(0)
        c1 = rng.standard_normal((20, 2)) + [0, 0]
        c2 = rng.standard_normal((20, 2)) + [5, 5]
        X = np.vstack([c1, c2])
        labels = cluster_spectral_k(X, 2)
        assert len(labels) == 40
        assert len(set(labels)) >= 2

    def test_cluster_knn_spectral(self):
        rng = np.random.default_rng(1)
        c1 = rng.standard_normal((15, 2))
        c2 = rng.standard_normal((15, 2)) + [4, 4]
        X = np.vstack([c1, c2])
        labels = cluster_knn_spectral(X, 2, k_neighbors=5)
        assert len(set(labels)) >= 2

    def test_choose_basis_auto(self):
        t = np.linspace(0, 1, 64)
        sig = np.sin(2 * np.pi * 10 * t)
        basis = choose_basis_auto(sig)
        assert basis in ("dct", "fourier", "identity")


class TestCarleman:
    def test_gf2(self):
        fam = operator_family_gf2(10, 4, seed=42)
        assert fam.N == 10
        assert fam.M == 4
        assert np.all(np.isfinite(fam.W))

    def test_complex_hf(self):
        W, W_imag = complex_hf_check(8, 3, seed=42)
        assert W.shape == (8, 3)
        assert np.max(np.abs(W)) > 0


class TestTransport:
    def test_optimal_transport(self):
        rng = np.random.default_rng(0)
        mu = rng.standard_normal(100)
        nu = rng.standard_normal(100) + 2.0
        T, w2 = optimal_transport_map(mu, nu)
        assert len(T) == 100
        assert w2 >= 0


class TestVerify:
    def test_verification_suite(self):
        results = run_verification_suite()
        assert "C1_result" in results
        assert "C8_result" in results
        assert results["C4_result"]["separable"] is True
        assert results["C8_result"] is True


class TestTasksAdditions:
    def test_sort_via_fiedler(self):
        x = np.array([5, 1, 3, 2, 4])
        result = sort_via_fiedler(x)
        assert len(result) == 5
        assert np.all(np.diff(result) >= -1e-10)

    def test_hash_injective_map(self):
        keys = np.array([3.0, 1.0, 3.0, 2.0, 1.0])
        result = hash_injective_map(keys)
        assert len(result) == 5
        assert result[0] == result[2]

    def test_compose_tasks(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(20)
        result = compose_tasks("filter via DCT", "sort", data)
        assert len(result) == 20

    def test_diagnose_mismatch(self):
        res = diagnose_mismatch("sort these values", np.eye(10))
        assert "genus" in res
        assert "phantom_detected" in res


class TestEPLocusND:
    def test_ep_locus_nd(self):
        fam = randfam(8, 4, seed=42)
        grid, gap_map = exceptional_point_locus_nd(fam, grid_resolution=10, axis_pair=(0, 2))
        assert grid.shape == (2, 10)
        assert gap_map.shape == (10, 10)
