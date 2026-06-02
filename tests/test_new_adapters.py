"""Tests for new domain adapters (video, voxel, pointcloud, molecular, financial, tabular, mesh)."""
import numpy as np
import pytest
from sft.adapters import (
    VideoAdapter, VoxelAdapter, PointCloudAdapter,
    MolecularAdapter, FinancialAdapter, TabularAdapter,
    MeshAdapter,
)


class TestVideoAdapter:
    def test_basic(self):
        rng = np.random.default_rng(0)
        frames = rng.standard_normal((20, 32, 32))
        adapter = VideoAdapter(frames, patch_t=4, patch_xy=8, n_regions=4)
        assert adapter.n_patches > 0
        assert adapter.kernel.shape[1] == 4
        assert adapter.rank >= 1

    def test_grayscale_from_rgb(self):
        rng = np.random.default_rng(1)
        frames = rng.standard_normal((10, 32, 32, 3))
        adapter = VideoAdapter(frames, patch_t=3, patch_xy=4, n_regions=4)
        assert adapter.frames.ndim == 3
        assert adapter.rank >= 1

    def test_complexity(self):
        rng = np.random.default_rng(2)
        frames = rng.standard_normal((15, 24, 24))
        adapter = VideoAdapter(frames, patch_t=3, patch_xy=4, n_regions=4)
        assert 0 <= adapter.complexity <= 1

    def test_single_frame(self):
        rng = np.random.default_rng(3)
        frames = rng.standard_normal((1, 16, 16))
        adapter = VideoAdapter(frames, patch_t=1, patch_xy=4, n_regions=4)
        assert adapter.n_patches > 0


class TestVoxelAdapter:
    def test_basic(self):
        rng = np.random.default_rng(0)
        vol = rng.standard_normal((16, 16, 16))
        adapter = VoxelAdapter(vol, patch_size=4, n_zones=4)
        assert adapter.n_patches > 0
        assert adapter.kernel.shape[1] == 4
        assert adapter.rank >= 1

    def test_not_3d_raises(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((16, 16))
        with pytest.raises(ValueError):
            VoxelAdapter(data)

    def test_complexity(self):
        rng = np.random.default_rng(2)
        vol = rng.standard_normal((12, 12, 12))
        adapter = VoxelAdapter(vol, patch_size=3, n_zones=4)
        assert 0 <= adapter.complexity <= 1

    def test_too_small_raises(self):
        rng = np.random.default_rng(3)
        vol = rng.standard_normal((2, 2, 2))
        with pytest.raises(ValueError):
            VoxelAdapter(vol, patch_size=4)


class TestPointCloudAdapter:
    def test_basic(self):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((50, 3))
        adapter = PointCloudAdapter(pts, k=10)
        assert adapter.n_points == 50
        assert adapter.kernel.shape[0] == 50
        assert adapter.rank >= 1

    def test_adjacency_is_symmetric(self):
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((30, 3))
        adapter = PointCloudAdapter(pts, k=8)
        adj = adapter.adjacency
        assert adj.shape == (30, 30)
        np.testing.assert_array_equal(adj, adj.T)

    def test_no_self_loops(self):
        rng = np.random.default_rng(2)
        pts = rng.standard_normal((20, 3))
        adapter = PointCloudAdapter(pts, k=5)
        assert np.all(np.diag(adapter.adjacency) == 0)

    def test_complexity(self):
        rng = np.random.default_rng(3)
        pts = rng.standard_normal((30, 3))
        adapter = PointCloudAdapter(pts, k=8)
        assert 0 <= adapter.complexity <= 1

    def test_not_2d_raises(self):
        rng = np.random.default_rng(4)
        data = rng.standard_normal(50)
        with pytest.raises(ValueError):
            PointCloudAdapter(data)


class TestMolecularAdapter:
    def test_basic(self):
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        atom_types = ["C", "H", "H", "H"]
        bonds = [(0, 1), (0, 2), (0, 3)]
        adapter = MolecularAdapter(positions, atom_types, bonds)
        assert adapter.n_atoms == 4
        assert adapter.kernel.shape[1] == 3

    def test_no_bonds(self):
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        atom_types = ["He", "He"]
        bonds = []
        adapter = MolecularAdapter(positions, atom_types, bonds)
        assert adapter.n_atoms == 2
        assert adapter.kernel.shape[1] == 0

    def test_complexity(self):
        positions = np.array([
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.2],
            [0.0, 1.2, 0.0], [1.2, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ])
        atom_types = ["C", "O", "O", "H", "Na"]
        bonds = [(0, 1), (0, 2), (0, 3)]
        adapter = MolecularAdapter(positions, atom_types, bonds)
        assert 0 <= adapter.complexity <= 1

    def test_kernel(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        atom_types = ["C", "C"]
        bonds = [(0, 1)]
        adapter = MolecularAdapter(positions, atom_types, bonds)
        assert adapter.kernel.shape[1] >= 1


class TestFinancialAdapter:
    def test_basic(self):
        rng = np.random.default_rng(0)
        returns = rng.standard_normal((200, 8))
        adapter = FinancialAdapter(returns)
        assert adapter.kernel.shape == (8, 1 + 8)

    def test_with_sectors(self):
        rng = np.random.default_rng(1)
        returns = rng.standard_normal((100, 6))
        sectors = ["tech", "tech", "tech", "fin", "fin", "fin"]
        adapter = FinancialAdapter(returns, sectors=sectors)
        assert adapter.kernel.shape[1] == 2 + 6

    def test_complexity(self):
        rng = np.random.default_rng(2)
        returns = rng.standard_normal((50, 5))
        adapter = FinancialAdapter(returns)
        assert 0 <= adapter.complexity <= 1

    def test_single_asset(self):
        rng = np.random.default_rng(3)
        returns = rng.standard_normal(100)
        adapter = FinancialAdapter(returns)
        assert adapter.kernel.shape == (1, 2)

    def test_corr_is_symmetric(self):
        rng = np.random.default_rng(4)
        returns = rng.standard_normal((100, 5))
        adapter = FinancialAdapter(returns)
        np.testing.assert_array_almost_equal(adapter.Corr, adapter.Corr.T)


class TestTabularAdapter:
    def test_basic(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 10))
        adapter = TabularAdapter(data)
        assert adapter.kernel.shape == (10, 1 + 10)

    def test_with_groups(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((50, 8))
        groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
        adapter = TabularAdapter(data, feature_groups=groups)
        assert adapter.kernel.shape[1] == 2 + 8

    def test_with_missing_values(self):
        rng = np.random.default_rng(2)
        data = rng.standard_normal((30, 5))
        data[5, 2] = np.nan
        data[10, 3] = np.nan
        adapter = TabularAdapter(data)
        assert adapter.kernel.shape == (5, 1 + 5)
        assert np.all(np.isfinite(adapter.kernel))

    def test_complexity(self):
        rng = np.random.default_rng(3)
        data = rng.standard_normal((50, 5))
        adapter = TabularAdapter(data)
        assert 0 <= adapter.complexity <= 1

    def test_single_feature(self):
        rng = np.random.default_rng(4)
        data = rng.standard_normal(50)
        adapter = TabularAdapter(data)
        assert adapter.kernel.shape == (1, 1 + 1)


class TestMeshAdapter:
    def test_tetrahedron(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ])
        adapter = MeshAdapter(vertices, faces)
        assert adapter.n_vertices == 4
        assert adapter.n_faces == 4
        assert adapter.n_edges >= 4
        assert adapter.kernel.shape == (4, adapter.n_edges)

    def test_icosahedron_like(self):
        phi = (1 + np.sqrt(5)) / 2
        a, b = 1.0, phi
        vertices = np.array([
            [-a, -b, 0], [a, -b, 0], [-a, b, 0], [a, b, 0],
            [0, -a, -b], [0, a, -b], [0, -a, b], [0, a, b],
            [-b, 0, -a], [-b, 0, a], [b, 0, -a], [b, 0, a],
        ]) / np.sqrt(a * a + b * b)
        faces = np.array([
            [0, 1, 4], [0, 2, 6], [0, 5, 4], [0, 6, 5],
            [1, 3, 7], [1, 4, 10], [1, 7, 10],
            [2, 3, 11], [2, 6, 9], [2, 9, 11],
            [3, 7, 11],
            [4, 5, 8], [4, 8, 10],
            [5, 6, 9], [5, 8, 9],
            [7, 10, 11],
            [6, 0, 9],
            [0, 4, 8], [0, 8, 5],
            [3, 10, 7],
            [3, 2, 11],
        ], dtype=int)
        adapter = MeshAdapter(vertices, faces)
        assert adapter.n_edges > 0
        assert adapter.kernel.shape[0] == 12
        assert 0 <= adapter.complexity <= 1

    def test_not_triangle_raises(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        faces = np.array([[0, 1, 2, 3]])
        with pytest.raises(ValueError):
            MeshAdapter(vertices, faces)

    def test_complexity(self):
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        adapter = MeshAdapter(vertices, faces)
        assert 0 <= adapter.complexity <= 1

    def test_spectrum(self):
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        adapter = MeshAdapter(vertices, faces)
        lam = adapter.spectrum
        assert lam.shape == (4,)
        assert lam[0] == pytest.approx(0.0, abs=1e-8)
