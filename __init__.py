"""
sft — Spectral Flow Transform v0.1.0

╔══════════════════════════════════════════════════════════════════════╗
║  MODULE DEPENDENCY GRAPH                                            ║
║                                                                     ║
║  ┌── numpy, scipy (LAPACK)  ←── everything depends on these         ║
║  │                                                                  ║
║  ├── sft.core              ←── central engine, no sft deps          ║
║  │   └── OperatorFamily, nullspace                                 ║
║  │                                                                  ║
║  ├── sft.families          ←── depends on core                      ║
║  │   └── random, graph_laplacian, toeplitz, diagonal, avoid_xing   ║
║  │                                                                  ║
║  ├── sft.algebra           ←── depends on core                      ║
║  ├── sft.topology          ←── depends on core (+joblib optional)   ║
║  ├── sft.hessian           ←── depends on core                      ║
║  ├── sft.tasks             ←── depends on core, compress            ║
║  ├── sft.constructor       ←── depends on tasks, core, families     ║
║  │                                                                  ║
║  ├── sft.adapters          ←── depends on core (12 domain adapters) ║
║  │   └── Audio, Image, Graph, Text, Timeseries, Video, Voxel,      ║
║  │       PointCloud, Molecular, Financial, Tabular, Mesh            ║
║  │                                                                  ║
║  ├── sft.graphop           ←── depends on numpy                     ║
║  ├── sft.embed             ←── depends on scipy.linalg              ║
║  ├── sft.order             ←── depends on numpy, scipy.stats        ║
║  ├── sft.cluster           ←── depends on scipy.linalg, cluster.vq  ║
║  ├── sft.compress          ←── depends on tasks (dct_matrix)        ║
║  ├── sft.streaming         ←── depends on numpy                     ║
║  ├── sft.carleman          ←── depends on core                      ║
║  ├── sft.transport         ←── depends on numpy                     ║
║  ├── sft.homotopy          ←── depends on core                      ║
║  ├── sft.invariants        ←── depends on core, hessian             ║
║  ├── sft.inversion         ←── depends on core                      ║
║  ├── sft.basis             ←── depends on scipy.linalg              ║
║  ├── sft.arnoldi           ←── depends on scipy.linalg              ║
║  └── sft.verify            ←── depends on tasks, order, topology,   ║
║                                hessian, embed, families              ║
║                                                                     ║
║  PUBLIC API SURFACE:                                                ║
║  ─────────────────                                                  ║
║  • 12 factory functions:  sft.audio(), sft.image(), ...            ║
║  • 12 adapter classes:    AudioAdapter, ..., MeshAdapter            ║
║  • 15 submodules:         algebra, topology, ..., arnoldi           ║
║  • 9 standalone classes:  OperatorFamily, GraphOperator, ...        ║
║  • 5 standalone funcs:    classify_task, cdf_rank_sort, ...         ║
║  • Total: 63 public names in __all__                                ║
╚══════════════════════════════════════════════════════════════════════╝

    import sft

    # Build an operator family
    fam = sft.families.random(N=100, M=30)
    W = fam.W; lam = fam.predict(dk); k, err, ok = fam.inverse(target)
    print(fam.complexity)

    # Domain adapters — killer feature
    sound = sft.audio(signal, sr=44100)
    pic   = sft.image(pixels, patch_size=8)
    net   = sft.graph(adjacency)

    # Algebra + topology
    fam_sum = sft.algebra.direct_sum(a, b)
    tracks, swaps = sft.topology.monodromy(fam, loop)
    holo = sft.topology.berry_holonomy(fam, loop)

    # Task classification
    genus = sft.classify_task("sort these numbers")
    sorted_arr = sft.cdf_rank_sort(arr)

    # Graph structural analysis
    gop = sft.graphop.GraphOperator(edges)
    if gop.is_bridge(0, 1): print("Critical edge!")
"""
__version__ = "0.1.0"

from .core import OperatorFamily, nullspace
from . import algebra, topology, hessian, families
from . import tasks, constructor, homotopy, graphop
from . import compress, streaming, order, embed
from . import cluster, carleman, transport, verify
from . import basis, graph_gen, arnoldi, inversion, invariants
from . import codec as codec
from .adapters import (
    AudioAdapter, ImageAdapter, GraphAdapter, TextAdapter, TimeseriesAdapter,
    VideoAdapter, VoxelAdapter, PointCloudAdapter, MolecularAdapter,
    FinancialAdapter, TabularAdapter, MeshAdapter,
)
from .graphop import GraphOperator
from .embed import GraphEmbedder, LogicalGraphEmbedder
from .order import UniversalRankOperator, DefectPrecomputedCDF, rank_defect_analysis, carleman_cdf
from .streaming import StreamingCDF, StreamingOrderOnline
from .tasks import OperatorGenus, classify_task, cdf_rank_sort, dct_matrix, filter_via_dct
from .constructor import from_task, plan_operator, construct, synthesize

def audio(s, **kw): return AudioAdapter(s, **kw)
def image(p, **kw): return ImageAdapter(p, **kw)
def graph(a, **kw): return GraphAdapter(a, **kw)
def text(t, **kw): return TextAdapter(t, **kw)
def timeseries(s, **kw): return TimeseriesAdapter(s, **kw)
def video(f, **kw): return VideoAdapter(f, **kw)
def voxel(v, **kw): return VoxelAdapter(v, **kw)
def pointcloud(p, **kw): return PointCloudAdapter(p, **kw)
def molecular(pos, at, bo, **kw): return MolecularAdapter(pos, at, bo, **kw)
def financial(r, **kw): return FinancialAdapter(r, **kw)
def tabular(d, **kw): return TabularAdapter(d, **kw)
def mesh(v, f, **kw): return MeshAdapter(v, f, **kw)

__all__ = [
    "OperatorFamily", "nullspace",
    "algebra", "topology", "hessian", "families",
    "tasks", "constructor", "homotopy", "graphop",
    "compress", "streaming", "order", "embed",
    "cluster", "carleman", "transport", "verify",
    "basis", "graph_gen", "arnoldi", "inversion", "invariants", "codec",
    "AudioAdapter", "ImageAdapter", "GraphAdapter", "TextAdapter", "TimeseriesAdapter",
    "VideoAdapter", "VoxelAdapter", "PointCloudAdapter", "MolecularAdapter",
    "FinancialAdapter", "TabularAdapter", "MeshAdapter",
    "GraphOperator", "GraphEmbedder", "LogicalGraphEmbedder",
    "UniversalRankOperator", "DefectPrecomputedCDF", "rank_defect_analysis",
    "carleman_cdf", "StreamingCDF", "StreamingOrderOnline",
    "OperatorGenus", "classify_task", "cdf_rank_sort",
    "dct_matrix", "filter_via_dct",
    "from_task", "plan_operator", "construct", "synthesize",
    "audio", "image", "graph", "text", "timeseries",
    "video", "voxel", "pointcloud", "molecular",
    "financial", "tabular", "mesh",
]
