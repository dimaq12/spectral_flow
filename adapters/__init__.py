"""
sft.adapters — 12 domain adapters: Audio, Image, Graph, Text,
              Timeseries, Video, Voxel, PointCloud, Molecular,
              Financial, Tabular, Mesh.

Each adapter takes RAW DATA and builds an OperatorFamily with
domain-interpretable parameters.
"""
from .base import BaseAdapter
from .audio import AudioAdapter
from .image import ImageAdapter
from .graph_adapter import GraphAdapter
from .text import TextAdapter
from .timeseries import TimeseriesAdapter
from .video import VideoAdapter
from .voxel import VoxelAdapter
from .pointcloud import PointCloudAdapter
from .molecular import MolecularAdapter
from .financial import FinancialAdapter
from .tabular import TabularAdapter
from .mesh import MeshAdapter
