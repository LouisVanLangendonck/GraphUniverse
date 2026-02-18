"""
GraphUniverse: Multi-graph Generation package.
"""

from .cli import launch_ui
from .dataset import GraphUniverseDataset
from .feature_generator import FeatureGenerator
from .graph_family import GraphFamilyGenerator
from .graph_sample import GraphSample
from .graph_universe import GraphUniverse
from .viz_utils import plot_graph_communities

__all__ = [
    "FeatureGenerator",
    "GraphFamilyGenerator",
    "GraphSample",
    "GraphUniverse",
    "GraphUniverseDataset",
    "launch_ui",
    "plot_graph_communities",
]

__version__ = "0.1.0"
