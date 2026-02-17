"""
GraphUniverse: Multi-graph Generation package.
"""

from graph_universe.cli import launch_ui
from graph_universe.dataset import GraphUniverseDataset
from graph_universe.feature_generator import FeatureGenerator
from graph_universe.graph_family import GraphFamilyGenerator
from graph_universe.graph_sample import GraphSample
from graph_universe.graph_universe import GraphUniverse
from graph_universe.viz_utils import plot_graph_communities

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
