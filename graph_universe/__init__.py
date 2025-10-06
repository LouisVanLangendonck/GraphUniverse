"""
GraphUniverse: Multi-graph Generation package.
"""

from graph_universe.dataset import GraphUniverseDataset
from graph_universe.feature_generator import FeatureGenerator
from graph_universe.graph_family import GraphFamilyGenerator
from graph_universe.graph_sample import GraphSample
from graph_universe.graph_universe import GraphUniverse

__all__ = [
    "FeatureGenerator",
    "GraphFamilyGenerator",
    "GraphSample",
    "GraphUniverse",
    "GraphUniverseDataset",
]

__version__ = "0.0.1"
