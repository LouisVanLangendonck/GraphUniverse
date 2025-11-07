"""
GraphUniverse: Multi-graph Generation package.
"""

from graph_universe.dataset import GraphUniverseDataset
from graph_universe.feature_generator import FeatureGenerator
from graph_universe.graph_family import GraphFamilyGenerator
from graph_universe.graph_sample import GraphSample
from graph_universe.graph_universe import GraphUniverse
from graph_universe.viz_utils import (
    plot_graph_communities,
    plot_property_validation,
    plot_universe_community_degree_propensity_vector,
    plot_universe_feature_centers,
)

__all__ = [
    "FeatureGenerator",
    "GraphFamilyGenerator",
    "GraphSample",
    "GraphUniverse",
    "GraphUniverseDataset",
    "plot_graph_communities",
    "plot_property_validation",
    "plot_universe_community_degree_propensity_vector",
    "plot_universe_feature_centers",
]

__version__ = "0.0.1"
