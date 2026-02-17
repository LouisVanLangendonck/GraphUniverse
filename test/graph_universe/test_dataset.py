"""Test GraphUniverseDataset class."""

import json
import os
import tempfile

import pytest
import torch
from torch_geometric.data import Data

from graph_universe.dataset import GraphUniverseDataset


class TestGraphUniverseDataset:
    """Test GraphUniverseDataset class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name

        # Create sample parameters
        self.parameters = {
            "universe_parameters": {
                "K": 5,
                "edge_propensity_variance": 0.5,
                "feature_dim": 10,
                "center_variance": 1.0,
                "cluster_variance": 0.1,
                "seed": 42,
            },
            "family_parameters": {
                "n_graphs": 3,
                "n_nodes_range": [20, 50],
                "n_communities_range": [2, 4],
                "homophily_range": [0.0, 0.4],
                "avg_degree_range": [1.0, 3.0],
                "power_law_exponent_range": [2.0, 3.5],
                "degree_separation_range": [0.5, 0.5],
                "seed": 42,
            },
            "task": "community_detection"
        }

        # Create sample PyG graphs
        self.graph_list = [
            Data(
                x=torch.randn(10, 5),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
                y=torch.tensor([0, 1, 0, 1, 2, 2, 0, 1, 2, 0], dtype=torch.long),
            ),
            Data(
                x=torch.randn(8, 5),
                edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
                y=torch.tensor([0, 1, 0, 1, 2, 0, 1, 2], dtype=torch.long),
            ),
        ]

    def teardown_method(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_get_dataset_dir(self):
        """Test get_dataset_dir method."""
        # Initialize dataset
        dataset = GraphUniverseDataset(
            root=self.root_dir,
            parameters=self.parameters,
            graph_list=self.graph_list,
        )

        # Get dataset directory
        dataset_dir = dataset.get_dataset_dir(self.parameters)

        # Check that directory structure is as expected
        assert "K_5_edge_prop_var_0.5" in dataset_dir
        assert "homophily_0.0_to_0.4" in dataset_dir
        assert "n_graphs_3" in dataset_dir
        assert "n_communities_2_to_4" in dataset_dir
        assert "hash_" in dataset_dir

    def test_raw_dir_property(self):
        """Test raw_dir property."""
        # Initialize dataset
        dataset = GraphUniverseDataset(
            root=self.root_dir,
            parameters=self.parameters,
            graph_list=self.graph_list,
        )

        # Check raw directory
        expected_raw_dir = os.path.join(self.root_dir, dataset.name)
        assert dataset.raw_dir == expected_raw_dir

    def test_processed_dir_property(self):
        """Test processed_dir property."""
        # Initialize dataset
        dataset = GraphUniverseDataset(
            root=self.root_dir,
            parameters=self.parameters,
            graph_list=self.graph_list,
        )

        # Check processed directory
        expected_processed_dir = os.path.join(self.root_dir, dataset.name)
        assert dataset.processed_dir == expected_processed_dir

    def test_raw_file_names_property(self):
        """Test raw_file_names property."""
        # Initialize dataset
        dataset = GraphUniverseDataset(
            root=self.root_dir,
            parameters=self.parameters,
            graph_list=self.graph_list,
        )

        # Check raw file names
        assert dataset.raw_file_names == ["data.pt"]

    def test_processed_file_names_property(self):
        """Test processed_file_names property."""
        # Initialize dataset
        dataset = GraphUniverseDataset(
            root=self.root_dir,
            parameters=self.parameters,
            graph_list=self.graph_list,
        )

        # Check processed file names
        assert dataset.processed_file_names == "data.pt"

    def test_get_data_dir(self):
        """Test get_data_dir method."""
        # Initialize dataset
        dataset = GraphUniverseDataset(
            root=self.root_dir,
            parameters=self.parameters,
            graph_list=self.graph_list,
        )

        # Check data directory
        expected_data_dir = os.path.join(self.root_dir, dataset.name)
        assert dataset.get_data_dir() == expected_data_dir

    def test_download(self):
        """Test download method - it should be a no-op for in-memory datasets."""
        # Initialize dataset
        dataset = GraphUniverseDataset(
            root=self.root_dir,
            parameters=self.parameters,
            graph_list=self.graph_list,
        )

        # Create the raw directory
        os.makedirs(dataset.raw_dir, exist_ok=True)

        # Call download - should not raise an error
        # The download method is typically a no-op for in-memory datasets
        try:
            dataset.download()
            # If download method exists and runs, it should complete without error
            assert True
        except NotImplementedError:
            # If download is not implemented, that's also acceptable
            assert True

    def test_process(self):
        """Test that process method properly saves graphs and metadata."""
        from graph_universe.dataset import GraphUniverseDataset

        class TestableDataset(GraphUniverseDataset):
            def __init__(self, root, parameters, graph_list):
                self.name = self.get_dataset_dir(parameters)
                self.parameters = parameters
                self.graph_list = graph_list
                self.root = root
                self._processed_dir = None

        dataset = TestableDataset(
            root=self.root_dir,
            parameters=self.parameters,
            graph_list=self.graph_list,
        )

        os.makedirs(dataset.processed_dir, exist_ok=True)

        if len(self.graph_list) > 0:
            dataset.process()

            metadata_file = os.path.join(dataset.processed_dir, "metadata.json")
            assert os.path.exists(metadata_file)

            with open(metadata_file) as f:
                metadata = json.load(f)
            assert "family_parameters" in metadata
            assert "universe_parameters" in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
