"""Test run module."""

import os
import tempfile
from unittest.mock import mock_open, patch

import pytest
import yaml

from graph_universe.run import load_config, main


class TestRun:
    """Test run module."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name

        # Create a sample config
        self.config = {
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
                "min_n_nodes": 20,
                "max_n_nodes": 50,
                "min_communities": 2,
                "max_communities": 4,
                "homophily_range": [0.0, 0.4],
                "avg_degree_range": [1.0, 3.0],
                "degree_distribution": "power_law",
                "power_law_exponent_range": [2.0, 3.5],
                "degree_separation_range": [0.5, 0.5],
                "seed": 42,
            },
        }

    def teardown_method(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_load_config(self):
        """Test load_config function."""
        # Create a temporary config file
        config_path = os.path.join(self.root_dir, "test_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

        # Load the config
        loaded_config = load_config(config_path)

        # Check that the loaded config matches the original
        assert loaded_config == self.config

    def test_load_config_file_not_found(self):
        """Test load_config function with non-existent file."""
        # Try to load a non-existent config file
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")

    @patch('graph_universe.run.GraphUniverseDataset')
    @patch('graph_universe.run.load_config')
    def test_main(self, mock_load_config, mock_dataset):
        """Test main function."""
        # Set up mock for load_config
        mock_load_config.return_value = self.config

        # Call main function
        main()

        # Check that load_config was called
        mock_load_config.assert_called_once()

        # Check that GraphUniverseDataset was initialized with correct parameters
        mock_dataset.assert_called_once_with(
            root="datasets",
            parameters=self.config,
        )

    @patch('builtins.open', new_callable=mock_open, read_data="invalid: yaml: :")
    def test_load_config_invalid_yaml(self, mock_file):
        """Test load_config function with invalid YAML."""
        # Try to load an invalid YAML file
        with pytest.raises(yaml.YAMLError):
            load_config("invalid_config.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
