import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml

from graph_universe.dataset import GraphUniverseDataset


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
            config_path (str): Path to the YAML configuration file.

    Returns:
            dict: Loaded configuration parameters.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point for the script."""
    config_path = os.path.join(os.path.dirname(__file__), "../configs/run.yaml")
    config = load_config(config_path)

    # Initialize GraphUniverseDataset
    GraphUniverseDataset(
        root="datasets",
        parameters=config,
    )


if __name__ == "__main__":
    main()
