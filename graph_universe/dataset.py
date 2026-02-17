import hashlib
import json
import os
import os.path as osp

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


class GraphUniverseDataset(InMemoryDataset):
    r"""Dataset class for GraphUniverse datasets.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        root: str,
        parameters: dict,
        name: str | None = None,
        graph_list: list[Data] | None = None,
        **kwargs,
    ) -> None:
        self.name = name if name is not None else self.get_dataset_dir(parameters)
        self.parameters = parameters
        self.graph_list = graph_list if graph_list is not None else []
        super().__init__(
            root,
        )
        data, self.slices, self.sizes, data_cls = fs.torch_load(self.processed_paths[0])
        self.data = data_cls.from_dict(data)
        assert isinstance(self._data, Data)

    def get_dataset_dir(self, config: dict) -> str:
        """Generate a unique dataset directory based on the configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            str: Unique dataset directory.
        """
        # Create a hash of the uniquely identifying metadata
        unique_hash = hashlib.sha256(str(config).encode()).hexdigest()

        # First level K_val_edge_prop_var_val
        dataset_dir = f"K_{config['universe_parameters']['K']}_edge_prop_var_{config['universe_parameters']['edge_propensity_variance']}"

        # Second level homophily_[minval_maxval]
        dataset_dir = os.path.join(
            dataset_dir,
            f"homophily_{config['family_parameters']['homophily_range'][0]}_to_{config['family_parameters']['homophily_range'][1]}",
        )
        # Third level n_graphs_val_n_nodes_[minval_maxval]
        n_nodes_range = config['family_parameters'].get('n_nodes_range',
                                                         [config['family_parameters'].get('min_n_nodes'),
                                                          config['family_parameters'].get('max_n_nodes')])
        dataset_dir = os.path.join(
            dataset_dir,
            f"n_graphs_{config['family_parameters']['n_graphs']}_n_nodes_{n_nodes_range[0]}_to_{n_nodes_range[1]}",
        )
        # Fourth level n_communities_[minval_maxval]
        n_communities_range = config['family_parameters'].get('n_communities_range',
                                                               [config['family_parameters'].get('min_communities'),
                                                                config['family_parameters'].get('max_communities')])
        dataset_dir = os.path.join(
            dataset_dir,
            f"n_communities_{n_communities_range[0]}_to_{n_communities_range[1]}",
        )

        # Fifth level task (if it exists)
        if 'task' in config and config['task'] is not None:
            dataset_dir = os.path.join(dataset_dir, f"task_{config['task']}")

        # Final level hash
        dataset_dir = os.path.join(dataset_dir, f"hash_{unique_hash}")
        return dataset_dir

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(
            self.root,
            self.name,
        )
        return self.processed_root

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return ["data.pt"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def get_data_dir(self) -> str:
        """Return the path to the data directory.

        Returns
        -------
        str
            Path to the data directory.
        """
        return osp.join(self.root, self.name)

    def download(self) -> None:
        r"""Generates the dataset"""
        from .graph_family import GraphFamilyGenerator
        from .graph_universe import GraphUniverse

        # Initialize GraphUniverse
        universe = GraphUniverse(
            **self.parameters["universe_parameters"],
        )

        # Initialize GraphFamilyGenerator
        family = GraphFamilyGenerator(
            universe=universe,
            **self.parameters["family_parameters"],
        )

        # Generate and save graph family
        family.generate_family(show_progress=True)
        self.graph_list = family.to_pyg_graphs(self.parameters.get("task", None))

    def process(self) -> None:
        r"""Handle the data for the dataset."""

        if not self.graph_list or len(self.graph_list) == 0:
            raise ValueError("Cannot process dataset: graph_list is empty. Generate or provide graphs before calling process().")

        self.data, self.slices = self.collate(self.graph_list)
        self.graph_list = []
        self._data_list = None
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
        metadata_file = os.path.join(self.processed_root, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(self.parameters, f, indent=2, default=str)
