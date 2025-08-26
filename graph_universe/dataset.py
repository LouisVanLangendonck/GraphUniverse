from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
import pickle
import os.path as osp
from omegaconf import DictConfig
import json
import os

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
        graph_list: list[Data],
        root: str,
        name: str,
        parameters: DictConfig,
        **kwargs,
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.graph_list = graph_list
        super().__init__(
            root,
        )

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(
            self.root,
            self.name
        )

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
        return ["raw_data.pt"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Nothing to download
        pass


    def process(self) -> None:
        r"""Handle the data for the dataset.
        """

        self.data, self.slices = self.collate(self.graph_list)
        self.graph_list = []    # Reset cache.
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
        # Save the metadata
        metadata_file = os.path.join(self.processed_root, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.parameters, f, indent=2, default=str)
