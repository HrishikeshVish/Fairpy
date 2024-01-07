from datasets.core.data_loader import HfDatasetLoader
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from typing import Union
from . import config

class StereoSetLoader(HfDatasetLoader):
    """
    Loader for the StereoSet dataset using the Hugging Face datasets library.

    This class inherits from HfDatasetLoader and is specifically tailored for
    loading the StereoSet dataset. It initializes the dataset loader with the 
    predefined dataset name from the configuration and loads the associated metadata.

    Attributes:
        _dataset_name (str): The name of the dataset.
        _builder (DatasetBuilder): The dataset builder from Hugging Face datasets library.
        _metadata (DatasetInfo): Metadata information about the dataset.

    Methods:
        __init__: Initializes the StereoSetLoader with the name of the StereoSet dataset.
                  Automatically loads the dataset metadata upon instantiation.

    Inherits all methods and properties from HfDatasetLoader, including:
        - load_hf_dataset
        - name
        - metadata
        - dataset_builder

    Example:
        >>> stereoset_loader = StereoSetLoader()
        >>> print(stereoset_loader.metadata)
    """
    def __init__(self):
        super().__init__(dataset_name=config.DATASET_NAME)
        self._load_hf_metadata()
        self._axes = config.AXES
        self._split = config.SPLIT

    def load_dataset(self, streaming=config.STREAMING, **kwargs) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        return self._load_hf_dataset(streaming=streaming, split=self._split, **kwargs)


    def standardize(self): # will be used to normalize dataset to work with all models of interest for 
        pass

    @property
    def axes(self) -> list:
        return self._axes
    
    
