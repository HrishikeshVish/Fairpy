import datasets
from datasets import load_dataset, DatasetInfo, DatasetBuilder, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from typing import Union

class HfDatasetLoader():
    """
    Base class for loading datasets using the Hugging Face datasets library.

    This class serves as a general loader for any dataset available in the Hugging Face datasets
    repository. It initializes the dataset loader with a specified dataset name and provides
    methods to load the dataset and its associated metadata.

    Attributes:
        _dataset_name (str): The name of the dataset to be loaded.
        _builder (DatasetBuilder): The dataset builder object from Hugging Face datasets library.
        _metadata (DatasetInfo): Metadata information about the dataset, loaded on demand.

    Methods:
        __init__(self, dataset_name): Initializes the dataset loader with a given dataset name.
                                      Raises ValueError if the dataset name is None.
        _load_hf_metadata(self): Loads metadata for the specified dataset using Hugging Face's 
                                 dataset builder.
        load_hf_dataset(self, *args, streaming=False, split=None, **kwargs): Loads the dataset 
                        with optional parameters for streaming and split, as well as additional 
                        arguments and keyword arguments.
        name (property): Returns the name of the dataset.
        metadata (property): Returns the loaded dataset metadata.
        dataset_builder (property): Returns the dataset builder object.

    Example:
        >>> dataset_loader = HfDatasetLoader('my_dataset')
        >>> dataset_loader.load_hf_dataset()
        >>> print(dataset_loader.metadata)

    """
    def __init__(self, dataset_name=None, config='default'):
        if dataset_name is None:
            raise ValueError("Dataset name cannot be None!")
        self._dataset_name: str = dataset_name
        self._default_config = config
        self._init_builder()


        self._metadata: DatasetInfo = None

    def _init_builder(self):
        try:
            self._builder: DatasetBuilder = datasets.load_dataset_builder(self._dataset_name) # only contains the hf metadata
        except FileNotFoundError:
            self._builder = None
            print("Dataset not found. Please check the dataset name.")
        except ValueError as e:
            self._builder = None
            print(f"An error occurred: {e}")
        except Exception as e:
            self._builder = None
            print(f"Unexpected error: {e}")
    
    def _load_hf_metadata(self) -> None:
        if self._builder is not None:
            self._metadata: DatasetInfo = self._builder.get_all_exported_dataset_infos()
        else:
            self._metadata = None
            print('Unable to load dataset info due to dataset builder not found.')
    def load_hf_dataset(self, *args, streaming=False, split=None, **kwargs) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        kwargs['path'] = self._dataset_name
        if streaming:
            if split is not None:
                kwargs['split'] = split
            
            return load_dataset(*args, streaming=True, **kwargs)
        else:
            if split is not None:
                kwargs['split'] = split
            
            return load_dataset(*args, **kwargs)

    def _get_builder_configs(self):
        if self._builder is not None:
            return self._builder.builder_configs.keys()
    
    @property
    def name(self) -> str:
        return self._dataset_name

    @property
    def metadata(self) -> DatasetInfo:
        return self._metadata
    
    @property
    def dataset_builder(self) -> DatasetBuilder:
        return self._builder
