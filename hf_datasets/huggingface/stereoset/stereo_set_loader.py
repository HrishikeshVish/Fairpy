
from datasets import load_dataset
from typing import Union
from . import config

class StereoSetLoader():
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
        self.dataset_name = 'McGill-NLP/stereoset'
        self.dataset = load_dataset(self.dataset_name)


    

    
