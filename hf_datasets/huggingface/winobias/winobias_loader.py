from hf_datasets.core.data_loader import DataLoader, DatasetConfig
from datasets import Dataset
import pandas as pd

class WinobiasLoader(DataLoader):
    CONFIG_NAMES = ['type1_anti', 'type2_anti', 'type1_pro', 'type2_pro']
    SPLITS = [None, 'train', 'validation']

    def __init__(self, config_name: str = None, split: str = 'validation'):
        """
        Initialize the loader with the specified split (train, validation, or test).
        """
        self.name = 'wino_bias'
        self.config = None
        assert config_name in self.CONFIG_NAMES, "Invalid config. Must be one of ['type1_anti', 'type2_anti', 'type1_pro', 'type2_pro']"
        assert split in self.SPLITS, "Invalid split. Must be one of ['train', 'validation', 'test']"
        
        if config_name is None:
            self.df = self._load_all_configs()
        else:
            self.config = DatasetConfig(dataset_name=self.name, split=split)
            super().__init__(self.config)
            self.dataset = self.dataset
            self.df = self.df

    
    def _load_all_configs(self) -> Dataset:
        """
        Load both the intrasentence and intersentence configurations of the dataset.
        """

        temp_df = pd.DataFrame()
        for split in self.SPLITS:
            for config_name in self.CONFIG_NAMES:
                temp_config = DatasetConfig(dataset_name=self.name, split=split, config_name=config_name)
                temp_loader = DataLoader(temp_config)
                temp_loader['config_name'] = config_name # meta data
                temp_loader['split'] = split # meta data
                temp_df = pd.concat([temp_df, temp_loader.df], ignore_index=True)

        self.config = None # None as we load the full dataset with both configs
        self.df = temp_df
        return Dataset.from_pandas(self.df)





