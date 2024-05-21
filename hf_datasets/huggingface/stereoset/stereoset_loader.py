from hf_datasets.core.data_loader import DataLoader, DatasetConfig
from datasets import Dataset
import pandas as pd

class StereosetLoader(DataLoader):
    CONFIG_NAMES = [None, 'intrasentence', 'intersentence']
    SPLITS = ['validation']
    VALID_BIAS_TYPES = {'race', 'religion', 'gender', 'profession'}

    def __init__(self, config_name: str = None):
        """
        Initialize the loader with the specified split (train, validation, or test).
        """
        self.name = 'McGill-NLP/stereoset'
        self.config = None
        assert config_name in self.CONFIG_NAMES, "Invalid config. Must be one of ['both', 'intrasentence', 'intersentence']"
        
        if config_name is None:
            self.dataset = self._load_both_configs()
        else:
            self.config = DatasetConfig(dataset_name=self.name, split='validation', config_name=config_name)
            super().__init__(self.config)
            self.dataset = self.dataset
            self.df = self.df


    def _load_all_configs(self) -> Dataset:
        """
        Load both the intrasentence and intersentence configurations of the dataset.
        """
        config_names = ['intrasentence', 'intersentence']
        splits = ['validation']
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


    def filter_by_bias_type(self, bias_type) -> Dataset:
        """
        Filter the dataset by a specific bias type.
        """
        assert bias_type in self.VALID_BIAS_TYPES, f"Invalid bias type. Must be one of {self.VALID_BIAS_TYPES}"
        filtered_data = self.df[self.df['bias_type'] == bias_type]
        return Dataset.from_pandas(filtered_data)

    def get_bias_type_distribution(self) -> pd.Series:
        """
        Get the distribution of bias types within the dataset.
        """
        distribution = self.df['bias_type'].value_counts()
        return distribution


    def visualize_bias_type_distribution(self) -> None:
        """
        Visualize the distribution of bias types using a bar plot.
        """
        import matplotlib.pyplot as plt
        distribution = self.get_bias_type_distribution()
        distribution.plot(kind='bar')
        plt.title('Bias Type Distribution')
        plt.xlabel('Bias Type')
        plt.ylabel('Frequency')
        plt.show()

    def export_data(self, filename) -> None:
        """
        Export the dataset or a subset of the dataset to a CSV file.
        """
        self.df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")


    
