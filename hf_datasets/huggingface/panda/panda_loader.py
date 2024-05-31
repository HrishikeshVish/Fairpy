from hf_datasets.core.data_loader import DataLoader, DatasetConfig
from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd

class PandaLoader(DataLoader):
    CONFIG_NAMES = [None]
    SPLITS = [None, 'train', 'validation']

    def __init__(self, split: str = 'train'):
        """
        Initialize the loader with the specified split (train, validation, or test).
        """
        self.name = 'facebook/panda'
        self.config = None
        assert split in self.SPLITS, "Invalid split. Must be one of ['train', 'validation']"
        
        if split is None:
            self.df = self._load_all_configs()
        else:
            self.config = DatasetConfig(dataset_name=self.name, split=split)
            super().__init__(self.config)
            self.dataset = self.dataset
            self.df = self.df

    
    def _load_all_data(self) -> Dataset:
        """
        Load the entire dataset
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


    def visualize_attribute_distribution(self):
        """
        Visualize the distribution of target attributes in the dataset.
        """
        attribute_counts = self.df['target_attribute'].value_counts()
        attribute_counts.plot(kind='bar')
        plt.title('Target Attribute Distribution')
        plt.xlabel('Target Attribute')
        plt.ylabel('Frequency')
        plt.show()

    def compare_original_perturbed(self, num_examples: int = 5):
        """
        Compare the original and perturbed texts to see how the perturbations affect the sentences.
        """
        examples = self.df.sample(num_examples)
        for idx, row in examples.iterrows():
            print(f"Original: {row['original']}")
            print(f"Perturbed: {row['perturbed']}")
            print("-" * 40)
    
    def extract_perturbation_examples(self, attribute: str, num_examples: int = 5):
        """
        Extract examples based on specific attributes or keywords.
        """
        examples = self.df[self.df['target_attribute'] == attribute].sample(num_examples)
        return examples[['original', 'selected_word', 'perturbed']]

    
    def export_data(self, filename: str) -> None:
        """
        Export the dataset to a .dat file.
        """
        self.df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
