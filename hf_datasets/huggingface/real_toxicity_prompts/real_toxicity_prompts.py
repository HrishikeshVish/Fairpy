from hf_datasets.core.data_loader import DataLoader, DatasetConfig
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt

class RealToxcityPromptsLoader(DataLoader):
    SPLITS = ['train']
    

    def __init__(self):
        """
        Initialize the loader with the specified split (train, validation, or test).
        """
        self.name = "allenai/real-toxicity-prompts"
        self.config = DatasetConfig(dataset_name=self.name, split=self.SPLITS[0])
        super().__init__(self.config)
        self.dataset = self.dataset
        self.df = self.df
        self.norm_df = self._normalize_df(self.df)


    def _normalize_df(self, df):
        """
        Normalize the 'prompt' and 'continuation' columns into separate columns.
        """
        prompt_df = pd.json_normalize(df['prompt'])
        continuation_df = pd.json_normalize(df['continuation'])
        
        prompt_df.columns = ['prompt_' + col for col in prompt_df.columns]
        continuation_df.columns = ['continuation_' + col for col in continuation_df.columns]
        
        df = df.drop(columns=['prompt', 'continuation'])
        df = pd.concat([df, prompt_df, continuation_df], axis=1)
        return df

    def export_data(self, filename: str) -> None:
        """
        Export the dataset to a .csv file.
        """
        self.df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")


    def get_toxicity_statistics(self):
        """
        Calculate and return basic statistics about the toxicity scores in the dataset.
        """
        toxicity_columns = [col for col in self.norm_df.columns if any(metric in col for metric in ['profanity', 'sexually_explicit', 'identity_attack', 'flirtation', 'threat', 'insult', 'severe_toxicity', 'toxicity'])]
        toxicity_stats = self.norm_df[toxicity_columns].describe()
        return toxicity_stats
    
    def filter_prompts(self, column: str, threshold: float):
        """
        Filter prompts based on specific toxicity thresholds or other criteria.
        """

        filtered_df = self.norm_df[self.norm_df[column] >= threshold]
        return filtered_df
    
    def visualize_toxicity_distributions(self):
        """
        Create visualizations of the distribution of various toxicity scores.
        """
        toxicity_columns = [col for col in self.norm_df.columns if any(metric in col for metric in ['profanity', 'sexually_explicit', 'identity_attack', 'flirtation', 'threat', 'insult', 'severe_toxicity', 'toxicity'])]
        self.df[toxicity_columns].hist(bins=20, figsize=(20, 15))
        plt.suptitle('Toxicity Distributions')
        plt.show()
    
    def compare_prompt_continuation(self):
        """
        Compare toxicity scores between prompts and their continuations.
        """
        toxicity_columns = ['profanity', 'sexually_explicit', 'identity_attack', 'flirtation', 'threat', 'insult', 'severe_toxicity', 'toxicity']
        comparisons = self.norm_df.apply(lambda row: pd.Series({f'{col}_diff': row[f'prompt_{col}'] - row[f'continuation_{col}'] for col in toxicity_columns}), axis=1)
        return comparisons.mean()
    
    def extract_examples(self, column: str, threshold: float, num_examples: int = 5):
        """
        Extract examples based on specific criteria, such as high toxicity or specific types of toxicity.
        """
        examples = self.norm_df[self.norm_df[column] >= threshold].sample(num_examples)
        return examples[['prompt_text', 'continuation_text']]
    
    def summarize_metadata(self):
        """
        Provide a summary of the metadata, such as the number of challenging prompts.
        """
        summary = self.norm_df['challenging'].value_counts()
        return summary
    
    @property
    def data(self):
        """
        Return the raw dataset.
        """
        return Dataset.from_pandas(self.df)