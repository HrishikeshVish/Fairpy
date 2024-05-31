import datasets
from datasets import Dataset
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Dict, Any, Union, Optional

class DatasetConfig(BaseModel):
    dataset_name: str
    split: str = 'train'
    config_name: Optional[str] = None


    @field_validator('dataset_name')
    def name_must_be_non_empty(cls, v):
        if not v:
            raise ValueError('dataset_name must be non-empty')
        return v

    @field_validator('split')
    def split_must_be_valid(cls, v):
        valid_splits = ['train', 'test', 'validation']
        if v not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}")
        return v

class DataLoader:
    def __init__(self, config: DatasetConfig):
        """
        Initialize the wrapper with the configuration for the Huggingface dataset.
        """
        self.config = config
        self.dataset = None
        self.df = None
        self.load_dataset()

    def load_dataset(self):
        """
        Load the dataset using the Huggingface datasets library.
        """
        try:
            if self.config.config_name:
                self.dataset = datasets.load_dataset(self.config.dataset_name, name=self.config.config_name, split=self.config.split)
            else:
                self.dataset = datasets.load_dataset(self.config.dataset_name, split=self.config.split)
            self.df = self.dataset.to_pandas()
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")


    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        """
        info = {
            "dataset_name": self.config.dataset_name,
            "number_of_rows": len(self.df),
            "number_of_columns": len(self.df.columns),
            "column_names": self.df.columns.tolist()
        }
        return info
    
    def get_missing_values(self) -> Dict[str, int]:
        """
        Get the number of missing values per column.
        """
        missing_values = self.df.isnull().sum()
        return missing_values.to_dict()
    
    def get_numeric_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Get basic descriptive statistics for numeric columns.
        """
        numeric_stats = self.df.describe().to_dict()
        return numeric_stats
    
    def get_column_stats(self, column_name: str) -> Dict[str, Union[int, float]]:
        """
        Get detailed statistics for a specific column.
        """
        if column_name in self.df.columns:
            column_stats = self.df[column_name].describe()
            return column_stats.to_dict()
        else:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")
    
    def get_unique_values(self, column_name: str) -> List[Any]:
        """
        Get unique values for a specific column.
        """
        if column_name in self.df.columns:
            unique_values = self.df[column_name].unique().tolist()
            return unique_values
        else:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")
    
    def get_value_counts(self, column_name: str) -> Dict[Any, int]:
        """
        Get value counts for a specific column.
        """
        if column_name in self.df.columns:
            value_counts = self.df[column_name].value_counts().to_dict()
            return value_counts
        else:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")
