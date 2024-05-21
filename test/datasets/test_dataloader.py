import unittest
from hf_datasets.core.data_loader import DataLoader, DatasetConfig
from pydantic import ValidationError
import pandas as pd

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.valid_config = DatasetConfig(dataset_name="stanfordnlp/imdb", split="train")
        #self.invalid_config_name = DatasetConfig(dataset_name="", split="train")
        # self.invalid_config_split = DatasetConfig(dataset_name="stanfordnlp/imdb", split="invalid_split")
        self.wrapper = DataLoader(self.valid_config)


    def test_valid_config_initialization(self):
        self.assertEqual(self.wrapper.config.dataset_name, "stanfordnlp/imdb")
        self.assertEqual(self.wrapper.config.split, "train")

    # def test_invalid_config_name(self):
    #     with self.assertRaises(ValidationError):
    #         DataLoader(self.invalid_config_name)

    # def test_invalid_config_split(self):
    #     with self.assertRaises(ValidationError):
    #         DataLoader(self.invalid_config_split)

    def test_load_dataset(self):
        self.assertIsNotNone(self.wrapper.dataset)
        self.assertIsInstance(self.wrapper.df, pd.DataFrame)

    def test_get_basic_info(self):
        basic_info = self.wrapper.get_basic_info()
        self.assertIn("dataset_name", basic_info)
        self.assertIn("number_of_rows", basic_info)
        self.assertIn("number_of_columns", basic_info)
        self.assertIn("column_names", basic_info)

    def test_get_missing_values(self):
        missing_values = self.wrapper.get_missing_values()
        self.assertIsInstance(missing_values, dict)

    def test_get_numeric_stats(self):
        numeric_stats = self.wrapper.get_numeric_stats()
        self.assertIsInstance(numeric_stats, dict)

    def test_get_column_stats(self):
        column_name = "label"
        column_stats = self.wrapper.get_column_stats(column_name)
        self.assertIsInstance(column_stats, dict)

    def test_get_column_stats_invalid(self):
        with self.assertRaises(ValueError):
            self.wrapper.get_column_stats("non_existent_column")

    def test_get_unique_values(self):
        column_name = "label"
        unique_values = self.wrapper.get_unique_values(column_name)
        self.assertIsInstance(unique_values, list)

    def test_get_unique_values_invalid(self):
        with self.assertRaises(ValueError):
            self.wrapper.get_unique_values("non_existent_column")

    def test_get_value_counts(self):
        column_name = "label"
        value_counts = self.wrapper.get_value_counts(column_name)
        self.assertIsInstance(value_counts, dict)

    def test_get_value_counts_invalid(self):
        with self.assertRaises(ValueError):
            self.wrapper.get_value_counts("non_existent_column")

if __name__ == "__main__":
    unittest.main()
