# test/datasets/test_mcgill_stereoset_loader.py
import unittest
import os
import pandas as pd
from pydantic import ValidationError
from hf_datasets.core.data_loader import DatasetConfig
from hf_datasets.huggingface.stereoset.stereoset_loader import StereosetLoader

class TestStereosetLoader(unittest.TestCase):

    def setUp(self):
        """
        Set up the dataset loader for tests.
        """
        self.intrasentence_loader = StereosetLoader(config_name='intrasentence')
        self.intersentence_loader = StereosetLoader(config_name='intersentence')
        self.both_loader = StereosetLoader(config_name=None)

    def test_initialization_intrasentence(self):
        """
        Test the initialization with intrasentence config.
        """
        self.assertEqual(self.intrasentence_loader.config.dataset_name, "McGill-NLP/stereoset")
        self.assertEqual(self.intrasentence_loader.config.split, "validation")
        self.assertEqual(self.intrasentence_loader.config.config_name, "intrasentence")

    def test_initialization_intersentence(self):
        """
        Test the initialization with intersentence config.
        """
        self.assertEqual(self.intersentence_loader.config.dataset_name, "McGill-NLP/stereoset")
        self.assertEqual(self.intersentence_loader.config.split, "validation")
        self.assertEqual(self.intersentence_loader.config.config_name, "intersentence")

    def test_initialization_both(self):
        """
        Test the initialization with both configs.
        """
        self.assertIsNone(self.both_loader.config)

    def test_invalid_config_name(self):
        """
        Test invalid config name.
        """
        with self.assertRaises(AssertionError):
            StereosetLoader(config_name='invalid_config')

    def test_load_intrasentence_data(self):
        """
        Test loading of intrasentence data.
        """
        intrasentence_data = self.intrasentence_loader.dataset
        self.assertIsNotNone(intrasentence_data)
        self.assertTrue('bias_type' in intrasentence_data.column_names)

    def test_load_intersentence_data(self):
        """
        Test loading of intersentence data.
        """
        intersentence_data = self.intersentence_loader.dataset
        self.assertIsNotNone(intersentence_data)
        self.assertTrue('bias_type' in intersentence_data.column_names)

    def test_load_both_data(self):
        """
        Test loading of both intrasentence and intersentence data.
        """
        both_data = self.both_loader.dataset
        self.assertIsNotNone(both_data)
        self.assertTrue('bias_type' in both_data.column_names)

    def test_filter_by_bias_type_valid(self):
        """
        Test filtering by valid bias types.
        """
        for bias_type in StereosetLoader.VALID_BIAS_TYPES:
            filtered_data = self.both_loader.filter_by_bias_type(bias_type)
            self.assertIsNotNone(filtered_data)
            self.assertTrue((filtered_data.to_pandas()['bias_type'] == bias_type).all())

    def test_filter_by_bias_type_invalid(self):
        """
        Test filtering by an invalid bias type.
        """
        with self.assertRaises(AssertionError):
            self.both_loader.filter_by_bias_type('invalid_bias_type')

    def test_get_bias_type_distribution(self):
        """
        Test getting the distribution of bias types.
        """
        distribution = self.both_loader.get_bias_type_distribution()
        self.assertIsNotNone(distribution)
        self.assertIsInstance(distribution, pd.Series)

if __name__ == "__main__":
    unittest.main()



    def test_visualize_bias_type_distribution(self):
        """
        Test visualizing the bias type distribution.
        """
        try:
            self.both_loader.visualize_bias_type_distribution()
        except Exception as e:
            self.fail(f"Visualization failed with exception: {e}")

    def test_export_data(self):
        """
        Test exporting data to a CSV file.
        """
        filename = 'test_stereoset_data.csv'
        try:
            self.both_loader.export_data(filename)
            self.assertTrue(os.path.exists(filename))
        finally:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    unittest.main()