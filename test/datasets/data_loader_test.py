import unittest
import sys
sys.path.append('D:\Projects\Python\FairPy_v2')
from hf_datasets.core.data_loader import HfDatasetLoader
from datasets import DatasetBuilder

class DataLoaderTest(unittest.TestCase):

    def test_init_valid_dataset(self):
        """ Test initialization with a valid dataset name. """
        loader = HfDatasetLoader('fka/awesome-chatgpt-prompts')  # Replace 'dataset_name' with a real dataset name
        self.assertIsNotNone(loader)

    def test_init_none_dataset(self):
        """ Test initialization with None as dataset name. """
        with self.assertRaises(ValueError):
            HfDatasetLoader(None)

    def test_metadata_loading(self):
        """ Test loading of metadata. """
        loader = HfDatasetLoader('fka/awesome-chatgpt-prompts')  # Replace 'dataset_name' with a real dataset name
        loader._load_hf_metadata()
        self.assertIsNotNone(loader.metadata)

    def test_load_hf_dataset(self):
        """ Test loading of Hugging Face dataset. """
        loader = HfDatasetLoader('fka/awesome-chatgpt-prompts')  # Replace 'dataset_name' with a real dataset name
        dataset = loader.load_hf_dataset()
        self.assertIsNotNone(dataset)

    def test_properties(self):
        """ Test properties of the class. """
        loader = HfDatasetLoader('fka/awesome-chatgpt-prompts')  # Replace 'dataset_name' with a real dataset name
        self.assertEqual(loader.name, 'fka/awesome-chatgpt-prompts')
        self.assertIsInstance(loader.dataset_builder, DatasetBuilder)

# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()