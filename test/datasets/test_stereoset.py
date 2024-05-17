import unittest
import sys

from hf_datasets.huggingface.stereoset import stereo_set_loader
from datasets import DatasetBuilder

class StereoSetTester(unittest.TestCase):
    def test_stereoset_loader(self):
        """
        Test the instantiation of the StereoSetLoader class.
        """
        stereoset_loader = HfDatasetLoader('McGill-NLP/stereoset')
        self.assertEqual(stereoset_loader.name, 'McGill-NLP/stereoset')
        self.assertIsInstance(stereoset_loader.dataset_builder, DatasetBuilder)
        self.assertIsNotNone(stereoset_loader.metadata)



# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()