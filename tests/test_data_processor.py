import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from automl_framework.core.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.sample_data = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'missing_numeric': [1, 2, np.nan, 4, 5],
            'missing_categorical': ['A', 'B', np.nan, 'C', 'A']
        })
    
    def test_missing_value_handling(self):
        processed = self.processor.handle_missing_values(self.sample_data)
        self.assertFalse(processed.isnull().any().any())
    
    def test_categorical_encoding(self):
        processed = self.processor.encode_categorical(self.sample_data)
        self.assertTrue(all(processed.dtypes != 'object'))
    
    def test_preprocess_pipeline(self):
        X_processed, y_processed = self.processor.preprocess_pipeline(self.sample_data)
        self.assertIsInstance(X_processed, pd.DataFrame)
        self.assertFalse(X_processed.isnull().any().any())

if __name__ == '__main__':
    unittest.main()