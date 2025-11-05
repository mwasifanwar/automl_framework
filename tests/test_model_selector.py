import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from automl_framework.core.model_selector import ModelSelector
from sklearn.datasets import make_classification

class TestModelSelector(unittest.TestCase):
    def setUp(self):
        self.selector = ModelSelector()
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        self.X = pd.DataFrame(X)
        self.y = y
    
    def test_problem_type_detection(self):
        problem_type = self.selector.detect_problem_type(self.y)
        self.assertEqual(problem_type, 'classification')
    
    def test_model_selection(self):
        best_model_name, best_score = self.selector.select_best_model(self.X, self.y)
        self.assertIsNotNone(best_model_name)
        self.assertGreater(best_score, 0)
    
    def test_model_pool(self):
        self.selector.detect_problem_type(self.y)
        models = self.selector.get_model_pool()
        self.assertGreater(len(models), 0)

if __name__ == '__main__':
    unittest.main()