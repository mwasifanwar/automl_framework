import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from automl_framework.core.data_processor import DataProcessor
from automl_framework.core.feature_engineer import FeatureEngineer
from automl_framework.core.model_selector import ModelSelector
from automl_framework.core.hyperparameter_optimizer import HyperparameterOptimizer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def basic_automl_example():
    print("Running Basic AutoML Example...")
    
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    data_processor = DataProcessor()
    X_processed, y_processed = data_processor.preprocess_pipeline(pd.DataFrame(X_train), y_train)
    
    feature_engineer = FeatureEngineer()
    X_engineered = feature_engineer.automated_feature_engineering(X_processed, y_processed)
    
    model_selector = ModelSelector()
    best_model_name, best_score = model_selector.select_best_model(X_engineered, y_processed)
    
    print(f"Best Model: {best_model_name}")
    print(f"Best Score: {best_score:.4f}")
    
    hyper_optimizer = HyperparameterOptimizer()
    optimized_model, optimized_score = hyper_optimizer.optimize_hyperparameters(
        model_selector.best_model, X_engineered, y_processed, best_model_name, 'classification'
    )
    
    print(f"Optimized Score: {optimized_score:.4f}")
    
    X_test_processed, _ = data_processor.preprocess_pipeline(pd.DataFrame(X_test))
    X_test_engineered = feature_engineer.automated_feature_engineering(X_test_processed)
    
    test_metrics = model_selector.get_model_performance(X_test_engineered, y_test)
    print("Test Metrics:", test_metrics)

if __name__ == "__main__":
    basic_automl_example()