import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from automl_framework.core.data_processor import DataProcessor
from automl_framework.core.feature_engineer import FeatureEngineer
from automl_framework.core.model_selector import ModelSelector
from automl_framework.core.hyperparameter_optimizer import HyperparameterOptimizer
from automl_framework.core.neural_architecture_search import NeuralArchitectureSearch
from automl_framework.models.ensemble_builder import EnsembleBuilder
import pandas as pd
from sklearn.datasets import make_classification

def advanced_automl_pipeline():
    print("Running Advanced AutoML Pipeline...")
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, n_classes=3, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    data_processor = DataProcessor()
    X_processed, y_processed = data_processor.preprocess_pipeline(pd.DataFrame(X_train), y_train)
    
    feature_engineer = FeatureEngineer()
    X_engineered = feature_engineer.automated_feature_engineering(X_processed, y_processed)
    
    model_selector = ModelSelector()
    best_model_name, best_score = model_selector.select_best_model(X_engineered, y_processed)
    
    hyper_optimizer = HyperparameterOptimizer()
    optimized_model, optimized_score = hyper_optimizer.optimize_hyperparameters(
        model_selector.best_model, X_engineered, y_processed, best_model_name, 'classification',
        method='bayesian', n_iter=30
    )
    
    nas = NeuralArchitectureSearch()
    nn_model, nn_score = nas.search_architecture(X_engineered.values, y_processed, 
                                                model_type='mlp', epochs=50)
    
    ensemble_builder = EnsembleBuilder()
    models = [optimized_model, nn_model]
    ensemble, ensemble_score = ensemble_builder.build_optimal_ensemble(
        models, X_engineered, y_processed, 'classification'
    )
    
    print(f"Individual Model Score: {optimized_score:.4f}")
    print(f"Neural Network Score: {nn_score:.4f}")
    print(f"Ensemble Score: {ensemble_score:.4f}")

if __name__ == "__main__":
    advanced_automl_pipeline()