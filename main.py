import argparse
import pandas as pd
from automl_framework.core.data_processor import DataProcessor
from automl_framework.core.feature_engineer import FeatureEngineer
from automl_framework.core.model_selector import ModelSelector
from automl_framework.core.hyperparameter_optimizer import HyperparameterOptimizer
from automl_framework.utils.config_loader import ConfigLoader
from automl_framework.utils.pipeline_utils import PipelineUtils

def main():
    parser = argparse.ArgumentParser(description='AutoML Framework')
    parser.add_argument('--data', type=str, required=True, help='Path to input data file')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    config_loader = ConfigLoader(args.config)
    
    print("Loading data...")
    data_processor = DataProcessor(config_loader.get('data_processing'))
    X, y = data_processor.load_data(args.data, args.target)
    
    print("Preprocessing data...")
    X_processed, y_processed = data_processor.preprocess_pipeline(X, y)
    
    print("Engineering features...")
    feature_engineer = FeatureEngineer(config_loader.get('feature_engineering'))
    X_engineered = feature_engineer.automated_feature_engineering(X_processed, y_processed)
    
    print("Selecting best model...")
    model_selector = ModelSelector(config_loader.get('model_selection'))
    best_model_name, best_score = model_selector.select_best_model(X_engineered, y_processed)
    
    print(f"Best model: {best_model_name} with score: {best_score:.4f}")
    
    print("Optimizing hyperparameters...")
    hyper_optimizer = HyperparameterOptimizer(config_loader.get('hyperparameter_optimization'))
    optimized_model, optimized_score = hyper_optimizer.optimize_hyperparameters(
        model_selector.best_model, X_engineered, y_processed, 
        best_model_name, model_selector.problem_type
    )
    
    print(f"Optimized score: {optimized_score:.4f}")
    
    pipeline_utils = PipelineUtils()
    timestamp = pipeline_utils.generate_timestamp()
    
    results = {
        'best_model': best_model_name,
        'initial_score': best_score,
        'optimized_score': optimized_score,
        'best_parameters': hyper_optimizer.best_params,
        'timestamp': timestamp
    }
    
    pipeline_utils.save_results(results, f'{args.output}/results_{timestamp}.json')
    pipeline_utils.save_pipeline(optimized_model, f'{args.output}/model_{timestamp}.pkl')
    
    print(f"Pipeline completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()