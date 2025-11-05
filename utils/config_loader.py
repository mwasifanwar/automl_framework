import yaml
import json

class ConfigLoader:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(file)
                elif self.config_path.endswith('.json'):
                    return json.load(file)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        return {
            'data_processing': {
                'missing_value_strategy': 'auto',
                'encoding_strategy': 'auto',
                'scaling_strategy': 'standard'
            },
            'feature_engineering': {
                'create_interactions': True,
                'create_polynomials': True,
                'feature_selection': True,
                'max_features': 50
            },
            'model_selection': {
                'cv_folds': 5,
                'scoring_metric': 'auto'
            },
            'hyperparameter_optimization': {
                'method': 'random',
                'n_iter': 50
            },
            'neural_architecture_search': {
                'max_epochs': 100,
                'patience': 10
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default