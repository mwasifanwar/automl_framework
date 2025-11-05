import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime

class PipelineUtils:
    def __init__(self):
        pass
    
    def save_pipeline(self, pipeline, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline, f)
    
    def load_pipeline(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_results(self, results, filepath):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_experiment_log(self, experiment_name, params, results, filepath):
        log_entry = {
            'experiment_name': experiment_name,
            'timestamp': self.generate_timestamp(),
            'parameters': params,
            'results': results
        }
        
        try:
            with open(filepath, 'r') as f:
                logs = json.load(f)
        except FileNotFoundError:
            logs = []
        
        logs.append(log_entry)
        
        with open(filepath, 'w') as f:
            json.dumps(logs, f, indent=2)