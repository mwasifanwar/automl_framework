import pandas as pd
import numpy as np
from datetime import datetime
import json

class ModelMonitor:
    def __init__(self, log_file='model_monitor.log'):
        self.log_file = log_file
        self.metrics_history = []
    
    def log_prediction(self, features, prediction, actual=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        }
        
        self.metrics_history.append(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def calculate_drift(self, reference_data, current_data):
        drift_metrics = {}
        
        for col in reference_data.columns:
            if reference_data[col].dtype in ['float64', 'int64']:
                ref_mean = reference_data[col].mean()
                curr_mean = current_data[col].mean()
                drift_metrics[col] = abs(ref_mean - curr_mean) / ref_mean
        
        return drift_metrics
    
    def generate_report(self):
        if len(self.metrics_history) == 0:
            return "No data available for monitoring report"
        
        df = pd.DataFrame(self.metrics_history)
        report = {
            'total_predictions': len(df),
            'timestamp_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
        return report