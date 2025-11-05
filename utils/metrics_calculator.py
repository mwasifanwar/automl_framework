import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

class MetricsCalculator:
    def __init__(self):
        pass
    
    def calculate_classification_metrics(self, y_true, y_pred, y_proba=None):
        metrics = {}
        
        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['precision'] = self.precision_score(y_true, y_pred)
        metrics['recall'] = self.recall_score(y_true, y_pred)
        metrics['f1_score'] = self.f1_score(y_true, y_pred)
        
        if y_proba is not None:
            metrics['log_loss'] = self.log_loss(y_true, y_proba)
            metrics['roc_auc'] = self.roc_auc_score(y_true, y_proba)
        
        return metrics
    
    def calculate_regression_metrics(self, y_true, y_pred):
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    def precision_score(self, y_true, y_pred):
        from sklearn.metrics import precision_score
        return precision_score(y_true, y_pred, average='weighted')
    
    def recall_score(self, y_true, y_pred):
        from sklearn.metrics import recall_score
        return recall_score(y_true, y_pred, average='weighted')
    
    def f1_score(self, y_true, y_pred):
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='weighted')
    
    def log_loss(self, y_true, y_proba):
        from sklearn.metrics import log_loss
        return log_loss(y_true, y_proba)
    
    def roc_auc_score(self, y_true, y_proba):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_proba, multi_class='ovr')
    
    def generate_report(self, y_true, y_pred, problem_type, class_names=None):
        if problem_type == 'classification':
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            return {'classification_report': report, 'confusion_matrix': cm}
        else:
            return self.calculate_regression_metrics(y_true, y_pred)