import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelSelector:
    def __init__(self, config=None):
        self.config = config or {}
        self.problem_type = None
        self.best_model = None
        self.model_scores = {}
        
    def detect_problem_type(self, y):
        unique_values = len(np.unique(y))
        
        if y.dtype == 'object' or unique_values < 10:
            self.problem_type = 'classification'
        else:
            self.problem_type = 'regression'
        
        return self.problem_type
    
    def get_model_pool(self):
        if self.problem_type == 'classification':
            return {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(random_state=42, probability=True),
                'knn': KNeighborsClassifier(),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'naive_bayes': GaussianNB(),
                'mlp': MLPClassifier(random_state=42, max_iter=1000),
                'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
                'lightgbm': LGBMClassifier(random_state=42)
            }
        else:
            return {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression(),
                'svm': SVR(),
                'knn': KNeighborsRegressor(),
                'decision_tree': DecisionTreeRegressor(random_state=42),
                'mlp': MLPRegressor(random_state=42, max_iter=1000),
                'xgboost': XGBRegressor(random_state=42),
                'lightgbm': LGBMRegressor(random_state=42)
            }
    
    def evaluate_model(self, model, X, y, cv=5):
        if self.problem_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return np.mean(cv_scores), np.std(cv_scores)
    
    def select_best_model(self, X, y, cv=5):
        self.detect_problem_type(y)
        models = self.get_model_pool()
        
        best_score = -np.inf
        best_model_name = None
        
        for name, model in models.items():
            try:
                mean_score, std_score = self.evaluate_model(model, X, y, cv)
                self.model_scores[name] = {
                    'mean_score': mean_score,
                    'std_score': std_score
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
                    self.best_model = model
                    
            except Exception as e:
                print(f"Model {name} failed: {str(e)}")
                continue
        
        self.best_model.fit(X, y)
        
        return best_model_name, best_score
    
    def get_model_performance(self, X_test, y_test):
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        y_pred = self.best_model.predict(X_test)
        
        if self.problem_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        return metrics