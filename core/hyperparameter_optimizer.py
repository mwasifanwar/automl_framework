import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, uniform
import optuna
from functools import partial

class HyperparameterOptimizer:
    def __init__(self, config=None):
        self.config = config or {}
        self.best_params = {}
        self.study = None
        
    def get_param_distributions(self, model_name, problem_type):
        if problem_type == 'classification':
            if model_name == 'random_forest':
                return {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 20),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            elif model_name == 'xgboost':
                return {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4)
                }
            elif model_name == 'svm':
                return {
                    'C': uniform(0.1, 10),
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5))
                }
        else:
            if model_name == 'random_forest':
                return {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 20),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10)
                }
        
        return {}
    
    def random_search_optimization(self, model, X, y, param_distributions, n_iter=50, cv=5):
        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=n_iter, cv=cv, 
            scoring='accuracy' if hasattr(model, 'predict_proba') else 'r2',
            random_state=42, n_jobs=-1
        )
        
        random_search.fit(X, y)
        self.best_params = random_search.best_params_
        
        return random_search.best_estimator_, random_search.best_score_
    
    def objective_function(self, trial, model, X, y, model_name, problem_type):
        if model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
        
        model.set_params(**params)
        
        from sklearn.model_selection import cross_val_score
        
        if problem_type == 'classification':
            score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        else:
            score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
        
        return score
    
    def bayesian_optimization(self, model, X, y, model_name, problem_type, n_trials=100):
        objective = partial(
            self.objective_function, 
            model=model, X=X, y=y, 
            model_name=model_name, problem_type=problem_type
        )
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials)
        
        model.set_params(**self.study.best_params)
        model.fit(X, y)
        
        self.best_params = self.study.best_params
        
        return model, self.study.best_value
    
    def optimize_hyperparameters(self, model, X, y, model_name, problem_type, method='random', n_iter=50):
        if method == 'random':
            param_distributions = self.get_param_distributions(model_name, problem_type)
            return self.random_search_optimization(model, X, y, param_distributions, n_iter)
        elif method == 'bayesian':
            return self.bayesian_optimization(model, X, y, model_name, problem_type, n_iter)
        else:
            raise ValueError("Optimization method must be 'random' or 'bayesian'")