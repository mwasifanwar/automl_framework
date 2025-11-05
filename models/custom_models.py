from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, VotingRegressor
import numpy as np

class AdvancedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
        
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        
        if self.weights:
            weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
            return np.round(weighted_predictions).astype(int)
        else:
            return np.round(np.mean(predictions, axis=0)).astype(int)
    
    def predict_proba(self, X):
        probas = np.array([model.predict_proba(X) for model in self.models])
        
        if self.weights:
            return np.average(probas, axis=0, weights=self.weights)
        else:
            return np.mean(probas, axis=0)

class StackingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        from sklearn.model_selection import cross_val_predict
        
        base_predictions = []
        for model in self.base_models:
            cv_pred = cross_val_predict(model, X, y, cv=5, method='predict_proba')
            base_predictions.append(cv_pred)
        
        X_meta = np.hstack(base_predictions)
        self.meta_model.fit(X_meta, y)
        
        for model in self.base_models:
            model.fit(X, y)
            
        return self
    
    def predict(self, X):
        base_preds = []
        for model in self.base_models:
            pred = model.predict_proba(X)
            base_preds.append(pred)
        
        X_meta = np.hstack(base_preds)
        return self.meta_model.predict(X_meta)
    
    def predict_proba(self, X):
        base_preds = []
        for model in self.base_models:
            pred = model.predict_proba(X)
            base_preds.append(pred)
        
        X_meta = np.hstack(base_preds)
        return self.meta_model.predict_proba(X_meta)