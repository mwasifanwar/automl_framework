import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import cross_val_score

class EnsembleBuilder:
    def __init__(self, config=None):
        self.config = config or {}
        self.best_ensemble = None
        
    def create_voting_ensemble(self, models, problem_type):
        if problem_type == 'classification':
            return VotingClassifier(
                estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
                voting='soft'
            )
        else:
            return VotingRegressor(
                estimators=[(f'model_{i}', model) for i, model in enumerate(models)]
            )
    
    def evaluate_ensemble(self, ensemble, X, y, cv=5):
        scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
        return np.mean(scores), np.std(scores)
    
    def build_optimal_ensemble(self, models, X, y, problem_type, top_k=3):
        model_scores = []
        
        for i, model in enumerate(models):
            try:
                score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
                model_scores.append((i, score, model))
            except:
                continue
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        best_models = [model for _, _, model in model_scores[:top_k]]
        
        ensemble = self.create_voting_ensemble(best_models, problem_type)
        ensemble_score = self.evaluate_ensemble(ensemble, X, y)[0]
        
        self.best_ensemble = ensemble
        return ensemble, ensemble_score