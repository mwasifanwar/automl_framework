import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import featuretools as ft

class FeatureEngineer:
    def __init__(self, config=None):
        self.config = config or {}
        self.feature_selector = None
        self.pca = None
        self.kmeans = {}
        
    def create_interaction_features(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
        
        return X
    
    def create_polynomial_features(self, X, degree=2):
        from sklearn.preprocessing import PolynomialFeatures
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(X[numeric_cols])
        
        poly_df = pd.DataFrame(poly_features, 
                             columns=poly.get_feature_names_out(numeric_cols),
                             index=X.index)
        
        X = pd.concat([X, poly_df], axis=1)
        return X
    
    def create_statistical_features(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        X['mean_features'] = X[numeric_cols].mean(axis=1)
        X['std_features'] = X[numeric_cols].std(axis=1)
        X['skew_features'] = X[numeric_cols].skew(axis=1)
        X['kurtosis_features'] = X[numeric_cols].kurtosis(axis=1)
        
        return X
    
    def create_cluster_features(self, X, n_clusters=3):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X[numeric_cols])
        
        X['cluster_feature'] = cluster_labels
        self.kmeans['main'] = kmeans
        
        for i in range(n_clusters):
            X[f'cluster_{i}_distance'] = np.linalg.norm(
                X[numeric_cols] - kmeans.cluster_centers_[i], axis=1
            )
        
        return X
    
    def automated_feature_engineering(self, X, y=None):
        X_engineered = X.copy()
        
        X_engineered = self.create_interaction_features(X_engineered)
        X_engineered = self.create_polynomial_features(X_engineered, degree=2)
        X_engineered = self.create_statistical_features(X_engineered)
        X_engineered = self.create_cluster_features(X_engineered)
        
        if y is not None:
            X_engineered = self.select_features(X_engineered, y)
        
        return X_engineered
    
    def select_features(self, X, y, k=50):
        if len(X.columns) > k:
            if y.dtype == 'object' or len(np.unique(y)) < 10:
                selector = SelectKBest(score_func=f_classif, k=k)
            else:
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            
            selected_columns = X.columns[selector.get_support()]
            return pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
        
        return X
    
    def apply_pca(self, X, n_components=0.95):
        if len(X.columns) > 10:
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            
            pca_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
            X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
            
            return pd.concat([X, X_pca_df], axis=1)
        
        return X