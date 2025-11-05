import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def load_data(self, file_path, target_column=None):
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
            
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            return X, y
        return data, None
    
    def detect_column_types(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
    
    def handle_missing_values(self, X, strategy='auto'):
        column_types = self.detect_column_types(X)
        
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if col in column_types['numeric']:
                    if strategy == 'auto':
                        imp_strategy = 'mean' if X[col].skew() < 2 else 'median'
                    else:
                        imp_strategy = strategy
                    
                    imputer = SimpleImputer(strategy=imp_strategy)
                    X[col] = imputer.fit_transform(X[[col]]).ravel()
                    self.imputers[col] = imputer
                    
                elif col in column_types['categorical']:
                    imputer = SimpleImputer(strategy='most_frequent')
                    X[col] = imputer.fit_transform(X[[col]]).ravel()
                    self.imputers[col] = imputer
        
        return X
    
    def encode_categorical(self, X, target=None):
        column_types = self.detect_column_types(X)
        
        for col in column_types['categorical']:
            if X[col].nunique() <= 10:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder
            else:
                X = pd.get_dummies(X, columns=[col], prefix=col)
        
        return X
    
    def scale_features(self, X):
        column_types = self.detect_column_types(X)
        
        for col in column_types['numeric']:
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(X[[col]]).ravel()
            self.scalers[col] = scaler
        
        return X
    
    def preprocess_pipeline(self, X, y=None):
        X_processed = X.copy()
        
        X_processed = self.handle_missing_values(X_processed)
        X_processed = self.encode_categorical(X_processed)
        X_processed = self.scale_features(X_processed)
        
        if y is not None:
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y_processed = label_encoder.fit_transform(y)
                self.encoders['target'] = label_encoder
            else:
                y_processed = y.values
        else:
            y_processed = None
        
        return X_processed, y_processed