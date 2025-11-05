import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class NeuralArchitectureSearch:
    def __init__(self, config=None):
        self.config = config or {}
        self.best_model = None
        
    def create_simple_cnn(self, input_shape, num_classes):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def create_advanced_cnn(self, input_shape, num_classes):
        model = keras.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def create_mlp(self, input_dim, num_classes, hidden_layers=[64, 32]):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        for units in hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model
    
    def search_architecture(self, X, y, model_type='mlp', epochs=50, validation_split=0.2):
        if len(X.shape) == 4:
            input_shape = X.shape[1:]
            num_classes = len(np.unique(y))
            
            if model_type == 'simple_cnn':
                model = self.create_simple_cnn(input_shape, num_classes)
            elif model_type == 'advanced_cnn':
                model = self.create_advanced_cnn(input_shape, num_classes)
            else:
                raise ValueError("Invalid model_type for image data")
                
        else:
            input_dim = X.shape[1]
            num_classes = len(np.unique(y))
            
            if model_type == 'mlp':
                model = self.create_mlp(input_dim, num_classes)
            else:
                raise ValueError("Invalid model_type for tabular data")
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.best_model = model
        best_accuracy = max(history.history['val_accuracy'])
        
        return model, best_accuracy