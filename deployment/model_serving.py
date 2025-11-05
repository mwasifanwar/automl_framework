from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

class ModelServer:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.app = Flask(__name__)
        self.setup_routes()
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def setup_routes(self):
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json()
                df = pd.DataFrame(data['instances'])
                predictions = self.model.predict(df)
                return jsonify({'predictions': predictions.tolist()})
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        self.app.run(host=host, port=port, debug=debug)