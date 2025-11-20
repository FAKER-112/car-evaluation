from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Run train.py first.")
    model = None

# Mapping dictionaries (same as training)
ordinal_mappings = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2}
}

# One-hot columns for 'lug_boot' (as in training)
lug_boot_columns = ['lug_boot_big', 'lug_boot_med', 'lug_boot_small']

# Reverse mapping for output
# Change the values here to what you want to see on the screen
class_mapping = {
    0: 'Unacceptable', 
    1: 'Acceptable', 
    2: 'Good', 
    3: 'Very Good'
}

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
        
    try:
        data = request.get_json()
        
        # Extract features
        X = pd.DataFrame([data])
        
        # Encode ordinal features
        for col, mapping in ordinal_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
            else:
                return jsonify({'error': f'Missing column: {col}'}), 400
        
        # One-hot encode 'lug_boot'
        # We use categorical type to force all columns to appear even if only one value is present
        if 'lug_boot' in X.columns:
            X = pd.get_dummies(X, columns=['lug_boot'], drop_first=False)
        else:
            return jsonify({'error': 'Missing column: lug_boot'}), 400
        
        # Ensure all expected columns exist (fill with 0 if missing from get_dummies)
        for col in lug_boot_columns:
            if col not in X:
                X[col] = 0
        
        # Reorder columns to match the training sequence exactly
        final_columns = ['buying', 'maint', 'doors', 'persons', 'safety'] + lug_boot_columns
        X = X[final_columns]
        
        # Predict
        pred_numeric = model.predict(X)[0]
        pred_label = class_mapping.get(pred_numeric, "Unknown")
        
        return jsonify({
            'prediction': pred_label,
            'class_id': int(pred_numeric)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)