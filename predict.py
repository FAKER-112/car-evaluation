from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

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
class_mapping = {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features
        X = pd.DataFrame([data])
        
        # Encode ordinal features
        for col, mapping in ordinal_mappings.items():
            X[col] = X[col].map(mapping)
        
        # One-hot encode 'lug_boot'
        X = pd.get_dummies(X, columns=['lug_boot'], drop_first=False)
        
        # Ensure all expected columns exist
        for col in lug_boot_columns:
            if col not in X:
                X[col] = 0
        
        # Reorder columns (important!)
        X = X[['buying', 'maint', 'doors', 'persons', 'safety'] + lug_boot_columns]
        
        # Predict
        pred_numeric = model.predict(X)[0]
        pred_label = class_mapping[pred_numeric]
        
        return jsonify({'prediction': pred_label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
