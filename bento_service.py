import bentoml
from bentoml.io import JSON
import pandas as pd

# Load your existing pickle model as a BentoML model artifact
# This will allow BentoML to handle versioning automatically
model = bentoml.sklearn.get("car_quality_model:latest")  # We'll save this model later

# Define the BentoML service
svc = bentoml.Service("car_quality_predictor", runners=[])

# Mapping dictionaries (same as your Flask version)
ordinal_mappings = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2}
}

lug_boot_columns = ['lug_boot_big', 'lug_boot_med', 'lug_boot_small']
class_mapping = {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'}

# Define the prediction API endpoint
@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    try:
        X = pd.DataFrame([input_data])

        # Encode ordinal features
        for col, mapping in ordinal_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
            else:
                return {"error": f"Missing column: {col}"}

        # One-hot encode 'lug_boot'
        if 'lug_boot' in X.columns:
            X = pd.get_dummies(X, columns=['lug_boot'], drop_first=False)
        else:
            return {"error": "Missing column: lug_boot"}

        # Ensure all expected columns exist
        for col in lug_boot_columns:
            if col not in X:
                X[col] = 0

        final_columns = ['buying', 'maint', 'doors', 'persons', 'safety'] + lug_boot_columns
        X = X[final_columns]

        # Predict
        pred_numeric = model.predict(X)[0]
        pred_label = class_mapping.get(pred_numeric, "Unknown")

        return {"prediction": pred_label, "class_id": int(pred_numeric)}

    except Exception as e:
        return {"error": str(e)}
