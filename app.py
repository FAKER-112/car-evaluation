import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Car Evaluation Classifier", page_icon="🚗")

st.title("🚗 Car Quality Evaluation")
st.write("Enter the car's features below to predict its evaluation class (Unacceptable, Acceptable, Good, Very Good).")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found. Please run 'train.py' first to generate the model.")
        return None

model = load_model()

# Define feature options (Mappings)
buying_opts = ['low', 'med', 'high', 'vhigh']
maint_opts = ['low', 'med', 'high', 'vhigh']
doors_opts = ['2', '3', '4', '5more']
persons_opts = ['2', '4', 'more']
lug_boot_opts = ['small', 'med', 'big']
safety_opts = ['low', 'med', 'high']

# Create Layout
col1, col2 = st.columns(2)

with col1:
    buying = st.selectbox("Buying Price", buying_opts)
    maint = st.selectbox("Maintenance Cost", maint_opts)
    doors = st.selectbox("Number of Doors", doors_opts)

with col2:
    persons = st.selectbox("Capacity (Persons)", persons_opts)
    lug_boot = st.selectbox("Luggage Boot Size", lug_boot_opts)
    safety = st.selectbox("Safety Rating", safety_opts)

# Prediction Logic
if st.button("Evaluate Car"):
    if model is not None:
        # 1. Prepare raw dataframe
        input_data = {
            'buying': buying,
            'maint': maint,
            'doors': doors,
            'persons': persons,
            'lug_boot': lug_boot,
            'safety': safety
        }
        
        X = pd.DataFrame([input_data])
        
        # 2. Mappings (Must match train.py exactly)
        ordinal_mappings = {
            'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
            'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
            'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
            'persons': {'2': 0, '4': 1, 'more': 2},
            'safety': {'low': 0, 'med': 1, 'high': 2}
        }

        # Apply Ordinal Encoding
        for col, mapping in ordinal_mappings.items():
            X[col] = X[col].map(mapping)

        # 3. One-Hot Encoding for lug_boot
        # We manually ensure columns exist because get_dummies on a single row 
        # might exclude categories not present in that specific row.
        lug_boot_mapping = {
            'small': [0, 0, 1], # big=0, med=0, small=1
            'med':   [0, 1, 0], # big=0, med=1, small=0
            'big':   [1, 0, 0]  # big=1, med=0, small=0
        }
        
        # The model expects: lug_boot_big, lug_boot_med, lug_boot_small
        lb_vals = lug_boot_mapping[lug_boot]
        X['lug_boot_big'] = lb_vals[0]
        X['lug_boot_med'] = lb_vals[1]
        X['lug_boot_small'] = lb_vals[2]
        
        # Drop original lug_boot string column
        X = X.drop('lug_boot', axis=1)

        # 4. Reorder columns strictly (Critical for XGBoost)
        expected_cols = ['buying', 'maint', 'doors', 'persons', 'safety', 
                         'lug_boot_big', 'lug_boot_med', 'lug_boot_small']
        X = X[expected_cols]

        # 5. Predict
        prediction = model.predict(X)[0]
        
        # Map result back to string
        class_mapping = {0: 'Unacceptable', 1: 'Acceptable', 2: 'Good', 3: 'Very Good'}
        result = class_mapping.get(prediction, "Unknown")

        # Display result
        st.success(f"Prediction: **{result}**")
        
        # Optional: Show probabilities if model supports it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            st.write("Confidence Scores:")
            probs_df = pd.DataFrame(proba, index=class_mapping.values(), columns=["Probability"])
            st.bar_chart(probs_df)