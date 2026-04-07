import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and scaler
# Using cache_resource ensures they are only loaded once, making the app faster
@st.cache_resource
def load_assets():
    model = joblib.load('RandomForest_model1 (1).pkl')
    scaler = joblib.load('RandomForest_scaler1 (1).pkl')
    return model, scaler

model, scaler = load_assets()

# 2. Define the Feature Layout exactly as expected by the scaler
features = [
    'volatile acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# 3. Build the User Interface
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("Wine Quality Predictor")
st.write("Adjust the chemical properties below to predict the quality score of the wine.")

st.divider()

# Organize inputs into three columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=150.0, value=15.0, step=1.0)
    pH = st.number_input("pH", min_value=2.0, max_value=4.5, value=3.2, step=0.01)

with col2:
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=400.0, value=46.0, step=1.0)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.5, value=0.6, step=0.01)

with col3:
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.08, step=0.001, format="%.3f")
    density = st.number_input("Density", min_value=0.980, max_value=1.050, value=0.996, step=0.001, format="%.4f")
    alcohol = st.number_input("Alcohol (%)", min_value=7.0, max_value=20.0, value=10.0, step=0.1)

st.divider()

# 4. Process Inputs & Predict
if st.button("Predict Wine Quality", type="primary", use_container_width=True):
    # Create a DataFrame with the exact column names and order
    input_data = pd.DataFrame([[
        volatile_acidity, 
        residual_sugar, 
        chlorides, 
        free_sulfur_dioxide,
        total_sulfur_dioxide, 
        density, 
        pH, 
        sulphates, 
        alcohol
    ]], columns=features)

    try:
        # Scale the inputs
        scaled_data = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(scaled_data)
        
        # Display the result
        # Wine quality is usually an integer score between 3 and 8 in this dataset
        predicted_score = round(prediction[0], 2)
        
        st.success(f"### Predicted Wine Quality Score: {predicted_score}/10")
        
        # Add a little visual flair based on the score
        if predicted_score >= 6.5:
            st.balloons()
            st.info("Excellent quality wine!")
        elif predicted_score < 5.0:
            st.warning("Lower quality wine.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")