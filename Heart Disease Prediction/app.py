import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('randomforest.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the StandardScaler
with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to predict heart disease
def predict_heart_disease(sample_values):
    # Use the loaded model to make predictions
    prediction = model.predict(sample_values)

    if prediction[0] == 1:
        return "The person is predicted to have heart disease."
    else:
        return "The person is predicted to not have heart disease."

# Main Streamlit application
st.title('Heart Disease Prediction')
st.write('Enter the following information to predict whether a person has heart disease or not.')

# Create input fields for user input
age = st.slider('Age', min_value=0, max_value=150, value=50)
trestbps = st.slider('Resting Blood Pressure', min_value=0, max_value=300, value=120)
chol = st.slider('Cholesterol', min_value=0, max_value=600, value=200)
thalch = st.slider('Max Heart Rate', min_value=0, max_value=300, value=150)
oldpeak = st.slider('ST Depression', min_value=0.0, max_value=10.0, value=0.0)
sex_1 = st.selectbox('Sex', [0, 1])
cp_1 = st.selectbox('Chest Pain Type - 1', [0, 1])
cp_2 = st.selectbox('Chest Pain Type - 2', [0, 1])
cp_3 = st.selectbox('Chest Pain Type - 3', [0, 1])
fbs_1 = st.selectbox('Fasting Blood Sugar', [0, 1])
restecg_1 = st.selectbox('Resting Electrocardiographic Results - 1', [0, 1])
restecg_2 = st.selectbox('Resting Electrocardiographic Results - 2', [0, 1])
exang_1 = st.selectbox('Exercise Induced Angina', [0, 1])
slope_1 = st.selectbox('Slope of Peak Exercise ST Segment - 1', [0, 1])
slope_2 = st.selectbox('Slope of Peak Exercise ST Segment - 2', [0, 1])
ca_1_0 = st.selectbox('Number of Major Vessels Colored by Flourosopy - 1', [0, 1])
ca_2_0 = st.selectbox('Number of Major Vessels Colored by Flourosopy - 2', [0, 1])
ca_3_0 = st.selectbox('Number of Major Vessels Colored by Flourosopy - 3', [0, 1])
thal_1 = st.selectbox('Thalassemia - 1', [0, 1])
thal_2 = st.selectbox('Thalassemia - 2', [0, 1])

# Standardize numerical input features
scaled_features = scaler.transform([[age, trestbps, chol, thalch, oldpeak]])

# Prepare sample values for prediction
sample_values = np.concatenate([scaled_features, [[sex_1, cp_1, cp_2, cp_3, fbs_1, restecg_1, restecg_2,
                                                  exang_1, slope_1, slope_2, ca_1_0, ca_2_0, ca_3_0, thal_1, thal_2]]], axis=1)

# Predict button
if st.button('Predict'):
    # Call the predict_heart_disease function
    prediction = predict_heart_disease(sample_values)
    st.write(prediction)
