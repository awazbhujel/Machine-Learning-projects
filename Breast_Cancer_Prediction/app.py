import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Load trained model and preprocessing objects
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# Define input interface
st.title('Breast Cancer Prediction')
st.write('Please enter the values for the following features:')

# Create input fields for each feature
input_features = []
for feature in ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst', 'concavity_worst', 'concave points_worst',
                'symmetry_worst', 'fractal_dimension_worst']:
    value = st.number_input(label=feature, format='%f')
    input_features.append(value)

# Make predictions
input_data = np.array(input_features).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)
input_data_pca = pca.transform(input_data_scaled)
prediction = model.predict(input_data_pca)

# Display results
st.write('Prediction:')
if prediction[0] == 1:
    st.write('The person is predicted to have breast cancer.')
else:
    st.write('The person is predicted not to have breast cancer.')
