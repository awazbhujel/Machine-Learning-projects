import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# Load StandardScaler and KMeans model
with open('scaled.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans_model = pickle.load(model_file)

# Streamlit app
st.title('KMeans Clustering App')

# Load DataFrame from pickle file
with open('df.pkl', 'rb') as df_file:
    df = pickle.load(df_file)

# Input form for new data point
st.header('Enter New Data Point')

income = st.slider('Income', min_value=0, max_value=200, step=1)
score = st.slider('Score', min_value=0, max_value=100, step=1)

# When the user submits the form
if st.button('Predict Cluster'):
    # Preprocess the input data
    new_data = np.array([[income, score]])
    new_data_scaled = scaler.transform(new_data)

    # Predict the cluster
    cluster = kmeans_model.predict(new_data_scaled)[0]

    # Display the predicted cluster
    st.write(f'The data point [{income}, {score}] belongs to cluster {cluster + 1}')

    # Plot existing clusters
    X = df[['Income', 'score']].values
    y_kmeans = kmeans_model.predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', label='Existing Clusters')

    # Plot the new data point with a different marker and color
    plt.scatter(income, score, c='red', marker='x', label='New Data Point')

    plt.title('Clustering Results')
    plt.xlabel('Income')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
