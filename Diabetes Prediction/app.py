import streamlit as st
import pickle

# Load the trained classifier
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    sc = pickle.load(f)


def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    preg = int(Pregnancies)
    glucose = float(Glucose)
    bp = float(BloodPressure)
    skint = float(SkinThickness)
    insulin = float(Insulin)
    bmi = float(BMI)
    dpf = float(DPF)
    age = int(Age)

    x = [[preg, glucose, bp, skint, insulin, bmi, dpf, age]]
    x = sc.transform(x)

#use the loaded classifier to make predictions
    prediction = classifier.predict(x)[0]
    return prediction

# Streamlit app
st.title("Diabetes Prediction App")

# Input fields for user to provide information
st.header("enter patient details:")

pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.slider("Glucose", min_value=0, max_value=300, value=0)
blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=200, value=0)
skin_thickness = st.slider("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.slider("Insulin", min_value=0, max_value=1000, value=0)
bmi = st.slider("BMI", min_value=0.0, max_value=60.0, value=0.0, step=0.1)
dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
age = st.slider("Age", min_value=0, max_value=120, value=0)

#predict button

if st.button("predict"):
    prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    if prediction:
        st.error("Oops! you have diabetes")
    else:
        st.success("Great! you dont have diabetes.")
