import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("Loan Eligibility Prediction")

# Input features
st.header("Enter Applicant Details:")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self-Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Preprocess input
property_area_mapping = {"Urban": 0, "Semiurban": 1, "Rural": 2}
gender_mapping = {"Male": 0, "Female": 1}
married_mapping = {"Yes": 1, "No": 0}
education_mapping = {"Graduate": 1, "Not Graduate": 0}
self_employed_mapping = {"Yes": 1, "No": 0}

input_data = np.array([
    gender_mapping[gender],
    married_mapping[married],
    education_mapping[education],
    self_employed_mapping[self_employed],
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    credit_history,
    property_area_mapping[property_area],
]).reshape(1, -1)

# Predict and display result
if st.button("Predict Loan Eligibility"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected.")
