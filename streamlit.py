import streamlit as st
import pandas as pd
import psycopg2
import numpy as np

# Function to connect to the PostgreSQL database
def get_db_connection():
    conn = psycopg2.connect(
        host="ec2-13-201-77-225.ap-south-1.compute.amazonaws.com",
        port=5432,
        database="retail_order",
        user="postgres",
        password="apple123"
    )
    return conn

# Function to execute a query and return the result as a pandas DataFrame
def run_query(query):
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None
    finally:
        conn.close()

import pickle
import os

# Function to load a model from a specified file path
def load_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Define paths for the .pkl files
#parkinson_model_path = r"G:/Data Science/project/models/XGBparkinsons.pkl"  
#kidney_model_path = r"G:/Data Science/project/models/XGBkidney.pkl"         
#liver_model_path = r"G:/Data Science/project/models/XGBliver.pkl"  
# Define paths for the .pkl files
parkinson_model_path = r"G:\Data Science\project\Streamlit\env\Scripts\XGBparkinson.pkl"
kidney_model_path = r"G:\Data Science\project\Streamlit\env\Scripts\XGBkidney.pkl"         
liver_model_path = r"G:\Data Science\project\Streamlit\env\Scripts\XGBliver.pkl"           

# Load the models
try:
    model_parkinson = load_model(parkinson_model_path)
    model_kidney = load_model(kidney_model_path)
    model_liver = load_model(liver_model_path)

    print("Models loaded successfully!")
except FileNotFoundError as e:
    print(e)

# Streamlit UI
st.title("Disease Prediction")

# Input fields for user
nav = st.sidebar.radio("Select Disease Prediction", ["Parkinson's Disease", "Kidney Disease", "Liver Disease"])

if nav == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")

    # Define input fields for Parkinson's disease prediction
    inputs = {
        "MDVP_Fo_Hz": st.number_input("Fundamental Frequency (MDVP:Fo(Hz))", min_value=0.0, value=0.0),
        "MDVP_Fhi_Hz": st.number_input("Maximum Frequency (MDVP:Fhi(Hz))", min_value=0.0, value=0.0),
        "MDVP_Flo_Hz": st.number_input("Minimum Frequency (MDVP:Flo(Hz))", min_value=0.0, value=0.0),
        "MDVP_Jitter_percent": st.number_input("Jitter (MDVP:Jitter(%))", min_value=0.0, value=0.0),
        "MDVP_Jitter_Abs": st.number_input("Absolute Jitter (MDVP:Jitter(Abs))", min_value=0.0, value=0.0),
        "MDVP_RAP": st.number_input("Relative Average Perturbation (MDVP:RAP)", min_value=0.0, value=0.0),
        "MDVP_PPQ": st.number_input("Pitch Period Perturbation Quotient (MDVP:PPQ)", min_value=0.0, value=0.0),
        "Jitter_DDP": st.number_input("Degree of Derivative Perturbation (Jitter:DDP)", min_value=0.0, value=0.0),
        "MDVP_Shimmer": st.number_input("Shimmer (MDVP:Shimmer)", min_value=0.0, value=0.0),
        "MDVP_Shimmer_dB": st.number_input("Shimmer in dB (MDVP:Shimmer(dB))", min_value=0.0, value=0.0),
        "Shimmer_APQ3": st.number_input("Amplitude Perturbation Quotient (Shimmer:APQ3)", min_value=0.0, value=0.0),
        "Shimmer_APQ5": st.number_input("Amplitude Perturbation Quotient (Shimmer:APQ5)", min_value=0.0, value=0.0),
        "MDVP_APQ": st.number_input("Amplitude Perturbation Quotient (MDVP:APQ)", min_value=0.0, value=0.0),
        "Shimmer_DDA": st.number_input("Difference of Average Amplitude (Shimmer:DDA)", min_value=0.0, value=0.0),
        "NHR": st.number_input("Noise-to-Harmonics Ratio (NHR)", min_value=0.0, value=0.0),
        "HNR": st.number_input("Harmonics-to-Noise Ratio (HNR)", min_value=0.0, value=0.0),
        "RPDE": st.number_input("Recurrence Period Density Entropy (RPDE)", min_value=0.0, value=0.0),
        "DFA": st.number_input("Detrended Fluctuation Analysis (DFA)", min_value=0.0, value=0.0),
        "spread1": st.number_input("Signal Spread 1 (spread1)", value=0.0),
        "spread2": st.number_input("Signal Spread 2 (spread2)", value=0.0),
        "D2": st.number_input("Correlation Dimension (D2)", min_value=0.0, value=0.0),
        "PPE": st.number_input("Pitch Period Entropy (PPE)", min_value=0.0, value=0.0),
    }

    # Button for prediction
    if st.button("Predict"):
        try:
            input_features = np.array([list(inputs.values())], dtype=float)
            prediction = model_parkinson.predict(input_features)
            if prediction[0] == 1:
                st.success("The model predicts that the individual has Parkinson's disease.")
            else:
                st.success("The model predicts that the individual does not have Parkinson's disease.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif nav == "Kidney Disease":
    st.header("Kidney Disease Prediction")

    # Define input fields for Kidney disease prediction
    kidney_inputs = {
        "Age": st.number_input("Age", min_value=1, max_value=120, value=30),
        "Blood Pressure": st.number_input("Blood Pressure", min_value=1, max_value=200, value=80),
        "Specific Gravity": st.selectbox("Specific Gravity", [1.005, 1.01, 1.015]),
        "Albumin": st.selectbox("Albumin", [1, 2, 3]),
        # Add other necessary features based on your model requirements...
    }

    # Button for prediction
    if st.button("Predict Kidney Disease"):
        try:
            kidney_features = np.array([list(kidney_inputs.values())], dtype=float)
            kidney_prediction = model_kidney.predict(kidney_features)
            if kidney_prediction[0] == 1:
                st.success("The model predicts that the individual has Kidney disease.")
            else:
                st.success("The model predicts that the individual does not have Kidney disease.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif nav == "Liver Disease":
    st.header("Liver Disease Prediction")

    # Define input fields for Liver disease prediction
    liver_inputs = {
        "Age": st.number_input("Age", min_value=1, max_value=120, value=30),
        "Gender": st.selectbox("Gender", ("Male", "Female")),
        "Total_Bilirubin": st.number_input("Total Bilirubin", min_value=1e-5, value=1e-5), 
        "Direct_Bilirubin": st.number_input("Direct Bilirubin", min_value=1e-5, value=1e-5), 
        # Add other necessary features based on your model requirements...
    }

    # Button for prediction
    if st.button("Predict Liver Disease"):
        try:
            liver_features = np.array([list(liver_inputs.values())], dtype=float)
            liver_prediction = model_liver.predict(liver_features)
            if liver_prediction[0] == 1:
                st.success("The model predicts that the individual has Liver disease.")
            else:
                st.success("The model predicts that the individual does not have Liver disease.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Navigation options for other diseases can be added here

st.text("Thank you for using the dashboard!")
