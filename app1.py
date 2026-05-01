import streamlit as st
import pandas as pd
import pickle

lr_best = pickle.load(open("lr_model.pkl", "rb"))
rf_best = pickle.load(open("rf_model.pkl", "rb"))
voting = pickle.load(open("voting_model.pkl", "rb"))

st.title(" Heart Disease Prediction System")

# Input fields
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
bp = st.number_input("Resting BP", 70, 300)
chol = st.number_input("Cholesterol", 0, 600)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
hr = st.number_input("Max Heart Rate", 60, 250)
angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", -3.0, 4.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert inputs
user_data = {
    'Age': age,
    'Sex': 1 if sex == "M" else 0,
    'ChestPainType': cp,
    'RestingBP': bp,
    'Cholesterol': chol,
    'FastingBS': fbs,
    'RestingECG': ecg,
    'MaxHR': hr,
    'ExerciseAngina': 1 if angina == "Y" else 0,
    'Oldpeak': oldpeak,
    'ST_Slope': slope
}

input_df = pd.DataFrame([user_data])


if st.button("Predict"):
    lr_pred = lr_best.predict(input_df)[0]
    lr_prob = lr_best.predict_proba(input_df)[0][1]

    rf_pred = rf_best.predict(input_df)[0]
    rf_prob = rf_best.predict_proba(input_df)[0][1]

    voting_pred = voting.predict(input_df)[0]
    voting_prob = voting.predict_proba(input_df)[0][1]

    st.subheader(" Results")

    # Logistic Regression
    st.markdown(" Logistic Regression")
    if lr_pred == 1:
        st.error(f"Disease Detected ({lr_prob*100:.2f}%)")
    else:
        st.success(f"No Disease ({(1-lr_prob)*100:.2f}%)")

    # Random Forest
    st.markdown("### Random Forest")
    if rf_pred == 1:
        st.error(f"Disease Detected ({rf_prob*100:.2f}%)")
    else:
        st.success(f"No Disease ({(1-rf_prob)*100:.2f}%)")

    # Voting Classifier
    st.markdown("Voting Classifier")
    if voting_pred == 1:
        st.error(f"Disease Detected ({voting_prob*100:.2f}%)")
    else:
        st.success(f"No Disease ({(1-voting_prob)*100:.2f}%)")