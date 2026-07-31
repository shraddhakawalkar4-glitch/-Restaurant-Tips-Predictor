import streamlit as st
import pandas as pd
import joblib

# Load trained model + features
model = joblib.load("tips_model.pkl")
all_features = joblib.load("tips_features.pkl")

st.title("💰 Restaurant Tips Prediction")
st.write("Enter details below to predict whether the tip will be High or Low:")

# User inputs
total_bill = st.number_input("Total Bill ($)", value=20.0)
size = st.number_input("Party Size", value=2, min_value=1)

sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])

# Build dataframe
input_dict = {
    "total_bill": total_bill,
    "size": size,
    "sex": sex,
    "smoker": smoker,
    "day": day,
    "time": time
}
sample = pd.DataFrame([input_dict])

# One-hot encode categorical features
sample_encoded = pd.get_dummies(sample)

# Align columns with training features
sample_encoded = sample_encoded.reindex(columns=all_features, fill_value=0)

if st.button("Predict"):
    prediction = model.predict(sample_encoded)[0]
    proba = model.predict_proba(sample_encoded)[0]

    if prediction == "Low":
        st.error(f"❌ Predicted Tip Class: Low (Confidence {proba[0]*100:.1f}%)")
    else:
        st.success(f"✅ Predicted Tip Class: High (Confidence {proba[1]*100:.1f}%)")
