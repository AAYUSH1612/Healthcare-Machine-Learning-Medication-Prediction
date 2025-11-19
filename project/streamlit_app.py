import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ---- Load Pickle Files ---- #
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))


# Manual target mapping
target_map = {0: 'Antibiotics', 1: 'Aspirin', 2: 'Chemotherapy', 3: 'Insulin', 4: 'Statins'}

# ---- Page Config ---- #
st.set_page_config(page_title="Medical Diagnosis Predictor", layout="wide")

# ---- Header ---- #
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🩺 Machine Learning-Based Medication Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter patient details and predict the most suitable medication</p>", unsafe_allow_html=True)
st.write("---")

# ---- Sidebar Inputs ---- #
st.sidebar.header("Patient Information")

def user_input():
    st.sidebar.subheader("Vital Signs")
    Age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)
    Blood_Pressure = st.sidebar.number_input("Blood Pressure", min_value=50, max_value=200)
    Heart_Rate = st.sidebar.number_input("Heart Rate", min_value=30, max_value=200)
    Temperature = st.sidebar.number_input("Temperature (°F)", min_value=90.0, max_value=110.0)

    st.sidebar.subheader("Medical History")
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Diagnosis = st.sidebar.selectbox("Past Diagnosis", ['Hypertension', 'Influenza', 'Heart Disease', 'Cancer', 'Diabetes'])
    Lab_Test_Results = st.sidebar.number_input("Lab Test Results")
    X_ray_Results = st.sidebar.selectbox("X-ray Result", ['Abnormal', 'Normal'])
    Surgery_Type = st.sidebar.selectbox("Surgery Type", ['Appendectomy', 'Knee Replacement', 'Gallbladder Removal',
                                                        'Cataract Surgery', 'Angioplasty'])
    Allergies = st.sidebar.selectbox("Allergies", ['Latex', 'Shellfish', 'No_Allergies', 'Penicillin', 'Peanuts'])
    Family_History = st.sidebar.selectbox("Family History", ['Heart Disease', 'Diabetes', 'Hypertension', 'Cancer', "Alzheimer's"])
    
    data = pd.DataFrame({
        "Age": [Age],
        "Gender": [Gender],
        "Blood_Pressure": [Blood_Pressure],
        "Heart_Rate": [Heart_Rate],
        "Temperature": [Temperature],
        "Diagnosis": [Diagnosis],
        "Lab_Test_Results": [Lab_Test_Results],
        "X_ray_Results": [X_ray_Results],
        "Surgery_Type": [Surgery_Type],
        "Allergies": [Allergies],
        "Family_History": [Family_History]
    })
    return data

user_data = user_input()
user_data_display = user_data.copy()

# ---- Encode categorical columns ---- #
categorical_cols = ["Gender", "Diagnosis", "X_ray_Results", "Surgery_Type", "Allergies", "Family_History"]
for col in categorical_cols:
    if col in user_data.columns:
        user_data[col] = encoders[col].transform(user_data[col])

# ---- Reorder columns to match training ---- #
user_data = user_data.reindex(columns=features)

# ---- Scale numerical columns ---- #
numerical_cols = scaler.feature_names_in_
user_data[numerical_cols] = scaler.transform(user_data[numerical_cols])

# ---- Prediction ---- #
if st.button("Predict"):
    # Predict
    pred_encoded = model.predict(user_data)[0]
    proba = model.predict_proba(user_data)[0]
    
    # Convert to names
    pred_label = target_map[pred_encoded]
    
    # ---- Highlight Top Prediction ---- #
    st.markdown(f"""
    <div style="
        background-color:#00CED1; 
        padding:20px; 
        border-radius:10px; 
        text-align:center;
        color:white;
        font-size:24px;
        font-weight:bold;">
        🏆 Top Predicted Medication: {pred_label} ({proba[pred_encoded]*100:.1f}% confidence)
    </div>
    """, unsafe_allow_html=True)
    st.write("---")
    
    # ---- Top Prediction Confidence Gauge ---- #
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba[pred_encoded]*100,
        title={'text': "Confidence in Top Prediction (%)", 'font': {'size': 18}},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#008080"},
               'steps': [
                   {'range': [0, 50], 'color': "#FFC0CB"},
                   {'range': [50, 75], 'color': "#FFD700"},
                   {'range': [75, 100], 'color': "#32CD32"}]}))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.write("---")
    
    # ---- Full Probability Bar Chart (All Medications) ---- #
    proba_df = pd.DataFrame({
        'Medication': [target_map[i] for i in range(len(proba))],
        'Probability': proba
    }).sort_values(by='Probability', ascending=False)
    
    proba_df_sorted = proba_df.sort_values(by='Probability', ascending=True)
    fig_full_bar = px.bar(proba_df_sorted, 
                          x='Probability', 
                          y='Medication', 
                          orientation='h',
                          text='Probability',
                          color='Probability',
                          color_continuous_scale=px.colors.sequential.Teal)
    
    # Highlight top 3 medications
    top3_meds = proba_df_sorted['Medication'].tail(3).tolist()
    fig_full_bar.update_traces(marker_color=[
        '#32CD32' if med in top3_meds else '#87CEFA' for med in proba_df_sorted['Medication']
    ])
    fig_full_bar.update_layout(yaxis={'categoryorder':'total ascending'},
                               title="Probability for All Medications")
    st.plotly_chart(fig_full_bar, use_container_width=True)
    st.write("---")
    
    # ---- Show Entered Data ---- #
    with st.expander("📝 Entered Patient Details"):
        st.table(user_data_display)
    
    # ---- Top 3 Probability Bar Chart & Pie Chart ---- #
    top3 = proba_df.head(3)
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.subheader("Prediction Confidence (Top 3)")
        fig_bar = px.bar(top3, x='Probability', y='Medication', orientation='h',
                         text='Probability', color='Probability', color_continuous_scale='Viridis')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Probability Distribution")
        fig_pie = px.pie(proba_df, names='Medication', values='Probability', color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("<p style='color: gray;'>The visualizations show the model's confidence for each medication. Top 3 are highlighted for clarity.</p>", unsafe_allow_html=True)
