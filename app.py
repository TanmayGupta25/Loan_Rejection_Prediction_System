
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Loan Rejection Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown( + css_content + , unsafe_allow_html=True)

# Resource Loading
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('models/best_model.pkl')
        features = joblib.load('models/feature_columns.pkl')
        with open('deployment/metadata.json', 'r') as f:
            meta = json.load(f)
        with open('deployment/deployment_config.json', 'r') as f:
            config = json.load(f)
        return model, features, meta, config
    except Exception as e:
        st.error(f"Error loading application resources: {e}")
        return None, None, None, None

model, features, meta, config = load_resources()

# Sidebar Navigation
st.sidebar.title("🏦 Loan System")
page = st.sidebar.radio("Navigation", ["🏠 Home", "📝 Loan Prediction", "📊 Dataset Information", "🦾 Model Information", "🖨 Developer"])

if page == "🏠 Home":
    st.title("Loan Rejection Prediction System 📈")
    st.write("Welcome to the production-ready loan rejection analysis platform.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Project Overview")
        st.info("This system uses advanced Machine Learning to evaluate loan applications based on financial and demographic indicators.")
    with col2:
        st.subheader("Quick Metrics")
        if meta:
            st.metric("Model Accuracy", f"{meta.get('accuracy', 0)*100:.2f}%")

elif page == "📝 Loan Prediction":
    st.title("Inference Engine 🦾")
    if features:
        input_data = {}
        cols = st.columns(3)
        for i, col_name in enumerate(features['original_features']):
            with cols[i % 3]:
                if col_name in features.get('numeric_features', []):
                    input_data[col_name] = st.number_input(f"{col_name}", value=0.0)
                else:
                    input_data[col_name] = st.selectbox(f"{col_name}", ["Yes", "No"])
        
        if st.button("Run Prediction Analysis"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            if prediction == 1:
                st.success(f"### APPROVED (Probability: {proba:.2%})")
            else:
                st.error(f"### REJECTED (Probability: {1-proba:.2%})")

elif page == "📊 Dataset Information":
    st.title("Dataset Insights")
    st.json(meta)

elif page == "🦾 Model Information":
    st.title("Model Details")
    st.write(f"**Algorithm:** {meta.get('best_model', 'N/A')}")

elif page == "🖨 Developer":
    st.title("Developer")
    st.write("Created for Loan Rejection Analysis.")

st.markdown("--- ")
st.caption("© 2024 Loan Rejection System")
