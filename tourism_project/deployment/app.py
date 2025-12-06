import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
    page_title="Tourism Package Predictor",
    layout="wide"
)

# Title and description
st.title("Tourism Package Purchase Predictor")
st.markdown("""
This application predicts whether a customer will purchase the **Wellness Tourism Package**
based on their profile and interaction data.
""")

# Load model and preprocessing objects from Hugging Face
@st.cache_resource
def load_model_artifacts():
    try:
        # Download from Hugging Face
        model_path = hf_hub_download(
            repo_id="RahulSingh211/TourismPackagePrediction",
            filename="best_model.pkl",
            repo_type="model"
        )
        scaler_path = hf_hub_download(
            repo_id="RahulSingh211/TourismPackagePrediction",
            filename="scaler.pkl",
            repo_type="model"
        )
        encoders_path = hf_hub_download(
            repo_id="RahulSingh211/TourismPackagePrediction",
            filename="label_encoders.pkl",
            repo_type="model"
        )

        # Use joblib to load the models
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)

        return model, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, label_encoders = load_model_artifacts()

if model is not None:
    st.success("Model loaded successfully!")

    # Create input form
    st.header("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)

    with col2:
        st.subheader("Contact & Location")
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

        st.subheader("Travel Details")
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        number_of_children_visiting = st.number_input("Number of Children (below 5)", min_value=0, max_value=5, value=0)
        preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

    with col3:
        st.subheader("Travel Profile")
        number_of_trips = st.number_input("Average Number of Trips per Year", min_value=0, max_value=20, value=2)
        passport = st.selectbox("Has Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns Car?", ["Yes", "No"])

        st.subheader("Sales Interaction")
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)

    # Prediction button
    if st.button("Predict Purchase Probability", type="primary", use_container_width=True):
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'TypeofContact': [type_of_contact],
                'CityTier': [city_tier],
                'DurationOfPitch': [duration_of_pitch],
                'Occupation': [occupation],
                'Gender': [gender],
                'NumberOfPersonVisiting': [number_of_person_visiting],
                'NumberOfFollowups': [number_of_followups],
                'ProductPitched': [product_pitched],
                'PreferredPropertyStar': [preferred_property_star],
                'MaritalStatus': [marital_status],
                'NumberOfTrips': [number_of_trips],
                'Passport': [1 if passport == "Yes" else 0],
                'PitchSatisfactionScore': [pitch_satisfaction_score],
                'OwnCar': [1 if own_car == "Yes" else 0],
                'NumberOfChildrenVisiting': [number_of_children_visiting],
                'Designation': [designation],
                'MonthlyIncome': [monthly_income]
            })

            # Encode categorical variables
            for col in label_encoders.keys():
                if col in input_data.columns:
                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

            # Scale features
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # Display results
            st.markdown("---")
            st.header("Prediction Results")

            col_res1, col_res2 = st.columns(2)

            with col_res1:
                if prediction == 1:
                    st.success("###High Likelihood of Purchase!")
                    st.markdown("This customer is **likely to purchase** the Wellness Tourism Package.")
                else:
                    st.warning("###Low Likelihood of Purchase")
                    st.markdown("This customer is **unlikely to purchase** the Wellness Tourism Package.")

            with col_res2:
                st.metric("Purchase Probability", f"{prediction_proba[1]:.2%}")
                st.metric("Non-Purchase Probability", f"{prediction_proba[0]:.2%}")

            # Visualization
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['Will Not Purchase', 'Will Purchase'],
                'Probability': prediction_proba
            })
            st.bar_chart(prob_df.set_index('Outcome'))

            # Recommendation
            st.markdown("---")
            st.subheader("💡 Recommendation")
            if prediction == 1 and prediction_proba[1] > 0.7:
                st.info("**High Priority**: Contact this customer immediately with a personalized offer!")
            elif prediction == 1:
                st.info("**Medium Priority**: Schedule a follow-up call within the next few days.")
            else:
                st.info("**Low Priority**: Consider nurturing this lead with email campaigns before direct contact.")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please ensure all fields are filled correctly.")

    # Additional information
    st.markdown("---")
    st.markdown("""
    ### About This Model
    This prediction model uses machine learning to analyze customer profiles and predict their likelihood
    of purchasing tourism packages. The model was trained on historical customer data and considers multiple
    factors including demographics, travel history, and sales interaction metrics.

    **Disclaimer**: This is a predictive model and should be used as a decision support tool alongside
    human judgment and business expertise.
    """)
else:
    st.error("Failed to load the prediction model. Please check the configuration.")
