"""
Script to deploy tourism prediction model to Hugging Face Spaces
"""
import os
from huggingface_hub import HfApi, create_repo
import shutil

def create_app_file():
    """Create Streamlit app file"""
    
    app_content = '''import streamlit as st
import pickle
import numpy as np
import os

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        with open("tourism_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        return model, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, label_encoders = load_model()

# App title
st.title("🌍 Tourism Prediction App")
st.markdown("Predict tourism demand level based on various factors")

# Input form
st.header("Enter Details")

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox("Month", list(range(1, 13)), 
                        format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
    visitors = st.number_input("Expected Visitors", min_value=0, max_value=50000, value=10000)
    avg_temp = st.number_input("Average Temperature (°C)", min_value=-30, max_value=40, value=20)

with col2:
    hotel_price = st.number_input("Average Hotel Price ($)", min_value=50, max_value=500, value=200)
    event_count = st.number_input("Number of Events", min_value=0, max_value=20, value=5)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    country = st.selectbox("Country", ["USA", "Canada", "UK", "Australia"])

# Prediction
if st.button("Predict Tourism Demand"):
    if model is not None:
        try:
            # Encode categorical variables
            season_encoded = label_encoders['season'].transform([season])[0]
            country_encoded = label_encoders['country'].transform([country])[0]
            
            # Prepare features
            features = np.array([[month, year, visitors, avg_temp, hotel_price, 
                                 event_count, season_encoded, country_encoded]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            
            # Get prediction label
            prediction_label = label_encoders['target'].inverse_transform([prediction])[0]
            
            # Display result
            st.success(f"Predicted Tourism Demand: **{prediction_label}**")
            
            # Display probabilities
            st.subheader("Confidence Levels")
            classes = label_encoders['target'].classes_
            for i, cls in enumerate(classes):
                st.progress(prediction_proba[i], text=f"{cls}: {prediction_proba[i]:.2%}")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model not loaded properly")

# Info section
st.markdown("---")
st.markdown("""
### About
This app predicts tourism demand levels (Low, Medium, High) based on:
- Temporal factors (month, year)
- Visitor statistics
- Weather conditions
- Hotel pricing
- Event counts
- Geographic location

The model uses a Random Forest Classifier trained on historical tourism data.
""")
'''
    
    return app_content

def create_requirements():
    """Create requirements file for the app"""
    requirements = '''streamlit>=1.28.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
'''
    return requirements

def push_to_hugging_face():
    """Push app files to Hugging Face Spaces"""
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found. Skipping upload.")
        return
    
    api = HfApi()
    space_id = "tourism-prediction-app"
    
    try:
        # Create space
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="streamlit",
            exist_ok=True,
            token=token
        )
        print(f"Space {space_id} created/verified successfully")
        
        # Create temporary directory for app files
        temp_dir = "/tmp/tourism_app"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create app.py
        with open(os.path.join(temp_dir, "app.py"), "w") as f:
            f.write(create_app_file())
        
        # Create requirements.txt
        with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
            f.write(create_requirements())
        
        # Copy model files
        model_dir = os.path.join(os.path.dirname(__file__), "..", "model_building", "models")
        preprocessing_dir = os.path.join(os.path.dirname(__file__), "..", "model_building", "preprocessing")
        
        if os.path.exists(os.path.join(model_dir, "tourism_model.pkl")):
            shutil.copy(os.path.join(model_dir, "tourism_model.pkl"), 
                       os.path.join(temp_dir, "tourism_model.pkl"))
        
        if os.path.exists(os.path.join(preprocessing_dir, "scaler.pkl")):
            shutil.copy(os.path.join(preprocessing_dir, "scaler.pkl"),
                       os.path.join(temp_dir, "scaler.pkl"))
        
        if os.path.exists(os.path.join(preprocessing_dir, "label_encoders.pkl")):
            shutil.copy(os.path.join(preprocessing_dir, "label_encoders.pkl"),
                       os.path.join(temp_dir, "label_encoders.pkl"))
        
        # Upload files
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=space_id,
                repo_type="space",
                token=token
            )
            print(f"Uploaded {filename}")
        
        print(f"\n✓ App deployed to Hugging Face Spaces: https://huggingface.co/spaces/{space_id}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"Error deploying to Hugging Face: {e}")

def main():
    """Main function"""
    push_to_hugging_face()

if __name__ == "__main__":
    main()
