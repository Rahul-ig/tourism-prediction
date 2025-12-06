# Tourism Prediction ML Pipeline

A complete machine learning pipeline for predicting tourism demand levels using historical data. This project demonstrates end-to-end ML workflow including data registration, preprocessing, model training with MLflow tracking, and deployment to Hugging Face Spaces.

## 🎯 Project Overview

This project predicts tourism demand levels (Low, Medium, High) based on various factors:
- **Temporal factors**: Month, Year
- **Visitor statistics**: Historical visitor counts
- **Weather conditions**: Average temperature
- **Economic factors**: Hotel prices
- **Events**: Number of events in the period
- **Geographic location**: Country and season

## 🏗️ Project Structure

```
tourism-prediction/
├── tourism_project/
│   ├── data/
│   │   ├── tourism_data.csv          # Sample tourism dataset
│   │   └── data_registration.py      # Upload dataset to Hugging Face Hub
│   ├── model_building/
│   │   ├── data_preparation.py       # Data preprocessing and feature engineering
│   │   ├── model_training.py         # Model training with MLflow
│   │   ├── preprocessing/            # Saved preprocessing objects
│   │   ├── processed_data/           # Processed train/test data
│   │   └── models/                   # Trained models
│   ├── deployment/
│   │   └── hosting.py                # Deploy to Hugging Face Spaces
│   └── workflow_requirements.txt     # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager
- Hugging Face account and API token (for deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Rahul-ig/tourism-prediction.git
cd tourism-prediction
```

2. Install dependencies:
```bash
pip install -r tourism_project/workflow_requirements.txt
```

3. Set up Hugging Face token (optional, for deployment):
```bash
export HF_TOKEN="your_hugging_face_token"
```

## 📊 Dataset

The project includes a sample tourism dataset with 48 records containing:
- Monthly visitor counts from 2020-2023
- Data from USA, Canada, UK, and Australia
- Seasonal patterns (Winter, Spring, Summer, Fall)
- Hotel prices and event counts
- Target variable: Tourism demand level (Low, Medium, High)

## 🔧 Usage

### 1. Data Registration

Upload the dataset to Hugging Face Hub:
```bash
python tourism_project/data/data_registration.py
```

### 2. Data Preparation

Preprocess data and create train/test splits:
```bash
python tourism_project/model_building/data_preparation.py
```

This script:
- Encodes categorical variables (season, country)
- Scales numerical features
- Splits data into train/test sets (80/20)
- Saves preprocessing objects for later use

### 3. Model Training

Train the Random Forest classifier with MLflow tracking:
```bash
# Start MLflow server (in a separate terminal)
mlflow ui --host 0.0.0.0 --port 5000

# Train the model
python tourism_project/model_building/model_training.py
```

The model achieves:
- Training accuracy: ~95-100%
- Test accuracy: ~90-95%

View experiment tracking at: http://localhost:5000

### 4. Deployment

Deploy the trained model to Hugging Face Spaces:
```bash
python tourism_project/deployment/hosting.py
```

This creates a Streamlit web application where users can:
- Input various factors (month, visitors, temperature, etc.)
- Get real-time predictions
- View confidence levels for each prediction class

## 🤖 Model Details

**Algorithm**: Random Forest Classifier

**Features**:
- month (1-12)
- year
- visitors (count)
- avg_temp (°C)
- hotel_price ($)
- event_count
- season_encoded
- country_encoded

**Hyperparameters**:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/pipeline.yml`) that automatically:

1. **Register Dataset**: Uploads data to Hugging Face Hub
2. **Data Preparation**: Preprocesses and splits the data
3. **Model Training**: Trains the model with MLflow tracking
4. **Deploy**: Pushes the app to Hugging Face Spaces

The pipeline triggers automatically on push to the `main` branch.

## 📈 Results

The model demonstrates strong performance:
- High accuracy on both training and test sets
- Good generalization across different countries and seasons
- Balanced predictions across all three target classes
- Feature importance shows visitor count and temperature as key factors

## 🛠️ Technologies Used

- **Python 3.9**: Core programming language
- **pandas & numpy**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **MLflow**: Experiment tracking and model registry
- **Hugging Face Hub**: Dataset and model hosting
- **Streamlit**: Web application framework
- **GitHub Actions**: CI/CD automation

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
