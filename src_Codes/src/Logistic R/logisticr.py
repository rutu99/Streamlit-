import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Streamlit cache compatibility across versions
_cache = getattr(st, "cache_resource", None) or getattr(st, "cache_data", None) or st.cache

# 1. Load and train the model (Caching ensures it doesn't retrain on every click)
@_cache
def train_model():
    df = pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv")
    df.columns = df.columns.str.strip().str.lower()

    # Normalize common diabetes dataset schemas to expected names
    rename_map = {
        "bloodpressure": "diastolic",
        "skinthickness": "triceps",
        "diabetespedigreefunction": "dpf",
        "outcome": "diabetes",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    feature_cols = ["pregnancies", "glucose", "diastolic", "triceps", "insulin", "bmi", "dpf", "age"]
    missing = [c for c in feature_cols + ["diabetes"] if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset is missing expected columns: {missing}. Available: {list(df.columns)}")

    X = df[feature_cols]
    y = df["diabetes"]
    
    # Train test split as per your notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)
    
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)
    return model

model = train_model()

# 2. UI Header
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter the patient's clinical data below to check for diabetes risk.")

# 3. Create Inputs (using two columns for a better look)
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
    diastolic = st.number_input("Blood Pressure (Diastolic)", min_value=0, max_value=200, value=70)
    triceps = st.number_input("Triceps Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# 4. Prediction Logic
if st.button("Predict Results", use_container_width=True):
    # Prepare input data
    input_df = pd.DataFrame([[pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age]], 
                            columns=['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age'])
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1] # Probability of being 1 (Diabetes)

    st.divider()
    
    if prediction[0] == 1:
        st.error(f"### Result: High Risk (Diabetes Positive)")
        st.write(f"Confidence: {probability:.2%}")
    else:
        st.success(f"### Result: Low Risk (Diabetes Negative)")
        st.write(f"Confidence: {(1-probability):.2%}")