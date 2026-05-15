import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_data
def load_and_train_model():
    # Loading dataset - Ensure heart.csv is in the same folder
    df = pd.read_csv("heart.csv")
    
    # Preprocessing: IQR Clipping as per your notebook
    cols_to_clip = ["trestbps", "chol", "fbs", "thalach", "oldpeak", "slope", "ca", "thal"]
    for col in cols_to_clip:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=low, upper=up)
    
    # Training
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy, X.columns

# Load model
model, score, feature_names = load_and_train_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Patient Data Input")

def get_user_inputs():
    age = st.sidebar.slider("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex (1=M, 0=F)", [1, 0])
    cp = st.sidebar.slider("Chest Pain Type (cp)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting BP", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholestoral", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])
    restecg = st.sidebar.slider("Resting ECG results", 0, 2, 1)
    thalach = st.sidebar.slider("Max Heart Rate", 70, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.slider("Slope of ST segment", 0, 2, 1)
    ca = st.sidebar.slider("Major vessels (0-4)", 0, 4, 0)
    thal = st.sidebar.slider("Thal (0-3)", 0, 3, 1)
    
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_inputs()

# --- MAIN PAGE ---
st.title("❤️ Heart Disease Classification")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Parameters")
    st.write(input_df.T)

with col2:
    st.subheader("Prediction Result")
    if st.button("Analyze Health Data"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        if prediction[0] == 1:
            st.error("🚨 High Risk: Heart Disease Detected")
        else:
            st.success("✅ Low Risk: No Heart Disease Detected")
            
        st.write(f"Confidence Level: **{np.max(probability)*100:.2f}%**")
        
        # Displaying probabilities
        prob_df = pd.DataFrame(probability, columns=["Healthy", "Heart Disease"])
        st.bar_chart(prob_df)
