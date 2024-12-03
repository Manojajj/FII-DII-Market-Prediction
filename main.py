import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import requests
from io import BytesIO

# Streamlit App Title
st.title("Market Sentiment Prediction")
st.write("Train and predict stock market sentiment based on FII and DII activity.")

# Download Dataset from GitHub
DATA_URL = "FIIDII_activity.xlsx"  # Replace with actual GitHub file URL
st.subheader("Loading Dataset")
st.write("The dataset is being fetched from GitHub...")

try:
    response = requests.get(DATA_URL)
    response.raise_for_status()  # Check for HTTP errors
    data = pd.read_excel(BytesIO(response.content))  # Load Excel data
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Display Dataset Preview
st.write("Dataset Preview:")
st.write(data.head())

# Validate Dataset Columns
required_columns = {'Previous Day FII Net', 'Previous Day DII Net', 'Market Sentiment'}
if not required_columns.issubset(data.columns):
    st.error(f"Dataset must contain the columns: {', '.join(required_columns)}")
    st.stop()

# Encode Target Column
data['Market Sentiment'] = data['Market Sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})

# Split Dataset
X = data[['Previous Day FII Net', 'Previous Day DII Net']]
y = data['Market Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
st.subheader("Model Performance")
st.text(report)

# Save Model and Scaler
joblib.dump(model, 'market_sentiment_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
st.success("Model trained and saved successfully!")

# Prediction Section
st.subheader("Predict Market Sentiment")
fii_input = st.number_input("Previous Day FII Net (in millions)", step=1.0)
dii_input = st.number_input("Previous Day DII Net (in millions)", step=1.0)

if st.button("Predict"):
    # Load Saved Model and Scaler
    loaded_model = joblib.load('market_sentiment_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    
    # Predict Sentiment
    input_data = np.array([[fii_input, dii_input]])
    input_data_scaled = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(input_data_scaled)
    sentiment = ['Negative', 'Neutral', 'Positive'][int(prediction[0])]
    
    st.subheader("Prediction Result")
    st.write(f"The market sentiment is likely to be **{sentiment}**.")
