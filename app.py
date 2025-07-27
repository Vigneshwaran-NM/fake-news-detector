import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import re
from nltk.corpus import stopwords
from datetime import datetime
import nltk
import os

# Download NLTK stopwords only if not already downloaded
try:
    from nltk.corpus import stopwords
    _ = stopwords.words('english')  # Try accessing to check if available
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

# ------------------------------------------
# Load model and vectorizer
# ------------------------------------------
model = joblib.load("model/fake_news_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

# ------------------------------------------
# Define stopwords (make sure to download in training)
# ------------------------------------------
stop_words = set(stopwords.words('english'))

# ------------------------------------------
# Text cleaning function (same as training)
# ------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ------------------------------------------
# Streamlit Page Settings
# ------------------------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection System")
st.subheader("Enter a news headline or article below:")

# ------------------------------------------
# Input Area
# ------------------------------------------
user_input = st.text_area("Paste news content here:", height=200)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Clean + Vectorize
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prob = model.predict_proba(vectorized_input)[0]

        # Prediction logic (you can tweak threshold)
        prediction = 1 if prob[1] >= 0.6 else 0
        label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        confidence = prob[1] if prediction == 1 else prob[0]

        # Show Result
        color_func = st.error if prediction == 1 else st.success
        color_func(f"🧠 Prediction: **{label}**")
        st.info(f"Confidence Score: **{confidence*100:.2f}%**")

        # Save History
        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": user_input,
            "prediction": label,
            "confidence": round(confidence, 4)
        }

        HISTORY_PATH = "data/prediction_history.csv"
        os.makedirs("data", exist_ok=True)

        if os.path.exists(HISTORY_PATH):
            df = pd.read_csv(HISTORY_PATH)
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_entry])

        df.to_csv(HISTORY_PATH, index=False)
