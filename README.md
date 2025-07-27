# 📰 Fake News Detection System using Machine Learning

![GitHub last commit](https://img.shields.io/github/last-commit/Vigneshwaran-NM/fake-news-detector)
![GitHub repo size](https://img.shields.io/github/repo-size/Vigneshwaran-NM/fake-news-detector)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Overview

This project is a **Fake News Detection System** built using **Machine Learning** and **Natural Language Processing (NLP)**. It can classify whether a given news headline or article is **Real** or **Fake** using a trained Logistic Regression model and TF-IDF vectorization.

## 🧠 Features

- ✅ Predict whether news is fake or real using text input
- ✅ Text preprocessing and cleaning pipeline
- ✅ TF-IDF feature extraction
- ✅ Logistic Regression model with high accuracy (~99%)
- ✅ Confidence score for each prediction
- ✅ Streamlit UI support (for web app)
- ✅ Historical prediction logging
- ✅ Clean and modular code

---

## 📂 Project Structure

```bash
fake-news-detector/
│
├── data/
│   ├── fake.csv                # Original fake news data
│   ├── true.csv                # Original real news data
│   ├── processed_news.csv      # Cleaned & preprocessed dataset
│   ├── prediction_history.csv  # User prediction history
│
├── model/
│   ├── fake_news_model.joblib  # Trained ML model
│   └── vectorizer.joblib       # TF-IDF vectorizer
│
├── pages/
│   └── 1_📊_Analytics_Dashboard.py # Streamlit dashboard
│
├── notebooks/
│   └── EDA, training notebooks
│
├── app.py                      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md                   # This file

## 🧪 How It Works

Dataset
Combines real and fake news data from Kaggle datasets (fake.csv and true.csv).

Preprocessing

Lowercasing

Removing punctuation, URLs, and stopwords

Token normalization

Feature Engineering

Text vectorized using TF-IDF (Term Frequency-Inverse Document Frequency)

Model

Uses Logistic Regression for binary classification

Achieves ~99% accuracy on test data

Prediction Logic

Takes in a news headline/article

Cleans it using the same preprocessing logic

Transforms using the trained vectorizer

Predicts and shows a confidence score

## 🛠️ Installation & Run Locally

1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/Vigneshwaran-NM/fake-news-detector.git
cd fake-news-detector
2. Create Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Mac/Linux
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Download NLTK Stopwords (Only once)
python
Copy
Edit
# Run this once in Python shell or at top of app.py
import nltk
nltk.download('stopwords')
5. Run the App (Streamlit)
bash
Copy
Edit
streamlit run app.py

## 🌍 Try it Online
🚀 [Click here to use the Fake News Detection App](https://fake-news-detector-b8ckqw633dp2hy4ujlktgr.streamlit.app/)

## 📊 Model Evaluation
text
Copy
Edit
Accuracy: 99%
Precision: 0.99
Recall: 0.99
F1 Score: 0.99
Confusion Matrix:

lua
Copy
Edit
[[4315   40]
 [  62 4563]]
## 📌 Sample Inputs
Try with:

✅ "Government announces new healthcare policies for 2025"

❌ "NASA confirms alien spaceship landed in Delhi"

✅ "Finance Minister announces tax cut for startups"

## 🔑 Keywords (SEO Optimized)
Fake News Detection, Machine Learning, NLP, Python, Streamlit, TF-IDF, Logistic Regression, Text Classification, News Classifier, Real or Fake News, GitHub Fake News Detector, Data Science Projects, Capstone Project, AI in Journalism, ML Project for Resume

## 📃 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributors
Vigneshwaran N M
GitHub
LinkedIn

## 🙌 Acknowledgements
Kaggle Fake News Dataset

Scikit-learn

NLTK

Streamlit
