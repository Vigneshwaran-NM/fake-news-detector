#  Fake News Detection System using Machine Learning

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
```
## 🧪 How It Works

###  Dataset
- Combines **real** and **fake** news articles from [Kaggle Datasets](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

###  Preprocessing
- Lowercasing
- Removing punctuation, URLs, and stopwords
- Token normalization using NLTK

###  Feature Engineering
- Uses **TF-IDF** (Term Frequency-Inverse Document Frequency)

###  Model
- Trained using **Logistic Regression**
- Achieves ~**99% accuracy**

###  Prediction Logic
1. Accepts user input (headline or article)
2. Cleans the text using the preprocessing pipeline
3. Vectorizes it using the trained TF-IDF vectorizer
4. Predicts label and displays confidence score

---

## 🛠️ Installation & Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Vigneshwaran-NM/fake-news-detector.git
cd fake-news-detector
```
### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
```
#### Windows
```venv\Scripts\activate```
#### Mac/Linux
```source venv/bin/activate```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Download NLTK Stopwords (Only Once)
Add the following at the top of app.py:
```python
import nltk
nltk.download('stopwords')
```
### 5. Run the App
```bash
streamlit run app.py
```
##  Sample Inputs

### ✅ Real News:
- "Indian government announces new policies to tackle inflation."
- "NASA successfully launches new Mars rover."
- "Finance Minister announces tax cut for startups."

### ❌ Fake News:
- "Aliens land in Delhi and sign peace treaty."
- "Bill Gates creates virus to control population."
- "Drinking bleach cures COVID-19 overnight."

---

## 🌍 Try It Online

🚀 **[Click here to use the Fake News Detection App](https://fake-news-detector-b8ckqw633dp2hy4ujlktgr.streamlit.app/)**  
_(Hosted via Streamlit Cloud)_

---

## 📊 Model Evaluation

| Metric    | Score |
|-----------|-------|
| Accuracy  | 99%   |
| Precision | 0.99  |
| Recall    | 0.99  |
| F1 Score  | 0.99  |

**Confusion Matrix:**
[[4315 40]
[ 62 4563]]


---

##  License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

##  Contributor

**Vigneshwaran N M**  
🔗 [GitHub](https://github.com/Vigneshwaran-NM)  
🔗 [LinkedIn](https://www.linkedin.com/in/vigneshwaran-nm)

---

##  Acknowledgements

- [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
- [Scikit-learn](https://scikit-learn.org/)  
- [NLTK](https://www.nltk.org/)  
- [Streamlit](https://streamlit.io/)
