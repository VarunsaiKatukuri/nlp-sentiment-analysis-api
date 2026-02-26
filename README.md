# 🎬 NLP Sentiment Analysis API

An end-to-end movie review sentiment analysis system built using traditional Machine Learning models and deployed with FastAPI.

---

## Project Overview

This project classifies movie reviews as **Positive** or **Negative** using Natural Language Processing techniques.

The system includes:

- Text preprocessing pipeline
- TF-IDF feature engineering
- Multiple ML models
- FastAPI backend
- Interactive UI
- Git version control

---

## Architecture

### 🔹 Traditional ML Pipeline

Text → Cleaning → TF-IDF → Model → Prediction

Models implemented:

- Logistic Regression (Baseline & Selected Model)
- Naive Bayes
- Support Vector Machine (SVM)

---

## Model Performance

| Model                 | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 0.8908 |
| SVM                   | 0.8868 |
| Naive Bayes           | 0.8650 |

**Chosen Model:** Logistic Regression  
Reason: Best balance of accuracy, speed, and probability calibration.

---

## Technologies Used

- Python
- Scikit-learn
- TF-IDF Vectorization
- FastAPI
- Jinja2 Templates
- Joblib
- Git & GitHub

---

## 📂 Project Structure

## 📁 Project Structure

```
nlp-sentiment-analysis-api/
│
├── api/
│   └── app.py
│
├── src/
│   ├── data_preprocessing.py
│   └── feature_engineering.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── requirements.txt
└── README.md
```

---

##  How to Run Locally

### 1️⃣ Clone Repository

git clone https://github.com/VarunsaiKatukuri/nlp-sentiment-analysis-api.git
cd nlp-sentiment-analysis-api

### 2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate   # Windows

### 3️⃣ Install Dependencies

pip install -r requirements.txt

## Open the browser now
http://127.0.0.1:8000/


## 🎯 Features

Multi-model selection via UI

Confidence score display

Clean modular architecture

Production-style folder structure

Version controlled project

## 📌 Future Improvements

Hyperparameter tuning

LSTM / Deep Learning model

Docker containerization

Cloud deployment

Performance monitoring dashboard


👨‍💻 Author

Varun Sai Katukuri
