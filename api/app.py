import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Response, Request, Form

from src.data_preprocessing import clean_text
from src.feature_engineering import transform_tfidf

# loading the saved models and vectorizer
logistic_model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.pkl"))
naive_bayes_model = joblib.load(os.path.join(BASE_DIR,"models", "naive_bayes.pkl"))
svm_model = joblib.load(os.path.join(BASE_DIR, "models", "svm.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models","tfidf_vectorizer.pkl"))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# creating model selection function
def get_model(model_name: str):
    if model_name == "logistic":
        return logistic_model
    elif model_name == "naive_bayes":
        return naive_bayes_model
    elif model_name == "svm":
        return svm_model
    else:
        return logistic_model   #default model

class ReviewRequest(BaseModel):
    review: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    review: str = Form(...),
    model_choice: str = Form(...)
):

    model = get_model(model_choice)

    cleaned = clean_text(review)
    vectorized = transform_tfidf(vectorizer, [cleaned])
    prediction = model.predict(vectorized)[0]

    # Handle probability safely
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(vectorized)[0].max()
        probability = round(float(probability), 3)
    else:
        probability = "Not Available"

    sentiment = "positive" if prediction == 1 else "negative"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "review": review,
        "prediction": sentiment,
        "confidence": probability,
        "selected_model": model_choice
    })