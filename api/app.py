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

# loading the saved model and vectorizer
model_path = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models","tfidf_vectorizer.pkl")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class ReviewRequest(BaseModel):
    review: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(request: Request, review: str = Form(...)):
    cleaned = clean_text(review)
    vectorized = transform_tfidf(vectorizer, [cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0].max()

    sentiment = "positive" if prediction == 1 else "negative"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "review": review,
        "prediction": sentiment,
        "confidence": round(float(probability), 3)
    })