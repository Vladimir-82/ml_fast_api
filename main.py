""""""

import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from model import load_model


config_path = Path(__file__).parent / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
app = FastAPI()


class SentimentResponse(BaseModel):
    """Модель для вывода предсказания."""

    text: str
    sentiment_label: str
    sentiment_score: float


# create a route
@app.get("/")
def index():
    """Приветствие."""
    return {"text": "Sentiment Analysis"}


@app.get('/predict')
def predict_sentiment(text: str):
    """Предсказание модели."""
    model = load_model()
    sentiment = model(text)

    return SentimentResponse(
        text=text,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
    )
