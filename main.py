import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.testclient import TestClient
from pathlib import Path

from model import load_model


config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)



# model = None
app = FastAPI()


class SentimentResponse(BaseModel):
    text: str
    sentiment_label: str
    sentiment_score: float


# create a route
@app.get("/")
def index():
    return {"text": "Sentiment Analysis"}


# Your FastAPI route handlers go here
@app.get("/predict")
def predict_sentiment(text: str):
    model = load_model()
    sentiment = model(text)

    response = SentimentResponse(
        text=text,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
    )

    return response

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host='localhost',
        port=8000,
        reload=True
    )
