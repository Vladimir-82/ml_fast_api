"""Модель прогнозирования настроений."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import pipeline


config_path = Path(__file__).parent / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


@dataclass
class SentimentPrediction:
    """Класс, представляющий результат прогнозирования настроений."""

    label: str
    score: float


def load_model():
    """Загрузка предварительно обученной модели анализа настроений.

    Returns:
        модель с результатом прогнозирования настроений.
    """
    model_hf = pipeline(config["task"], model=config["model"], device=-1)

    def model(text: str) -> SentimentPrediction:
        pred = model_hf(text)
        pred_best_class = pred[0]
        return SentimentPrediction(
            label=pred_best_class["label"],
            score=pred_best_class["score"],
        )
    return model
