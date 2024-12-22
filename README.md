## Веб-сервер для определения настроения текста на основе NLP-модели.
### Описание
* Модель анализирует настроения текста на основе NLP-модели.
* Используется модель из библиотеки Hugging Face Hub
```
cointegrated/rubert-tiny-sentiment-balanced
```


### Запуск
uvicorn main:app --reload


### Использование
* Приветственная страница
```
{"text":"Sentiment Analysis"}
```
* Страница предсказания
```
<host>/predict?text=отлично
```
выдаст текст, оценку предсказания, вероятность предсказания:
```
{"text":"отлично","sentiment_label":"positive","sentiment_score":0.9955776929855347}
```

### Тестирование
```
poetry run pytest test.py
```
или
```
pytest test.py
```


### Совместимость
Python 3.8 +
