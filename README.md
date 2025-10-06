# Сервис кредитного скоринга 

<img width="2559" height="1299" alt="image" src="https://github.com/user-attachments/assets/72269071-faa5-4566-88ea-4d04b42f1028" />

Задача кредитного скоринга дает ответ на вопрос "Стоит ли давать кредит клиенту?" 

## Описание проекта

### 1. Обучение модели

Самый первый этап - обучение модели для задачи бинарной классификации. 

Все необходимые библиотеки для обучения:
* pandas позволяет работать с табличными данными из .csv файла
* scikit-learn дает готовые алгоритмы для обучения модели
* joblib импортирует модель в .pkl формат для связи с сервисной частью

```shell
pip3 install pandas scikit-learn joblib
```

Датасет с тренировочной и тестовой выборкой лежит в <a href="https://github.com/neuezeldaa/credit_scoring_service/blob/main/data/scoring.csv">папке data</a>

Обучение моделей происходит в <a href="https://github.com/neuezeldaa/credit_scoring_service/blob/main/train.py">train.py</a>

В обучении применялись алгоритмы:

* Логистической регрессии (F1-score test = 0.2309)
* Случайного леса (F1-score test = 0.2303
* Градиентного бустинга (F1-score test = 0.2334)

F1-score метрики на всех трех обученых моделях оказались сильно низкими. Это можно списать на проблемный датасет: слишком большой дизбаланс классов.

---

### 2. Описание сервиса скоринга

Сервисная часть проекта описана в <a href="https://github.com/neuezeldaa/credit_scoring_service/blob/main/service.py">service.py</a>

Установление модулей: 
```shell
pip3 install fastapi uvicorn
```
Сервис включается с помощью команды после описания объекта app:
```shell
uvicorn service:app
```
Запросы к сервису скоринга можно выполнить со страницы [Swagger UI](http://localhost:8000/docs) или другого HTTP клиента.


### 3. Веб-страница с помощью StreamLit

Устанавливаем streamlit и requests для отправки http запросов:
```shell
pip3 install streamlit requests
```
Описание формы заявки описано в <a href="https://github.com/neuezeldaa/credit_scoring_service/blob/main/app.py">app.py</a>

Запуск веб-страницы:
```shell
streamlit run app.py
```
Веб-страница откроется автоматически в вашем браузере. Если не открылась, то вы можете воспользоваться URL в консоли:

<img width="966" height="209" alt="image" src="https://github.com/user-attachments/assets/717daa8f-b276-4555-9b0a-1fd5ae06d27e" />


