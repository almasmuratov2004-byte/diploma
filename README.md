# Credit Scoring System
Система кредитного скоринга на основе анализа банковских транзакций.

Загрузите PDF выписку — получите оценку кредитоспособности с объяснением решения на основе ML-модели и SHAP-анализа.

---

## Как это работает

1. Пользователь загружает PDF банковской выписки
2. Система парсит транзакции и категоризирует их
3. Извлекаются признаки (доходы, расходы, ставки, кредиты и др.)
4. XGBoost-модель предсказывает кредитный риск
5. SHAP объясняет, какие факторы повлияли на решение
6. AI генерирует текстовое объяснение на русском языке

---

## Стек технологий

- **Backend:** Python, Flask
- **ML:** XGBoost, scikit-learn, SHAP
- **PDF парсинг:** pdfplumber
- **Frontend:** HTML/CSS/JS

---

## Установка и запуск

### 1. Клонировать репозиторий
```bash
git clone https://github.com/almasmuratov2004-byte/diploma.git
cd diploma
```

### 2. Создать виртуальное окружение (рекомендуется)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Установить зависимости
```bash
pip install -r requirements.txt
```

### 4. Запустить приложение
```bash
python app.py
```

Открыть в браузере: [http://localhost:5000](http://localhost:5000)

---

## Зависимости

```
flask
werkzeug
pandas
numpy<2.0
joblib
xgboost
scikit-learn
shap
pdfplumber
requests
```

Или установить одной командой:
```bash
pip install flask werkzeug pandas "numpy<2.0" joblib xgboost scikit-learn shap pdfplumber requests
```

---

## Структура проекта

```
diploma/
├── app.py                        # Flask-приложение, основной сервер
├── parser/
│   ├── pdf_parser.py             # Парсинг PDF выписок
│   ├── categorizer.py            # Категоризация транзакций
│   ├── feature_engineering.py    # Извлечение признаков для модели
│   ├── ai_explanation.py         # Генерация AI-объяснения
│   └── config.py                 # Конфигурация (API ключи, пути)
├── templates/
│   └── index.html                # Веб-интерфейс
├── model_xgboost.pkl             # Обученная модель
├── feature_names.pkl             # Названия признаков
├── scaler.pkl                    # Нормализатор данных
├── ml_dataset.csv                # Датасет для обучения
├── tests/
│   └── test_project.py           # Unit-тесты
├── 1Выгрузка.py                  # Скрипт подготовки данных
├── 2Категоризация.py             # Скрипт категоризации
├── 3Feat.py                      # Скрипт feature engineering
├── 4Обучение и выбор модели.py   # Скрипт обучения моделей
├── 5shap_explanation.py          # Скрипт SHAP-анализа
└── requirements.txt
```

---

## Запуск тестов

```bash
pip install pytest
pytest tests/ -v
```

---

## Примечание по данным

Датасет (`ml_dataset.csv`) является **синтетическим** — сгенерирован для демонстрации работы системы. Реальные банковские данные не использовались.
