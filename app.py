from flask import Flask, render_template, request, jsonify
import os
import sys

# Добавляем parser в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'parser'))

from werkzeug.utils import secure_filename

# Импортируем твои модули
from pdf_parser import extract_text_from_pdf, parse_user_info, parse_transactions
from categorizer import recategorize_transactions
from feature_engineering import extract_features
from ai_explanation import generate_explanation
from config import CATEGORIES_FILE

import json
import pandas as pd
import joblib
import shap

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загружаем категории
def load_categories():
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Загружаем модель при старте
try:
    model = joblib.load('model_xgboost.pkl')
    feature_names = joblib.load('feature_names.pkl')
    explainer = shap.TreeExplainer(model)
    print("✓ Модель загружена")
except Exception as e:
    print(f"✗ Ошибка загрузки модели: {e}")
    model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Модель не загружена'}), 500
    
    if 'pdf' not in request.files:
        return jsonify({'error': 'PDF файл не найден'}), 400
    
    file = request.files['pdf']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Выберите PDF файл'}), 400
    
    try:
        # Сохраняем файл
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # === ИСПОЛЬЗУЕМ ТВОИ ФУНКЦИИ ===
        
        # 1. Парсинг PDF (твой pdf_parser.py)
        text = extract_text_from_pdf(filepath)
        user_info = parse_user_info(text)
        transactions = parse_transactions(text)
        
        if len(transactions) == 0:
            os.remove(filepath)
            return jsonify({'error': 'Транзакции не найдены в PDF'}), 400
        
        # 2. Категоризация (твой categorizer.py)
        categories = load_categories()
        category_stats, uncategorized = recategorize_transactions(transactions, categories)
        
        # 3. Feature Engineering (твой feature_engineering.py)
        features, counts, sums = extract_features(transactions, categories)
        
        # Заполняем отсутствующие признаки
        for f in feature_names:
            if f not in features:
                features[f] = 0
        
        X = pd.DataFrame([features])[feature_names]
        
        # 4. Предсказание
        pred_proba = model.predict_proba(X)[0]
        pred_label = int(model.predict(X)[0])
        risk_probability = float(pred_proba[1] * 100)
        
        # 5. SHAP
        shap_values_raw = explainer.shap_values(X)
        
        if len(shap_values_raw.shape) == 3:
            shap_values = shap_values_raw[0, :, 1]
        elif isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1][0]
        else:
            shap_values = shap_values_raw[0]
        
        factors = list(zip(feature_names, shap_values.tolist(), X.iloc[0].values.tolist()))
        factors_sorted = sorted(factors, key=lambda x: abs(x[1]), reverse=True)
        
        # Для JSON ответа
        positive_factors_json = [
            {'feature': f, 'value': v, 'impact': round(s, 4)}
            for f, s, v in factors_sorted if s > 0.005
        ][:5]
        
        negative_factors_json = [
            {'feature': f, 'value': v, 'impact': round(s, 4)}
            for f, s, v in factors_sorted if s < -0.005
        ][:5]
        
        # Для AI объяснения (tuple формат как в scoring.py)
        positive_factors_tuple = [(f, s, v) for f, s, v in factors_sorted if s > 0.005][:5]
        negative_factors_tuple = [(f, s, v) for f, s, v in factors_sorted if s < -0.005][:5]
        
        # 6. AI объяснение (твой ai_explanation.py)
        print("\n[5] Генерация AI объяснения...")
        ai_text = generate_explanation(
            user_info=user_info,
            risk_probability=risk_probability,
            pred_label=pred_label,
            features=features,
            sums=sums,
            positive_factors=positive_factors_tuple,
            negative_factors=negative_factors_tuple,
            transaction_count=len(transactions)
        )
        
        # Удаляем временный файл
        os.remove(filepath)
        
        # Возвращаем JSON для веб-интерфейса
        return jsonify({
            'success': True,
            'user': {
                'name': user_info.get('full_name', 'Неизвестно'),
                'period_from': user_info.get('period_from', '—'),
                'period_to': user_info.get('period_to', '—')
            },
            'transactions_count': len(transactions),
            'risk': {
                'probability': round(risk_probability, 1),
                'label': 'HIGH_RISK' if pred_label == 1 else 'LOW_RISK',
                'decision': 'ВЫСОКИЙ РИСК' if pred_label == 1 else 'НИЗКИЙ РИСК'
            },
            'factors': {
                'positive': positive_factors_json,
                'negative': negative_factors_json
            },
            'stats': {
                'total_income': features['total_income'],
                'total_expense': features['total_expense'],
                'expense_to_income_ratio': features['expense_to_income_ratio'],
                'betting_sum': sums.get('betting', 0),
                'betting_expense_ratio': features['betting_expense_ratio'],
                'credit_sum': sums.get('credit', 0),
                'credit_expense_ratio': features['credit_expense_ratio'],
                'utility_sum': sums.get('utility', 0),
                'utility_count_ratio': features['utility_count_ratio'],
                'credit_taken_count': features['credit_taken_count'],
                'credit_payment_count': features['credit_payment_count']
            },
            'categories': {
                cat: {'count': stats['count'], 'sum': stats['sum']}
                for cat, stats in category_stats.items()
            },
            'ai_explanation': ai_text  # <-- AI текст для веб-интерфейса
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)