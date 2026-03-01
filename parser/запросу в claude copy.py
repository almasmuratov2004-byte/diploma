import os
import re
import json
import pdfplumber
import numpy as np
import pandas as pd
import joblib
import shap
from collections import Counter
import requests

# ===================== ПАРАМЕТРЫ =====================
PDF_PATH = "files/Алмас.pdf"
CATEGORIES_FILE = "parser/categories.json"
TOP_N_UNCATEGORIZED = 10
OTHER_THRESHOLD_PERCENT = 10

AI_API_KEY = "api key"
AI_MODEL = "claude-haiku-4-5-20251001"

# ===================== ЗАГРУЗКА КАТЕГОРИЙ =====================
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
        CATEGORIES = json.load(f)
else:
    print(f"Файл категорий не найден: {CATEGORIES_FILE}")
    CATEGORIES = {}

# ===================== ФУНКЦИИ =====================
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def parse_user_info(text):
    """Извлекает информацию о пользователе"""
    user = {}
    
    # ФИО с ИИН
    iin_pattern = re.compile(r"что\s+(.+?),\s*ИИН\s*(\d{12})")
    iin_match = iin_pattern.search(text)
    if iin_match:
        user["full_name"] = iin_match.group(1).strip()
        user["iin"] = iin_match.group(2)
    
    # ФИО из ВЫПИСКИ
    if "full_name" not in user:
        vypiska_pattern = re.compile(
            r"ВЫПИСКА\s+по Kaspi Gold за период.*?\n(\S+)\s*\n",
            re.DOTALL
        )
        vypiska_match = vypiska_pattern.search(text)
        if vypiska_match:
            user["full_name"] = vypiska_match.group(1).strip()
    
    # Период
    period_pattern = re.compile(r"за период с (\d{2}\.\d{2}\.\d{2}) по (\d{2}\.\d{2}\.\d{2})")
    period_match = period_pattern.search(text)
    if period_match:
        user["period_from"] = period_match.group(1)
        user["period_to"] = period_match.group(2)
    
    return user

def parse_transactions(text):
    pattern = re.compile(r"(\d{2}\.\d{2}\.\d{2})\s*([+-])\s*([\d\s,]+)\s*₸\s*(\S+)\s*(.+)")
    records = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            date = match.group(1)
            sign = match.group(2)
            amount_str = match.group(3).replace(" ", "").replace(",", ".")
            amount = float(amount_str)
            if sign == "-":
                amount = -amount
            operation = match.group(4)
            details = match.group(5).strip()
            records.append({
                "date": date,
                "amount": amount,
                "operation": operation,
                "details": details
            })
    return records

def categorize(details, categories_dict):
    d = details.lower()
    for cat, keywords in categories_dict.items():
        for kw in keywords:
            if kw.lower() in d:
                return cat
    if d.startswith('ип ') or d.startswith('ip '):
        return 'small_business'
    if re.search(r' .\.$', details):
        return 'person_transfer'
    return "other"

def ai_categorize(top_words, categories_keys):
    prompt = (
        f"У меня есть следующие категории:\n{categories_keys}\n\n"
        f"Попробуй распределить эти слова по категориям:\n{top_words}\n"
        f"Если слово не подходит ни под одну категорию, создай новую категорию.\n"
        f"Выведи результат в формате JSON, где ключ — категория, "
        f"значение — массив слов."
    )
    headers = {
        "x-api-key": AI_API_KEY,
        "Content-Type": "application/json",
        "Anthropic-Version": "2023-06-01"
    }
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
    if response.status_code != 200:
        print("Ошибка AI:", response.text)
        return {}

    content = response.json().get("content", [])
    text = ""
    for c in content:
        if c.get("type") == "text":
            text += c.get("text", "")

    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        text = code_block.group(1)

    try:
        result_json = json.loads(text)
    except json.JSONDecodeError:
        print("Не удалось распарсить JSON от AI, вот что вернулось:")
        print(text)
        result_json = {}
    return result_json

def recategorize_transactions(transactions, categories):
    category_stats = {}
    uncategorized = []
    for t in transactions:
        cat = categorize(t['details'], categories)
        t['category'] = cat
        category_stats.setdefault(cat, {"count":0, "sum":0})
        category_stats[cat]["count"] += 1
        category_stats[cat]["sum"] += abs(t["amount"])
        if cat == "other":
            uncategorized.append(t)
    return category_stats, uncategorized

# ===================== FEATURE ENGINEERING =====================
def extract_features(transactions, categories):
    """Извлекает признаки из транзакций для модели"""
    
    counts = {}
    sums = {}
    
    # Инициализация для всех категорий из JSON
    for cat in categories.keys():
        counts[cat] = 0
        sums[cat] = 0.0
    
    # Дополнительные категории
    counts['person_transfer'] = 0
    counts['small_business'] = 0
    counts['other'] = 0
    sums['person_transfer'] = 0.0
    sums['small_business'] = 0.0
    sums['other'] = 0.0
    
    credit_taken_count = 0
    credit_payment_count = 0
    
    total_income = 0.0
    total_expense = 0.0
    max_income = 0.0
    max_expense = 0.0
    amounts = []
    
    for t in transactions:
        amount = t['amount']
        category = t.get('category', 'other')
        
        amounts.append(amount)
        
        if amount > 0:
            total_income += amount
            if amount > max_income:
                max_income = amount
        else:
            total_expense += abs(amount)
            if abs(amount) > max_expense:
                max_expense = abs(amount)
        
        if category in counts:
            counts[category] += 1
            sums[category] += abs(amount)
        
        if category == 'credit':
            if amount > 0:
                credit_taken_count += 1
            else:
                credit_payment_count += 1
    
    # Формируем признаки
    features = {}
    
    for cat in categories.keys():
        features[f'{cat}_count'] = counts.get(cat, 0)
        features[f'{cat}_sum'] = round(sums.get(cat, 0), 2)
    
    features['person_transfer_count'] = counts['person_transfer']
    features['person_transfer_sum'] = round(sums['person_transfer'], 2)
    features['small_business_count'] = counts['small_business']
    features['small_business_sum'] = round(sums['small_business'], 2)
    
    features['credit_taken_count'] = credit_taken_count
    features['credit_payment_count'] = credit_payment_count
    
    features['total_income'] = round(total_income, 2)
    features['total_expense'] = round(total_expense, 2)
    features['transaction_count'] = len(transactions)
    features['max_income'] = round(max_income, 2)
    features['max_expense'] = round(max_expense, 2)
    
    if amounts:
        features['avg_transaction'] = round(np.mean([abs(a) for a in amounts]), 2)
    else:
        features['avg_transaction'] = 0
    
    # Относительные признаки
    if total_income > 0:
        features['expense_to_income_ratio'] = round(total_expense / total_income, 4)
    else:
        features['expense_to_income_ratio'] = 10.0
    
    if total_expense > 0:
        features['betting_expense_ratio'] = round(sums.get('betting', 0) / total_expense, 4)
        features['credit_expense_ratio'] = round(sums.get('credit', 0) / total_expense, 4)
        features['entertainment_expense_ratio'] = round(sums.get('entertainment', 0) / total_expense, 4)
        features['hotels_expense_ratio'] = round(sums.get('hotels', 0) / total_expense, 4)
        features['cafes_expense_ratio'] = round(sums.get('cafes', 0) / total_expense, 4)
        features['utility_expense_ratio'] = round(sums.get('utility', 0) / total_expense, 4)
        features['shops_expense_ratio'] = round(sums.get('shops', 0) / total_expense, 4)
    else:
        features['betting_expense_ratio'] = 0
        features['credit_expense_ratio'] = 0
        features['entertainment_expense_ratio'] = 0
        features['hotels_expense_ratio'] = 0
        features['cafes_expense_ratio'] = 0
        features['utility_expense_ratio'] = 0
        features['shops_expense_ratio'] = 0
    
    total_count = len(transactions)
    if total_count > 0:
        features['betting_count_ratio'] = round(counts.get('betting', 0) / total_count, 4)
        features['credit_count_ratio'] = round(counts.get('credit', 0) / total_count, 4)
        features['utility_count_ratio'] = round(counts.get('utility', 0) / total_count, 4)
        features['transport_count_ratio'] = round(counts.get('transport', 0) / total_count, 4)
    else:
        features['betting_count_ratio'] = 0
        features['credit_count_ratio'] = 0
        features['utility_count_ratio'] = 0
        features['transport_count_ratio'] = 0
    
    return features, counts, sums

# ===================== КРЕДИТНЫЙ СКОРИНГ С SHAP =====================
def run_credit_scoring(transactions, categories, user_info, category_stats):
    """Запускает кредитный скоринг и SHAP объяснение"""
    
    print("\n" + "=" * 60)
    print("КРЕДИТНЫЙ СКОРИНГ")
    print("=" * 60)
    
    # 1. Загружаем модель
    print("\n[1] Загрузка модели...")
    try:
        model = joblib.load('model_xgboost.pkl')
        feature_names = joblib.load('feature_names.pkl')
        print(f"  ✓ Модель загружена (XGBoost)")
        print(f"  ✓ Признаков в модели: {len(feature_names)}")
    except FileNotFoundError as e:
        print(f"  ✗ Ошибка: {e}")
        print("  Убедись что файлы model_xgboost.pkl и feature_names.pkl в папке")
        return
    
    # 2. Извлекаем признаки
    print("\n[2] Извлечение признаков...")
    features, counts, sums = extract_features(transactions, categories)
    
    # Проверяем что все признаки есть
    missing = [f for f in feature_names if f not in features]
    if missing:
        print(f"  ⚠ Отсутствуют {len(missing)} признаков, заполняем нулями")
        for f in missing:
            features[f] = 0
    
    # Создаём DataFrame в правильном порядке
    X = pd.DataFrame([features])[feature_names]
    print(f"  ✓ Признаков извлечено: {len(features)}")
    
    # 3. Предсказание
    print("\n[3] Предсказание модели...")
    pred_proba = model.predict_proba(X)[0]
    pred_label = model.predict(X)[0]
    risk_probability = pred_proba[1] * 100
    
    # 4. SHAP объяснение
    print("\n[4] SHAP анализ...")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)
    
    if len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[0, :, 1]
    elif isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1][0]
    else:
        shap_values = shap_values_raw[0]
    
    # Топ факторы
    factors = list(zip(feature_names, shap_values, X.iloc[0].values))
    factors_sorted = sorted(factors, key=lambda x: abs(x[1]), reverse=True)
    
    # ===== ВЫВОД РЕЗУЛЬТАТОВ =====
    print("\n")
    print("=" * 60)
    print("РЕЗУЛЬТАТ КРЕДИТНОГО СКОРИНГА")
    print("=" * 60)
    
    print(f"\n👤 Клиент: {user_info.get('full_name', 'Неизвестно')}")
    print(f"📅 Период анализа: {user_info.get('period_from', '?')} — {user_info.get('period_to', '?')}")
    print(f"📊 Проанализировано транзакций: {len(transactions)}")
    
    print("\n" + "-" * 60)
    
    if pred_label == 1:
        print(f"\n🔴 РЕШЕНИЕ: ВЫСОКИЙ РИСК")
    else:
        print(f"\n🟢 РЕШЕНИЕ: НИЗКИЙ РИСК")
    
    print(f"\n📈 Вероятность риска: {risk_probability:.1f}%")
    
    # Шкала риска
    bar_filled = int(risk_probability / 5)
    bar_empty = 20 - bar_filled
    print(f"\n   Шкала риска:")
    print(f"   [{'█' * bar_filled}{'░' * bar_empty}] {risk_probability:.1f}%")
    print(f"   0%                 50%                100%")
    
    print("\n" + "-" * 60)
    print("\n📋 ОБОСНОВАНИЕ РЕШЕНИЯ (SHAP):")
    print("-" * 60)
    
    positive_factors = [(f, s, v) for f, s, v in factors_sorted if s > 0.005][:5]
    negative_factors = [(f, s, v) for f, s, v in factors_sorted if s < -0.005][:5]
    
    if positive_factors:
        print("\n🔴 Факторы ПОВЫШАЮЩИЕ риск:")
        for feat, shap_val, feat_val in positive_factors:
            if 'ratio' in feat:
                val_str = f"{feat_val*100:.1f}%"
            elif feat_val > 1000:
                val_str = f"{feat_val:,.0f} ₸"
            else:
                val_str = f"{feat_val:.0f}"
            print(f"   • {feat}: {val_str} (влияние: +{shap_val:.3f})")
    
    if negative_factors:
        print("\n🟢 Факторы СНИЖАЮЩИЕ риск:")
        for feat, shap_val, feat_val in negative_factors:
            if 'ratio' in feat:
                val_str = f"{feat_val*100:.1f}%"
            elif feat_val > 1000:
                val_str = f"{feat_val:,.0f} ₸"
            else:
                val_str = f"{feat_val:.0f}"
            print(f"   • {feat}: {val_str} (влияние: {shap_val:.3f})")
    
    print("\n" + "-" * 60)
    print("\n📊 КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:")
    print("-" * 60)
    print(f"   Доходы:              {features['total_income']:>15,.0f} ₸")
    print(f"   Расходы:             {features['total_expense']:>15,.0f} ₸")
    print(f"   Расходы/Доходы:      {features['expense_to_income_ratio']*100:>14.1f}%")
    print(f"   ")
    print(f"   Ставки (betting):    {sums.get('betting', 0):>15,.0f} ₸ ({features['betting_expense_ratio']*100:.1f}% расходов)")
    print(f"   Кредиты:             {sums.get('credit', 0):>15,.0f} ₸ ({features['credit_expense_ratio']*100:.1f}% расходов)")
    print(f"   Коммуналка:          {sums.get('utility', 0):>15,.0f} ₸ ({features['utility_count_ratio']*100:.1f}% транзакций)")
    print(f"   ")
    print(f"   Кредитов взято:      {features['credit_taken_count']:>15}")
    print(f"   Платежей по кредиту: {features['credit_payment_count']:>15}")
    
    return risk_probability, pred_label

# ===================== ОСНОВНОЙ ЦИКЛ =====================
def main():
    global CATEGORIES
    
    if not os.path.exists(PDF_PATH):
        print(f"Файл не найден: {PDF_PATH}")
        return

    print("=" * 60)
    print("АУДИТ КАТЕГОРИЗАЦИИ ТРАНЗАКЦИЙ")
    print("=" * 60)

    print("\nКлючи из categories.json:")
    print(", ".join(CATEGORIES.keys()))
    print("-"*60)

    print("Извлекаем текст из PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    user_info = parse_user_info(text)
    print(f"Клиент: {user_info.get('full_name', 'Неизвестно')}")
    print("Текст извлечён\n")

    print("Парсим транзакции...")
    transactions = parse_transactions(text)
    total = len(transactions)
    print(f"Найдено {total} транзакций\n")

    iteration = 1
    category_stats = {}
    
    while True:
        print(f"\n--- Итерация {iteration} ---")
        category_stats, uncategorized = recategorize_transactions(transactions, CATEGORIES)
        other_percent = len(uncategorized) / total * 100

        print("Распределение по категориям:\n")
        sorted_stats = sorted(category_stats.items(), key=lambda x: x[1]["count"], reverse=True)
        for cat, stats in sorted_stats:
            percent = stats["count"] / total * 100
            print(f"{cat:20} {stats['count']:5} шт {percent:6.2f}% {stats['sum']:15,.0f} ₸")

        print("\nНе распознано (other):", len(uncategorized), f"({other_percent:.2f}%)")

        if other_percent <= OTHER_THRESHOLD_PERCENT:
            print("\n✓ OTHER меньше порога. Цикл завершён.")
            break

        # --- топ N деталей для AI ---
        uncategorized_details = [t['details'].lower() for t in uncategorized]
        top_uncategorized = [word for word, _ in Counter(uncategorized_details).most_common(TOP_N_UNCATEGORIZED)]
        print(f"\nOTHER > {OTHER_THRESHOLD_PERCENT}% ({other_percent:.2f}%). Топ {TOP_N_UNCATEGORIZED} деталей:")
        for word in top_uncategorized:
            print("  ", word)

        # --- вызываем AI ---
        result_json = ai_categorize(top_uncategorized, list(CATEGORIES.keys()))
        print("\nРезультат распределения от AI:")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))

        # --- обновляем CATEGORIES ---
        for cat, words in result_json.items():
            if cat in CATEGORIES:
                for w in words:
                    if w not in CATEGORIES[cat]:
                        CATEGORIES[cat].append(w)
            else:
                CATEGORIES[cat] = words

        print("\nОбновлённые категории после AI:")
        for cat, words in CATEGORIES.items():
            print(f"{cat}: {len(words)} слов")

        # --- сохраняем обратно в JSON ---
        with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump(CATEGORIES, f, ensure_ascii=False, indent=2)
        print(f"\nФайл категорий обновлён: {CATEGORIES_FILE}")

        iteration += 1
        
        if iteration > 5:
            print("\n⚠ Достигнут лимит итераций (5). Продолжаем...")
            break

    print("\n" + "="*60)
    print("АУДИТ ЗАВЕРШЁН")
    print("="*60)
    
    # ===================== КРЕДИТНЫЙ СКОРИНГ =====================
    run_credit_scoring(transactions, CATEGORIES, user_info, category_stats)
    
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)


if __name__ == "__main__":
    main()