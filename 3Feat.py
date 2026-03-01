import pandas as pd
import numpy as np
import re
import json
import os

# ===== ЗАГРУЗКА ДАННЫХ =====
print("Загрузка данных...")
df = pd.read_csv('all_transactions.csv')
print(f"  Транзакций: {len(df):,}")


# =============================================================================
# КАТЕГОРИИ — ВСТАВЬ СВОИ КЛЮЧЕВЫЕ СЛОВА СЮДА
# =============================================================================
CATEGORIES_FILE = "parser/categories.json"

# ===== РАСШИРЕННЫЕ КЛЮЧЕВЫЕ СЛОВА ДЛЯ КАТЕГОРИЙ =====
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
        CATEGORIES = json.load(f)
else:
    print(f"Файл категорий не найден: {CATEGORIES_FILE}")
    CATEGORIES = {}
# =============================================================================


def categorize(details):
    """Определяет категорию транзакции"""
    d = details.lower()
    
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in d:
                return cat
    
    # Переводы на людей (Имя Б.)
    if re.search(r' .\.$', details):
        return 'person_transfer'
    
    # ИП и магазины
    if d.startswith('ип ') or d.startswith('ip '):
        return 'small_business'
    if 'магазин' in d or 'маг ' in d or 'дукен' in d:
        return 'small_business'
    
    return 'other'


def extract_features(user_id, transactions):
    """Извлекает признаки из транзакций юзера"""
    
    # Счётчики по категориям
    counts = {cat: 0 for cat in CATEGORIES.keys()}
    counts['person_transfer'] = 0
    counts['small_business'] = 0
    counts['other'] = 0
    
    sums = {cat: 0.0 for cat in CATEGORIES.keys()}
    sums['person_transfer'] = 0.0
    sums['small_business'] = 0.0
    sums['other'] = 0.0
    
    # Для кредитов отдельно
    credit_taken_count = 0
    credit_payment_count = 0
    
    # Общие
    total_income = 0.0
    total_expense = 0.0
    max_income = 0.0
    max_expense = 0.0
    amounts = []
    
    for _, t in transactions.iterrows():
        amount = float(t['amount'])
        details = str(t['details'])
        
        amounts.append(amount)
        
        # Доходы/расходы
        if amount > 0:
            total_income += amount
            if amount > max_income:
                max_income = amount
        else:
            total_expense += abs(amount)
            if abs(amount) > max_expense:
                max_expense = abs(amount)
        
        # Категоризация
        category = categorize(details)
        counts[category] += 1
        sums[category] += abs(amount)
        
        # Для кредитов: брал или платил
        if category == 'credit':
            if amount > 0:
                credit_taken_count += 1
            else:
                credit_payment_count += 1
    
    # === ФОРМИРУЕМ ПРИЗНАКИ ===
    features = {'user_id': user_id}
    
    # Количество и суммы по категориям
    for cat in CATEGORIES.keys():
        features[f'{cat}_count'] = counts[cat]
        features[f'{cat}_sum'] = round(sums[cat], 2)
    
    # Дополнительные категории
    features['person_transfer_count'] = counts['person_transfer']
    features['person_transfer_sum'] = round(sums['person_transfer'], 2)
    features['small_business_count'] = counts['small_business']
    features['small_business_sum'] = round(sums['small_business'], 2)
    
    # Кредиты детально
    features['credit_taken_count'] = credit_taken_count
    features['credit_payment_count'] = credit_payment_count
    
    # Общие финансовые
    features['total_income'] = round(total_income, 2)
    features['total_expense'] = round(total_expense, 2)
    features['transaction_count'] = len(transactions)
    features['max_income'] = round(max_income, 2)
    features['max_expense'] = round(max_expense, 2)
    
    if amounts:
        features['avg_transaction'] = round(np.mean([abs(a) for a in amounts]), 2)
    else:
        features['avg_transaction'] = 0
    
    # === ОТНОСИТЕЛЬНЫЕ ПРИЗНАКИ (самое важное!) ===
    
    # Соотношение расходов к доходам
    if total_income > 0:
        features['expense_to_income_ratio'] = round(total_expense / total_income, 4)
    else:
        features['expense_to_income_ratio'] = 10.0  # Нет дохода = плохо
    
    # Проценты от расходов
    if total_expense > 0:
        features['betting_expense_ratio'] = round(sums['betting'] / total_expense, 4)
        features['credit_expense_ratio'] = round(sums['credit'] / total_expense, 4)
        features['entertainment_expense_ratio'] = round(sums['entertainment'] / total_expense, 4)
        features['hotels_expense_ratio'] = round(sums['hotels'] / total_expense, 4)
        features['cafes_expense_ratio'] = round(sums['cafes'] / total_expense, 4)
        features['utility_expense_ratio'] = round(sums['utility'] / total_expense, 4)
        features['shops_expense_ratio'] = round(sums['shops'] / total_expense, 4)
    else:
        features['betting_expense_ratio'] = 0
        features['credit_expense_ratio'] = 0
        features['entertainment_expense_ratio'] = 0
        features['hotels_expense_ratio'] = 0
        features['cafes_expense_ratio'] = 0
        features['utility_expense_ratio'] = 0
        features['shops_expense_ratio'] = 0
    
    # Проценты от количества транзакций
    total_count = len(transactions)
    if total_count > 0:
        features['betting_count_ratio'] = round(counts['betting'] / total_count, 4)
        features['credit_count_ratio'] = round(counts['credit'] / total_count, 4)
        features['utility_count_ratio'] = round(counts['utility'] / total_count, 4)
        features['transport_count_ratio'] = round(counts['transport'] / total_count, 4)
    else:
        features['betting_count_ratio'] = 0
        features['credit_count_ratio'] = 0
        features['utility_count_ratio'] = 0
        features['transport_count_ratio'] = 0
    
    return features


def calculate_risk_label(f):
    """
    Рассчитывает метку риска на основе ОТНОСИТЕЛЬНЫХ признаков.
    """
    
    risk_score = 0
    
    # =========================================================================
    # КРАСНЫЕ ФЛАГИ (повышают риск)
    # =========================================================================
    
    # СТАВКИ — % от расходов
    if f['betting_expense_ratio'] > 0.30:      # >30% расходов на ставки
        risk_score += 35
    elif f['betting_expense_ratio'] > 0.20:    # >20%
        risk_score += 25
    elif f['betting_expense_ratio'] > 0.10:    # >10%
        risk_score += 15
    elif f['betting_expense_ratio'] > 0.05:    # >5%
        risk_score += 10
    
    # СТАВКИ — % от количества транзакций
    if f['betting_count_ratio'] > 0.15:        # >15% транзакций — ставки
        risk_score += 15
    elif f['betting_count_ratio'] > 0.08:
        risk_score += 10
    elif f['betting_count_ratio'] > 0.03:
        risk_score += 5
    
    # КРЕДИТЫ — сколько раз брал
    if f['credit_taken_count'] > 10:
        risk_score += 25
    elif f['credit_taken_count'] > 5:
        risk_score += 15
    elif f['credit_taken_count'] > 2:
        risk_score += 5
    
    # КРЕДИТЫ — % от расходов
    if f['credit_expense_ratio'] > 0.25:
        risk_score += 15
    elif f['credit_expense_ratio'] > 0.15:
        risk_score += 10
    
    # РАСХОДЫ > ДОХОДЫ
    if f['expense_to_income_ratio'] > 1.5:
        risk_score += 25
    elif f['expense_to_income_ratio'] > 1.2:
        risk_score += 15
    elif f['expense_to_income_ratio'] > 1.0:
        risk_score += 5
    
    # =========================================================================
    # ЖЁЛТЫЕ ФЛАГИ (немного повышают риск)
    # =========================================================================
    
    # ОТЕЛИ — % от расходов (живёт не по средствам?)
    if f['hotels_expense_ratio'] > 0.10:
        risk_score += 10
    elif f['hotels_expense_ratio'] > 0.05:
        risk_score += 5
    
    # РАЗВЛЕЧЕНИЯ — % от расходов
    if f['entertainment_expense_ratio'] > 0.15:
        risk_score += 10
    elif f['entertainment_expense_ratio'] > 0.08:
        risk_score += 5
    
    # КАФЕ/РЕСТОРАНЫ — % от расходов
    if f['cafes_expense_ratio'] > 0.20:
        risk_score += 10
    elif f['cafes_expense_ratio'] > 0.12:
        risk_score += 5
    
    # =========================================================================
    # ЗЕЛЁНЫЕ ФЛАГИ (снижают риск)
    # =========================================================================
    
    # КОММУНАЛКА — % от количества транзакций
    if f['utility_count_ratio'] > 0.10:
        risk_score -= 15
    elif f['utility_count_ratio'] > 0.05:
        risk_score -= 10
    elif f['utility_count_ratio'] > 0.02:
        risk_score -= 5
    
    # ГОСУСЛУГИ — количество
    if f['government_count'] > 15:
        risk_score -= 15
    elif f['government_count'] > 8:
        risk_score -= 10
    elif f['government_count'] > 3:
        risk_score -= 5
    
    # ТРАНСПОРТ — % (ездит на работу = занятость)
    if f['transport_count_ratio'] > 0.08:
        risk_score -= 10
    elif f['transport_count_ratio'] > 0.04:
        risk_score -= 5
    
    # ЗДОРОВЬЕ — заботится о себе
    if f['health_count'] > 10:
        risk_score -= 10
    elif f['health_count'] > 5:
        risk_score -= 5
    
    # ПЛАТИТ ПО КРЕДИТАМ (ответственный)
    if f['credit_payment_count'] > f['credit_taken_count'] * 3:
        risk_score -= 10
    elif f['credit_payment_count'] > f['credit_taken_count'] * 2:
        risk_score -= 5
    
    # ДОХОДЫ > РАСХОДЫ (финансово здоров)
    if f['expense_to_income_ratio'] < 0.6:
        risk_score -= 15
    elif f['expense_to_income_ratio'] < 0.8:
        risk_score -= 10
    elif f['expense_to_income_ratio'] < 0.95:
        risk_score -= 5
    
    # =========================================================================
    # МЕТКА
    # =========================================================================
    # high_risk = 1, low_risk = 0
    # Порог = 20
    label = 1 if risk_score >= 20 else 0
    
    return label, risk_score


def main():
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING (ОТНОСИТЕЛЬНЫЕ ПРИЗНАКИ)")
    print("=" * 60)
    
    # Группируем по юзерам
    print("\nГруппировка транзакций по юзерам...")
    grouped = df.groupby('user_id')
    user_ids = df['user_id'].unique()
    print(f"  Уникальных юзеров: {len(user_ids)}")
    
    # Извлекаем признаки
    print("\nИзвлечение признаков...")
    all_features = []
    
    for idx, user_id in enumerate(user_ids):
        user_transactions = grouped.get_group(user_id)
        
        features = extract_features(user_id, user_transactions)
        label, risk_score = calculate_risk_label(features)
        features['risk_score'] = risk_score
        features['label'] = label
        
        all_features.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"  Обработано: {idx + 1}/{len(user_ids)}")
    
    # Создаём DataFrame
    print("\nСоздание датасета...")
    result_df = pd.DataFrame(all_features)
    
    # === СТАТИСТИКА ===
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)
    
    print(f"\nВсего юзеров: {len(result_df)}")
    print(f"Признаков: {len(result_df.columns)}")
    
    low_risk = len(result_df[result_df['label'] == 0])
    high_risk = len(result_df[result_df['label'] == 1])
    print(f"\nРаспределение меток:")
    print(f"  low_risk (0):  {low_risk:>4} ({low_risk / len(result_df) * 100:.1f}%)")
    print(f"  high_risk (1): {high_risk:>4} ({high_risk / len(result_df) * 100:.1f}%)")
    
    print(f"\nКлючевые ОТНОСИТЕЛЬНЫЕ признаки (средние):")
    print(f"  betting_expense_ratio:      {result_df['betting_expense_ratio'].mean()*100:>6.2f}%")
    print(f"  betting_count_ratio:        {result_df['betting_count_ratio'].mean()*100:>6.2f}%")
    print(f"  credit_expense_ratio:       {result_df['credit_expense_ratio'].mean()*100:>6.2f}%")
    print(f"  utility_count_ratio:        {result_df['utility_count_ratio'].mean()*100:>6.2f}%")
    print(f"  expense_to_income_ratio:    {result_df['expense_to_income_ratio'].mean():>6.2f}")
    
    print(f"\nРиск скор:")
    print(f"  Минимум: {result_df['risk_score'].min()}")
    print(f"  Максимум: {result_df['risk_score'].max()}")
    print(f"  Среднее: {result_df['risk_score'].mean():.1f}")
    
    # Сохраняем
    result_df.to_csv('ml_dataset.csv', index=False)
    print(f"\n✓ Датасет сохранён в ml_dataset.csv")
    
    # Примеры
    print("\n" + "=" * 60)
    print("ПРИМЕРЫ HIGH_RISK ЮЗЕРОВ")
    print("=" * 60)
    high_risk_examples = result_df[result_df['label'] == 1].head(3)
    for _, row in high_risk_examples.iterrows():
        print(f"\nUser {int(row['user_id'])} (risk_score={row['risk_score']}):")
        print(f"  betting: {row['betting_expense_ratio']*100:.1f}% расходов, {row['betting_count_ratio']*100:.1f}% транзакций")
        print(f"  credit: taken={row['credit_taken_count']}, payments={row['credit_payment_count']}")
        print(f"  expense/income: {row['expense_to_income_ratio']:.2f}")
    
    print("\n" + "=" * 60)
    print("ПРИМЕРЫ LOW_RISK ЮЗЕРОВ")
    print("=" * 60)
    low_risk_examples = result_df[result_df['label'] == 0].head(3)
    for _, row in low_risk_examples.iterrows():
        print(f"\nUser {int(row['user_id'])} (risk_score={row['risk_score']}):")
        print(f"  betting: {row['betting_expense_ratio']*100:.1f}% расходов")
        print(f"  utility: {row['utility_count_ratio']*100:.1f}% транзакций")
        print(f"  expense/income: {row['expense_to_income_ratio']:.2f}")


if __name__ == "__main__":
    main()