import numpy as np


def extract_features(transactions, categories):
    """Извлекает признаки из транзакций для модели"""

    counts = {}
    sums = {}

    # Инициализация для всех категорий из JSON
    for cat in categories.keys():
        counts[cat] = 0
        sums[cat] = 0.0

    # Дополнительные категории
    for extra in ['person_transfer', 'small_business', 'other']:
        counts[extra] = 0
        sums[extra] = 0.0

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

    # === Формируем признаки ===
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

    # === Относительные признаки ===
    if total_income > 0:
        features['expense_to_income_ratio'] = round(total_expense / total_income, 4)
    else:
        features['expense_to_income_ratio'] = 10.0

    if total_expense > 0:
        for cat in ['betting', 'credit', 'entertainment', 'hotels', 'cafes', 'utility', 'shops']:
            features[f'{cat}_expense_ratio'] = round(sums.get(cat, 0) / total_expense, 4)
    else:
        for cat in ['betting', 'credit', 'entertainment', 'hotels', 'cafes', 'utility', 'shops']:
            features[f'{cat}_expense_ratio'] = 0

    total_count = len(transactions)
    if total_count > 0:
        for cat in ['betting', 'credit', 'utility', 'transport']:
            features[f'{cat}_count_ratio'] = round(counts.get(cat, 0) / total_count, 4)
    else:
        for cat in ['betting', 'credit', 'utility', 'transport']:
            features[f'{cat}_count_ratio'] = 0

    return features, counts, sums
