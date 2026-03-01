import pandas as pd
import joblib
import shap

from feature_engineering import extract_features


def run_credit_scoring(transactions, categories, user_info, category_stats):
    """Запускает кредитный скоринг и SHAP объяснение"""

    print("\n" + "=" * 60)
    print("КРЕДИТНЫЙ СКОРИНГ")
    print("=" * 60)

    # 1. Загрузка модели
    print("\n[1] Загрузка модели...")
    try:
        model = joblib.load('model_xgboost.pkl')
        feature_names = joblib.load('feature_names.pkl')
        print(f"  ✓ Модель загружена (XGBoost)")
        print(f"  ✓ Признаков в модели: {len(feature_names)}")
    except FileNotFoundError as e:
        print(f"  ✗ Ошибка: {e}")
        print("  Убедись что файлы model_xgboost.pkl и feature_names.pkl в папке")
        return None, None

    # 2. Извлечение признаков
    print("\n[2] Извлечение признаков...")
    features, counts, sums = extract_features(transactions, categories)

    # Заполняем отсутствующие признаки нулями
    missing = [f for f in feature_names if f not in features]
    if missing:
        print(f"  ⚠ Отсутствуют {len(missing)} признаков, заполняем нулями")
        for f in missing:
            features[f] = 0

    X = pd.DataFrame([features])[feature_names]
    print(f"  ✓ Признаков извлечено: {len(features)}")

    # 3. Предсказание
    print("\n[3] Предсказание модели...")
    pred_proba = model.predict_proba(X)[0]
    pred_label = model.predict(X)[0]
    risk_probability = pred_proba[1] * 100

    # 4. SHAP
    print("\n[4] SHAP анализ...")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)

    if len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[0, :, 1]
    elif isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1][0]
    else:
        shap_values = shap_values_raw[0]

    factors = list(zip(feature_names, shap_values, X.iloc[0].values))
    factors_sorted = sorted(factors, key=lambda x: abs(x[1]), reverse=True)

    positive_factors = [(f, s, v) for f, s, v in factors_sorted if s > 0.005][:5]
    negative_factors = [(f, s, v) for f, s, v in factors_sorted if s < -0.005][:5]

    # 5. Вывод
    print_results(user_info, transactions, risk_probability, pred_label,
                  features, sums, positive_factors, negative_factors)

    # 6. AI объяснение для клиента
    print("\n[5] Генерация объяснения для клиента...")
    try:
        from ai_explanation import generate_explanation

        explanation = generate_explanation(
            user_info=user_info,
            risk_probability=risk_probability,
            pred_label=pred_label,
            features=features,
            sums=sums,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            transaction_count=len(transactions)
        )

        print("\n" + "=" * 60)
        print("📝 АНАЛИЗ ДЛЯ КЛИЕНТА (AI)")
        print("=" * 60)
        print(explanation)

    except Exception as e:
        print(f"\n⚠ Не удалось сгенерировать AI объяснение: {e}")

    return risk_probability, pred_label


def print_results(user_info, transactions, risk_probability, pred_label,
                  features, sums, positive_factors, negative_factors):
    """Выводит результаты скоринга в консоль"""

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

    bar_filled = int(risk_probability / 5)
    bar_empty = 20 - bar_filled
    print(f"\n   Шкала риска:")
    print(f"   [{'█' * bar_filled}{'░' * bar_empty}] {risk_probability:.1f}%")
    print(f"   0%                 50%                100%")

    print("\n" + "-" * 60)
    print("\n📋 ОБОСНОВАНИЕ РЕШЕНИЯ (SHAP):")
    print("-" * 60)

    if positive_factors:
        print("\n🔴 Факторы ПОВЫШАЮЩИЕ риск:")
        for feat, shap_val, feat_val in positive_factors:
            val_str = _format_value(feat, feat_val)
            print(f"   • {feat}: {val_str} (влияние: +{shap_val:.3f})")

    if negative_factors:
        print("\n🟢 Факторы СНИЖАЮЩИЕ риск:")
        for feat, shap_val, feat_val in negative_factors:
            val_str = _format_value(feat, feat_val)
            print(f"   • {feat}: {val_str} (влияние: {shap_val:.3f})")

    print("\n" + "-" * 60)
    print("\n📊 КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:")
    print("-" * 60)
    print(f"   Доходы:              {features['total_income']:>15,.0f} ₸")
    print(f"   Расходы:             {features['total_expense']:>15,.0f} ₸")
    print(f"   Расходы/Доходы:      {features['expense_to_income_ratio'] * 100:>14.1f}%")
    print(f"   ")
    print(f"   Ставки (betting):    {sums.get('betting', 0):>15,.0f} ₸ ({features['betting_expense_ratio'] * 100:.1f}% расходов)")
    print(f"   Кредиты:             {sums.get('credit', 0):>15,.0f} ₸ ({features['credit_expense_ratio'] * 100:.1f}% расходов)")
    print(f"   Коммуналка:          {sums.get('utility', 0):>15,.0f} ₸ ({features['utility_count_ratio'] * 100:.1f}% транзакций)")
    print(f"   ")
    print(f"   Кредитов взято:      {features['credit_taken_count']:>15}")
    print(f"   Платежей по кредиту: {features['credit_payment_count']:>15}")


def _format_value(feat, feat_val):
    """Форматирует значение признака для вывода"""
    if 'ratio' in feat:
        return f"{feat_val * 100:.1f}%"
    elif feat_val > 1000:
        return f"{feat_val:,.0f} ₸"
    else:
        return f"{feat_val:.0f}"
