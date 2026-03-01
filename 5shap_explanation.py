import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# ===== ЗАГРУЗКА =====
print("=" * 60)
print("SHAP — ОБЪЯСНЕНИЕ ПРЕДСКАЗАНИЙ МОДЕЛИ")
print("=" * 60)

print("\nЗагрузка данных и модели...")
df = pd.read_csv('ml_dataset.csv')
model = joblib.load('model_random_forest.pkl')
feature_names = joblib.load('feature_names.pkl')

# Подготовка данных
X = df.drop(['user_id', 'risk_score', 'label'], axis=1)
y = df['label']

print(f"  Юзеров: {len(df)}")
print(f"  Признаков: {len(feature_names)}")

# ===== SHAP EXPLAINER =====
print("\nСоздание SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X)

print(f"  SHAP values type: {type(shap_values_raw)}")
print(f"  SHAP values shape: {np.array(shap_values_raw).shape}")

# Обработка разных форматов SHAP values
if isinstance(shap_values_raw, list):
    # Старый формат: список [class_0, class_1]
    shap_values_risk = shap_values_raw[1]
elif len(shap_values_raw.shape) == 3:
    # Новый формат: (n_samples, n_features, n_classes)
    shap_values_risk = shap_values_raw[:, :, 1]
else:
    shap_values_risk = shap_values_raw

print(f"  SHAP values для класса 1: {shap_values_risk.shape}")
print("  ✓ SHAP values рассчитаны")

# ===== 1. ОБЩАЯ ВАЖНОСТЬ ПРИЗНАКОВ =====
print("\n" + "=" * 60)
print("1. ГЛОБАЛЬНАЯ ВАЖНОСТЬ ПРИЗНАКОВ (SHAP)")
print("=" * 60)

# Средняя абсолютная важность
mean_shap = np.abs(shap_values_risk).mean(axis=0)

print(f"  mean_shap shape: {mean_shap.shape}")

# Создаём DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_shap
}).sort_values('importance', ascending=False)

print("\nТоп-15 признаков по SHAP:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}")

# Сохраняем график
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_risk, X, feature_names=feature_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ График сохранён: shap_summary.png")

# ===== 2. ОБЪЯСНЕНИЕ КОНКРЕТНЫХ ЮЗЕРОВ =====
print("\n" + "=" * 60)
print("2. ОБЪЯСНЕНИЕ КОНКРЕТНЫХ ПРЕДСКАЗАНИЙ")
print("=" * 60)

def explain_user(user_idx, X, shap_values, feature_names, df):
    """Объясняет предсказание для конкретного юзера"""
    user_id = df.iloc[user_idx]['user_id']
    actual_label = df.iloc[user_idx]['label']
    risk_score = df.iloc[user_idx]['risk_score']
    
    # SHAP values для этого юзера
    user_shap = shap_values[user_idx]
    
    # Предсказание модели
    pred_proba = model.predict_proba(X.iloc[[user_idx]])[0][1]
    pred_label = 1 if pred_proba >= 0.5 else 0
    
    print(f"\n{'─' * 50}")
    print(f"User ID: {int(user_id)}")
    print(f"{'─' * 50}")
    print(f"  Факт:         {'HIGH RISK' if actual_label == 1 else 'LOW RISK'}")
    print(f"  Предсказание: {'HIGH RISK' if pred_label == 1 else 'LOW RISK'} ({pred_proba:.1%})")
    print(f"  Risk Score:   {risk_score:.1f}")
    
    # Топ факторы
    factors = list(zip(feature_names, user_shap, X.iloc[user_idx].values))
    factors_sorted = sorted(factors, key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n  Почему такое решение:")
    print(f"  {'Признак':<28} {'Значение':>12} {'Влияние':>10}")
    print(f"  {'-'*52}")
    
    for feat, shap_val, feat_val in factors_sorted[:8]:
        direction = "↑ риск" if shap_val > 0 else "↓ риск"
        
        # Форматирование значения
        if 'ratio' in feat:
            val_str = f"{feat_val*100:.1f}%"
        elif feat_val > 1000:
            val_str = f"{feat_val:,.0f}"
        else:
            val_str = f"{feat_val:.2f}"
        
        print(f"  {feat:<28} {val_str:>12} {shap_val:>+.4f} {direction}")
    
    return user_id, actual_label, pred_label


# Находим примеры
high_risk_idx = df[df['label'] == 1].index[:3].tolist()
low_risk_idx = df[df['label'] == 0].index[:3].tolist()

# Находим ошибки модели
predictions = model.predict(X)
false_positives = df[(df['label'] == 0) & (predictions == 1)].index[:2].tolist()
false_negatives = df[(df['label'] == 1) & (predictions == 0)].index[:2].tolist()

print("\n>>> HIGH RISK юзеры:")
for idx in high_risk_idx:
    explain_user(idx, X, shap_values_risk, feature_names, df)

print("\n>>> LOW RISK юзеры:")
for idx in low_risk_idx:
    explain_user(idx, X, shap_values_risk, feature_names, df)

if false_negatives:
    print("\n>>> ОШИБКИ: Пропущенный риск (False Negative)")
    print("    Модель сказала LOW, но на самом деле HIGH")
    for idx in false_negatives[:2]:
        explain_user(idx, X, shap_values_risk, feature_names, df)

if false_positives:
    print("\n>>> ОШИБКИ: Ложная тревога (False Positive)")
    print("    Модель сказала HIGH, но на самом деле LOW")
    for idx in false_positives[:2]:
        explain_user(idx, X, shap_values_risk, feature_names, df)

# ===== 3. WATERFALL PLOT ДЛЯ ОДНОГО ЮЗЕРА =====
print("\n" + "=" * 60)
print("3. ДЕТАЛЬНЫЙ ГРАФИК (WATERFALL)")
print("=" * 60)

# Берём первого high_risk юзера
example_idx = high_risk_idx[0]

# Определяем base_value
if isinstance(explainer.expected_value, (list, np.ndarray)):
    if len(explainer.expected_value) > 1:
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value[0]
else:
    base_value = explainer.expected_value

plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_risk[example_idx],
        base_values=base_value,
        data=X.iloc[example_idx].values,
        feature_names=feature_names
    ),
    show=False,
    max_display=12
)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Waterfall график сохранён: shap_waterfall.png")

# ===== 4. BAR PLOT =====
print("\n" + "=" * 60)
print("4. BAR PLOT — СРЕДНЯЯ ВАЖНОСТЬ")
print("=" * 60)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_risk, X, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Bar график сохранён: shap_bar.png")

# ===== ИТОГИ =====
print("\n" + "=" * 60)
print("ИТОГИ")
print("=" * 60)
print("""
Сохранённые файлы:
  • shap_summary.png   — точечный график важности
  • shap_bar.png       — столбчатый график важности  
  • shap_waterfall.png — детальный разбор одного юзера

SHAP показывает:
  • Какие признаки влияют на решение
  • Насколько сильно влияют
  • В какую сторону (↑ риск или ↓ риск)

Это нужно для:
  • Объяснения клиенту ("вам отказано потому что...")
  • Регуляторов (банки обязаны объяснять)
  • Отладки модели (почему ошибается)
""")