import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ===== ЗАГРУЗКА ДАННЫХ =====
print("=" * 60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ КРЕДИТНОГО СКОРИНГА")
print("=" * 60)

print("\nЗагрузка данных...")
df = pd.read_csv('ml_dataset.csv')
print(f"  Всего записей: {len(df)}")
print(f"  Признаков: {len(df.columns)}")

# ===== ПОДГОТОВКА ДАННЫХ =====
print("\nПодготовка данных...")

# Убираем ненужные колонки
X = df.drop(['user_id', 'risk_score', 'label'], axis=1)
y = df['label']

# Названия признаков
feature_names = X.columns.tolist()
print(f"  Признаков для обучения: {len(feature_names)}")

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# Масштабирование для Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== МОДЕЛИ =====
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

# ===== ОБУЧЕНИЕ И ОЦЕНКА =====
results = []

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
print("=" * 60)

for name, model in models.items():
    print(f"\n{'─' * 40}")
    print(f"  {name}")
    print(f"{'─' * 40}")
    
    # Используем scaled данные для LR, обычные для остальных
    if name == 'Logistic Regression':
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # Обучение
    model.fit(X_tr, y_train)
    
    # Предсказания
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    if name == 'Logistic Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Сохраняем результаты
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'CV ROC-AUC': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    # Выводим
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"  Actual 0    {cm[0,0]:>4}   {cm[0,1]:>4}")
    print(f"  Actual 1    {cm[1,0]:>4}   {cm[1,1]:>4}")

# ===== СРАВНЕНИЕ МОДЕЛЕЙ =====
print("\n" + "=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Лучшая модель по ROC-AUC
best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_roc_auc = results_df['ROC-AUC'].max()
print(f"\n✓ Лучшая модель по ROC-AUC: {best_model_name} ({best_roc_auc:.4f})")

# ===== FEATURE IMPORTANCE (для лучшей модели) =====
print("\n" + "=" * 60)
print("ВАЖНОСТЬ ПРИЗНАКОВ (Random Forest)")
print("=" * 60)

rf_model = models['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nТоп-15 важных признаков:")
for i in range(min(15, len(feature_names))):
    idx = indices[i]
    print(f"  {i+1:>2}. {feature_names[idx]:<30} {importances[idx]:.4f}")

# ===== СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ =====
import joblib

# Сохраняем все модели
for name, model in models.items():
    filename = f"model_{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    print(f"\n✓ Модель сохранена: {filename}")

# Сохраняем scaler
joblib.dump(scaler, 'scaler.pkl')
print("✓ Scaler сохранён: scaler.pkl")

# Сохраняем названия признаков
joblib.dump(feature_names, 'feature_names.pkl')
print("✓ Feature names сохранены: feature_names.pkl")

# ===== ИТОГИ =====
print("\n" + "=" * 60)
print("ИТОГИ")
print("=" * 60)
print(f"""
Обучено 3 модели:
  • Logistic Regression — базовая, интерпретируемая
  • Random Forest — ансамбль деревьев
  • XGBoost — градиентный бустинг

Лучшая модель: {best_model_name}
ROC-AUC: {best_roc_auc:.4f}

Следующий шаг: SHAP для объяснения предсказаний
""")