import pandas as pd
import os
import re
from collections import defaultdict
import json


# Читаем CSV
df = pd.read_csv('all_transactions.csv')
CATEGORIES_FILE = "parser/categories.json"

# ===== РАСШИРЕННЫЕ КЛЮЧЕВЫЕ СЛОВА ДЛЯ КАТЕГОРИЙ =====
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
        CATEGORIES = json.load(f)
else:
    print(f"Файл категорий не найден: {CATEGORIES_FILE}")
    CATEGORIES = {}

def categorize(details, operation):
    """Определяет категорию транзакции"""
    d = details.lower()
    
    # Проверяем по ключевым словам
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in d:
                return cat
    
    # Проверяем переводы на людей (формат "Имя Б." или "Имя Фамилия")
    # Паттерн: заканчивается на " X." где X - одна-две буквы
    if re.search(r' .\.$', details):  # Любой символ перед точкой
        return 'person_transfer'
    
    # ИП и магазины
    if details.lower().startswith('ип ') or details.lower().startswith('ip '):
        return 'small_business'
    if 'магазин' in d or 'маг ' in d or 'дукен' in d:
        return 'small_business'
    
    return 'other'

# Категоризируем
print("Категоризация транзакций...")
df['category'] = df.apply(lambda row: categorize(str(row['details']), str(row['operation'])), axis=1)

# Статистика
print("\n" + "=" * 70)
print("РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ")
print("=" * 70)

total = len(df)
stats = df['category'].value_counts()

for cat, count in stats.items():
    pct = count / total * 100
    print(f"{cat:20} | {count:>10,} | {pct:>5.1f}%")

print("-" * 70)
print(f"{'ВСЕГО':20} | {total:>10,}")

# Считаем сколько осталось в other
other_count = stats.get('other', 0)
other_pct = other_count / total * 100
print(f"\n{'НЕКАТЕГОРИЗОВАНО':20} | {other_count:>10,} | {other_pct:>5.1f}%")

# Топ-50 некатегоризованных
if other_count > 0:
    print("\n" + "=" * 70)
    print("ТОП-50 НЕКАТЕГОРИЗОВАННЫХ (other)")
    print("=" * 70)

    other_df = df[df['category'] == 'other']
    other_counts = other_df['details'].value_counts().head(50)

    for details, count in other_counts.items():
        print(f"{count:>6} | {details[:60]}")

    # Сохраняем other в файл
    other_df['details'].value_counts().to_csv('other_details_v2.csv')
    print(f"\nВсе 'other' сохранены в other_details_v2.csv")