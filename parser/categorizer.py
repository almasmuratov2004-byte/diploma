import re
import json
import requests
from collections import Counter

from config import AI_API_KEY, AI_MODEL, AI_API_URL


def categorize(details, categories_dict):
    """Определяет категорию транзакции по ключевым словам"""
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


def recategorize_transactions(transactions, categories):
    """Категоризирует все транзакции, возвращает статистику и нераспознанные"""
    category_stats = {}
    uncategorized = []
    for t in transactions:
        cat = categorize(t['details'], categories)
        t['category'] = cat
        category_stats.setdefault(cat, {"count": 0, "sum": 0})
        category_stats[cat]["count"] += 1
        category_stats[cat]["sum"] += abs(t["amount"])
        if cat == "other":
            uncategorized.append(t)
    return category_stats, uncategorized


def ai_categorize(top_words, categories_keys):
    """Отправляет нераспознанные слова в Claude API для категоризации"""
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
    response = requests.post(AI_API_URL, headers=headers, json=payload)
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
        print("Не удалось распарсить JSON от AI:")
        print(text)
        result_json = {}
    return result_json


def update_categories(categories, ai_result):
    """Обновляет словарь категорий результатами от AI"""
    for cat, words in ai_result.items():
        if cat in categories:
            for w in words:
                if w not in categories[cat]:
                    categories[cat].append(w)
        else:
            categories[cat] = words
    return categories
