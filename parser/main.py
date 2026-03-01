import os
import json
from collections import Counter

from config import (
    PDF_PATH, CATEGORIES_FILE, TOP_N_UNCATEGORIZED,
    OTHER_THRESHOLD_PERCENT, MAX_ITERATIONS
)
from pdf_parser import extract_text_from_pdf, parse_user_info, parse_transactions
from categorizer import recategorize_transactions, ai_categorize, update_categories
from scoring import run_credit_scoring


def load_categories():
    """Загружает категории из JSON"""
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"Файл категорий не найден: {CATEGORIES_FILE}")
        return {}


def save_categories(categories):
    """Сохраняет категории в JSON"""
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"\nФайл категорий обновлён: {CATEGORIES_FILE}")


def run_audit(transactions, categories):
    """Цикл аудита категоризации с AI расширением"""
    total = len(transactions)
    iteration = 1
    category_stats = {}

    while True:
        print(f"\n--- Итерация {iteration} ---")
        category_stats, uncategorized = recategorize_transactions(transactions, categories)
        other_percent = len(uncategorized) / total * 100

        # Вывод статистики
        print("Распределение по категориям:\n")
        sorted_stats = sorted(category_stats.items(), key=lambda x: x[1]["count"], reverse=True)
        for cat, stats in sorted_stats:
            percent = stats["count"] / total * 100
            print(f"{cat:20} {stats['count']:5} шт {percent:6.2f}% {stats['sum']:15,.0f} ₸")

        print("\nНе распознано (other):", len(uncategorized), f"({other_percent:.2f}%)")

        if other_percent <= OTHER_THRESHOLD_PERCENT:
            print("\n✓ OTHER меньше порога. Цикл завершён.")
            break

        # Топ нераспознанных деталей
        uncategorized_details = [t['details'].lower() for t in uncategorized]
        top_uncategorized = [
            word for word, _ in Counter(uncategorized_details).most_common(TOP_N_UNCATEGORIZED)
        ]
        print(f"\nOTHER > {OTHER_THRESHOLD_PERCENT}% ({other_percent:.2f}%). Топ {TOP_N_UNCATEGORIZED} деталей:")
        for word in top_uncategorized:
            print("  ", word)

        # AI категоризация
        result_json = ai_categorize(top_uncategorized, list(categories.keys()))
        print("\nРезультат распределения от AI:")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))

        # Обновляем категории
        categories = update_categories(categories, result_json)

        print("\nОбновлённые категории после AI:")
        for cat, words in categories.items():
            print(f"{cat}: {len(words)} слов")

        save_categories(categories)

        iteration += 1
        if iteration > MAX_ITERATIONS:
            print(f"\n⚠ Достигнут лимит итераций ({MAX_ITERATIONS}). Продолжаем...")
            break

    return categories, category_stats


def main():
    if not os.path.exists(PDF_PATH):
        print(f"Файл не найден: {PDF_PATH}")
        return

    print("=" * 60)
    print("АУДИТ КАТЕГОРИЗАЦИИ ТРАНЗАКЦИЙ")
    print("=" * 60)

    # Загрузка
    categories = load_categories()
    print("\nКлючи из categories.json:")
    print(", ".join(categories.keys()))
    print("-" * 60)

    # Парсинг PDF
    print("Извлекаем текст из PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    user_info = parse_user_info(text)
    print(f"Клиент: {user_info.get('full_name', 'Неизвестно')}")
    print("Текст извлечён\n")

    print("Парсим транзакции...")
    transactions = parse_transactions(text)
    print(f"Найдено {len(transactions)} транзакций\n")

    # Аудит категоризации
    categories, category_stats = run_audit(transactions, categories)

    print("\n" + "=" * 60)
    print("АУДИТ ЗАВЕРШЁН")
    print("=" * 60)

    # Кредитный скоринг
    run_credit_scoring(transactions, categories, user_info, category_stats)

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)


if __name__ == "__main__":
    main()
