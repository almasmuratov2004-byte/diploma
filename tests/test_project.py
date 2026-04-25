import sys
import os
import pytest

# Добавляем папку parser в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'parser'))

from categorizer import categorize, recategorize_transactions
from feature_engineering import extract_features


# ─────────────────────────────────────────────
# Тесты для categorizer.py
# ─────────────────────────────────────────────

CATEGORIES = {
    "betting": ["olimp", "1xbet", "melbet", "ставка"],
    "utility": ["kegoc", "водоканал", "газ", "свет"],
    "credit": ["kaspi кредит", "jusan кредит", "погашение кредита"],
    "cafes": ["mcdonalds", "kfc", "burger king", "starbucks"],
    "transport": ["яндекс такси", "uber", "bolt"],
    "shops": ["магазин", "market", "store"],
}


class TestCategorize:

    def test_betting_keyword(self):
        """Транзакция с ключевым словом 'olimp' → категория betting"""
        result = categorize("Пополнение OLIMP казино", CATEGORIES)
        assert result == "betting"

    def test_utility_keyword(self):
        """Транзакция с 'водоканал' → категория utility"""
        result = categorize("Оплата Водоканал г.Алматы", CATEGORIES)
        assert result == "utility"

    def test_cafe_keyword(self):
        """Транзакция с 'kfc' → категория cafes"""
        result = categorize("KFC Достык", CATEGORIES)
        assert result == "cafes"

    def test_ip_prefix_returns_small_business(self):
        """Транзакция начинается с 'ИП ' → small_business"""
        result = categorize("ИП Иванов Строительство", CATEGORIES)
        assert result == "small_business"

    def test_unknown_transaction_returns_other(self):
        """Неизвестная транзакция → other"""
        result = categorize("Неизвестный получатель XYZ123", CATEGORIES)
        assert result == "other"

    def test_case_insensitive(self):
        """Категоризация не зависит от регистра"""
        result = categorize("MELBET пополнение", CATEGORIES)
        assert result == "betting"


class TestRecategorizeTransactions:

    def test_returns_stats_and_uncategorized(self):
        """recategorize_transactions возвращает статистику и список other"""
        transactions = [
            {"details": "Оплата Водоканал", "amount": -3500},
            {"details": "Неизвестно", "amount": -100},
        ]
        stats, uncategorized = recategorize_transactions(transactions, CATEGORIES)
        assert "utility" in stats
        assert len(uncategorized) == 1

    def test_category_assigned_to_transaction(self):
        """Транзакция получает поле category после обработки"""
        transactions = [{"details": "KFC Mega", "amount": -2000}]
        recategorize_transactions(transactions, CATEGORIES)
        assert transactions[0]["category"] == "cafes"

    def test_stats_count_and_sum(self):
        """Статистика правильно считает count и sum"""
        transactions = [
            {"details": "Оплата Водоканал", "amount": -1000},
            {"details": "KEGOC оплата", "amount": -500},
        ]
        stats, _ = recategorize_transactions(transactions, CATEGORIES)
        assert stats["utility"]["count"] == 2
        assert stats["utility"]["sum"] == 1500


# ─────────────────────────────────────────────
# Тесты для feature_engineering.py
# ─────────────────────────────────────────────

class TestExtractFeatures:

    def _make_transactions(self):
        return [
            {"amount": 150000, "category": "other"},       # доход
            {"amount": -5000,  "category": "cafes"},        # расход кафе
            {"amount": -3000,  "category": "utility"},      # расход коммуналка
            {"amount": -10000, "category": "betting"},      # расход ставки
            {"amount": -20000, "category": "credit"},       # платёж по кредиту
            {"amount": 50000,  "category": "credit"},       # взятый кредит
        ]

    def test_total_income(self):
        """total_income = сумма всех положительных транзакций"""
        transactions = self._make_transactions()
        features, _, _ = extract_features(transactions, CATEGORIES)
        assert features["total_income"] == 200000.0

    def test_total_expense(self):
        """total_expense = сумма всех отрицательных транзакций"""
        transactions = self._make_transactions()
        features, _, _ = extract_features(transactions, CATEGORIES)
        assert features["total_expense"] == 38000.0

    def test_transaction_count(self):
        """transaction_count = общее число транзакций"""
        transactions = self._make_transactions()
        features, _, _ = extract_features(transactions, CATEGORIES)
        assert features["transaction_count"] == 6

    def test_expense_to_income_ratio(self):
        """expense_to_income_ratio = total_expense / total_income"""
        transactions = self._make_transactions()
        features, _, _ = extract_features(transactions, CATEGORIES)
        expected = round(38000 / 200000, 4)
        assert features["expense_to_income_ratio"] == expected

    def test_credit_taken_and_payment_count(self):
        """credit_taken_count и credit_payment_count считаются корректно"""
        transactions = self._make_transactions()
        features, _, _ = extract_features(transactions, CATEGORIES)
        assert features["credit_taken_count"] == 1
        assert features["credit_payment_count"] == 1

    def test_empty_transactions(self):
        """Пустой список транзакций не вызывает ошибку"""
        features, counts, sums = extract_features([], CATEGORIES)
        assert features["total_income"] == 0
        assert features["total_expense"] == 0
        assert features["transaction_count"] == 0
        assert features["expense_to_income_ratio"] == 10.0  # fallback

    def test_avg_transaction(self):
        """avg_transaction считается как среднее от abs(amount)"""
        transactions = [
            {"amount": 100, "category": "other"},
            {"amount": -300, "category": "cafes"},
        ]
        features, _, _ = extract_features(transactions, CATEGORIES)
        assert features["avg_transaction"] == 200.0
