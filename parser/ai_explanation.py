import json
import requests

from config import AI_API_KEY, AI_MODEL, AI_API_URL


def _to_float(val):
    """Конвертирует numpy типы в обычный Python float"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def generate_explanation(user_info, risk_probability, pred_label,
                         features, sums, positive_factors, negative_factors,
                         transaction_count):
    """Генерирует человекочитаемое объяснение скоринга через Claude API"""

    print("  Подготовка данных для AI...")

    scoring_data = {
        "client": user_info.get("full_name", "Неизвестно"),
        "period": f"{user_info.get('period_from', '?')} — {user_info.get('period_to', '?')}",
        "transactions_analyzed": int(transaction_count),
        "risk_probability": round(_to_float(risk_probability), 1),
        "decision": "ВЫСОКИЙ РИСК" if pred_label == 1 else "НИЗКИЙ РИСК",
        "key_metrics": {
            "total_income": _to_float(features.get("total_income", 0)),
            "total_expense": _to_float(features.get("total_expense", 0)),
            "expense_to_income_ratio": _to_float(features.get("expense_to_income_ratio", 0)),
            "betting_expense_ratio": _to_float(features.get("betting_expense_ratio", 0)),
            "credit_expense_ratio": _to_float(features.get("credit_expense_ratio", 0)),
            "utility_count_ratio": _to_float(features.get("utility_count_ratio", 0)),
            "utility_expense_ratio": _to_float(features.get("utility_expense_ratio", 0)),
            "credit_taken_count": int(features.get("credit_taken_count", 0)),
            "credit_payment_count": int(features.get("credit_payment_count", 0)),
        },
        "category_sums": {
            "betting": _to_float(sums.get("betting", 0)),
            "credit": _to_float(sums.get("credit", 0)),
            "utility": _to_float(sums.get("utility", 0)),
            "cafes": _to_float(sums.get("cafes", 0)),
            "entertainment": _to_float(sums.get("entertainment", 0)),
            "shops": _to_float(sums.get("shops", 0)),
        },
        "shap_positive_factors": [
            {"feature": f,
             "shap_value": round(_to_float(s), 3),
             "value": round(_to_float(v), 4) if _to_float(v) < 1 else round(_to_float(v), 0)}
            for f, s, v in positive_factors
        ],
        "shap_negative_factors": [
            {"feature": f,
             "shap_value": round(_to_float(s), 3),
             "value": round(_to_float(v), 4) if _to_float(v) < 1 else round(_to_float(v), 0)}
            for f, s, v in negative_factors
        ],
    }

    print(f"  ✓ Данные подготовлены: {len(scoring_data)} полей")

    try:
        json_str = json.dumps(scoring_data, ensure_ascii=False, indent=2)
        print(f"  ✓ JSON сериализован ({len(json_str)} символов)")
    except Exception as e:
        print(f"  ✗ Ошибка сериализации JSON: {e}")
        print(f"  Типы данных: {[(k, type(v).__name__) for k, v in scoring_data.items()]}")
        return f"Ошибка сериализации: {e}"

    prompt = f"""Ты — аналитик кредитного скоринга. Проанализируй результаты оценки клиента.

Данные:
```json
{json_str}
```

СТРОГИЕ ПРАВИЛА по факторам — НЕ ПУТАЙ:

В поле "shap_positive_factors" перечислены факторы которые ПОВЫШАЮТ вероятность дефолта.
→ Опиши их в секции "Что повышает риск".
Вот они: {', '.join(f['feature'] for f in scoring_data['shap_positive_factors'])}

В поле "shap_negative_factors" перечислены факторы которые СНИЖАЮТ вероятность дефолта.
→ Опиши их в секции "Что снижает риск".  
Вот они: {', '.join(f['feature'] for f in scoring_data['shap_negative_factors'])}

НЕ ПЕРЕМЕЩАЙ факторы между секциями. Каждый фактор описывай ТОЛЬКО в своей секции.

Структура ответа:

1. **Общий вердикт** — решение и вероятность, одним предложением.
2. **Финансовый профиль** — доходы/расходы, основные траты, ставки/кредиты.
3. **Что повышает риск** — ТОЛЬКО факторы из shap_positive_factors. Если фактор нелогичный (например низкая коммуналка повышает риск), объясни: "модель обучена на данных где у надёжных клиентов этот показатель выше".
4. **Что снижает риск** — ТОЛЬКО факторы из shap_negative_factors. Пример: отсутствие ставок = хорошо, низкая закредитованность = хорошо.
5. **Итоговая рекомендация** — краткий вывод.

Правила:
- Русский язык, кратко (до 2000 символов)
- Без технического жаргона (не упоминай XGBoost, SHAP, ML, модель)
- Конкретные цифры в тенге (₸)
- Если expense_to_income_ratio ≈ 1.0 — упомяни что внутренние переводы раздувают обе стороны
- Эмодзи для наглядности"""

    headers = {
        "x-api-key": AI_API_KEY,
        "Content-Type": "application/json",
        "Anthropic-Version": "2023-06-01",
    }

    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2500,
    }

    try:
        print(f"  Отправляем запрос в Claude API ({AI_MODEL})...")
        response = requests.post(AI_API_URL, headers=headers, json=payload, timeout=30)

        print(f"  Статус ответа: {response.status_code}")

        if response.status_code != 200:
            error_text = response.text[:500]
            print(f"  ✗ Ошибка API: {error_text}")
            return f"Ошибка API: {response.status_code} — {error_text}"

        content = response.json().get("content", [])
        text = ""
        for c in content:
            if c.get("type") == "text":
                text += c.get("text", "")

        print(f"  ✓ Ответ получен ({len(text)} символов)")
        return text

    except requests.exceptions.Timeout:
        print("  ✗ Таймаут запроса (30 сек)")
        return "Ошибка: таймаут запроса к AI"
    except requests.exceptions.ConnectionError:
        print("  ✗ Нет соединения с API")
        return "Ошибка: нет соединения с API"
    except Exception as e:
        print(f"  ✗ Неожиданная ошибка: {type(e).__name__}: {e}")
        return f"Ошибка при вызове AI: {str(e)}"