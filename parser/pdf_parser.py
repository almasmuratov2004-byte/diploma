import re
import pdfplumber


def extract_text_from_pdf(path):
    """Извлекает весь текст из PDF"""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def parse_user_info(text):
    """Извлекает информацию о пользователе из текста выписки"""
    user = {}

    # ФИО с ИИН (формат справки)
    iin_pattern = re.compile(r"что\s+(.+?),\s*ИИН\s*(\d{12})")
    iin_match = iin_pattern.search(text)
    if iin_match:
        user["full_name"] = iin_match.group(1).strip()
        user["iin"] = iin_match.group(2)

    # ФИО из ВЫПИСКИ (формат Kaspi Gold)
    if "full_name" not in user:
        vypiska_pattern = re.compile(
            r"ВЫПИСКА\s+по Kaspi Gold за период.*?\n(\S+)\s*\n",
            re.DOTALL
        )
        vypiska_match = vypiska_pattern.search(text)
        if vypiska_match:
            user["full_name"] = vypiska_match.group(1).strip()

    # Период
    period_pattern = re.compile(r"за период с (\d{2}\.\d{2}\.\d{2}) по (\d{2}\.\d{2}\.\d{2})")
    period_match = period_pattern.search(text)
    if period_match:
        user["period_from"] = period_match.group(1)
        user["period_to"] = period_match.group(2)

    return user


def parse_transactions(text):
    """Парсит транзакции из текста выписки"""
    pattern = re.compile(r"(\d{2}\.\d{2}\.\d{2})\s*([+-])\s*([\d\s,]+)\s*₸\s*(\S+)\s*(.+)")
    records = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            date = match.group(1)
            sign = match.group(2)
            amount_str = match.group(3).replace(" ", "").replace(",", ".")
            amount = float(amount_str)
            if sign == "-":
                amount = -amount
            operation = match.group(4)
            details = match.group(5).strip()
            records.append({
                "date": date,
                "amount": amount,
                "operation": operation,
                "details": details
            })
    return records
