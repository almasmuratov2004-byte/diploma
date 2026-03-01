import csv
import time
from supabase import create_client

# ===== НАСТРОЙКИ =====
SUPABASE_URL = "https://oebyxstruskskjasfwsc.supabase.co"
SUPABASE_KEY = "sb_publishable_gok3B5m8SU_AtC4TEhNERQ_aAjrDKdT"

OUTPUT_FILE = "all_transactions.csv"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_with_retry(offset, max_retries=5):
    """Загружает данные с повторными попытками"""
    for attempt in range(max_retries):
        try:
            batch = supabase.table("transactions") \
                .select("id, user_id, date, amount, operation, details") \
                .order("id") \
                .range(offset, offset + 999) \
                .execute().data
            return batch
        except Exception as e:
            print(f"  Ошибка на offset {offset}, попытка {attempt + 1}/{max_retries}: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def main():
    print("Выгрузка транзакций в CSV (с паузами)...")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'user_id', 'date', 'amount', 'operation', 'details'])
        
        offset = 0
        total = 0
        
        while True:
            batch = fetch_with_retry(offset)
            
            if batch is None:
                print(f"  Не удалось загрузить offset {offset}, пропускаем...")
                offset += 1000
                continue
            
            if not batch:
                break
            
            for t in batch:
                writer.writerow([
                    t['id'],
                    t['user_id'],
                    t['date'],
                    t['amount'],
                    t['operation'],
                    t['details']
                ])
            
            total += len(batch)
            offset += 1000
            
            if offset % 50000 == 0:
                print(f"  Выгружено: {total:,}")
            
            # Пауза между запросами
            time.sleep(0.2)
        
        print(f"\nГотово! Выгружено {total:,} транзакций в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()