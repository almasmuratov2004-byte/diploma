import os

# Текущая рабочая директория
cwd = os.getcwd()
print(f"Текущая директория: {cwd}\n")

# Список всех файлов и папок
for root, dirs, files in os.walk(cwd):
    print(f"Папка: {root}")
    if dirs:
        print("  Подпапки:", dirs)
    if files:
        print("  Файлы:", files)
    print("-"*40)