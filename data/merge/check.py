import pandas as pd

# Загружаем объединенный файл
df = pd.read_csv('inat_birds_combined.csv')

print("=== ОБЗОР ДАННЫХ ===")
print(f"Всего записей: {len(df)}")
print(f"Колонки: {list(df.columns)}")

print("\n=== ПРОВЕРКА НА ДУБЛИКАТЫ ===")
duplicates = df.duplicated(subset=['id']).sum()
print(f"Найдено дубликатов по ID: {duplicates}")

print("\n=== СТАТИСТИКА ПО КЛАССАМ ===")
class_stats = df['class_name'].value_counts()
print(class_stats)

print("\n=== ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n=== ПРОВЕРКА ЛИЦЕНЗИЙ ===")
if 'license' in df.columns:
    print(df['license'].value_counts())
else:
    print("Столбец 'license' отсутствует - это может быть проблемой для легального использования!")

print("\n=== ПРОВЕРКА КАЧЕСТВА (quality_grade) ===")
if 'quality_grade' in df.columns:
    print(df['quality_grade'].value_counts())
else:
    print("Столбец 'quality_grade' отсутствует")