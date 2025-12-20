import pandas as pd
from pathlib import Path

# Прочитаем CSV
df = pd.read_csv('data/image_paths.csv')

# Создадим абсолютные пути
absolute_paths = []
missing_files = 0
found_files = 0

for idx, row in df.iterrows():
    # Пробуем разные варианты путей
    possible_paths = [
        # Абсолютный путь
        Path(f"C:/Users/Lada/Documents/DS/naturalist/data/raw/{row['class_name']}/{row['id']}.jpg"),
        # Относительный путь из CSV
        Path(row['local_path']),
        # С заменой слешей
        Path(str(row['local_path']).replace('/', '\\')),
    ]

    found = False
    for path in possible_paths:
        if path.exists():
            absolute_paths.append(str(path))
            found_files += 1
            found = True
            break

    if not found:
        # Если файл не найден, оставляем оригинальный путь
        absolute_paths.append(row['local_path'])
        missing_files += 1
        print(f"Файл не найден: {row['class_name']}/{row['id']}.jpg")

# Обновляем DataFrame
df['local_path'] = absolute_paths

# Сохраняем исправленный CSV
df.to_csv('data/image_paths_absolute.csv', index=False)

print(f"\nСтатистика:")
print(f"  Всего записей: {len(df)}")
print(f"  Найдено файлов: {found_files}")
print(f"  Не найдено: {missing_files}")
print(f"\nИсправленный CSV сохранен: data/image_paths_absolute.csv")

# Проверим первые 3 пути
print("\nПроверка первых 3 путей:")
for i in range(3):
    path = Path(df.iloc[i]['local_path'])
    print(f"  {path.name}: {path.exists()}")
