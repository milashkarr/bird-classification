import pandas as pd
import os

# Словарь для соответствия файлов и названий классов
# (используем английские названия для удобства кодирования)
file_class_mapping = {
    'goldfinch.csv': 'goldfinch',  # щегол
    'penochka.csv': 'chiffchaff',  # пеночка
    'popolzen.csv': 'nuthatch',  # поползень
    'swallow.csv': 'swallow',  # ласточка
    'zyablik.csv': 'brambling'  # зяблик
}

# Путь к папке с вашими CSV файлами
csv_folder = '.'  # если файлы в текущей папке, иначе укажите путь

# Список для хранения всех данных
all_data = []

# Читаем каждый файл и добавляем метку класса
for filename, class_name in file_class_mapping.items():
    filepath = os.path.join(csv_folder, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['class_name'] = class_name  # добавляем столбец с названием класса
        all_data.append(df)
        print(f"Загружен {filename}: {len(df)} записей")
    else:
        print(f"Файл {filename} не найден!")

# Объединяем все данные
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)

    # Сохраняем в один файл
    output_file = 'inat_birds_combined.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nОбъединенный файл создан: {output_file}")
    print(f"Всего записей: {len(combined_df)}")

    # Показываем статистику по классам
    print("\nКоличество записей по классам:")
    print(combined_df['class_name'].value_counts())

    # Показываем первые строки
    print("\nПервые 3 строки данных:")
    print(combined_df.head(3))
else:
    print("Нет данных для объединения!")