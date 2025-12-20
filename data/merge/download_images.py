import pandas as pd
import requests
import os
from tqdm import tqdm
import time


def download_images(csv_path='inat_birds_combined.csv', output_dir='data/raw', max_per_class=100):
    """
    Скачивает изображения из CSV файла
    """
    # Читаем данные
    df = pd.read_csv(csv_path)
    print(f"Всего записей в CSV: {len(df)}")

    # Создаем папки для каждого класса
    for class_name in df['class_name'].unique():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # Счетчики
    downloaded = 0
    errors = 0
    skipped = 0

    # Ограничиваем количество изображений на класс
    class_counts = {}

    # Проходим по всем записям
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Скачивание изображений"):
        class_name = row['class_name']

        # Ограничиваем количество на класс
        if class_name not in class_counts:
            class_counts[class_name] = 0

        if class_counts[class_name] >= max_per_class:
            skipped += 1
            continue

        # Формируем путь для сохранения
        image_id = str(row['id'])
        filename = f"{image_id}.jpg"
        save_path = os.path.join(output_dir, class_name, filename)

        # Пропускаем, если уже скачано
        if os.path.exists(save_path):
            skipped += 1
            continue

        # Скачиваем изображение
        try:
            response = requests.get(row['image_url'], timeout=10)
            response.raise_for_status()  # Проверяем на ошибки

            # Сохраняем изображение
            with open(save_path, 'wb') as f:
                f.write(response.content)

            downloaded += 1
            class_counts[class_name] += 1

            # Небольшая задержка, чтобы не перегружать сервер
            time.sleep(0.1)

        except Exception as e:
            print(f"\nОшибка при скачивании {row['image_url']}: {e}")
            errors += 1

    # Выводим статистику
    print("\n" + "=" * 50)
    print("СТАТИСТИКА СКАЧИВАНИЯ:")
    print(f"Успешно скачано: {downloaded}")
    print(f"С ошибками: {errors}")
    print(f"Пропущено (уже есть/лимит): {skipped}")

    print("\nПо классам:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} изображений")

    # Сохраняем информацию о скачанных файлах в новый CSV
    print("\nСоздаю файл с путями к изображениям...")
    create_image_paths_csv(df, output_dir, 'data/image_paths.csv')


def create_image_paths_csv(df, image_dir, output_csv):
    """
    Создает CSV файл с путями к локальным изображениям
    """
    records = []

    for _, row in df.iterrows():
        image_id = str(row['id'])
        class_name = row['class_name']
        local_path = os.path.join(image_dir, class_name, f"{image_id}.jpg")

        # Проверяем, существует ли файл
        if os.path.exists(local_path):
            records.append({
                'id': row['id'],
                'class_name': class_name,
                'scientific_name': row['scientific_name'],
                'common_name': row['common_name'],
                'image_url': row['image_url'],
                'local_path': local_path,
                'quality_grade': row['quality_grade']
            })

    # Создаем DataFrame и сохраняем
    paths_df = pd.DataFrame(records)
    paths_df.to_csv(output_csv, index=False)
    print(f"Создан файл: {output_csv}")
    print(f"Записей с существующими изображениями: {len(paths_df)}")

    return paths_df


if __name__ == "__main__":
    # Установите меньшее число для теста, например 20
    download_images(max_per_class=100)  # Можно поставить 20 для быстрого теста