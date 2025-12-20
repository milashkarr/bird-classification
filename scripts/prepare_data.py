import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import json
from pathlib import Path


def main():
    print("=" * 50)
    print("ПОДГОТОВКА ДАННЫХ")
    print("=" * 50)

    # 1. ПАРАМЕТРЫ
    SPLIT_RATIO = [0.7, 0.15, 0.15]  # По умолчанию

    # Пробуем прочитать params.yaml
    if os.path.exists("params.yaml"):
        try:
            with open("params.yaml", 'r') as f:
                params = yaml.safe_load(f)
                if 'prepare' in params and 'split_ratio' in params['prepare']:
                    SPLIT_RATIO = params['prepare']['split_ratio']
            print(f"Параметры загружены: split_ratio = {SPLIT_RATIO}")
        except Exception as e:
            print(f"Ошибка чтения params.yaml: {e}")

    # 2. ЗАГРУЗКА ДАННЫХ
    print("\nЗагрузка данных...")

    # Собираем все файлы из папок
    data_records = []
    raw_path = Path("data/raw")

    for class_folder in raw_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg"))

            for img_file in image_files:
                data_records.append({
                    'id': img_file.stem,
                    'class_name': class_name,
                    'local_path': str(img_file.absolute())
                })

    df = pd.DataFrame(data_records)
    print(f"Найдено изображений: {len(df)}")

    if len(df) == 0:
        print("ОШИБКА: Нет изображений в data/raw/")
        return False

    # 3. РАЗДЕЛЕНИЕ НА TRAIN/VAL/TEST
    print(f"\nРазделение данных (train/val/test): {SPLIT_RATIO}")

    train_dfs, val_dfs, test_dfs = [], [], []

    for class_name in df['class_name'].unique():
        class_df = df[df['class_name'] == class_name]

        if len(class_df) < 10:
            print(f"  {class_name}: {len(class_df)} - СЛИШКОМ МАЛО, пропускаем")
            continue

        # Простое разделение
        n_test = int(len(class_df) * SPLIT_RATIO[2])
        n_val = int(len(class_df) * SPLIT_RATIO[1])

        # Случайная выборка
        test = class_df.sample(n=n_test, random_state=42)
        remaining = class_df.drop(test.index)
        val = remaining.sample(n=n_val, random_state=42)
        train = remaining.drop(val.index)

        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

        print(f"  {class_name}: {len(class_df)} -> train:{len(train)} val:{len(val)} test:{len(test)}")

    # 4. СОХРАНЕНИЕ
    print("\nСохранение данных...")

    # Создаем папки
    Path("data/processed").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    # Объединяем
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

    # Сохраняем разделенные данные
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"\nИТОГИ:")
    print(f"  Train: {len(train_df)} изображений")
    print(f"  Val:   {len(val_df)} изображений")
    print(f"  Test:  {len(test_df)} изображений")
    print(f"  Всего: {len(train_df) + len(val_df) + len(test_df)} изображений")

    # 5. ИНФОРМАЦИЯ О КЛАССАХ
    classes = sorted(df['class_name'].unique().tolist())
    class_info = {
        'class_names': classes,
        'class_to_idx': {cls: idx for idx, cls in enumerate(classes)},
        'idx_to_class': {idx: cls for idx, cls in enumerate(classes)}
    }

    with open("data/processed/class_info.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(class_info, f, allow_unicode=True)

    print(f"\nКлассы ({len(classes)}): {', '.join(classes)}")

    # 6. МЕТРИКИ ДАННЫХ
    data_metrics = {
        'total_images': len(df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'classes': classes,
        'class_distribution': {cls: len(df[df['class_name'] == cls]) for cls in classes}
    }

    with open("metrics/data_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(data_metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    main()