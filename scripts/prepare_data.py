import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
from pathlib import Path


def prepare_data():
    # Читаем параметры
    with open('../params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    split_ratio = params['prepare']['split_ratio']

    # Читаем ИСПРАВЛЕННЫЙ CSV
    csv_path = '../data/image_paths_fixed.csv'
    if not os.path.exists(csv_path):
        csv_path = '../data/image_paths.csv'  # fallback

    df = pd.read_csv(csv_path)
    print(f"Загружен {csv_path}: {len(df)} записей")

    # Проверяем существование файлов
    existing_files = []
    for idx, row in df.iterrows():
        path_obj = Path(row['local_path'])
        if path_obj.exists():
            existing_files.append(True)
        else:
            # Пробуем альтернативный формат пути
            alt_path = Path(str(path_obj).replace('\\', '/'))
            if alt_path.exists():
                df.at[idx, 'local_path'] = str(alt_path)
                existing_files.append(True)
            else:
                existing_files.append(False)

    df_exists = df[existing_files].copy()
    print(f"\nНайдено существующих файлов: {len(df_exists)} из {len(df)}")

    if len(df_exists) == 0:
        print("ОШИБКА: Не найдено ни одного файла!")
        return

    # Разделяем данные по классам
    train_dfs, val_dfs, test_dfs = [], [], []

    for class_name in df_exists['class_name'].unique():
        class_df = df_exists[df_exists['class_name'] == class_name]

        print(f"Класс {class_name}: {len(class_df)} изображений")

        if len(class_df) < 5:
            print(f"  Пропускаем - слишком мало данных")
            continue

        # Разделяем данные этого класса
        try:
            train_val, test = train_test_split(
                class_df,
                test_size=split_ratio[2],
                random_state=42,
                stratify=class_df['class_name']
            )

            train, val = train_test_split(
                train_val,
                test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
                random_state=42,
                stratify=train_val['class_name']
            )

            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)

        except Exception as e:
            print(f"  Ошибка при разделении класса {class_name}: {e}")
            # Простое разделение если stratify не работает
            train_val, test = train_test_split(class_df, test_size=split_ratio[2], random_state=42)
            train, val = train_test_split(train_val, test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
                                          random_state=42)
            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)

    # Объединяем обратно
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

    # Сохраняем
    os.makedirs('data/processed', exist_ok=True)

    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)

    print("\n" + "=" * 50)
    print("ДАННЫЕ ПОДГОТОВЛЕНЫ:")
    print(f"  Train: {len(train_df)} изображений")
    print(f"  Val: {len(val_df)} изображений")
    print(f"  Test: {len(test_df)} изображений")

    # Сохраняем информацию о классах
    class_info = {
        'class_names': sorted(df_exists['class_name'].unique().tolist()),
        'class_to_idx': {cls: idx for idx, cls in enumerate(sorted(df_exists['class_name'].unique()))},
        'idx_to_class': {idx: cls for idx, cls in enumerate(sorted(df_exists['class_name'].unique()))},
        'total_images': len(df_exists),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }

    with open('data/processed/class_info.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(class_info, f, allow_unicode=True)

    print(f"\nКлассы: {class_info['class_names']}")

    # Создаем упрощенный CSV для быстрой загрузки
    df_exists[['local_path', 'class_name']].to_csv('data/processed/all_images.csv', index=False)

    print(f"\nСоздан файл data/processed/all_images.csv")


if __name__ == "__main__":
    prepare_data()
