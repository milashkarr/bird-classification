import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import json
from pathlib import Path

print("=" * 50)
print("ПРОСТАЯ ПОДГОТОВКА ДАННЫХ")
print("=" * 50)

# 1. Собираем все файлы
data = []
raw_path = Path("data/raw")

for class_folder in raw_path.iterdir():
    if class_folder.is_dir():
        class_name = class_folder.name
        for img_file in class_folder.glob("*.jpg"):
            data.append({
                'id': img_file.stem,
                'class_name': class_name,
                'local_path': str(img_file.absolute())
            })

df = pd.DataFrame(data)
print(f"Найдено изображений: {len(df)}")

if len(df) == 0:
    print("ОШИБКА: Нет изображений!")
    exit(1)

# 2. Разделяем данные (70/15/15)
print("\nРазделение данных...")
train_dfs, val_dfs, test_dfs = [], [], []

for class_name in df['class_name'].unique():
    class_df = df[df['class_name'] == class_name]

    if len(class_df) >= 10:
        # Простое разделение
        n_test = int(len(class_df) * 0.15)
        n_val = int(len(class_df) * 0.15)

        test = class_df.sample(n=n_test, random_state=42)
        remaining = class_df.drop(test.index)
        val = remaining.sample(n=n_val, random_state=42)
        train = remaining.drop(val.index)

        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

        print(f"  {class_name}: {len(class_df)} -> train:{len(train)} val:{len(val)} test:{len(test)}")
    else:
        print(f"  {class_name}: {len(class_df)} - СЛИШКОМ МАЛО, пропускаем")

# 3. Сохраняем
print("\nСохранение данных...")
Path("data/processed").mkdir(exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print(f"\nИТОГИ:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")
print(f"  Всего: {len(train_df) + len(val_df) + len(test_df)}")

# 4. Информация о классах
classes = sorted(df['class_name'].unique().tolist())
class_info = {
    'class_names': classes,
    'class_to_idx': {cls: idx for idx, cls in enumerate(classes)},
    'idx_to_class': {idx: cls for idx, cls in enumerate(classes)}
}

with open("data/processed/class_info.yaml", 'w', encoding='utf-8') as f:
    yaml.dump(class_info, f, allow_unicode=True)

print(f"\nКлассы: {classes}")

# 5. Метрики данных
data_metrics = {
    'total_images': len(df),
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df),
    'classes': classes
}

with open("metrics/data_metrics.json", 'w', encoding='utf-8') as f:
    json.dump(data_metrics, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 50)
print("ГОТОВО!")
print("=" * 50)