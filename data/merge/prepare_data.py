import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml

# Создаем абсолютные пути
df = pd.read_csv('data/image_paths.csv')

print(f"Загружено записей: {len(df)}")

# Создаем абсолютные пути
absolute_paths = []
for idx, row in df.iterrows():
    abs_path = Path(f"C:/Users/Lada/Documents/DS/naturalist/data/raw/{row['class_name']}/{row['id']}.jpg")
    if abs_path.exists():
        absolute_paths.append(str(abs_path))
    else:
        # Пробуем jpeg
        abs_path = Path(f"C:/Users/Lada/Documents/DS/naturalist/data/raw/{row['class_name']}/{row['id']}.jpeg")
        if abs_path.exists():
            absolute_paths.append(str(abs_path))
        else:
            absolute_paths.append(row['local_path'])

df['local_path'] = absolute_paths

# Сохраняем с абсолютными путями
df.to_csv('data/image_paths_absolute.csv', index=False)
print(f"Создан файл с абсолютными путями: {len(df)} записей")

# Разделяем данные
train_dfs, val_dfs, test_dfs = [], [], []

for class_name in df['class_name'].unique():
    class_df = df[df['class_name'] == class_name]

    print(f"Класс {class_name}: {len(class_df)} изображений")

    if len(class_df) < 5:
        print(f"  Пропускаем - мало данных")
        continue

    # Разделяем
    train_val, test = train_test_split(
        class_df,
        test_size=0.15,
        random_state=42
    )

    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=42
    )

    train_dfs.append(train)
    val_dfs.append(val)
    test_dfs.append(test)

# Объединяем
train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

# Сохраняем
output_dir = Path("data/processed")
output_dir.mkdir(exist_ok=True)

train_df.to_csv(output_dir / 'train.csv', index=False)
val_df.to_csv(output_dir / 'val.csv', index=False)
test_df.to_csv(output_dir / 'test.csv', index=False)

print(f"\nДанные подготовлены:")
print(f"  Train: {len(train_df)} изображений")
print(f"  Val: {len(val_df)} изображений")
print(f"  Test: {len(test_df)} изображений")

# Информация о классах
class_info = {
    'class_names': sorted(df['class_name'].unique().tolist()),
    'class_to_idx': {cls: idx for idx, cls in enumerate(sorted(df['class_name'].unique()))},
    'idx_to_class': {idx: cls for idx, cls in enumerate(sorted(df['class_name'].unique()))}
}

with open(output_dir / 'class_info.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(class_info, f, allow_unicode=True)

print(f"\nКлассы: {class_info['class_names']}")