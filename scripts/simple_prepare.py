import os
import pandas as pd
from pathlib import Path
import json
import random
from collections import defaultdict


def prepare_dataset():
    """Подготовка датасета"""
    print("=" * 50)
    print("ПОДГОТОВКА ДАТАСЕТА")
    print("=" * 50)

    # Путь к сырым данным
    raw_data_path = "data/raw"

    # Собираем все изображения по классам
    image_paths = defaultdict(list)

    for class_name in os.listdir(raw_data_path):
        class_path = os.path.join(raw_data_path, class_name)

        if os.path.isdir(class_path):
            # Собираем все изображения
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(class_path, img_name)
                    image_paths[class_name].append(img_path)

    print(f"Всего изображений: {sum(len(imgs) for imgs in image_paths.values())}")

    # Разделение на train/val/test
    train_data = []
    val_data = []
    test_data = []

    print("\nРазделение данных...")
    for class_name, images in image_paths.items():
        random.shuffle(images)
        total = len(images)

        # 70% train, 15% val, 15% test
        train_split = int(0.7 * total)
        val_split = int(0.85 * total)

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        print(f"  {class_name}: {total} -> train:{len(train_images)} val:{len(val_images)} test:{len(test_images)}")

        # Добавляем в train
        for img_path in train_images:
            train_data.append({
                'id': Path(img_path).stem,
                'class_name': class_name,
                'local_path': img_path,
                'label': list(image_paths.keys()).index(class_name)
            })

        # Добавляем в val
        for img_path in val_images:
            val_data.append({
                'id': Path(img_path).stem,
                'class_name': class_name,
                'local_path': img_path,
                'label': list(image_paths.keys()).index(class_name)
            })

        # Добавляем в test
        for img_path in test_images:
            test_data.append({
                'id': Path(img_path).stem,
                'class_name': class_name,
                'local_path': img_path,
                'label': list(image_paths.keys()).index(class_name)
            })

    # Создаем DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Сохраняем CSV файлы
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    # Сохраняем информацию о классах
    class_info = {
        'classes': list(image_paths.keys()),
        'class_to_idx': {class_name: idx for idx, class_name in enumerate(image_paths.keys())}
    }

    with open("data/processed/class_info.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(class_info, f, default_flow_style=False, allow_unicode=True)

    # Сохраняем метрики данных
    data_metrics = {
        'total_images': len(train_df) + len(val_df) + len(test_df),
        'train_count': len(train_df),
        'val_count': len(val_df),
        'test_count': len(test_df),
        'classes': list(image_paths.keys()),
        'class_distribution': {
            class_name: {
                'total': len(images),
                'train': len([d for d in train_data if d['class_name'] == class_name]),
                'val': len([d for d in val_data if d['class_name'] == class_name]),
                'test': len([d for d in test_data if d['class_name'] == class_name])
            }
            for class_name in image_paths.keys()
        }
    }

    with open("metrics/data_metrics.json", "w", encoding="utf-8") as f:
        json.dump(data_metrics, f, ensure_ascii=False, indent=2)

    print("\nСводка:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    print(f"  Всего: {len(train_df) + len(val_df) + len(test_df)}")

    print(f"\nКлассы: {list(image_paths.keys())}")

    print("\n" + "=" * 50)
    print("ГОТОВО!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    prepare_dataset()
