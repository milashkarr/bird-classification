import pandas as pd
import yaml
import json
from pathlib import Path


def calculate_data_metrics():
    print("Расчет метрик данных...")

    # Загружаем информацию о классах
    with open('data/processed/class_info.yaml', 'r') as f:
        class_info = yaml.safe_load(f)

    # Загружаем разделенные данные
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    # Рассчитываем метрики
    metrics = {
        'dataset_info': {
            'total_classes': len(class_info['class_names']),
            'class_names': class_info['class_names'],
            'total_images': len(train_df) + len(val_df) + len(test_df)
        },
        'split_sizes': {
            'train': int(len(train_df)),
            'validation': int(len(val_df)),
            'test': int(len(test_df))
        },
        'split_percentages': {
            'train': round(100 * len(train_df) / (len(train_df) + len(val_df) + len(test_df)), 1),
            'validation': round(100 * len(val_df) / (len(train_df) + len(val_df) + len(test_df)), 1),
            'test': round(100 * len(test_df) / (len(train_df) + len(val_df) + len(test_df)), 1)
        },
        'class_distribution': {}
    }

    # Распределение по классам
    for class_name in class_info['class_names']:
        train_count = len(train_df[train_df['class_name'] == class_name])
        val_count = len(val_df[val_df['class_name'] == class_name])
        test_count = len(test_df[test_df['class_name'] == class_name])

        metrics['class_distribution'][class_name] = {
            'train': int(train_count),
            'validation': int(val_count),
            'test': int(test_count),
            'total': int(train_count + val_count + test_count)
        }

    # Сохраняем метрики
    Path('metrics').mkdir(exist_ok=True)
    with open('metrics/data_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Метрики данных сохранены в metrics/data_metrics.json")

    # Выводим сводку
    print(f"\nСводка данных:")
    print(f"Классы: {len(class_info['class_names'])}")
    print(f"Всего изображений: {metrics['dataset_info']['total_images']}")
    print(f"Train: {metrics['split_sizes']['train']} ({metrics['split_percentages']['train']}%)")
    print(f"Validation: {metrics['split_sizes']['validation']} ({metrics['split_percentages']['validation']}%)")
    print(f"Test: {metrics['split_sizes']['test']} ({metrics['split_percentages']['test']}%)")

    return metrics


if __name__ == "__main__":
    calculate_data_metrics()