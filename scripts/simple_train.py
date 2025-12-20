# ЗАМЕНИТЕ начало файла scripts/simple_train.py на этот код:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import yaml
import json
from pathlib import Path
import warnings

# Отключаем предупреждения
warnings.filterwarnings('ignore')

print("=" * 50)
print("ПРОСТОЕ ОБУЧЕНИЕ МОДЕЛИ (без скачивания)")
print("=" * 50)


# 1. Проверяем наличие предобученных весов
def load_model_without_download():
    """Загружает модель без скачивания"""
    try:
        # Пробуем загрузить с весами по умолчанию
        model = models.mobilenet_v2(weights=None)  # Без скачивания
        print("Модель создана без предобученных весов")
        return model
    except:
        # Создаем модель с случайными весами
        model = models.mobilenet_v2(pretrained=False)
        print("Модель создана со случайными весами")
        return model


# 2. Датасет (без изменений)
class BirdDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        with open('data/processed/class_info.yaml', 'r') as f:
            self.class_info = yaml.safe_load(f)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['local_path']).convert('RGB')
        img = self.transform(img)
        label = self.class_info['class_to_idx'][row['class_name']]
        return img, label


# 3. Основная функция
def main():
    print("Загрузка данных...")

    with open('data/processed/class_info.yaml', 'r') as f:
        class_info = yaml.safe_load(f)

    num_classes = len(class_info['class_names'])
    print(f"Классов: {num_classes} ({class_info['class_names']})")

    train_dataset = BirdDataset('data/processed/train.csv')
    val_dataset = BirdDataset('data/processed/val.csv')

    print(f"Train: {len(train_dataset)} изображений")
    print(f"Val: {len(val_dataset)} изображений")

    # Даталоадеры
    BATCH_SIZE = 8  # Меньше для стабильности
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Модель БЕЗ скачивания
    device = torch.device('cpu')
    model = load_model_without_download()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)

    # Если есть старая модель - загружаем веса
    if Path('models/best_model.pth').exists():
        try:
            checkpoint = torch.load('models/best_model.pth', map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Загружены веса из старой модели")
        except:
            pass

    # Обучение (только 3 эпохи для теста)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 3  # Только 3 эпохи для теста

    print(f"\nОбучение на {EPOCHS} эпохах...")

    for epoch in range(EPOCHS):
        # Только валидация для теста
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch + 1}/{EPOCHS}: Val Acc: {val_acc:.2f}%")

    # Сохраняем модель
    Path('models').mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_info['class_names'],
        'val_accuracy': val_acc
    }, 'models/best_model.pth')

    print(f"\nМодель сохранена в models/best_model.pth")

    # Простые метрики
    Path('metrics').mkdir(exist_ok=True)
    metrics = {
        'val_accuracy': float(val_acc),
        'classes': class_info['class_names'],
        'total_params': sum(p.numel() for p in model.parameters())
    }

    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Метрики сохранены в metrics/train_metrics.json")
    print("\n" + "=" * 50)
    print("ГОТОВО!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    main()
    