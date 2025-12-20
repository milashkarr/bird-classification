import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import sys


# ========== КОНФИГУРАЦИЯ ==========
class Config:
    def __init__(self):
        # Пути
        self.data_dir = Path("data")
        self.model_dir = Path("models")
        self.metrics_dir = Path("metrics")

        # Параметры данных
        self.image_size = (224, 224)
        self.batch_size = 16  # Меньше для CPU
        self.num_workers = 0  # 0 для Windows чтобы избежать ошибок

        # Параметры модели
        self.model_name = "mobilenet_v2"
        self.num_classes = 5
        self.pretrained = True

        # Параметры обучения
        self.learning_rate = 0.001
        self.epochs = 10
        self.patience = 5

        # Устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Создаем директории
        self.model_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)


# ========== ДАТАСЕТ ==========
class BirdDataset(Dataset):
    def __init__(self, csv_path, transform=None, class_info_path=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Загружаем информацию о классах
        if class_info_path and Path(class_info_path).exists():
            with open(class_info_path, 'r') as f:
                class_info = yaml.safe_load(f)
            self.class_to_idx = class_info['class_to_idx']
            self.idx_to_class = class_info['idx_to_class']
            self.class_names = class_info['class_names']
        else:
            # Создаем mapping классов
            self.class_names = sorted(self.df['class_name'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.class_names)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row['local_path'])
        class_name = row['class_name']

        # Загружаем изображение
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path}: {e}")
            # Возвращаем черное изображение такого же размера
            image = Image.new('RGB', (224, 224), color='black')

        # Применяем трансформации
        if self.transform:
            image = self.transform(image)

        # Метка класса
        label = self.class_to_idx[class_name]

        return image, label


# ========== ТРАНСФОРМАЦИИ ==========
def get_transforms(image_size=(224, 224), augment=False):
    """Возвращает трансформации для изображений"""
    if augment:
        # Для тренировочных данных
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # Для валидации и теста
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# ========== МОДЕЛЬ ==========
class BirdClassifier(nn.Module):
    def __init__(self, num_classes=5, model_name='mobilenet_v2', pretrained=True):
        super().__init__()
        self.model_name = model_name

        if model_name == 'mobilenet_v2':
            # MobileNetV2
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)

            # Заменяем классификатор
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_features, num_classes)
            )
        else:
            raise ValueError(f"Модель {model_name} не поддерживается")

    def forward(self, x):
        return self.backbone(x)


# ========== ТРЕНЕР ==========
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)

        # Функция потерь и оптимизатор
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # История
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        self.best_val_acc = 0.0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Статистика
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Обновляем progress bar
            pbar.set_postfix({
                'loss': f"{total_loss / len(train_loader):.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, data_loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / len(data_loader), 100. * correct / total

    def train(self, train_loader, val_loader):
        print(f"Устройство: {self.device}")
        print(f"Батч: {self.config.batch_size}, Эпох: {self.config.epochs}")

        patience_counter = 0

        for epoch in range(self.config.epochs):
            print(f"\nЭпоха {epoch + 1}/{self.config.epochs}")

            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader)

            # Валидация
            val_loss, val_acc = self.validate(val_loader)

            # Сохраняем историю
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Сохраняем лучшую модель
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_model.pth')
                patience_counter = 0
                print(f" Лучшая модель: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f" Нет улучшения {patience_counter}/{self.config.patience}")

            print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
            print(f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}%")

            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"\n⚠️ Early stopping")
                break

        # Сохраняем последнюю модель
        self.save_model('last_model.pth')

        return self.history

    def save_model(self, filename):
        """Сохраняет модель"""
        model_path = self.config.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': vars(self.config),
            'history': self.history
        }, model_path)


# ========== ОСНОВНАЯ ФУНКЦИЯ ==========
def main():
    # Конфигурация
    config = Config()
    print("=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)

    # Трансформации
    train_transform = get_transforms(config.image_size, augment=True)
    val_transform = get_transforms(config.image_size, augment=False)

    # Датасеты
    print("\nЗагрузка данных...")
    train_dataset = BirdDataset(
        'data/processed/train.csv',
        transform=train_transform,
        class_info_path='data/processed/class_info.yaml'
    )

    val_dataset = BirdDataset(
        'data/processed/val.csv',
        transform=val_transform,
        class_info_path='data/processed/class_info.yaml'
    )

    print(f"Train: {len(train_dataset)} изображений")
    print(f"Val: {len(val_dataset)} изображений")

    # Даталоадеры
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # Модель
    print(f"\nСоздание модели {config.model_name}...")
    model = BirdClassifier(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=config.pretrained
    )

    # Обучение
    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader)

    # Тестирование
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ")
    print("=" * 50)

    test_dataset = BirdDataset(
        'data/processed/test.csv',
        transform=val_transform,
        class_info_path='data/processed/class_info.yaml'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Сохраняем метрики
    final_metrics = {
        'best_val_accuracy': float(trainer.best_val_acc),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss)
    }

    metrics_path = config.metrics_dir / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"\nМетрики сохранены в {metrics_path}")

    # Простой график точности
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss during training')

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy during training')

        plt.tight_layout()
        plt.savefig('metrics/training_history.png', dpi=100)
        print("График обучения сохранен в metrics/training_history.png")

    except ImportError:
        print("Matplotlib не установлен, пропускаем создание графиков")

    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Лучшая точность на валидации: {trainer.best_val_acc:.2f}%")
    print(f"Точность на тесте: {test_acc:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()

    # ... существующий код ...

    # Сохраняем метрики обучения
    train_metrics = {
        'best_val_accuracy': float(trainer.best_val_acc),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_train_accuracy': float(history['train_acc'][-1]),
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']]
        }
    }

    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=4)

    print(f"\nМетрики обучения сохранены в metrics/train_metrics.json")