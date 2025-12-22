import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os
from pathlib import Path
import json

class BirdDataset:
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.classes = sorted(self.df['class_name'].unique())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['local_path']

        if not Path(img_path).exists():
            filename = Path(img_path).name
            img_path = f"data/raw/{row['class_name']}/{filename}"

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.classes.index(row['class_name'])
        return image, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # аугментация
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\nЗагрузка данных...")
    train_data = BirdDataset("data/processed/train.csv", train_transform)
    val_data = BirdDataset("data/processed/val.csv", val_transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    num_classes = len(train_data.classes)
    print(f"Классов: {num_classes} ({train_data.classes})")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    print("\nСоздание модели...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Замораживаем слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем последние слои
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Классификатор
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.0005, weight_decay=1e-4)

    print("\nОбучение...")
    best_acc = 0.0
    history = []

    for epoch in range(30):
        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        history.append((train_acc, val_acc))

        print(f'Epoch {epoch + 1}/30: Train: {train_acc:.1f}%, Val: {val_acc:.1f}%')

        if val_acc > 65:
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'epoch': epoch,
                'classes': train_data.classes
            }, f"models/model_epoch{epoch + 1}_acc{val_acc:.1f}.pth")
            print(f'  -> Сохранена модель {val_acc:.1f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'epoch': epoch,
                'classes': train_data.classes
            }, "models/best_model.pth")

    print(f"\nЛучшая точность: {best_acc:.2f}%")

    print("\nТестирование лучшей модели...")
    test_data = BirdDataset("data/processed/test.csv", val_transform)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

    metrics = {
        'best_val_acc': best_acc,
        'test_acc': test_acc,
        'epochs': 30,
        'classes': train_data.classes,
        'model': 'ResNet18_optimal',
        'note': 'Возврат к параметрам которые давали 67%'
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nГотово!")
    print(f"\nВсе модели сохранены в папке models/")
    print(f"Лучшая модель: models/best_model.pth ({best_acc:.1f}%)")


if __name__ == "__main__":
    main()