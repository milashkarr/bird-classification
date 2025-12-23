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
import yaml

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

EPOCHS = params['train']['epochs']
BATCH_SIZE = params['train']['batch_size']
LEARNING_RATE = params['train']['learning_rate']
WEIGHT_DECAY = params['train']['weight_decay']
IMAGE_SIZE = params['train']['image_size']
RESIZE_SIZE = params['train']['resize_size']
MODEL_NAME = params['model']['name']
NUM_CLASSES = params['model']['num_classes']
PRETRAINED = params['model']['pretrained']
FREEZE_LAYERS = params['train']['freeze_layers']
UNFREEZE_LAST_N = params['train']['unfreeze_last_n_layers']

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
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []

    os.makedirs("metrics/tmp", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # аугментация
    train_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE),
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

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_data.classes)
    print(f"Классов: {num_classes} ({train_data.classes})")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    print("\nСоздание модели...")

    if MODEL_NAME == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None)
    else:
        raise ValueError(f"Модель {MODEL_NAME} не поддерживается")

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
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("\nОбучение...")
    best_acc = 0.0
    history = []

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        epoch_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = epoch_train_loss / len(train_loader)

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

        print(
            f'Epoch {epoch + 1}/{EPOCHS}: Train: {train_acc:.1f}%, Val: {val_acc:.1f}%, Loss: {avg_train_loss:.4f}')

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

        history_metrics = {
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': avg_train_loss
        }

        with open(f"metrics/tmp/epoch_{epoch + 1}.json", "w", encoding="utf-8") as f:
            json.dump(history_metrics, f, ensure_ascii=False, indent=2)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(avg_train_loss)

    print("\nТестирование лучшей модели...")

    checkpoint = torch.load("models/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_data = BirdDataset("data/processed/test.csv", val_transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

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

    history_data = []
    for i in range(EPOCHS):
        history_data.append({
            'epoch': i + 1,
            'train_acc': train_acc_history[i],
            'val_acc': val_acc_history[i],
            'train_loss': train_loss_history[i]
        })

    with open("metrics/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    final_metrics = {
        'best_val_acc': best_acc,
        'test_acc': test_acc,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'classes': train_data.classes,
        'model': MODEL_NAME,
        'params': params
    }

    with open("metrics/train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nЛучшая точность на валидации: {best_acc:.2f}%")
    print(f"Точность на тесте: {test_acc:.2f}%")
    print(f"\nВсе модели сохранены в папке models/")
    print(f"Лучшая модель: models/best_model.pth ({best_acc:.1f}%)")


if __name__ == "__main__":
    main()