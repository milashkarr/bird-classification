"""
УПРОЩЕННЫЙ СКРИПТ ОБУЧЕНИЯ
"""

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
import numpy as np

class SimpleDataset(Dataset):
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

def train():
    print("="*50)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("="*50)
    
    # Параметры
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 0.001
    
    # Загрузка классов
    with open('data/processed/class_info.yaml', 'r') as f:
        class_info = yaml.safe_load(f)
    
    num_classes = len(class_info['class_names'])
    print(f"Классов: {num_classes}")
    print(f"Классы: {class_info['class_names']}")
    
    # Датасеты
    train_dataset = SimpleDataset('data/processed/train.csv')
    val_dataset = SimpleDataset('data/processed/val.csv')
    
    print(f"Train: {len(train_dataset)} изображений")
    print(f"Val: {len(val_dataset)} изображений")
    
    # Даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Модель
    device = torch.device('cpu')
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)
    
    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Обучение
    print("\nНачало обучения...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_info['class_names'],
                'val_accuracy': val_acc
            }, 'models/best_model.pth')
            print(f"  -> Сохранена лучшая модель (точность: {val_acc:.2f}%)")
    
    # Сохраняем последнюю модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_info['class_names'],
        'val_accuracy': val_acc
    }, 'models/last_model.pth')
    
    # Тестирование
    print("\nТестирование на тестовой выборке...")
    test_dataset = SimpleDataset('data/processed/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    test_correct, test_total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
    
    # Сохраняем метрики
    metrics = {
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'train_history': history
    }
    
    Path('metrics').mkdir(exist_ok=True)
    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nМетрики сохранены в metrics/train_metrics.json")
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*50)
    
    return True

if __name__ == "__main__":
    train()
