import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

class Trainer:
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
        
        return total_loss/len(train_loader), 100.*correct/total
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss/len(val_loader), 100.*correct/total
    
    def train(self, train_loader, val_loader):
        print(f"Начинаем обучение на устройстве: {self.device}")
        print(f"Размер батча: {self.config.batch_size}")
        print(f"Эпох: {self.config.epochs}")
        
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            print(f"\nЭпоха {epoch+1}/{self.config.epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            self.scheduler.step(val_loss)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_model.pth')
                patience_counter = 0
                print(f"Сохранена лучшая модель с точностью: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"Нет улучшения {patience_counter}/{self.config.patience}")
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping на эпохе {epoch+1}")
                break
        
        self.save_model('last_model.pth')
        
        return self.history
    
    def save_model(self, filename):
        model_path = self.config.model_dir / filename
        self.config.model_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, model_path)
    
    def save_history(self, filename='training_history.json'):
        history_path = Path('metrics') / filename
        Path('metrics').mkdir(exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)