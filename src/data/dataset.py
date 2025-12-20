import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
import yaml

class BirdDataset(Dataset):
    def __init__(self, csv_path, transform=None, class_info_path=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        if class_info_path and Path(class_info_path).exists():
            with open(class_info_path, 'r') as f:
                class_info = yaml.safe_load(f)
            self.class_to_idx = class_info['class_to_idx']
            self.idx_to_class = class_info['idx_to_class']
            self.class_names = class_info['class_names']
        else:
            # mapping классов
            self.class_names = sorted(self.df['class_name'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.class_names)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['local_path']
        class_name = row['class_name']
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[class_name]
        
        return image, label
    
    def get_class_distribution(self):
        return self.df['class_name'].value_counts()
    