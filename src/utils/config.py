from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    # Пути
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    
    # Параметры данных
    image_size: tuple = (224, 224)
    batch_size: int = 32
    num_workers: int = 2
    
    # Параметры модели
    model_name: str = "mobilenet_v2"
    num_classes: int = 5
    pretrained: bool = True
    
    # Параметры обучения
    learning_rate: float = 0.001
    epochs: int = 10
    patience: int = 5 

    device: str = "cpu" 
    
    @classmethod
    def from_yaml(cls, path: Path):
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        
        config = cls()
        
        if 'train' in params:
            config.batch_size = params['train'].get('batch_size', config.batch_size)
            config.epochs = params['train'].get('epochs', config.epochs)
            config.learning_rate = params['train'].get('learning_rate', config.learning_rate)
            config.model_name = params['train'].get('model_name', config.model_name)
        
        return config