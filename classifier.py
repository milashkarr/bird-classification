import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

class BirdClassifier(nn.Module):
    def __init__(self, num_classes=5, model_name='mobilenet_v2', pretrained=True):
        super().__init__()
        
        self.model_name = model_name
        
        if model_name == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_features, num_classes)
            )
            
        elif model_name == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(f"Модель {model_name} не поддерживается")
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        if self.model_name == 'mobilenet_v2':
            # Для MobileNetV2
            x = self.backbone.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x
        elif self.model_name == 'resnet18':
            # Для ResNet18
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x