import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
