import torch.nn as nn
from torchvision import models

def initialize_model(num_classes, use_pretrained=True):
    # Load ResNet-18 model pre-trained on ImageNet
    model = models.resnet18(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Replace the final layer for our number of classes
    
    return model
