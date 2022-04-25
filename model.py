import torch
from torch import nn
import torchvision

class ObjectDetector(nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()
        self.baseModel = torchvision.models.mobilenet_v3_large(pretrained = True)
        
        self.regressor = nn.Sequential(nn.Linear(1000, 128), #self.baseModel.fc.in_features
                                       nn.ReLU(), 
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, 4),
                                       nn.Sigmoid())
                                       
        self.classifier = nn.Sequential(nn.BatchNorm1d(1000),
                                     nn.Linear(1000, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 1),
                                     nn.Sigmoid())
        
    def forward(self, x):
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        return (bboxes, classLogits)
