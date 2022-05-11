import torch
from torch import nn
import torchvision

class ObjectDetector(nn.Module):
    def __init__(self, max_detections):
        super(ObjectDetector, self).__init__()
        self.max_detections = max_detections
        #self.baseModel = torchvision.models.mobilenet_v3_large(pretrained = True)
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained_backbone=True)
        modules = list(model.children())[:-3]
        backbone = nn.Sequential(*modules)
        self.baseModel = backbone
        
        self.regressor = nn.Sequential(nn.Linear(672, 128), #self.baseModel.fc.in_features
                                       nn.ReLU(), 
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, 2 * max_detections),
                                       nn.Sigmoid())
                                       
        self.classifier = nn.Sequential(nn.BatchNorm1d(672),
                                     nn.Linear(672, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 1 * max_detections),
                                     nn.Sigmoid())
        
    def forward(self, x):
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        bboxes = bboxes.view(-1, self.max_detections, 2)
        classLogits = classLogits.view(-1, self.max_detections)
        return (bboxes, classLogits)

if __name__ == '__main__':
    model = ObjectDetector(5)
    model.eval()
    batch_size = 15
    device = 'cpu'
    inputs = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
    output = model(inputs)
    pred_boxes = output[0]
    pred_labels = output[1]
    print(pred_labels.size(), pred_boxes.size())
    print(pred_boxes)
