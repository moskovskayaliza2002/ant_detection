import torch
from torch import nn
import torchvision

class ObjectDetector(nn.Module):
    def __init__(self, max_detections):
        super(ObjectDetector, self).__init__()
        self.max_detections = max_detections
        self.baseModel = torchvision.models.mobilenet_v3_large(pretrained = True)
        
        self.regressor = nn.Sequential(nn.Linear(1000, 128), #self.baseModel.fc.in_features
                                       nn.ReLU(), 
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, 4 * max_detections),
                                       nn.ReLU())
                                       
        self.classifier = nn.Sequential(nn.BatchNorm1d(1000),
                                     nn.Linear(1000, 512),
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
        bboxes = bboxes.view(-1, self.max_detections, 4)
        classLogits = classLogits.view(-1, self.max_detections)
        return (bboxes, classLogits)

if __name__ == '__main__':
    model = ObjectDetector(5)
    model.eval()
    batch_size = 15
    input_im = torch.rand(batch_size, 3, 224, 224)
    output = model(input_im)
    pred_boxes = output[0]
    pred_labels = output[1]
    print(pred_labels.size(), pred_boxes.size())
