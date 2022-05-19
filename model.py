import torch
from torch import nn
import torchvision

class ObjectDetector(nn.Module):
    def __init__(self, max_detections):
        super(ObjectDetector, self).__init__()
        self.max_detections = max_detections
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained_backbone=True).backbone
        
        self.baseModel = model
        
        self.upModel = nn.Sequential(  nn.ReLU(), 
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, 3 * max_detections),
                                       nn.Sigmoid())
         
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0.7)
        
        self.upModel.apply(init_weights)
        
        
    def forward(self, x):
        features = self.baseModel(x)
        features = torch.squeeze(features['5'])
        output = self.upModel(features)
        output = output.view(-1, self.max_detections, 3)
        classifier = output[:,:,0]
        regressor = output[:,:,1:]
        
        return (regressor, classifier)

if __name__ == '__main__':
    model = ObjectDetector(5)
    model.eval()
    batch_size = 15
    device = 'cpu'
    inputs = torch.rand(batch_size,3, 320, 320)
    output = model(inputs)
    pred_boxes = output[0]
    pred_labels = output[1]
    print(pred_labels.size(), pred_boxes.size())
