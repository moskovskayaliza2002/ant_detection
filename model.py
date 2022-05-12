import torch
from torch import nn
import torchvision

class ObjectDetector(nn.Module):
    def __init__(self, max_detections):
        super(ObjectDetector, self).__init__()
        self.max_detections = max_detections
        #self.baseModel = torchvision.models.mobilenet_v3_large(pretrained = True)
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained_backbone=True).backbone
        
        #print(model)
        #modules = list(model.children())[:-3]
        #backbone = nn.Sequential(*modules)
        self.baseModel = model#backbone
        
        self.upModel = nn.Sequential( #self.baseModel.fc.in_features
                                       nn.ReLU(), 
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, 3 * max_detections),
                                       nn.Sigmoid())
        
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                #m.weight.data.fill_(0.01)
                m.bias.data.fill_(0.7)
            
            
        self.upModel.apply(init_weights)
        #self.classifier = nn.Sequential(nn.BatchNorm1d(128),
                                     #nn.Linear(672, 512),
                                     #nn.ReLU(),
                                     #nn.Dropout(),
                                     #nn.Linear(512, 512),
                                     #nn.ReLU(),
                                     #nn.Dropout(),
                                     #nn.Linear(512, 1 * max_detections),
                                     #nn.Sigmoid())
        
    def forward(self, x):
        features = self.baseModel(x)
        #print(features.keys())
        features = torch.squeeze(features['5'])
        #print(features.size())
        #print("*************************************\n\n\n\n\n\n")
        #bboxes = self.regressor(features)
        #classLogits = self.classifier(features)
        #bboxes = bboxes.view(-1, self.max_detections, 2)
        #classLogits = classLogits.view(-1, self.max_detections)
        #return (bboxes, classLogits)
        output = self.upModel(features)
        output = output.view(-1, self.max_detections, 3)
        classifier = output[:,:,0]
        regressor = output[:,:,1:]
        #print("class\n", classifier.size())
        #print("reg\n", regressor.size())
        #return output[:,], output[:,1:]
        return (regressor, classifier)

if __name__ == '__main__':
    model = ObjectDetector(5)
    model.eval()
    batch_size = 15
    device = 'cpu'
    inputs = torch.rand(batch_size,3, 320, 320)#[torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
    output = model(inputs)
    pred_boxes = output[0]
    pred_labels = output[1]
    print(pred_labels.size(), pred_boxes.size())
    #print(pred_boxes)
