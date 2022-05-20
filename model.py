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
        
        self.conv = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1, 0), dilation=(1,1), bias=False)
        
        self.analyzer_c = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, max_detections),
            nn.Sigmoid())
        
        self.analyzer_r = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, max_detections),
            nn.Sigmoid())
        
        #self.conv.bias.data = torch.FloatTensor(self.conv.bias.size()).uniform_(-3, 3)
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
        #self.upModel = nn.Sequential(  nn.Linear(256*3, 128),#self.baseModel.fc.in_features
                                       #nn.ReLU(), 
                                       #nn.Linear(128, 64),
                                       #nn.ReLU(),
                                       #nn.Linear(64, 3 * max_detections),
                                       #nn.Sigmoid())
            
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
        features = torch.squeeze(features['5'])
        output = self.upModel(features)
        output = output.view(-1, self.max_detections, 3)
        classifier = output[:,:,0]
        regressor = output[:,:,1:]
        
        #features = self.baseModel(x)
        #features = features['3']
        #convLayer = self.conv(features)
        #classifier = (convLayer[:,:,0,0])
        #regressor = (convLayer[:,:,1:,0])
        #regressor = regressor.permute(0,2,1) # swap last and pre last
        #classifier = self.analyzer_c(classifier)
        #regressor = self.analyzer_r(regressor)
        #regressor = regressor.permute(0,2,1) # back
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
