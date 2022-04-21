import torch
import torchvision
from torch import nn

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
    
def custom_loss(predC, predB, targetC, targetB, C = 1):
    classLoss = nn.BCELoss()(predC, targetC)
    bboxLoss = torch.sum(nn.MSELoss(reduction='none')(predB, targetB),dim = 1)    
    bboxLoss = torch.matmul(bboxLoss , targetC)
    totalLoss = classLoss + bboxLoss.mean() / C
    print(classLoss, bboxLoss)
    return totalLoss
    

    
if __name__ == '__main__':
    
    Batch = 12
    test_input = torch.rand(Batch, 3, 224, 224)
    
    model = ObjectDetector()
    
    test_output = model(test_input)
    bboxes = torch.rand(Batch, 4)
    labels = torch.randint(high = 2, size=(Batch, 1)).float()
    print(labels)
    loss = custom_loss(test_output[1], test_output[0], labels, bboxes)
    print(loss)
