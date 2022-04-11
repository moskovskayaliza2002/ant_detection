import torch
import torchvision
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

# return FloatTensor of boxes for each file in directory
def read_boxes(directory):
    
    all_files = []
    
    for f in os.scandir(directory):
        if f.is_file() and f.path.split('.')[-1].lower() == 'xml':
            all_files.append(f.path)
       
    list_with_all_boxes = []
    
    for xml_file in all_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for boxes in root.iter('object'):

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)

            list_with_single_boxes = [xmin, ymin, xmax, ymax]
            list_with_all_boxes.append(list_with_single_boxes)
        

    return torch.FloatTensor(list_with_all_boxes)

# common image transforms
def image_transform(or_im):
    #resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return normalize(to_tensor(or_im))

def read_labels(N):
    
    return torch.tensor(np.full((N,), 1))


if __name__ == '__main__':
    model = torchvision.models.detection.ssd300_vgg16(num_classes = 2, pretrained_backbone = True)
    train_dir = '/home/ubuntu/ant_detection/FILE0001'
    boxes = read_boxes(train_dir)
    labels = read_labels(boxes.size(dim=0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #target = list({boxes, labels})
    target = {'boxes': boxes, 'labels': labels}
    list_input = []
    for f in os.scandir(train_dir):
        if f.is_file() and f.path.split('.')[-1].lower() == 'jpg':
            original_image = Image.open(f.path, mode='r')
            original_image = original_image.convert('RGB')
            transf_image = image_transform(original_image)
            list_input.append(transf_image)
    
    model.train()
    #list_input = list_input.to(device)
    list_input = [i.to(device) for i in list_input] # (batch_size (N), 3, 300, 300)
    boxes = [b.to(device) for b in boxes]
    labels = [l.to(device) for l in labels]
    #output = model(list_input).squeeze() # predict mode
    results = model(list_input, target)
    
    
