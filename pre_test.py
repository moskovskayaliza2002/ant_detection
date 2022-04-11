import torch
import torchvision
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import random
import glob
#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

# return FloatTensor of boxes for each file in directory
def read_boxes(directory, indexes):
    
    all_files = []
    
    for f in os.scandir(directory):
        if f.is_file() and f.path.split('.')[-1].lower() == 'xml':
            all_files.append(f.path)
       
    all_dicts = []
    
    
    for xml_file in all_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        

        list_with_single_boxes = []
        list_of_labels = []

        for boxes in root.iter('object'):

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)

            list_with_single_boxes.append([xmin, ymin, xmax, ymax])
            #list_with_all_boxes.append(list_with_single_boxes)
            list_of_labels.append(1)
        
        out_dict = {'boxes':torch.FloatTensor(list_with_single_boxes), 'labels': torch.tensor(list_of_labels)}        
        all_dicts.append(out_dict)
    
    batch_dicts = []
    for i in indexes:
        batch_dicts.append(all_dicts[i])
    
    return batch_dicts

# common image transforms
def image_transform(or_im):
    #resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return normalize(to_tensor(or_im))

def read_image(directory, indexes):
    im_input = []
    for f in os.scandir(train_dir):
        if f.is_file() and f.path.split('.')[-1].lower() == 'jpg':
            original_image = Image.open(f.path, mode='r')
            original_image = original_image.convert('RGB')
            transf_image = image_transform(original_image)
            im_input.append(transf_image)
            
    batch_im = []
    for i in indexes:
        batch_im.append(im_input[i])
    
    return batch_im

if __name__ == '__main__':
    num_epoch = 10
    batch_size = 10
    train_dir = '/home/ubuntu/ant_detection/FILE0001'
    dir_size = int(len(glob.glob(train_dir + '/*')) / 2)
    #indexes = random.sample(range(dir_size), batch_size)
    
    model = torchvision.models.detection.ssd300_vgg16(num_classes = 2, pretrained_backbone = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #target = list({boxes, labels})
    #target = read_boxes(train_dir, indexes)
    #im_input = read_image(train_dir, indexes)
    
    model.train()
    #im_input = im_input.to(device)
    #im_input = [i.to(device) for i in im_input] # (batch_size (N), 3, 300, 300)
    #boxes = [b.to(device) for b in boxes]
    #labels = [l.to(device) for l in labels]
    #output = model(im_input).squeeze() # predict mode
    for _ in range(num_epoch):
        indexes = random.sample(range(dir_size), batch_size)
        target = read_boxes(train_dir, indexes)
        im_input = read_image(train_dir, indexes)
        results = model(im_input, target)
        print(results)
    
