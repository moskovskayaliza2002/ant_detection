import torch
import torchvision
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image 
from torchvision.utils import draw_bounding_boxes
from datetime import datetime

from train import image_transform
import argparse

def load_model(path):
    model = torchvision.models.detection.ssd300_vgg16(num_classes = 2, pretrained_backbone = True)
    model.load_state_dict(torch.load(path))
    return model
    
  
def read_xml(path):
    tree = ET.parse(path)
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
        list_of_labels.append(1)
        
    return list_with_single_boxes, list_of_labels


# draw real and predicted bboxes    
def test_model(model_path, image_path):
    model = load_model(model_path)
    model.eval()
    xml_path = image_path[:-3] + 'xml'
    print(xml_path)
    real_boxes, real_labels = read_xml(xml_path)
    real_boxes = torch.tensor(real_boxes, dtype=torch.int)
    input_im = Image.open(image_path, mode='r')
    input_im = input_im.convert('RGB')
    input_im = image_transform(input_im)
    results = model([input_im])
    pred_boxes = [bx['boxes'] for bx in results[:]]
    scores = [bx['scores'] for bx in results[:]]
    indexes = []
    
    pred_boxes = pred_boxes[0].long()
    pred_labels = results[:][0]['labels']
    
    img = read_image(image_path)
    
    img = draw_bounding_boxes(img, pred_boxes, width=3, colors=(255,0,0))
    img = draw_bounding_boxes(img, real_boxes, width=3, colors=(0,255,0))
        
    img = torchvision.transforms.ToPILImage()(img) 
    img
    img.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('load_path', nargs='?', default='/home/lizamoscow/ant_detection/models/20220516-194408/best_model.pt', help="Specify full path to model to load.", type=str)
    parser.add_argument('image_path', nargs='?', default='/home/lizamoscow/ant_detection/FILE0001/FILE0001.MOV_snapshot_15.22.521.jpg', help="Specify full path to image you going to test", type=str)
    args = parser.parse_args()
    
    load_path = args.load_path
    image_path = args.image_path
    test_model(load_path, image_path)

