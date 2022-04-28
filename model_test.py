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
from model import ObjectDetector

from model_train import image_transform
import argparse
import cv2



def read_im_size(path = '/home/ubuntu/ant_detection/FILE0001/FILE0001.MOV_snapshot_02.22.953.jpg'):
    im = cv2.imread(path)
    height = im.shape[0]
    width = im.shape[1]
    return height, width


def load_model(path, max_objects):
    model = torchvision.models.detection.ssd300_vgg16(num_classes = 2, pretrained_backbone = True)
    model = ObjectDetector(max_objects)
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
        
        h = ymax - ymin
        w = xmax - xmin

        list_with_single_boxes.append([xmin, ymin, xmax, ymax])
        list_of_labels.append(1)
        
    return list_with_single_boxes, list_of_labels


# draw real and predicted bboxes    
def test_model(model_path, image_path, max_objects):
    model = load_model(model_path, max_objects)
    model.eval()
    xml_path = image_path[:-3] + 'xml'
    print(xml_path)
    real_boxes, real_labels = read_xml(xml_path)
    real_boxes = torch.tensor(real_boxes, dtype=torch.int)
    print(real_boxes.size())
    input_im = Image.open(image_path, mode='r')
    input_im = input_im.convert('RGB')
    input_im = image_transform(input_im)
    input_im = torch.unsqueeze(input_im, 0)
    with torch.no_grad():
        results = model(input_im)
    
    #pred_boxes = results[0]
    pred_boxes = torch.squeeze(results[0])
    print(pred_boxes)
    height, width = read_im_size()
    a = pred_boxes[:,0] * height
    b = pred_boxes[:,1] * width
    c = pred_boxes[:,2] * height
    d = pred_boxes[:,3] * width
    x2 = c - a 
    y2 = d - b
    pred_boxes = torch.stack([a, b, x2, y2], 1)
    pred_labels = results[1]
    indexes = []
    print(pred_boxes.size())
    print(pred_boxes)
    print(pred_labels)
    #pred_boxes = pred_boxes[0].long()
    #pred_labels = results[:][0]['labels']
    
    img = read_image(image_path)
    
    img = draw_bounding_boxes(img, pred_boxes, width=3, colors=(255,0,0))
    img = draw_bounding_boxes(img, real_boxes, width=3, colors=(0,255,0))
        
    img = torchvision.transforms.ToPILImage()(img) 
    img
    img.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('load_path', nargs='?', default='/home/ubuntu/ant_detection/models/20220428-145111/best_model.pt', help="Specify full path to model to load.", type=str)
    parser.add_argument('image_path', nargs='?', default='/home/ubuntu/ant_detection/FILE0001/FILE0001.MOV_snapshot_15.22.521.jpg', help="Specify full path to image you going to test", type=str)
    parser.add_argument('max_objects', nargs='?', default=10, help="Enter maximum number of objects detected per image", type=int)
    args = parser.parse_args()
    
    load_path = args.load_path
    image_path = args.image_path
    max_objects = args.max_objects
    test_model(load_path, image_path, max_objects) 
