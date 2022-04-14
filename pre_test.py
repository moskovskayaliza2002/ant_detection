import torch
import torchvision
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import random
import glob
import matplotlib.pyplot as plt
from torchvision.io import read_image 
from torchvision.utils import draw_bounding_boxes
from datetime import datetime
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

# read one image boxes
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
    

# common image transforms
def image_transform(or_im):
    #resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return normalize(to_tensor(or_im))

def read_image(directory, indexes):
    im_input = []
    for f in os.scandir(directory):
        if f.is_file() and f.path.split('.')[-1].lower() == 'jpg':
            original_image = Image.open(f.path, mode='r')
            original_image = original_image.convert('RGB')
            transf_image = image_transform(original_image)
            im_input.append(transf_image)
            
    batch_im = []
    for i in indexes:
        batch_im.append(im_input[i])
    
    return batch_im

def save_model(model, path):
    torch.save(model.state_dict(), path)
    

def load_model(path):
    model = torchvision.models.detection.ssd300_vgg16(num_classes = 2, pretrained_backbone = True)
    model.load_state_dict(torch.load(path))

# draw real and predicted bboxes    
def test_model(model_path, image_path):
    load_model(model_path)
    model.eval()
    #xml_path = '.'.join(image_path.split('-')[:-1]) + '.xml'
    xml_path = image_path[:-3] + 'xml'
    print(xml_path)
    real_boxes, real_labels = read_xml(xml_path)
    real_boxes = torch.tensor(real_boxes, dtype=torch.int)
    input_im = Image.open(image_path, mode='r')
    input_im = input_im.convert('RGB')
    input_im = image_transform(input_im)
    results = model([input_im])
    #print(results[:])
    pred_boxes = [bx['boxes'] for bx in results[:]]
    scores = [bx['scores'] for bx in results[:]]
    indexes = []
    
    print(pred_boxes)
    pred_boxes = torch.tensor(pred_boxes, dtype=torch.int)
    pred_labels = results[:][0]['labels']
    
    img = read_image(image_path)
    
    for i in pred_boxes:
        img = draw_bounding_boxes(img, pred_boxes[i], width=3, colors=(255,0,0))
    
    for i in real_boxes:
        img = raw_bounding_boxes(img, real_boxes[i], width=3, colors=(0,255,0))
        
    img = torchvision.transforms.ToPILImage()(img) 
    img.show()
    
if __name__ == '__main__':
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = os.path.join('/home/ubuntu/ant_detection/models/', time_str)
    if not os.path.exists(dir):
        os.mkdir(dir)
    saving_path = '/home/ubuntu/ant_detection/models/' + time_str
    num_epoch = 30
    batch_size = 10
    train_dir = '/home/ubuntu/ant_detection/FILE0001'
    dir_size = int(len(glob.glob(train_dir + '/*')) / 2)
    
    model = torchvision.models.detection.ssd300_vgg16(num_classes = 2, pretrained_backbone = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #test_model(saving_path, '/home/ubuntu/ant_detection/FILE0001/FILE0001.MOV_snapshot_15.22.521.jpg')
    
    model.train()

    reg_loss = []
    class_loss = []
    epoch = []
    fig, axs = plt.subplots(2)
    min_reg_loss = float('inf')
    for i in range(num_epoch):
        indexes = random.sample(range(dir_size), batch_size)
        target = read_boxes(train_dir, indexes)
        im_input = read_image(train_dir, indexes)
        results = model(im_input, target)
        reg_loss.append(results['bbox_regression'].item())
        
        if reg_loss[-1] < min_reg_loss:
            min_reg_loss = reg_loss[-1]
            save_model(model, saving_path + '/best_model.pt')
            
        class_loss.append(results['classification'].item())
        epoch.append(i)
        plt.cla()
        axs[0].plot(epoch, reg_loss, color = 'green', label = 'bbox_regression') 
        axs[1].plot(epoch, class_loss, color = 'red', label = 'classification')
        axs[0].legend(loc='best')
        axs[1].legend(loc='best')
        plt.xlabel('no. epoch')
        plt.pause(0.0001)
    save_model(model, saving_path + '/full_trained_model.pt')
    plt.savefig(saving_path + '/loss.png')
    plt.show()
    
    
