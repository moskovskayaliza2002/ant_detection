import torch
import torchvision
import xml.etree.ElementTree as ET
import os
import numpy as np
from torchvision.transforms import transforms
import random
import glob
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def read_boxes(directory, indexes):
    
    all_files = []
    
    for f in os.scandir(directory):
        if f.is_file() and f.path.split('.')[-1].lower() == 'txt':
            all_files.append(f.path)
    
    all_dicts = []
    
    for txt_file in all_files:

        list_with_single_boxes = []
        list_of_labels = []
        
        with open(txt_file) as f:
            for line in f:
                xmin, ymin, xmax, ymax = map(int, line[:-1].split(" "))
                list_with_single_boxes.append([xmin, ymin, xmax, ymax])
                list_of_labels.append(1)
        
        out_dict = {'boxes':torch.tensor(list_with_single_boxes), 'labels': torch.tensor(list_of_labels)}        
        all_dicts.append(out_dict)
    
    batch_dicts = []
    for i in indexes:
        batch_dicts.append(all_dicts[i])
    
    return batch_dicts


def read_input_image(directory, indexes):
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


def image_transform(or_im):
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return normalize(to_tensor(or_im))


def save_model(model, path):
    torch.save(model.state_dict(), path)
    
    
def training(model, models_path, train_dir):
    
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = os.path.join(models_path, time_str)
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    saving_path = models_path + time_str
    reg_loss = []
    class_loss = []
    epoch = []
    fig, axs = plt.subplots(2)
    min_reg_loss = float('inf')
    for i in range(num_epoch):
        indexes = random.sample(range(dir_size), batch_size)
        target = read_boxes(train_dir, indexes)
        im_input = read_input_image(train_dir, indexes)
        results = model(im_input, target)
        reg_loss.append(results['bbox_regression'].item())
        
        if reg_loss[-1] < min_reg_loss:
            min_reg_loss = reg_loss[-1]
            save_model(model, saving_path + '/best_model.pt')
            
        class_loss.append(results['classification'].item())
        epoch.append(i)
        axs[0].cla()
        axs[1].cla()
        axs[0].plot(epoch, reg_loss, color = 'green', label = 'bbox_regression') 
        axs[1].plot(epoch, class_loss, color = 'red', label = 'classification')
        axs[0].legend(loc='best')
        axs[1].legend(loc='best')
        plt.xlabel('no. epoch')
        plt.pause(0.0001)
    save_model(model, saving_path + '/full_trained_model.pt')
    plt.savefig(saving_path + '/loss.png')
    plt.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_epoch', nargs='?', default=100, help="Enter number of epoch to train.", type=int)
    parser.add_argument('batch_size', nargs='?', default=16, help="Enter the batch size", type=int)
    parser.add_argument('train_dir', nargs='?', default='/home/lizamoscow/ant_detection/croped_data', help="Specify training directory.", type=str)
    parser.add_argument('models_path', nargs='?', default='/home/lizamoscow/ant_detection/models/', help="Specify directory where models will be saved.", type=str)
    args = parser.parse_args()
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    train_dir = args.train_dir
    dir_size = int(len(glob.glob(train_dir + '/*')) / 2)
    models_path = args.models_path
    model = torchvision.models.detection.ssd300_vgg16(num_classes = 2, pretrained_backbone = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    training(model, models_path, train_dir)
