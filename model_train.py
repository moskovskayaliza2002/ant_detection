import argparse
from model import ObjectDetector
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import glob
from torch.optim import Adam
import random
from torchvision.transforms import transforms
import xml.etree.ElementTree as ET
import os
import torch
from torch import nn
import torchvision
import numpy as np

def read_boxes(directory, indexes, max_objs=10):
    
    all_files = []
    all_boxes = []#torch.tensor(())
    all_labels = torch.tensor(())
    
    for f in os.scandir(directory):
        if f.is_file() and f.path.split('.')[-1].lower() == 'xml':
            all_files.append(f.path)
    for xml_file in all_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        one_im_l = []
        one_im_b = torch.tensor(())
        
        some_boxes = []
        for boxes in root.iter('object'):

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            
            single_box = torch.tensor([xmin, ymin, xmax, ymax]).float()
            some_boxes.append(single_box)
            one_im_b = torch.cat((one_im_b, single_box), 0)
            all_labels = torch.cat((all_labels, torch.tensor([1])), 0)
            
            if len(some_boxes) == max_objs:
                print("WARN! bla bla")
                break
            # brake if cnt exeeds max_object
        # add to tensor values for max_object
        for i in range(len(some_boxes),max_objs):
            some_boxes.append(torch.tensor([0,0,0,0]).float())
        
        
        all_boxes.append(torch.stack(some_boxes))
        
    all_boxes = torch.stack(all_boxes)
    print(all_boxes.size())
            
    batch_list_l = []
    batch_list_b = []
    for i in indexes:
        batch_list_b.append(all_boxes[i,:,:])
        batch_list_l.append(list_of_labels[i,:])

    return torch.FloatTensor(batch_list_b), torch.FloatTensor(batch_list_l)


def image_transform(or_im):
    resize = transforms.Resize((224,224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return normalize(to_tensor(resize(or_im)))


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
    
    return torch.stack(batch_im)


def save_model(model, path):

    torch.save(model.state_dict(), path)
    

def custom_loss(predC, predB, targetC, targetB, C = 10**9):
    #print(f"predC {predC}")
    #print(f"targetC {targetC}")
    classLoss = nn.BCELoss()(predC, targetC)
    #print(classLoss)
    bboxLoss = torch.sum(nn.MSELoss(reduction='none')(predB, targetB),dim = 1)    
    bboxLoss = torch.matmul(bboxLoss , targetC)
    totalLoss = C * classLoss + bboxLoss.mean()

    return totalLoss, C * classLoss, bboxLoss.mean()

def train(num_epoch, batch_size, train_dir, models_path, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs = []
    dir = os.path.join(models_path, time_str)
    if not os.path.exists(dir):
        os.mkdir(dir)
    saving_path = models_path + time_str
    model = ObjectDetector(5)
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    min_loss = float('inf')
    dir_size = int(len(glob.glob(train_dir + '/*')) / 2)
    totalTrainLoss = []
    classTrainLoss = []
    bboxTrainLoss = []
    plt.ion()
    plt.show(block = False)
    for epoch in range(num_epoch):
        model.train()
        indexes = random.sample(range(dir_size), batch_size)
        bboxes, labels = read_boxes(train_dir, indexes)
        images = read_input_image(train_dir, indexes)
        (images, labels, bboxes) = (images.to(device), labels.to(device), bboxes.to(device))
        print(images.size())
        predictions = model(images)
        print(f"predC {predictions[1].size()}, predB {predictions[0].size()}, targetC {labels.size()}, targetB {bboxes.size()}")
        totalLoss, classLoss, bboxLoss = custom_loss(predictions[1].float(), predictions[0].float(), labels, bboxes)
        classTrainLoss.append(round(classLoss.item(), 3))
        print(f"classLoss {classLoss}")
        print(f"bboxLoss {bboxLoss}")
        bboxTrainLoss.append(bboxLoss.item())
        if totalLoss < min_loss:
            min_loss = totalLoss
            save_model(model, saving_path + '/best_model.pt')
        
        totalTrainLoss.append(totalLoss.item())
        epochs.append(epoch)
        plt.cla()
        plt.xlabel("epoch")
        plt.plot(epochs, classTrainLoss, label="Classification", linestyle = '--', color = 'green')
        plt.plot(epochs, bboxTrainLoss, label="Regression", linestyle = '-.', color = 'red')
        plt.plot(epochs, totalTrainLoss, label="Total", linestyle = ':', color = 'darkblue')
        plt.legend(loc="best")
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.3)
        
        opt.zero_grad()
        totalLoss.backward()
        opt.step()
    
    plt.savefig(saving_path + '/loss.png')
    plt.show()
    save_model(model, saving_path + '/full_trained_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_epoch', nargs='?', default=10, help="Enter number of epoch to train.", type=int)
    parser.add_argument('batch_size', nargs='?', default=15, help="Enter the batch size", type=int)
    parser.add_argument('train_dir', nargs='?', default='/home/ubuntu/ant_detection/FILE0001', help="Specify training directory.", type=str)
    parser.add_argument('models_path', nargs='?', default='/home/ubuntu/ant_detection/models/', help="Specify directory where models will be saved.", type=str)
    parser.add_argument('learning_rate', nargs='?', default=1e-4, help="Enter learning rate for optimizer", type=int)
    args = parser.parse_args()
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    train_dir = args.train_dir
    models_path = args.models_path
    lr = args.learning_rate
    train(num_epoch, batch_size, train_dir, models_path, lr)
