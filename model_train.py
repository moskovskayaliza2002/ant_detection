import argparse
from model import ObjectDetector
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import glob
from torch.optim import Adam
from torchvision.io import read_image 
import random
from torchvision.transforms import transforms
import xml.etree.ElementTree as ET
import os
import torch
from torch import nn
import torchvision
import numpy as np
import cv2
from giou_loss import giou_loss
from torchvision.utils import draw_bounding_boxes, draw_keypoints
from PIL import Image
from PIL import ImageDraw, ImageFont
from torchvision.transforms.functional import hflip, vflip, to_tensor


def read_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
        
    list_with_single_boxes = []
    all_boxes = torch.tensor(1)
    list_of_labels = []
    height, width = read_im_size()

    for boxes in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text) / height
        xmin = int(boxes.find("bndbox/xmin").text) / width
        ymax = int(boxes.find("bndbox/ymax").text) / height
        xmax = int(boxes.find("bndbox/xmax").text) / width
        
        h = ymax - ymin
        w = xmax - xmin

        #list_with_single_boxes.append([xmin, ymin, xmax, ymax])
        single_box = torch.tensor([(xmin + xmax)/2, (ymin+ymax)/2]).float() #/ 224
        list_with_single_boxes.append(single_box)
        #list_with_single_boxes.append([[(xmin + xmax)/2, (ymin + ymax)/2]])
        list_of_labels.append(1)
        
    all_boxes = torch.stack(list_with_single_boxes)
    #print(f"all bboxes {all_boxes}")
        
    return all_boxes, list_of_labels


def read_im_size(path = '/home/ubuntu/ant_detection/FILE0001/FILE0001.MOV_snapshot_02.22.953.jpg'):
    im = cv2.imread(path)
    height = im.shape[0]
    width = im.shape[1]
    return height, width
    
    
def read_boxes(directory, indexes, max_objs):
    
    all_files = []
    all_boxes = []
    all_labels = []
    
    height, width = read_im_size()
    
    for f in os.scandir(directory):
        if f.is_file() and f.path.split('.')[-1].lower() == 'xml':
            all_files.append(f.path)
    
    #i = 0
    for i, xml_file in enumerate(all_files):
        if i not in indexes:
            #i+=1
            continue
        #i+=1
        tree = ET.parse(xml_file)
        root = tree.getroot()
        one_im_l = []
        one_im_b = torch.tensor(())
        
        one_im_labels = []
        one_im_boxes = []
        for boxes in root.iter('object'):

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text) * 320 / height
            xmin = int(boxes.find("bndbox/xmin").text) * 320 / width
            ymax = int(boxes.find("bndbox/ymax").text) * 320 / height
            xmax = int(boxes.find("bndbox/xmax").text) * 320 / width
            
            h = ymax - ymin
            w = xmax - xmin
            
            single_label = torch.tensor([1]).float()
            #single_box = torch.tensor([xmin, ymin, w, h]).float()
            single_box = torch.tensor([(xmin + xmax)/2, (ymin+ymax)/2]).float() / 320
            one_im_boxes.append(single_box)
            one_im_labels.append(single_label)

            if len(one_im_boxes) == max_objs:
                print("WARN! the number of real objects exceeds the maximum limit")
                break
            # break if cnt exeeds max_object
        # add to tensor values for max_object
        #if len(one_im_boxes) < max_objs:
            #for i in range(len(one_im_boxes),max_objs):
                ##one_im_boxes.append(torch.tensor([0,0,0,0]).float())
                #one_im_boxes.append(torch.tensor([0,0]).float())
                #one_im_labels.append(torch.tensor([0]).float())
                
        
        all_boxes.append(torch.stack(one_im_boxes))
        all_labels.append(torch.stack(one_im_labels))
        
    #all_boxes = torch.stack(all_boxes)
    #all_labels = torch.stack(all_labels)
    #all_labels = all_labels.view(-1, max_objs)
    
    #batch_list_l = []
    #batch_list_b = []
    #for i in indexes:
        #batch_list_b.append(all_boxes[i,:,:])
        #batch_list_l.append(all_labels[i,:])
        
    #batch_list_b = torch.stack(batch_list_b)
    #batch_list_l = torch.stack(batch_list_l)
    #batch_list_l = batch_list_l.view(-1, max_objs)
    
    #all_boxes = rescale(all_boxes)
    
    return all_boxes, all_labels

def rescale(bbox):
    for box in bbox:
        box = box / 320
    #print(bbox)
    return bbox


def aumentation(im_type, im):
    #nothing
    if im_type == 0:
        new_im = im
    #mirror h
    elif im_type == 1:
        new_im = hflip(im)
        #new_im = torch.flip(im, (1,))
        #img = torchvision.transforms.ToPILImage()(new_im)
        #img.show()
        
    #mirror v
    elif im_type == 2:
        new_im = vflip(im)
    #both
    else:
        new_im = hflip(vflip(im))
        
    return new_im


def bboxes_flip(bboxes, aug):
    new_bboxes = bboxes.copy()
    for i in range(len(aug)):
        #nothing
        if aug[i] == 0:
            new_bboxes[i] = bboxes[i]
        #mirror h
        elif aug[i] == 1:
            #new_bboxes[i,:,0] = torch.ones(bboxes.size()[2]) - bboxes[i,:,0]
            new_bboxes[i][:,0] = 1 - bboxes[i][:,0]
        #mirror v
        elif aug[i] == 2:
            #new_bboxes[i,:,1] = 1 - bboxes[i,:,1]
            new_bboxes[i][:,1] = 1 - bboxes[i][:,1]
        #both
        else:
            new_bboxes[i][:,0] = 1 - bboxes[i][:,0]
            new_bboxes[i][:,1] = 1 - bboxes[i][:,1]
            
    return new_bboxes

def image_transform(or_im):
    resize = transforms.Resize((320,320))
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return normalize(totensor(resize(or_im)))


def read_input_image(directory, indexes, flag):
    im_input = []
    for f in os.scandir(directory):
        if f.is_file() and f.path.split('.')[-1].lower() == 'jpg':
            original_image = Image.open(f.path, mode='r')
            original_image = original_image.convert('RGB')
            transf_image = image_transform(original_image)
            im_input.append(transf_image)
         
    if flag:
        aug_type = torch.randint(low=0,high=4, size=(len(indexes),))
    else:
        aug_type = torch.zeros(len(indexes), dtype=torch.uint8)
    batch_im = []
    for i, no in enumerate(indexes):
        batch_im.append(aumentation(aug_type[i], im_input[no]))
    
    return torch.stack(batch_im), aug_type


def save_model(model, path):

    torch.save(model.state_dict(), path)
    

def custom_loss(predC, predB, targetC, targetB, C = 1):
    classLoss = nn.BCELoss()(predC, targetC)
    bboxLoss = torch.sum(nn.MSELoss(reduction='none')(predB, targetB),dim = 2)
    #bboxLoss = giou_loss(predB, targetB, reduction='none')
    
    #print(f'bboxLoss size {bboxLoss.size()}')
    #print(f'targetC size {targetC.size()}')
    bboxLoss = torch.matmul(bboxLoss , targetC.T)
    totalLoss = classLoss + bboxLoss.mean() / C

    return totalLoss, classLoss, bboxLoss.mean() / C

# predC - tensor [batch, max_obj]
# predB - tensor [batch, max_obj, 2]
# targetC - list (x batch) of tensors [n_real_objects]
# targetB - list (x batch) of tensors [n_real_objects, 2]
# mse_thresh - threshold to count objects as positive
def best_point_loss(predC, predB, targetC, targetB, mse_thresh = 0.01, C = 100):
    MSE = nn.MSELoss(reduction='sum')
    BCE = nn.BCELoss(reduction='mean')
    BCE_nr = nn.BCELoss(reduction='none')
    mse_loss = 0
    bce_loss = 0
    # for each batch element construct matrix [max_obj, n_real_objects] of mse between predicted points to all reals
    for b in range(predB.size()[0]):
        pred = predB[b,:,:] # [max_obj, 2]
        targ = targetB[b] # [n_real_objects, 2]
        
        mse = torch.zeros(pred.size()[0], targ.size()[0])
        #print(pred, targ)
        for i in range(pred.size()[0]):
            for j in range(targ.size()[0]):
                mse[i,j] = MSE(pred[i], targ[j])
        #print(mse)
        maxes, max_inds = torch.min(mse, dim = 1)
        zeros = torch.zeros(pred.size()[0])
        mse_losses = torch.where(maxes < mse_thresh, maxes, zeros)
        #print(mse_losses)
                
        mean_non_zero = torch.nan_to_num(torch.sum(mse_losses) / torch.count_nonzero(mse_losses))                
        #mean = torch.mean(mse_losses)        
        #print(mean, mean_non_zero)
        mse_loss += mean_non_zero
        
        # supd dupa bce loss calc        
        
        bce_targets = (maxes < mse_thresh).float()
        
        pos_index = (maxes < mse_thresh)
        neg_index = (maxes >= mse_thresh)
        
        n_pos = torch.count_nonzero(pos_index)
        
        if n_pos == 0:
            bce_pos = 0                        
            
            bce_neg_hard = BCE(predC[b,:], torch.zeros(predC.size()[1]))
        else:
            bce_pos = BCE(predC[b,pos_index], torch.ones(n_pos))
            
            bce_neg_all = BCE_nr(predC[b,neg_index], torch.zeros(torch.count_nonzero(neg_index)))
            
            bce_neg_all_sorted, _ = torch.sort(bce_neg_all)
            
            bce_neg_hard = torch.mean(bce_neg_all_sorted[:n_pos*3])
        
        #bce_loss += BCE(predC[b,:], bce_targets)
        bce_loss += (bce_pos+bce_neg_hard)
    
    #mse_loss = mse_loss / C
    bce_loss = bce_loss / C
    total_loss = mse_loss + bce_loss
    return total_loss, bce_loss, mse_loss
    #return mse_loss, torch.zeros(1), mse_loss
    
    
def open_im(path = '/home/ubuntu/ant_detection/FILE0001/FILE0001.MOV_snapshot_02.22.953.jpg'):
    input_im = Image.open(path, mode='r')
    input_im = input_im.convert('RGB')
    input_im = image_transform(input_im)
    input_im = torch.unsqueeze(input_im, 0)
    return input_im
    
    
def mse_of_test_image(predC, predB, targetC, targetB):
    pred = torch.squeeze(predB)
    targ = torch.squeeze(targetB)
    
    MSE = nn.MSELoss(reduction='sum')
    mse = torch.zeros(pred.size()[0], targ.size()[0])
    for i in range(pred.size()[0]):
        for j in range(targ.size()[0]):
            mse[i,j] = MSE(pred[i], targ[j])
    #print('mse',mse)
    minies, min_inds = torch.min(mse, dim = 1)
    zeros = torch.zeros(pred.size()[0])
    #mse_losses = torch.where(minies < mse_thresh, minies, zeros)
    #print(f"minies {minies}")
    return minies
    
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        #m.weight.data.fill_(0.01)
        #bias = random.random()
        #bias = round(bias, 2)
        #m.bias.data.fill_(bias)
        size = m.bias.size()
        new_values = torch.FloatTensor(size).uniform_(-3, 3)
        m.bias.data = new_values
                

def train(num_epoch, batch_size, train_dir, models_path, lr, max_objects):
    image_path = '/home/ubuntu/ant_detection/FILE0001/FILE0001.MOV_snapshot_03.13.259.jpg'
    input_im = open_im()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs = []
    dir = os.path.join(models_path, time_str)
    if not os.path.exists(dir):
        os.mkdir(dir)
    saving_path = models_path + time_str
    model = ObjectDetector(max_objects)
    model.apply(init_weights)
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
        model.eval()
        with torch.no_grad():
            output_1 = model(input_im)
            xml_path = image_path[:-3] + 'xml'
            real_boxes_1, real_labels_1 = read_xml(xml_path)
            real_boxes_1 = real_boxes_1.unsqueeze(0)
            pred_boxes_1 = torch.squeeze(output_1[0])
            norm_pred_bboxes_1 = pred_boxes_1
            norm_real_bboxes = torch.clone(real_boxes_1)
            height_1, width_1 = read_im_size()
            a = pred_boxes_1[:,0] * width_1
            b = pred_boxes_1[:,1] * height_1
            pred_boxes_1 = torch.stack([a, b], 1)
            pred_boxes_1 = pred_boxes_1.unsqueeze(0) #1
            pred_labels_1 = output_1[1]
            mse_losses_1 =  mse_of_test_image(pred_labels_1, norm_pred_bboxes_1, real_labels_1, norm_real_bboxes)
            #print(f"predC {pred_labels_1} \npredB {norm_pred_bboxes} \ntargetC {real_labels_1} \ntargetB {real_boxes_1}")
            img = read_image(image_path)
            real_boxes_1[:, :, 0] = real_boxes_1[:, :, 0] * width_1
            real_boxes_1[:, :, 1] = real_boxes_1[:, :, 1] * height_1
            img = draw_keypoints(img, pred_boxes_1, width=3, colors=(255,0,0), radius = 5)
            img = draw_keypoints(img, real_boxes_1, width=3, colors=(0,255,0), radius = 4)
            
            img = torchvision.transforms.ToPILImage()(img)
            #print(f"predB {pred_boxes_1.size()}")
            for i in range(pred_boxes_1.size()[1]):
                ImageDraw.Draw(img).text((pred_boxes_1[0,i,0]+10, pred_boxes_1[0,i,1]),'(b){:.2f},{:.4f}'.format(pred_labels_1[0,i], mse_losses_1[i]),(255, 0, 0))
        
        model.train()
        aug_flag = True
        indexes = random.sample(range(dir_size), batch_size)
        bboxes, labels = read_boxes(train_dir, indexes, max_objects)
        images, aug_types = read_input_image(train_dir, indexes, aug_flag)
        aug_boxes = bboxes_flip(bboxes, aug_types)
        #(images, labels, bboxes) = (images.to(device), labels.to(device), bboxes.to(device))
        images = images.to(device)
        #print(images.size())
        predictions = model(images)
        #print(predictions[0])
        #print(f"predC {predictions[1].size()}, predB {predictions[0].size()}, targetC {labels.size()}, targetB {bboxes.size()}")
        totalLoss, classLoss, bboxLoss = best_point_loss(predictions[1].float(), predictions[0].float(), labels, aug_boxes)
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
        
        model.eval()
        with torch.no_grad():
            output_2 = model(input_im)
            pred_boxes_2 = torch.squeeze(output_2[0])
            height_2, width_2 = read_im_size()
            norm_pred_bboxes_2 = pred_boxes_2
            a = pred_boxes_2[:,0] * width_2
            b = pred_boxes_2[:,1] * height_2
            pred_boxes_2 = torch.stack([a, b], 1)
            pred_boxes_2 = pred_boxes_2.unsqueeze(0)
            pred_labels_2 = output_2[1]
            mse_losses_2 =  mse_of_test_image(pred_labels_2, norm_pred_bboxes_2, real_labels_1, norm_real_bboxes)
            
            
            transform = transforms.Compose([transforms.PILToTensor()])
            img = transform(img)
            
            img = draw_keypoints(img, pred_boxes_2, width=3, colors=(0,0,255), radius = 5)
            
            img = torchvision.transforms.ToPILImage()(img)
            
            for i in range(pred_boxes_2.size()[1]):
                ImageDraw.Draw(img).text((pred_boxes_2[0,i,0]+10, pred_boxes_2[0,i,1]),'(e){:.2f},{:.4f}'.format(pred_labels_2[0,i], mse_losses_2[i]),(0, 0, 255))
    
            img.show()
        model.train()
    
    plt.savefig(saving_path + '/loss.png')
    plt.show()
    save_model(model, saving_path + '/full_trained_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_epoch', nargs='?', default=400, help="Enter number of epoch to train.", type=int)
    parser.add_argument('batch_size', nargs='?', default=32, help="Enter the batch size", type=int)
    parser.add_argument('train_dir', nargs='?', default='/home/ubuntu/ant_detection/FILE0001', help="Specify training directory.", type=str)
    parser.add_argument('models_path', nargs='?', default='/home/ubuntu/ant_detection/models/', help="Specify directory where models will be saved.", type=str)
    parser.add_argument('learning_rate', nargs='?', default=1e-2, help="Enter learning rate for optimizer", type=float)
    parser.add_argument('max_objects', nargs='?', default=32, help="Enter maximum number of objects detected per image", type=int)

    args = parser.parse_args()
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    train_dir = args.train_dir
    models_path = args.models_path
    lr = args.learning_rate
    max_objects = args.max_objects
    
    train(num_epoch, batch_size, train_dir, models_path, lr, max_objects)
