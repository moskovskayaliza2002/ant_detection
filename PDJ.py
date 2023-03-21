import numpy as np
import glob
import os
import argparse
from universal_intersection import find_kp, read_boxes
from RCNN_model import get_model
from universal_RCNN_test import one_image_test
import torch
import matplotlib.pyplot as plt
from mAP import intersection_over_union

def get_real_bboxes(im_path):
    #По пути изображения находит аннотации ограничивающих рамок
    number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
    im_root = im_path[:im_path.rfind('/')]
    test_root = im_root[:im_root.rfind('/') + 1]
    b_path = test_root + 'bboxes/bbox' + number + '.txt'
    orig_bb = read_boxes(b_path)
    return orig_bb


def cleanup_iou_matrix(mat, annot_bboxes, aanot_kp, pred_kp):
    #Выбирает наибольшие совпадения
    new_annot_bboxes = []
    new_aanot_kp = []
    new_pred_kp = []
    for i in range(len(mat)):
        if len(np.where(mat[i]==max(mat[i]))[0].tolist()) != 1:
            ind = int(np.where(mat[i]==max(mat[i]))[0].tolist()[0])
            new_pred_kp.append(pred_kp[i])
            new_annot_bboxes.append(annot_bboxes[ind])
            new_aanot_kp.append(aanot_kp[ind])
            #conf_cand.append(mat[i][ind])
        else:
            ind = int(np.where(mat[i]==max(mat[i]))[0])
            new_pred_kp.append(pred_kp[i])
            new_annot_bboxes.append(annot_bboxes[ind])
            new_aanot_kp.append(aanot_kp[ind])
            #conf_cand.append(mat[i][ind])
    return new_pred_kp, new_annot_bboxes, new_aanot_kp

def count_pdj(root, model_path, overlay_w, overlay_h, conf_threshold, nms_threshold, iou_threshold, splits_vertical, splits_horizontal):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(2, model_path)
    dir_size = len(glob.glob(root + "/images" + '/*'))
    mean_pdj = 0
    counter = 1
    for f in os.scandir(root + "/images"):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            print(f'Изображение №{counter} из {dir_size}')
            counter += 1
            number = int(f.path[f.path.rfind('e') + 1 : f.path.rfind('.')])
            _, pred_b, pred_kp, pred_sc = one_image_test(f.path, model, device, False, conf_threshold, nms_threshold, iou_threshold, overlay_w, overlay_h, splits_vertical, splits_horizontal, False)
            annot_bboxes = np.array(get_real_bboxes(f.path))
            aanot_kp, _ = find_kp(number, root)
            pdj = pdj_one_image(pred_b, annot_bboxes, aanot_kp, pred_kp)
            #print(pdj)
            mean_pdj += pdj
            
    print(f"PDJ: {mean_pdj / dir_size}")
            
def distance(kpA, kpB):
    pr_x_a = kpA[0][0]
    pr_y_a = kpA[0][1]
    pr_x_h = kpA[1][0]
    pr_y_h = kpA[1][1]
    
    or_x_a = kpB[0][0]
    or_y_a = kpB[0][1]
    or_x_h = kpB[1][0]
    or_y_h = kpB[1][1]
    
    dist_A = ((pr_x_a - or_x_a) ** 2 + (pr_y_a - or_y_a) ** 2) ** 0.5
    dist_H = ((pr_x_h - or_x_h) ** 2 + (pr_y_h - or_y_h) ** 2) ** 0.5
    
    return dist_A, dist_H
    
def diagonal(bbox):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    
    return diag

    
def pdj_one_image(pred_b, annot_bboxes, aanot_kp, pred_kp):
    iou_matrix = np.zeros((len(pred_b), len(annot_bboxes)))
    for i in range(len(pred_b)):
        for j in range(len(annot_bboxes)):
            iou_matrix[i][j] = intersection_over_union(pred_b[i], annot_bboxes[j])
    
    new_pred_kp, new_annot_bboxes, new_aanot_kp = cleanup_iou_matrix(iou_matrix, annot_bboxes, aanot_kp, pred_kp)
    #annot_bboxes = np.array(annot_bboxes[ind_gt])
    #aanot_kp = np.array(aanot_kp[ind_pred])
    #pred_kp = pred_kp[ind_pred]
    pdj = 0
    for i in range(len(new_annot_bboxes)):
        A, H = distance(new_pred_kp[i], new_aanot_kp[i])
        if A >= 0.05 * diagonal(new_annot_bboxes[i]):
            pdj += 1
        if H >= 0.05 * diagonal(new_annot_bboxes[i]):
            pdj += 1
    pdj = pdj / (len(new_pred_kp) * 2)
    return pdj
            
if __name__ == '__main__':        
    parser = argparse.ArgumentParser()  
    parser.add_argument('--test_path', nargs='?', default='/home/ubuntu/ant_detection/new_dataset/Test_data', help="path to folder with images and annot", type=str)
    parser.add_argument('--model_path', nargs='?', default='/home/ubuntu/ant_detection/new_dataset/rcnn_models/20230216-180517/best_weights.pth', help="path to weights", type=str)
    parser.add_argument('conf_threshold', nargs='?', default=0.7, help="Confident threshold for boxes", type=float)
    parser.add_argument('nms_threshold', nargs='?', default=0.3, help="Non maximum suppression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.3, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    parser.add_argument('splits_vertical', nargs='?', default=3, help="Num of pictures in w-axis", type=int)
    parser.add_argument('splits_horizontal', nargs='?', default=2, help="Num of pictures in h-axis", type=int)
    
    args = parser.parse_args()
    
    count_pdj(args.test_path, args.model_path, args.overlay_w, args.overlay_h, args.conf_threshold, args.nms_threshold, args.iou_threshold, args.splits_vertical, args.splits_horizontal)
