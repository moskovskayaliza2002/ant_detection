import pandas as pd
import numpy as np
import argparse
from overlay_intersection import read_boxes
import glob
import os
from RCNN_model import get_model
from RCNN_overlay_test import one_image_test
import torch
import matplotlib.pyplot as plt
import pandas as pd


def intersection_over_union(boxA, boxB):
    #Считает IoU для двух боксов
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = round(iou, 2)
    return iou 


def get_real_bboxes(im_path):
    #По пути изображения находит аннотации ограничивающих рамок
    number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
    im_root = im_path[:im_path.rfind('/')]
    test_root = im_root[:im_root.rfind('/') + 1]
    b_path = test_root + 'bboxes/bbox' + number + '.txt'
    orig_bb = read_boxes(b_path)
    return orig_bb

def P_R_counter(scores, obj, num_obj):
    # Считает точность и отклик
    flat_scores = [item for sublist in scores for item in sublist]
    flat_obj = [item for sublist in obj for item in sublist]
    
    x = zip(flat_scores,flat_obj)
    xs = sorted(x, key=lambda tup: tup[0], reverse=True)
    new_sc = [x[0] for x in xs]
    new_obj= [x[1] for x in xs]
    
    all_prec = []
    all_recall = []
    TP = 0 # количество перекрытий больше порога
    FP = 0 # количество перекрытий меньше порога
    
    for i in new_obj:
        if i:
            TP += 1
        else:
            FP += 1
        try:
            precision = TP / (FP + TP)
            recall = TP / num_obj
        
        except ZeroDivisionError:
        
            precision = recall = 0   
            
        all_prec.append(precision)
        all_recall.append(recall)
        
    return all_prec, all_recall

def cleanup_iou_matrix(mat):
    #Выбирает наибольшие совпадения
    conf_cand = []
    for i in range(len(mat)):
        if len(np.where(mat[i]==max(mat[i]))[0].tolist()) != 1:
            ind = int(np.where(mat[i]==max(mat[i]))[0].tolist()[0])
            conf_cand.append(mat[i][ind])
        else:
            ind = int(np.where(mat[i]==max(mat[i]))[0])
            conf_cand.append(mat[i][ind])
    return conf_cand

def is_object(gt, scores, pred, tresh_iou):
    #Проверяет, является ли объектом (по пороговому значению iou)
    result = []
    iou_matrix = np.zeros((pred.shape[0], gt.shape[0]))
    for i in range(pred.shape[0]):
        for j in range(gt.shape[0]):
            iou_matrix[i][j] = intersection_over_union(pred[i], gt[j])
    
    conf_cand = cleanup_iou_matrix(iou_matrix)
    for i in conf_cand:
        if i >= tresh_iou:
            result.append(True)
        else:
            result.append(False)
            
    return result
 
def count_mAP(conf_threshold, nms_threshold, iou_threshold, images_path, iuo_tresh, model_path, overlay_w, overlay_h):
    average_precision = 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(2, model_path)
    counter = 1
    all_scores = []
    all_is_objects = []
    number_of_real_obj = 0
    dir_size = len(glob.glob(images_path + '/*'))
    for f in os.scandir(images_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            print(f'Изображение №{counter} из {dir_size}')
            counter += 1
            _, pred_b, _, pred_sc = one_image_test(f.path, model, device, False, conf_threshold, nms_threshold, iou_threshold, overlay_w, overlay_h, False)
            annot_bboxes = np.array(get_real_bboxes(f.path))
            number_of_real_obj += annot_bboxes.shape[0]
            is_obj = is_object(annot_bboxes, pred_sc, np.array(pred_b), iuo_tresh)
            all_is_objects.append(is_obj)
            all_scores.append(pred_sc)
            
    
    pr, rec = P_R_counter(all_scores, all_is_objects, number_of_real_obj)
    
    inter_pr = np.maximum.accumulate(pr[::-1])[::-1]
    ap = np.trapz(inter_pr, rec)
    return ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    file_ = 'cut50s'
    parser.add_argument('--test_path', nargs='?', default='/home/ubuntu/ant_detection/dataset/Test_data', help="path to folder with images and annot", type=str)
    parser.add_argument('--iuo_tresh', nargs='?', default=0.5, help="treshold for TP and FP", type=float)
    parser.add_argument('--model_path', nargs='?', default='/home/ubuntu/ant_detection/dataset/rcnn_models/20221226-111349/full_weights.pth', help="path to weights", type=str)
    parser.add_argument('conf_threshold', nargs='?', default=0.7, help="Confident threshold for boxes", type=float)
    parser.add_argument('nms_threshold', nargs='?', default=0.3, help="Non maximum suppression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.3, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    
    # the best weights 20221122-172249
    
    
    args = parser.parse_args()
    conf_threshold = args.conf_threshold
    nms_threshold = args.nms_threshold
    iou_threshold = args.iou_threshold
    overlay_w = args.overlay_w
    overlay_h = args.overlay_h
    real_bboxes_path = args.test_path + '/bboxes'
    images_path = args.test_path + '/images'
    #dir_size = len(glob.glob(real_bboxes_path + '/*'))
    iuo_tresh = args.iuo_tresh
    model_path = args.model_path
    
    mAP = count_mAP(conf_threshold, nms_threshold, iou_threshold, images_path, iuo_tresh, model_path, overlay_w, overlay_h)
    print(f'AP: {mAP}')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(rec, pr, label="P/R curve", color='blue')
    plt.plot(rec, inter_pr, label="Interpolated P/R curve", linestyle = '--', color='red')
    plt.grid(axis='x', color='0.95')
    plt.legend(loc="best")
    plt.show()
