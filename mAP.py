import pandas as pd
import numpy as np
import argparse
from overlay_intersection import read_boxes
import glob
import os
from RCNN_model import get_model
from RCNN_overlay_test import one_image_test
import torch
'''
boxA, boxB shape[4]
'''
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

'''
GT_values - [N, 4]
pred_values - [M, 4] - прямоугольники сортируются в порядке убывания достоверности наличия в них объектов
'''
def IoU_matrix(GT_values, pred_values, iou_tresh):
    correct = []
    all_prec = []
    all_recall = []
    iou_matrix = np.zeros((GT_values.shape[0], pred_values.shape[0]))
    TP = 0 # количество перекрытий больше порога
    FP = 0 # количество перекрытий меньше порога
    FN = 0 # количество необраруженных объектов
    for i in range(GT_values.shape[0]):
        for j in range(pred_values.shape[0]):
            iou_matrix[i][j] = intersection_over_union(GT_values[i], pred_values[j])
            if iou_matrix[i][j] >= iou_tresh:
                correct.append(1)
            else:
                correct.append(0)
        
        precision = np.count_nonzero(correct) / (i + 1)
        recall = np.count_nonzero(correct) / GT_values.shape[0]
        all_prec.append(precision)
        all_recall.append(recall)
        #np.append(table['precision'], precision)
        #np.append(table['recall'], recall)
        
    area = np.trapz(all_prec, all_recall)
    return area
    
def get_real_bboxes(im_path):
    number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
    im_root = im_path[:im_path.rfind('/')]
    test_root = im_root[:im_root.rfind('/') + 1]
    b_path = test_root + 'bboxes/bbox' + number + '.txt'
    orig_bb = read_boxes(b_path)
    return orig_bb
    
# сортируются в порядке убывания достоверности наличия в них объектов
def sort_bboxes(scores, bboxes):
    x = zip(scores,bboxes)
    xs = sorted(x, key=lambda tup: tup[0], reverse=True)
    a1 = [x[0] for x in xs]
    b1 = [x[1] for x in xs]
    
    #scores_bboxes = np.stack((pred_sc, pred_b), axis=-1)
    #scores_bboxes.tolist()
    #scores_bboxes.sort(reverse=True)
    #new_bboxes = scores_bboxes[:, 1]
    return np.array(b1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    file_ = 'cut50s'
    parser.add_argument('--test_path', nargs='?', default='/home/ubuntu/ant_detection/polygon_data/Test_data', help="path to folder with images and annot", type=str)
    parser.add_argument('--iuo_tresh', nargs='?', default=0.5, help="treshold for TP and FP", type=float)
    parser.add_argument('--model_path', nargs='?', default='/home/ubuntu/ant_detection/polygon_data/rcnn_models/20221102-124538/full_weights.pth', help="path to weights", type=str)
    parser.add_argument('conf_threshold', nargs='?', default=0.4, help="Confident threshold for boxes", type=float)
    parser.add_argument('nms_threshold', nargs='?', default=0.6, help="Non maximum suppression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.5, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    
    args = parser.parse_args()
    conf_threshold = args.conf_threshold
    nms_threshold = args.nms_threshold
    iou_threshold = args.iou_threshold
    overlay_w = args.overlay_w
    overlay_h = args.overlay_h
    real_bboxes_path = args.test_path + '/bboxes'
    images_path = args.test_path + '/images'
    dir_size = len(glob.glob(real_bboxes_path + '/*'))
    iuo_tresh = args.iuo_tresh
    average_precision = 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(2, args.model_path)
    counter = 1
    for f in os.scandir(images_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            print(f'Изображение №{counter} из {dir_size}')
            counter += 1
            _, pred_b, _, pred_sc = one_image_test(f.path, model, device, False, conf_threshold, nms_threshold, iou_threshold, overlay_w, overlay_h, False)
            #print(len(pred_b), len(pred_sc))
            annot_bboxes = np.array(get_real_bboxes(f.path))
            
            pred_sort_bboxes = sort_bboxes(pred_sc, pred_b)
            average_precision += IoU_matrix(annot_bboxes, pred_sort_bboxes, iuo_tresh)
            
    mAP = average_precision / dir_size
    print(f"Mean average precision: {mAP}")
