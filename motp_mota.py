from mAP import get_real_bboxes,is_object
import numpy as np
import argparse
import glob
import os
from RCNN_model import get_model
from RCNN_overlay_test import one_image_test
import torch
import matplotlib.pyplot as plt

def get_iou(a, b, epsilon=1e-5, intersection_check=False):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        if intersection_check:
            return 0.0, False
        else:
            return 0.0
    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)
    if intersection_check:
        return iou, bool(area_overlap)
    else:
        return iou


def calc_conditions(gt_boxes, pred_boxes, iou_thresh=0.5, hard_fp=True):
    gt_class_ids_ = np.zeros(len(gt_boxes))
    pred_class_ids_ = np.zeros(len(pred_boxes))

    tp, fp, fn, dt = 0, 0, 0, 0
    for i in range(len(gt_class_ids_)):
        iou = []
        for j in range(len(pred_class_ids_)):
            now_iou, intersect = get_iou(gt_boxes[i], pred_boxes[j], intersection_check=True)
            dt += 1 - now_iou
            if now_iou >= iou_thresh and intersect:
                iou.append(now_iou)
                gt_class_ids_[i] = 1
                pred_class_ids_[j] = 1
        if len(iou) > 0:   # если gt box пересекает с нужным больше 1 pred
            tp += 1   # один с наивысшим IoU - TP
            fp += len(iou) - 1   # все остальные - FP
    fn += np.count_nonzero(np.array(gt_class_ids_) == 0)
    fp += np.count_nonzero(np.array(pred_class_ids_) == 0)

    return tp, fp, fn, dt
    
def count_mota(conf_threshold, nms_threshold, iou_threshold, images_path, iuo_tresh, model_path, overlay_w, overlay_h):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(2, model_path)
    counter = 1
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_dt = 0
    number_of_det = 0
    number_of_real_obj = 0
    dir_size = len(glob.glob(images_path + '/*'))
    for f in os.scandir(images_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            print(f'Изображение №{counter} из {dir_size}')
            counter += 1
            _, pred_b, _, pred_sc = one_image_test(f.path, model, device, False, conf_threshold, nms_threshold, iou_threshold, overlay_w, overlay_h, False)
            annot_bboxes = np.array(get_real_bboxes(f.path))
            number_of_real_obj += annot_bboxes.shape[0]
            number_of_det += len(pred_b)
            tp, fp, fn, dt = calc_conditions(annot_bboxes, pred_b, iuo_tresh)
            all_tp += tp
            all_fp += fp
            all_fn += fn
            all_dt += dt
    
    #Object in ground truth is falsely related to some other object due to false tracking.
    mismatch_error = 0
    
    mota = 1 - (all_fp + all_fn + mismatch_error)/number_of_real_obj
    motp = all_dt / all_tp
    return mota, motp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', nargs='?', default='/home/ubuntu/ant_detection/dataset/Test_data/images', help="path to folder with images and annot", type=str)
    parser.add_argument('--iuo_tresh', nargs='?', default=0.5, help="treshold for TP and FP", type=float)
    parser.add_argument('--model_path', nargs='?', default='/home/ubuntu/ant_detection/dataset/rcnn_models/20221226-111349/full_weights.pth', help="path to weights", type=str)
    parser.add_argument('conf_threshold', nargs='?', default=0.7, help="Confident threshold for boxes", type=float)
    parser.add_argument('nms_threshold', nargs='?', default=0.3, help="Non maximum suppression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.3, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    
    args = parser.parse_args()
    MOTA, MOTP = count_mota(args.conf_threshold, args.nms_threshold, args.iou_threshold, args.test_path, args.iuo_tresh, args.model_path, args.overlay_w, args.overlay_h)
    
    print(f"MOTA: {MOTA}, MOTP: {MOTP}")
