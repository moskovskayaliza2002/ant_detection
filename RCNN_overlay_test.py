import torchvision
from overlay_intersection import crop_one_im, read_boxes, resize_bboxes_kps 
from RCNN_test import get_out_kp_bb, visualize, find_bbox, find_kp
import numpy as np
import argparse
from RCNN_model import get_model
import cv2
import os
import glob
import torch
from torchvision.transforms import functional as F


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
    
 
def chose_right_bbox(bbox, keypoints):
    #Отбор не совпадающих с другим изображением боксов
    new_bb = []
    new_kp = []
    for i in range(len(bbox)):
        if type(bbox[i]) != int:
            new_bb.append(bbox[i])
            new_kp.append(keypoints[i])
    return new_bb, new_kp
            
    
def selection(boxA, boxB, kpA, kpB, treshould = 0.75):
    #Функция, фильтрующая накладывающиеся боксы одного объекта разных частей изображения
    if len(boxA) == 0 or len(boxB) == 0:
        if len(boxA) and len(boxB) == 0:
            return [], []
        elif len(boxA) == 0:
            return boxB, kpB
        elif len(boxB) == 0:
            return boxA, kpA
    else:
        #Матрица Джакардова индекса для всех элементов двух частей изображения
        iou_matrix = [[0 for i in range(len(boxB))] for j in range(len(boxA))]

        for i in range(len(iou_matrix)):
            for j in range(len(iou_matrix[0])):
                iou_matrix[i][j] = intersection_over_union(boxA[i], boxB[j])
        
        mean_bb = []
        mean_kp = []
        
        while max(map(max, iou_matrix)) >= treshould:
            for i in range(len(boxA)):
                for j in range(len(boxB)):
                    if iou_matrix[i][j] == max(map(max, iou_matrix)):
                        #Удаляем совпадающие боксы и заменяем их средним значением
                        xmin = int((boxA[i][0] + boxB[j][0]) / 2)
                        ymin = int((boxA[i][1] + boxB[j][1]) / 2)
                        xmax = int((boxA[i][2] + boxB[j][2]) / 2)
                        ymax = int((boxA[i][3] + boxB[j][3]) / 2)
                        mean_bb.append([xmin, ymin, xmax, ymax])
                        #То же самое с ключевыми точками
                        x_a = int((kpA[i][0][0] + kpB[j][0][0]) / 2)
                        y_a = int((kpA[i][0][1] + kpB[j][0][1]) / 2)
                        x_h = int((kpA[i][1][0] + kpB[j][1][0]) / 2)
                        y_h = int((kpA[i][1][1] + kpB[j][1][1]) / 2)
                        mean_kp.append([[x_a, y_a], [x_h, y_h]])
                        
                        boxA[i] = -1
                        boxB[j] = -1
                        #"Зануляем" соответствущие элементы в матрице
                        iou_matrix[i] = [-1] * len(boxB)
                        for row in iou_matrix:
                            row[j] = -1
        
        #Отбираем несовпадающие боксы и ключевые точки
        new_boxB, new_kpB = chose_right_bbox(boxB, kpB)
        new_boxA, new_kpA = chose_right_bbox(boxA, kpA)
        
        bb = np.vstack([np.array(new_boxA), np.array(new_boxB)]) 
        kp = np.vstack([np.array(new_kpA), np.array(new_kpB)]) 
        
        #Объединяем с новыми и несовпадающими боксами
        if len(mean_bb) != 0:
            bb = np.vstack([bb, np.array(mean_bb)]) 
            kp = np.vstack([kp, np.array(mean_kp)]) 
        
        return bb.tolist(), kp.tolist()
    
    
    
def iou_filter(kp_l1, bb_l1, kp_l2, bb_l2, kp_r1, bb_r1, kp_r2, bb_r2, c_w, c_h, delta_w, delta_h):
    #Функция, объединяющая предсказанные ключевые точки и боксы
    
    #Объединение двух верхних частей
    sel_l1_r1_b, sel_l1_r1_kp = selection(bb_l1, bb_r1, kp_l1, kp_r1)
    #Объединение двух нижних частей
    sel_l2_r2_b, sel_l2_r2_kp = selection(bb_l2, bb_r2, kp_l2, kp_r2)
    #Объединение двух получившихся частей
    sel_all_b, sel_all_kp = selection(sel_l1_r1_b, sel_l2_r2_b, sel_l1_r1_kp, sel_l2_r2_kp)
    
    return(sel_all_b, sel_all_kp)
        
    
def one_image_test(im_path, model, flag, nms_threshold, iou_threshold, delta_w, delta_h):
    #Функция показывающая предсказывания модели на отдельном изображении (Для модели, что делит на 4 исходное изображение)
    
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    orig_bb = []
    orig_kp = []
    #Получаем настоящие координаты для целого изображения
    number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
    im_root = im_path[:im_path.rfind('/')]
    test_root = im_root[:im_root.rfind('/') + 1]
    b_path = test_root + 'bboxes/bbox' + number + '.txt'
    orig_bb = read_boxes(b_path)
    k_path = test_root + 'keypoints/keypoint' + number + '.txt'
    kp = read_boxes(k_path)
    for i in kp:
        orig_kp.append([[i[0], i[1]], [i[2], i[3]]])
    #Разрезаем изображение на 4    
    left_1, left_2, right_1, right_2, c_w, c_h = crop_one_im(img, delta_w, delta_h)
    
    with torch.no_grad():
        model.to(device)
        model.eval()
        out_l1 = model([F.to_tensor(cv2.cvtColor(left_1, cv2.COLOR_BGR2RGB))])
        out_l2 = model([F.to_tensor(cv2.cvtColor(left_2, cv2.COLOR_BGR2RGB))])
        out_r1 = model([F.to_tensor(cv2.cvtColor(right_1, cv2.COLOR_BGR2RGB))])
        out_r2 = model([F.to_tensor(cv2.cvtColor(right_2, cv2.COLOR_BGR2RGB))])
    #Получаем предсказанные значения для каждой из 4 картинок, с пороговыми NMS и IOU        
    kp_l1, bb_l1 = get_out_kp_bb(out_l1, 0, 0, [], [], nms_threshold, iou_threshold)
    kp_l2, bb_l2 = get_out_kp_bb(out_l2, 0, c_h - delta_h, [], [], nms_threshold, iou_threshold)
    kp_r1, bb_r1 = get_out_kp_bb(out_r1, c_w - delta_w, 0, [], [], nms_threshold, iou_threshold)
    kp_r2, bb_r2 = get_out_kp_bb(out_r2, c_w - delta_w, c_h - delta_h, [], [], nms_threshold, iou_threshold)
    #Объединяем предсказанные значения    
    pred_b, pred_kp = iou_filter(kp_l1, bb_l1, kp_l2, bb_l2, kp_r1, bb_r1, kp_r2, bb_r2, c_w, c_h, delta_w, delta_h)
    #Визуализация и таргетов, и предсказаний    
    if flag:
        visualize(img, pred_b, pred_kp, img, orig_bb, orig_kp)
    #Визуализация только предсказаний
    else:
        visualize(img, pred_b, pred_kp)
        

def batch_test(root, model, flag, nms_threshold, iou_threshold, delta_w, delta_h):
    #Функция показывающая предсказывания модели на пакете изображений (Для модели, что делит на 4 исходное изображение
    image_data_path = root + '/images'
    dir_size = len(glob.glob(image_data_path + '/*'))
    counter = 0
    for f in os.scandir(image_data_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            print(f'Изображение №{counter} из {dir_size}')
            image_path = f.path
            one_image_test(image_path, model, flag, nms_threshold, iou_threshold, delta_w, delta_h)
            counter += 1
    

if __name__ == '__main__':       
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/TEST_ACC_DATA', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    #parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/TEST_ACC_DATA/images/0a302e52-image202.png', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('model_path', nargs='?', default='/home/ubuntu/ant_detection/rcnn_models/20220628-124306/best_weights.pth', help="Specify weights path", type=str)
    parser.add_argument('draw_targets', nargs='?', default=True, help="True - will draw targets, False - will not", type=bool)
    parser.add_argument('nms_threshold', nargs='?', default=0.5, help="Non maximum supression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.2, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    
    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path
    draw_targets = args.draw_targets
    nms_threshold = args.nms_threshold
    iou_threshold = args.iou_threshold
    overlay_w = args.overlay_w
    overlay_h = args.overlay_h
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_model = get_model(2, model_path)
    
    if test_data_path[-3:] == 'png':
        one_image_test(test_data_path, test_model, draw_targets, nms_threshold, iou_threshold, overlay_w, overlay_h)
    else:
        batch_test(test_data_path, test_model, draw_targets, nms_threshold, iou_threshold, overlay_w, overlay_h)
