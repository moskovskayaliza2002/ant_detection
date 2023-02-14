import torch
import torchvision
from RCNN_model import get_model
from universal_RCNN_test import intersection_over_union, one_image_test
from overlay_intersection import read_boxes
import numpy as np
import argparse
import glob
import os
import matplotlib.pyplot as plt


def plot_gist(data):
    fig, ax = plt.subplots()
    ax.set_title('Распределение ошибок')
    parts = ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    medians = np.percentile(data, [50], axis=1)

    ax.scatter([1, 2, 3], medians, marker='o', color='white', s=30, zorder=3)
    labels = ['X', 'Y', 'ANGLE']
    
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
    plt.show()

def get_orig_annot(im_path, root_path):
    #Из пути изображение берет номер и по этому номеру находит аннотации для этого изображения
    orig_bb = []
    orig_kp = []
    number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
    b_path = root_path + '/bboxes/bbox' + number + '.txt'
    orig_bb = read_boxes(b_path)
    k_path = root_path + '/keypoints/keypoint' + number + '.txt'
    kp = read_boxes(k_path)
    for i in kp:
        orig_kp.append([[i[0], i[1]], [i[2], i[3]]])
        
    return orig_bb, orig_kp
    

def get_right_from_predict(orig_bb, pred_bb, orig_kp, pred_kp, treshold):
    # Подбирает для каждого оригинального объекта, предсказанный, с большей уверенностью
    iou_matrix = [[0 for i in range(len(orig_bb))] for j in range(len(pred_bb))]
    #Заполнение матрицы IOU
    for i in range(len(iou_matrix)):
            for j in range(len(iou_matrix[0])):
                iou_matrix[i][j] = intersection_over_union(pred_bb[i], orig_bb[j])
                
    new_pred_bb = [[0, 0, 0, 0]] * len(orig_bb)
    new_pred_kp = [[[0, 0], [0, 0]]] * len(orig_kp)
    
    if not iou_matrix:
        return new_pred_bb, new_pred_kp
    
    while max(map(max, iou_matrix)) >= treshold:
        for i in range(len(pred_bb)):
            for j in range(len(orig_bb)):
                if iou_matrix[i][j] == max(map(max, iou_matrix)) and iou_matrix[i][j] >= treshold:
                    #print("новое значение")
                    new_pred_bb[j] = pred_bb[i]
                    new_pred_kp[j] = pred_kp[i]
                    
                    iou_matrix[i] = [-1] * len(orig_bb)
                    for row in iou_matrix:
                        row[j] = -1
    #print(new_pred_bb, new_pred_kp)
    return new_pred_bb, new_pred_kp


def substract_angles(target, source):
    return np.arctan2(np.sin(target-source), np.cos(target-source))


def loss(orig_bb, pred_bb, orig_kp, pred_kp, alpha_error, point_error_x, point_error_y):
    #Возвращает массив ошибок для всех значений одного изображения
    #alpha_error = []
    #point_error = []
    for i in range(len(orig_bb)):
        if pred_bb[i] != [0, 0, 0, 0]:
            #Высчитывае центров объектов из боксов
            center_point_pr = [(pred_bb[i][0] + pred_bb[i][2]) / 2, (pred_bb[i][1] + pred_bb[i][3]) / 2]
            center_point_or = [(orig_bb[i][0] + orig_bb[i][2]) / 2, (orig_bb[i][1] + orig_bb[i][3]) / 2]
            #Ошибка по x и y
            point_error_x.append(center_point_or[0] - center_point_pr[0])
            point_error_y.append(center_point_or[1] - center_point_pr[1])
            #Высчитывание углов
            alpha_orig = np.arctan2(orig_kp[i][1][1] - orig_kp[i][0][1], orig_kp[i][1][0] - orig_kp[i][0][0])
            alpha_pr = np.arctan2(pred_kp[i][1][1] - pred_kp[i][0][1], pred_kp[i][1][0] - pred_kp[i][0][0])
            alpha_error.append(substract_angles(alpha_orig, alpha_pr))
            #Ошибка углов
            #a_error = alpha_orig - alpha_pr
            #while a_error < 0:
            #    a_error += 2 * math.pi
            #while a_error > 2 * math.pi:
            #    a_error -= 2 * math.pi
            
            #if a_error > math.pi:
            #    a_error -= math.pi
            #    a_error = math.pi - a_error
                
            #alpha_error.append(a_error)
    #print(f'alpha_error: {alpha_error}\npoint_error_x: {point_error_x}\npoint_error_y: {point_error_y}')
    return point_error_x, point_error_y, alpha_error

    
def batch_test(root, model, device, conf_threshold, iou_threshold, nms_threshold, delta_w, delta_h, splits_vertical, splits_horizontal, treshold = 0.5):
    #Функция показывающая предсказывания модели на пакете изображений (Для модели, что делит на 4 исходное изображение
    image_data_path = root + '/images'
    dir_size = len(glob.glob(image_data_path + '/*'))
    counter = 0
    #Массивы ошибок по всем изображениям
    total_x_error = []
    total_y_error = []
    total_angle_error = []
    
    for f in os.scandir(image_data_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            print(f'Изображение №{counter + 1} из {dir_size}')
            image_path = f.path
            #Получаем настоящие аннотации
            orig_bboxes, orig_keypoints = get_orig_annot(image_path, root)
            #Предсказанные значения
            _, pred_b, pred_kp, pred_sc = one_image_test(image_path, model, device, False, conf_threshold, nms_threshold, iou_threshold, delta_w, delta_h, splits_vertical, splits_horizontal, False)
            #Сопоставленные с оригиналом предсказания
            new_pred_bboxes, new_pred_keypoints = get_right_from_predict(orig_bboxes, pred_b, orig_keypoints, pred_kp, treshold)
            #Ошибки по изображению
            total_x_error, total_y_error, total_angle_error = loss(orig_bboxes, new_pred_bboxes, orig_keypoints, new_pred_keypoints, total_angle_error, total_x_error, total_y_error)
            #total_location_error.append(location_error)
            #total_angle_error.append(angle_error)
            counter += 1
    #x = [row[0] for row in total_location_error]
    #y = [row[1] for row in total_location_error]
    x_error = sum([i ** 2 for i in total_x_error]) / len(total_x_error)
    y_error = sum([i ** 2 for i in total_y_error]) / len(total_y_error)
    angle_error = sum([i ** 2 for i in total_angle_error]) / len(total_angle_error)
    plot_gist([total_x_error, total_y_error, total_angle_error])
    return x_error, y_error, angle_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/new_dataset/Train_data', help="Specify the full path to the folder with test data", type=str)
    parser.add_argument('model_path', nargs='?', default='/home/ubuntu/ant_detection/new_dataset/rcnn_models/20230207-161545/best_weights.pth', help="Specify weights path", type=str)
    parser.add_argument('conf_threshold', nargs='?', default=0.7, help="Confident threshold for boxes", type=float)
    parser.add_argument('nms_threshold', nargs='?', default=0.3, help="Non maximum suppression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.18, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    parser.add_argument('splits_vertical', nargs='?', default=3, help="Num of pictures in w-axis", type=int)
    parser.add_argument('splits_horizontal', nargs='?', default=2, help="Num of pictures in h-axis", type=int)
    
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path
    conf_threshold = args.conf_threshold
    nms_threshold = args.nms_threshold
    iou_threshold = args.iou_threshold
    overlay_w = args.overlay_w
    overlay_h = args.overlay_h
    
    if torch.cuda.is_available():
        print('*****************************DEVICE: GPU*****************************')
    else:
        print('*****************************DEVICE: CPU*****************************')
        
    test_model = get_model(2, model_path)
    x_err, y_err, angle_err = batch_test(test_data_path, test_model, DEVICE, conf_threshold, iou_threshold, nms_threshold, overlay_w, overlay_h, args.splits_vertical, args.splits_horizontal)
    print(f'X ERROR: {x_err}\nY ERROR: {y_err}\nANGLE ERROR: {angle_err}')
