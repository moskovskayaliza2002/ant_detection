import numpy as np
import cv2
import argparse
import os
import shutil

def read_boxes(bbox_path):
    # Считывает боксы
    bboxes_original = []
    with open(bbox_path) as f:
        for i in f:
            x_min, y_min, x_max, y_max = map(int, i[:-1].split(' '))
            bboxes_original.append([x_min, y_min, x_max, y_max])
    return bboxes_original


def find_bbox(num, root_path):
    # Находит файл с боксами, относящийся к изображению и возвращает считанные значения
    bb = []
    bbox_path = root_path + '/bboxes'
    for f in os.scandir(bbox_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'txt':
            if int(f.path[f.path[:-3].rfind('x') + 1 : f.path[:-3].rfind('.')]) == num:
                bb = read_boxes(f.path)
    
    return bb


def find_kp(num, root_path):
    # Находит файл с ключевыми точками, относящийся к изображению и возвращает считанные значения
    kp = []
    orig_kp = []
    kp_path = root_path + '/keypoints'
    for f in os.scandir(kp_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'txt':
            if int(f.path[f.path[:-3].rfind('t') + 1 : f.path[:-3].rfind('.')]) == num:
                kp = read_boxes(f.path)
    for i in kp:
        orig_kp.append([[i[0], i[1]], [i[2], i[3]]])
    
    return orig_kp, kp


def conv_x(old):
    # Переводит координату х в новую систему координат
    old_min = new_min = 0
    old_range = 224 - 0 #1920 - 0
    new_range = 1 - 0 #
    
    converted = (((old - old_min) * new_range) / old_range) + new_min
    return converted


def conv_y(old):
    # Переводит координату у в новую систему координат
    old_min = new_min = 0
    old_range = 224 - 0 # 1080 - 0 
    new_range = 1 - 0
    
    converted = (((old - old_min) * new_range) / old_range) + new_min
    return converted


def create(dataset_path, yolo_images, yolo_labels):
    image_path = dataset_path + "/images"
    counter = 481
    for f in os.scandir(image_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            filename = f.path[f.path.rfind('/')+1:]
            number = filename[filename.index('e')+1:-4]
            original_image = cv2.imread(f.path)
            h, w, _ = original_image.shape
            original_bboxs = find_bbox(int(number), dataset_path)
            _, original_keypoints = find_kp(int(number), dataset_path)
            cv2.imwrite(yolo_images + '/' + str(counter) + '.png', original_image)
            convert_to_str(original_bboxs, original_keypoints, yolo_labels + '/' + str(counter) + '.txt')
            counter += 1
            
            
def convert_to_str(bbs, kps, filename):
    str_list = []
    for i in range(len(bbs)):
        full_str = '0 '
        xmin, ymin, xmax, ymax = conv_x(bbs[i][0]), conv_y(bbs[i][1]), conv_x(bbs[i][2]), conv_y(bbs[i][3])
        x_a, y_a, x_h, y_h = conv_x(kps[i][0]), conv_y(kps[i][1]), conv_x(kps[i][2]), conv_y(kps[i][3])
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        str_bbox = str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height)
        str_kps = str(x_a) + ' ' + str(y_a) + ' ' + str(2) + ' ' + str(x_h) + ' ' + str(y_h) + ' ' + str(2)
        full_str += str_bbox + ' '
        full_str += str_kps + '\n'
        #chech_convert(bbs[i], [x_center, y_center, width, height], kps[i], [x_a, y_a, x_h, y_h])
        str_list.append(full_str)
    with open(filename, 'w') as file:
        file.writelines(str_list)
        file.close()
    
def chech_convert(orig_bb, pred_bb, orig_kps, pred_kps):
    print(f"Оригинальный бокс: {orig_bb}")
    x_center = int((((pred_bb[0] - 0) * 1920) / 1) + 0)
    y_center = int((((pred_bb[1] - 0) * 1080) / 1) + 0)
    width = int((((pred_bb[2] - 0) * 1920) / 1) + 0)
    height = int((((pred_bb[3] - 0) * 1080) / 1) + 0)
    xmin = x_center - width//2
    ymin = y_center - height//2
    xmax = x_center + width//2
    ymax = y_center + height//2
    print(f"Сконвертированный бокс: {[xmin, ymin, xmax, ymax]}")
    print(f"Оригинальные точки: {orig_kps}")
    x_a = int((((pred_kps[0] - 0) * 1920) / 1) + 0)
    y_a = int((((pred_kps[1] - 0) * 1080) / 1) + 0)
    x_h = int((((pred_kps[2] - 0) * 1920) / 1) + 0)
    y_h = int((((pred_kps[3] - 0) * 1080) / 1) + 0)
    
    print(f"Сконвертированные точки: {[x_a, y_a, x_h, y_h]}")
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', nargs='?', default='/home/ubuntu/ant_detection/new_dataset', help="Specify the full path to dataset to convert to yolo type", type=str)
    parser.add_argument('--yolo_dataset_path', nargs='?', default='/home/ubuntu/ant_detection/yolo_aug', help="Specify the full path to directory to store ", type=str)
    args = parser.parse_args()
    
    #create(args.dataset_path + '/Train_data_not_cropped', args.yolo_dataset_path + '/images/train', args.yolo_dataset_path + '/labels/train')
    create(args.dataset_path + '/Train_data_with_aug/augmentation', args.yolo_dataset_path + '/images/train', args.yolo_dataset_path + '/labels/train')
    #create(args.dataset_path + '/Test_data_not_cropped', args.yolo_dataset_path + '/images/val', args.yolo_dataset_path + '/labels/val')
            
# перебрать все изображения в папке
# для каждого из них считать боксы и точки
# нормализовать данные
# составить строчку
# записать в файл и переименовать изображение просто числом
