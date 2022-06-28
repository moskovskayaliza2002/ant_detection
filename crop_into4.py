import os
import cv2
import argparse
import torch
import glob
import time
import matplotlib.pyplot as plt
import shutil

def read_boxes(bbox_path):
    # Считывает боксы
    bboxes_original = []
    with open(bbox_path) as f:
        for i in f:
            x_min, y_min, x_max, y_max = map(int, i[:-1].split(' '))
            bboxes_original.append([x_min, y_min, x_max, y_max])
    return bboxes_original
     
     
def write_bbox(bbox, filename):
    # Записывает список координат в файл
    str_list = []
    for i in bbox:
        s = ' '.join(map(str, i)) + "\n"
        str_list.append(s)
    with open(filename, 'w') as file:
        file.writelines(str_list)
        file.close()
        
     
def resize_bboxes(bboxes, left_x, left_y, right_x, right_y):
    # Изменяет размеры боксов
    new_list_bboxes = []
    k = 0
    for i in bboxes:
        # [xmin, ymin, xmax, ymax]
        ymin, xmin, ymax, xmax = i[1], i[0], i[3], i[2]
        # проверка на то, что это не бокс за границей обрезанной области
        if not(xmax < left_x or ymax < left_y or ymin > right_y or xmin > right_x):
            new_list_bboxes.append([xmin - left_x, ymin - left_y, xmax - left_x, ymax - left_y])
            k += 1
    return k, new_list_bboxes


def resize_keypoints(kp, left_x, left_y, right_x, right_y):
    # Изменяет размеры ключевых точек
    new_list_kp = []
    k = 0
    for i in kp:
        x_a, y_a, x_h, y_h = i[0], i[1], i[2], i[3]
        if left_x < x_a < right_x and left_y < y_a < right_y:
            if left_x < x_h < right_x and left_y < y_h < right_y:
                new_list_kp.append([x_a - left_x, y_a - left_y, x_h - left_x, y_h - left_y])
                k += 1
    print(f'Муравьев найдено: {k}')
    return new_list_kp


def crop_one_im(img):
    # Возвращает 4 куска изображения и границы разреза
    crop_w = 0
    crop_h = 0
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    
    crop_w = width_cutoff
    
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]
    # finish vertical devide image
    # At first Horizontal devide left1 image #
    #rotate image LEFT1 to 90 CLOCKWISE
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    
    crop_h = width_cutoff
    
    l1 = img[:, width_cutoff:]
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    l2 = img[:, :width_cutoff]
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    
    r1 = img[:, width_cutoff:]
    r1 = cv2.rotate(r1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    r2 = img[:, :width_cutoff]
    r2 = cv2.rotate(r2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #f, axarr = plt.subplots(2,2)
    #axarr[0,0].imshow(l1)
    #axarr[0,0].set_title('l1')
    #axarr[0,1].imshow(r1)
    #axarr[0,1].set_title('r1')
    #axarr[1,0].imshow(l2)
    #axarr[1,0].set_title('l2')
    #axarr[1,1].imshow(r2)
    #axarr[1,1].set_title('r2')
    #plt.show()
    return l1, l2, r1, r2, crop_w, crop_h
    

def verification(bboxes, kp, l1, l2, r1, r2, crop_w, crop_h, save_path, count):
    # Выполняет проверку, что боксы не пересекают линии разреза, сохранение в случе успешной проверки
    flag_l1 = True
    flag_l2 = True
    flag_r1 = True
    flag_r2 = True
    for i in bboxes:
        xmin = i[0]
        ymin = i[1]
        xmax = i[2]
        ymax = i[3]
        if xmin < crop_w < xmax and ymax < crop_h:
            flag_l1 = flag_r1 = False
        if xmin < crop_w < xmax and ymin > crop_h:
            flag_l2 = flag_r2 = False
        if ymin < crop_h < ymax and xmax < crop_w:
            flag_l1 = flag_l2 = False
        if ymin < crop_h < ymax  and xmin > crop_w:
            flag_r1 = flag_r2 = False
        if ymin < crop_h < ymax and xmin < crop_w < xmax:
            flag_l1 = flag_r1 = flag_l2 = flag_r2 = False
    print("новое изображение")
    if flag_l1:
        ants_counter, b_l1 = resize_bboxes(bboxes, 0, 0, crop_w, crop_h)
        if ants_counter != 0:
            print('l1', count + 1)
            cv2.imwrite(save_path + '/images' + '/image' + str(count + 1) + '.png', l1)
            kp_l1 = resize_keypoints(kp, 0, 0, crop_w, crop_h)
            write_bbox(b_l1, save_path + '/bboxes' + '/bbox' + str(count + 1) + '.txt')
            write_bbox(kp_l1, save_path + '/keypoints' + '/keypoint' + str(count + 1) + '.txt')
            count += 1
    if flag_l2:
        ants_counter, b_l2 = resize_bboxes(bboxes, 0, crop_h, crop_w, 2 * crop_h)
        if ants_counter != 0:
            print('l2', count + 1)
            cv2.imwrite(save_path + '/images' + '/image' + str(count + 1) + '.png', l2)
            kp_l2 = resize_keypoints(kp, 0, crop_h, crop_w, 2 * crop_h)
            write_bbox(b_l2, save_path + '/bboxes' + '/bbox' + str(count + 1) + '.txt')
            write_bbox(kp_l2, save_path + '/keypoints' + '/keypoint' + str(count + 1) + '.txt')
            count += 1
    if flag_r1:
        ants_counter, b_r1 = resize_bboxes(bboxes, crop_w, 0, 2 * crop_w, crop_h)
        if ants_counter != 0:
            print('r1', count + 1)
            cv2.imwrite(save_path + '/images' + '/image' + str(count + 1) + '.png', r1)
            kp_r1 = resize_keypoints(kp, crop_w, 0, 2 * crop_w, crop_h)
            write_bbox(b_r1, save_path + '/bboxes' + '/bbox' + str(count + 1) + '.txt')
            write_bbox(kp_r1, save_path + '/keypoints' + '/keypoint' + str(count + 1) + '.txt')
            count += 1
    if flag_r2:
        ants_counter, b_r2 = resize_bboxes(bboxes, crop_w, crop_h, 2 * crop_w, 2 * crop_h)
        if ants_counter != 0:
            print('r2', count + 1)
            cv2.imwrite(save_path + '/images' + '/image' + str(count + 1) + '.png', r2)
            kp_r2 = resize_keypoints(kp, crop_w, crop_h, 2 * crop_w, 2 * crop_h)
            write_bbox(b_r2, save_path + '/bboxes' + '/bbox' + str(count + 1) + '.txt')
            write_bbox(kp_r2, save_path + '/keypoints' + '/keypoint' + str(count + 1) + '.txt')
            count += 1
    return count

    
def crop_data(root_path):
    # Создает директории, основная процедура
    save_p = root_path + '/1to4'
        
    new_keypoints_path = save_p + '/keypoints'
    new_bboxes_path = save_p + '/bboxes'
    new_images_path = save_p + '/images'

    for i in [new_keypoints_path, new_bboxes_path, new_images_path]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i)    
    # Read images
    data_path = root_path + '/TRAIN_on_real'
    orig_image_path = data_path + '/images'
    counter = -1
    dir_size = len(glob.glob(orig_image_path + '/*'))
    all_images = [0] * (dir_size + 1) # если ошибка с новым датасетом, то она тут
    for f in os.scandir(orig_image_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            number = int(f.path[f.path.rfind('e') + 1 : f.path.rfind('.')])
            all_images[number] = f.path
    all_images.remove(0)
    for img in all_images:
            # read orig image and bboxes
            original_image = cv2.imread(img)
            number = int(img[img.rfind('e') + 1 : img.rfind('.')])
            orig_bboxes = read_boxes(data_path + '/bboxes' + '/bbox' + str(number) + '.txt')
            orig_keyp = read_boxes(data_path + '/keypoints' + '/keypoint' + str(number) + '.txt')
            # crop
            left1, left2, right1, right2, w, h = crop_one_im(original_image)
            # check crossing
            counter = verification(orig_bboxes, orig_keyp, left1, left2, right1, right2, w, h, save_p, counter)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', nargs='?', default='/home/ubuntu/ant_detection', help="Specify directory to create dataset", type=str)
    args = parser.parse_args()
    ROOT = args.root_path
    crop_data(ROOT)
            
