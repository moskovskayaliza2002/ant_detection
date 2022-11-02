import os
import cv2
import matplotlib.pyplot as plt #для отладки, убери потом
import shutil
import time #для отладки, убери потом
import argparse

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
        

def crop_one_im(img, delta_w, delta_h):
    # Возвращает 4 куска изображения и границы разреза
    crop_w = 0
    crop_h = 0
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    
    crop_w = width_cutoff
    
    left1 = img[:, :width_cutoff + delta_w]
    right1 = img[:, width_cutoff - delta_w:]
    # finish vertical devide image
    # At first Horizontal devide left1 image #
    #rotate image LEFT1 to 90 CLOCKWISE
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    
    crop_h = width_cutoff
    
    l1 = img[:, width_cutoff - delta_h:]
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    l2 = img[:, :width_cutoff + delta_h]
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    
    r1 = img[:, width_cutoff - delta_h:]
    r1 = cv2.rotate(r1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    r2 = img[:, :width_cutoff + delta_h]
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


def conv_x(old, old_min, new_min, old_max, new_max):
    old_range = old_max - old_min  
    new_range = new_max - new_min 
    
    converted = (((old - old_min) * new_range) / old_range) + new_min
    if converted < 0:
        print(f'{converted} = ((({old} - {old_min}) * {new_range}) / {old_range}) + {new_min}')
    return int(converted)


def conv_y(old, old_min, new_min, old_max, new_max):
    old_range = old_max - old_min  
    new_range = new_max - new_min 
    
    converted = (((old - old_min) * new_range) / old_range) + new_min
    if converted < 0:
        print(f'{converted} = ((({old} - {old_min}) * {new_range}) / {old_range}) + {new_min}')
    return int(converted)


def resize_bboxes_kps(bboxes, kps, left_x, left_y, right_x, right_y):
    # Изменяет размеры боксов
    new_list_bboxes = []
    new_list_kps = []
    k = 0
    flag = True
    neg = 0
    for i in range(len(bboxes)):
        # [xmin, ymin, xmax, ymax]
        ymin, xmin, ymax, xmax = bboxes[i][1], bboxes[i][0], bboxes[i][3], bboxes[i][2]
        
        # проверка на то, что границы не пересекают муравья
        if (ymin < left_y < ymax or ymin < right_y < ymax) and (xmax > left_x and xmin < right_x):
            flag = False
        if (xmin < left_x < xmax or xmin < right_x < xmax) and (ymax > left_y and ymin < right_y): #left_y < ymin < right_y:
            flag = False
        # проверка на то, что это не бокс за границей обрезанной области
        if flag == True and not(xmax <= left_x or ymax <= left_y or ymin >= right_y or xmin >= right_x):
            #new_list_bboxes.append([xmin - left_x, ymin - left_y, xmax - left_x, ymax - left_y])
            #resize to 300 300
            #x_f = conv_x(xmin - left_x, left_x, 0, right_x, 300)
            #y_f = conv_y(ymin - left_y, left_y, 0, right_y, 300)
            #x_s = conv_x(xmax - left_x, left_x, 0, right_x, 300)
            #y_s = conv_y(ymax - left_y, left_y, 0, right_y, 300)
            x_f = conv_x(xmin, left_x, 0, right_x, 300)
            y_f = conv_y(ymin, left_y, 0, right_y, 300)
            x_s = conv_x(xmax, left_x, 0, right_x, 300)
            y_s = conv_y(ymax, left_y, 0, right_y, 300)
            new_list_bboxes.append([x_f, y_f, x_s, y_s])
            l = [x_f, y_f, x_s, y_s]
            neg += sum([num for num in l if num < 0])
            #x_a = conv_x(kps[i][0] - left_x, left_x, 0, right_x, 300)
            #y_a = conv_y(kps[i][1] - left_y, left_y, 0, right_y, 300)
            #x_h = conv_x(kps[i][2] - left_x, left_x, 0, right_x, 300)
            #y_h = conv_y(kps[i][3] - left_y, left_y, 0, right_y, 300)
            x_a = conv_x(kps[i][0], left_x, 0, right_x, 300)
            y_a = conv_y(kps[i][1], left_y, 0, right_y, 300)
            x_h = conv_x(kps[i][2], left_x, 0, right_x, 300)
            y_h = conv_y(kps[i][3], left_y, 0, right_y, 300)
            new_list_kps.append([x_a, y_a, x_h, y_h])
            #new_list_kps.append([x_a - left_x, y_a - left_y, x_h - left_x, y_h - left_y])
            k += 1
    print(f"Negative numbers {neg}")
    return k, new_list_bboxes, new_list_kps


def verification(bboxes, kp, l1, l2, r1, r2, crop_w, crop_h, delta_w, delta_h, s_path, counter):
    l1_ants_counter, l1_bb, l1_kps = resize_bboxes_kps(bboxes, kp, 0, 0, crop_w + delta_w, crop_h + delta_h)
    #neg_nos = [num for num in l1_bb if num < 0]
    #print("Negative numbers in l1: ", *neg_nos)
    if l1_ants_counter != 0:
        print('l1')
        print(counter + 1)
        resized_l1 = cv2.resize(l1, (300, 300), interpolation = cv2.INTER_AREA)
        cv2.imwrite(s_path + '/images' + '/image' + str(counter + 1) + '.png', resized_l1)
        write_bbox(l1_bb, s_path + '/bboxes' + '/bbox' + str(counter + 1) + '.txt')
        write_bbox(l1_kps, s_path + '/keypoints' + '/keypoint' + str(counter + 1) + '.txt')
        counter += 1
    
    l2_ants_counter, l2_bb, l2_kps = resize_bboxes_kps(bboxes, kp, 0, crop_h - delta_h, crop_w + delta_w, 2 * crop_h)
    #neg_nos = [num for num in l2_bb if num < 0]
    #print("Negative numbers in l2: ", *neg_nos)
    if l2_ants_counter != 0:
        print('l2')
        print(counter + 1)
        resized_l2 = cv2.resize(l2, (300, 300), interpolation = cv2.INTER_AREA)
        cv2.imwrite(s_path + '/images' + '/image' + str(counter + 1) + '.png', resized_l2)
        write_bbox(l2_bb, s_path + '/bboxes' + '/bbox' + str(counter + 1) + '.txt')
        write_bbox(l2_kps, s_path + '/keypoints' + '/keypoint' + str(counter + 1) + '.txt')
        counter += 1
    
    r1_ants_counter, r1_bb, r1_kps = resize_bboxes_kps(bboxes, kp, crop_w - delta_w, 0, 2 * crop_w, crop_h + delta_h)
    #neg_nos = [num for num in r1_bb if num < 0]
    #print("Negative numbers in r1: ", *neg_nos)
    if r1_ants_counter != 0:
        print('r1')
        print(counter + 1)
        resized_r1 = cv2.resize(r1, (300, 300), interpolation = cv2.INTER_AREA)
        cv2.imwrite(s_path + '/images' + '/image' + str(counter + 1) + '.png', resized_r1)
        write_bbox(r1_bb, s_path + '/bboxes' + '/bbox' + str(counter + 1) + '.txt')
        write_bbox(r1_kps, s_path + '/keypoints' + '/keypoint' + str(counter + 1) + '.txt')
        counter += 1
    
    r2_ants_counter, r2_bb, r2_kps = resize_bboxes_kps(bboxes, kp, crop_w - delta_w, crop_h - delta_h, 2 * crop_w, 2 * crop_h)
    #neg_nos = [num for num in r2_bb if num < 0]
    #print("Negative numbers in r2: ", *neg_nos)
    if r2_ants_counter != 0:
        print('r2')
        print(counter + 1)
        resized_r2 = cv2.resize(r2, (300, 300), interpolation = cv2.INTER_AREA)
        cv2.imwrite(s_path + '/images' + '/image' + str(counter + 1) + '.png', resized_r2)
        write_bbox(r2_bb, s_path + '/bboxes' + '/bbox' + str(counter + 1) + '.txt')
        write_bbox(r2_kps, s_path + '/keypoints' + '/keypoint' + str(counter + 1) + '.txt')
        counter += 1
        
    return counter


def cropper(new_root_path, old_root_path, overlay_w, overlay_h):
    
    new_keypoints_path = new_root_path + '/keypoints'
    new_bboxes_path = new_root_path + '/bboxes'
    new_images_path = new_root_path + '/images'

    for i in [new_keypoints_path, new_bboxes_path, new_images_path]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i) 
    
    #c count + 1 начнется нумерация
    count = -1
    for f in os.scandir(old_root_path + '/images'):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            original_image = cv2.imread(f.path)
            #image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            number = int(f.path[f.path.rfind('e') + 1 : f.path.rfind('.')])
            #print(number)
            original_bboxs = find_bbox(number, old_root_path)
            _, original_keypoints = find_kp(number, old_root_path)
            left1, left2, right1, right2, w, h = crop_one_im(original_image, overlay_w, overlay_h)
            count = verification(original_bboxs, original_keypoints, left1, left2, right1, right2, w, h, overlay_w, overlay_h, new_root_path, count)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('old_root', nargs='?', default='/home/ubuntu/ant_detection/real_im_annot/train_data', help="Specify directory with old dataset, there should be such folders as images, keypoints and bboxes", type=str)
    parser.add_argument('new_root', nargs='?', default='/home/ubuntu/ant_detection/real_im_annot/train_data/crop', help="Specify path for new data directory", type=str)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    args = parser.parse_args()
    
    old_root = args.old_root
    new_root = args.new_root
    overlay_w = args.overlay_w
    overlay_h = args.overlay_h
    
    cropper(new_root, old_root, overlay_w, overlay_h)
    
