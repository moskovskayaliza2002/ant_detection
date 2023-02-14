import os
import cv2
import matplotlib.pyplot as plt #для отладки, убери потом
import shutil
import argparse
import numpy as np

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
    #neg = 0
    nouse_ant = 0
    for i in range(len(bboxes)):
        # [xmin, ymin, xmax, ymax]
        ymin, xmin, ymax, xmax = bboxes[i][1], bboxes[i][0], bboxes[i][3], bboxes[i][2]
        
        # проверка на то, что границы не пересекают муравья
        if ((ymin < left_y < ymax or ymin < right_y < ymax) and (xmax > left_x and xmin < right_x)) or ((xmin < left_x < xmax or xmin < right_x < xmax) and (ymax > left_y and ymin < right_y)):
            flag = False
            nouse_ant += 1
        else:
            flag = True
        #if (xmin < left_x < xmax or xmin < right_x < xmax) and (ymax > left_y and ymin < right_y): #left_y < ymin < right_y:
        #    flag = False
        # проверка на то, что это не бокс за границей обрезанной области
        if flag == True and not(xmax <= left_x or ymax <= left_y or ymin >= right_y or xmin >= right_x):
            #new_list_bboxes.append([xmin - left_x, ymin - left_y, xmax - left_x, ymax - left_y])
            #resize to 224 224
            x_f = conv_x(xmin, left_x, 0, right_x, 224)
            y_f = conv_y(ymin, left_y, 0, right_y, 224)
            x_s = conv_x(xmax, left_x, 0, right_x, 224)
            y_s = conv_y(ymax, left_y, 0, right_y, 224)
            new_list_bboxes.append([x_f, y_f, x_s, y_s])
            l = [x_f, y_f, x_s, y_s]
            #neg += sum([num for num in l if num < 0])
            x_a = conv_x(kps[i][0], left_x, 0, right_x, 224)
            y_a = conv_y(kps[i][1], left_y, 0, right_y, 224)
            x_h = conv_x(kps[i][2], left_x, 0, right_x, 224)
            y_h = conv_y(kps[i][3], left_y, 0, right_y, 224)
            new_list_kps.append([x_a, y_a, x_h, y_h])
            #new_list_kps.append([x_a - left_x, y_a - left_y, x_h - left_x, y_h - left_y])
            k += 1
    #print(f"Negative numbers {neg}")
    return k, nouse_ant, new_list_bboxes, new_list_kps


def crop_one_im(img, splits_vertical, splits_horizontal, delta_w, delta_h):
    crop_w = 0
    crop_h = 0
    
    height = img.shape[0]
    width = img.shape[1]
    
    vertical_split_images = []
    
    # start vertical devide image
    width_cutoff = width // splits_vertical
    crop_w = width_cutoff
    start = 0
    finish = width_cutoff + delta_w
    for i in range(splits_vertical):
        vertical_split_images.append(img[:, start:finish])
        if len(vertical_split_images) < splits_vertical - 1:
            start = finish - 2 * delta_w
            finish += width_cutoff
        else:
            start = finish - 2 * delta_w
            finish = width
    
    all_images = [[0 for i in range(splits_vertical)] for j in range(splits_horizontal)]
    
    for i, ver_im in enumerate(vertical_split_images):
        img = cv2.rotate(ver_im, cv2.ROTATE_90_CLOCKWISE)
        height = img.shape[0]
        width = img.shape[1]
        width_cutoff = width // splits_horizontal
        crop_h = width_cutoff
        start = 0
        finish = width_cutoff + delta_h
        counter_splits = 0
        for j in range(splits_horizontal):
            new_image = img[:, start:finish]
            new_image = cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            counter_splits += 1
            if counter_splits < splits_horizontal - 1:
                start = finish - 2 * delta_h
                finish += width_cutoff
            else:
                start = finish - 2 * delta_h
                finish = width
            all_images[j][i] = new_image
    all_images.reverse()
    '''
    f, axarr = plt.subplots(splits_horizontal,splits_vertical)
    axarr[0,0].imshow(all_images[0][0])
    axarr[0,0].set_title('[0][0]')
    axarr[0,1].imshow(all_images[0][1])
    axarr[0,1].set_title('[0][1]')
    axarr[0,2].imshow(all_images[0][2])
    axarr[0,2].set_title('[0][2]')
    axarr[1,0].imshow(all_images[1][0])
    axarr[1,0].set_title('[1][0]')
    axarr[1,1].imshow(all_images[1][1])
    axarr[1,1].set_title('[1][1]')
    axarr[1,2].imshow(all_images[1][2])
    axarr[1,2].set_title('[1][2]')
    axarr[2,0].imshow(all_images[2][0])
    axarr[2,0].set_title('[2][0]')
    axarr[2,1].imshow(all_images[2][1])
    axarr[2,1].set_title('[2][1]')
    axarr[2,2].imshow(all_images[2][2])
    axarr[2,2].set_title('[2][2]')
    plt.show()
    '''
    return all_images, crop_w, crop_h


def verification(bb, kps, images, crop_w, crop_h, delta_w, delta_h, w, h, s_path, counter):
    num_lines = len(images)
    num_rows = len(images[0]) 
    for i, line in enumerate(images):
        for j, image in enumerate(line):
            if j == 0:
                left_x = 0
                right_x = crop_w + delta_w
            elif j == num_rows - 1:
                left_x = w - crop_w - delta_w
                right_x = w
            else:
                left_x = j * crop_w - delta_w
                right_x = left_x + crop_w + 2 * delta_w
                
            if i == 0:
                left_y = 0
                right_y = crop_h + delta_h
            elif i == num_lines - 1:
                left_y = h - crop_h - delta_h
                right_y = h
            else:
                left_y = i * crop_h - delta_h
                right_y = left_y + crop_h + 2 * delta_h
                
            ants_counter, noise, bboxes, keypoints = resize_bboxes_kps(bb, kps, left_x, left_y, right_x, right_y)
            if ants_counter != 0 and noise == 0:
                resized = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
                cv2.imwrite(s_path + '/images' + '/image' + str(counter + 1) + '.png', resized)
                write_bbox(bboxes, s_path + '/bboxes' + '/bbox' + str(counter + 1) + '.txt')
                write_bbox(keypoints, s_path + '/keypoints' + '/keypoint' + str(counter + 1) + '.txt')
                counter += 1
                
    return counter


def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 1
    keypoints_classes_ids2names = {0: 'A', 1: 'H'}
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (255,0,0), 1)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 2, (255,0,0), 2)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40))
        plt.imshow(image)
        plt.show(block=True)

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 2, (0,255,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)
        plt.show(block=True)
        
        
def cropper(old_root_path, new_root_path, splits_vertical, splits_horizontal, delta_w, delta_h):
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
            h, w, _ = original_image.shape
            number = int(f.path[f.path.rfind('e') + 1 : f.path.rfind('.')])
            original_bboxs = find_bbox(number, old_root_path)
            _, original_keypoints = find_kp(number, old_root_path)
            new_images, cr_w, cr_h = crop_one_im(original_image, splits_vertical, splits_horizontal, delta_w, delta_h)
            count = verification(original_bboxs, original_keypoints, new_images, cr_w, cr_h, delta_w, delta_h, w, h, new_root_path, count)
    
    
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('old_root', nargs='?', default='/home/ubuntu/ant_detection/dataset/Train_not_cropped', help="Specify directory with old dataset, there should be such folders as images, keypoints and bboxes", type=str)
    parser.add_argument('new_root', nargs='?', default='/home/ubuntu/ant_detection/new_dataset/Train_data', help="Specify path for new data directory", type=str)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    parser.add_argument('splits_vertical', nargs='?', default=4, help="Num of pictures in w-axis", type=int)
    parser.add_argument('splits_horizontal', nargs='?', default=3, help="Num of pictures in h-axis", type=int)
    args = parser.parse_args()
    
    cropper(args.old_root, args.new_root, args.splits_vertical, args.splits_horizontal, args.overlay_w, args.overlay_h)
    
    #test_image = cv2.imread('/home/ubuntu/ant_detection/new_dataset/Train_data/images/image9.png')
    #test_bb = read_boxes('/home/ubuntu/ant_detection/new_dataset/Train_data/bboxes/bbox9.txt')
    #test_kp, _ = find_kp(9, '/home/ubuntu/ant_detection/new_dataset/Train_data')
    #visualize(test_image, test_bb, test_kp)
