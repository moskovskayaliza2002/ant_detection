import json
import numpy as np
import glob
import os
import shutil
import argparse
import time


def write_txt(list_of_lists, filename):
    # Записывает лист в txt файл
    str_list = []
    for i in list_of_lists:
        if i != 0:
            int_s = [int(a) for a in i]
            s = ' '.join(map(str, int_s)) + "\n"
            str_list.append(s)
    with open(filename, 'w') as file:
        file.writelines(str_list)
        file.close()


def read_json(root_path, path, max_obj = 50):
    # Основная функция, перебирает все изображения и сохраняет измененные на новый диапазон координаты боксов и ключевых точек в формат, для считывания модели.
    
    keypoints_path = root_path + '/keypoints'
    bboxes_path = root_path + '/bboxes'
    dir_size = len(glob.glob(root_path + '/images' + '/*'))
    for i in [keypoints_path, bboxes_path]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i)
    
    #head_list = [] # shape [N_im, max_obj, 2]
    #abdomen_list = [] # shape [N_im, max_obj, 2]
    #bboxes_list = [] # shape [N_im, max_obj, 4]
    
    with open(path) as f:
        data = json.load(f)
    for img in data:
        number = int(img['img'][img['img'].rfind('e') + 1 : img['img'].rfind('.')])
        single_head_list = [0] * max_obj
        single_abdomen_list = [0] * max_obj
        count_head = 0
        count_abdimen = 0
        
        if not 'kp-1' in img:
            print(f"Skipped {img['img']}")
            continue
        for kps in img['kp-1']:
            label = kps['keypointlabels'][0]
            kp_x = kps['x']
            kp_y = kps['y']
            
            if label == 'Abdomen':
                single_abdomen_list[count_abdimen] = [conv_x(kp_x), conv_y(kp_y)]
                count_abdimen += 1
            elif label == 'Head':
                single_head_list[count_head] = [conv_x(kp_x), conv_y(kp_y)]
                count_head += 1
                
        
        single_bboxes_list = [0] * max_obj
        for i, bboxes in enumerate(img['label']):
            xmin = bboxes['x']
            ymin = bboxes['y']
            xmax = bboxes['x'] + bboxes['width']
            ymax = bboxes['y'] + bboxes['height']
            
            if bboxes['rectanglelabels'][0] == 'Ant':
                single_bboxes_list[i] = [conv_x(xmin), conv_y(ymin), conv_x(xmax), conv_y(ymax)]
        
        
        # ТУТ СОПОСТАВЛЕНИЕ И СОХРАНЕНИЕ ПО ИМЕНИ ИЗОБРАЖЕНИЯ + объединение
        if separator(single_head_list, single_bboxes_list) and separator(single_abdomen_list, single_bboxes_list):
            compared_h, compared_a, _ = correct_comparison(single_head_list, single_abdomen_list, single_bboxes_list)
            single_keypoints_list = join_keypoints(compared_h, compared_a)
            bb_filename = bboxes_path + '/bbox' + str(number) + '.txt'
            write_txt(single_bboxes_list, bb_filename)
            k_filename = keypoints_path + '/keypoint' + str(number) + '.txt'
            write_txt(single_keypoints_list, k_filename)
            print(f'№ {number}, ants: {np.count_nonzero(single_bboxes_list)}')
        else:
            print(f"Skipped {img['img']}")
    
    return 0

def separator(h, b):
    #Функция, что ищет ключевые точки, попадающие в несколько боксов.
    d_h = {}
    N_ob = len(h)
    single_im_h = [0] * N_ob
    for j in range(N_ob):
        if b[j] == 0:
            break
        else:
            xmin, ymin, xmax, ymax = b[j][0], b[j][1], b[j][2], b[j][3]
            all_head_in = []
            for g in range(N_ob):
                if h[g] != 0:
                    x_h, y_h = h[g][0], h[g][1]
                    if (xmin < x_h < xmax) and (ymin < y_h < ymax):
                        #single_im_h[j] = [x_h, y_h]
                        all_head_in.append([x_h, y_h])
            if len(all_head_in) == 1:
                single_im_h[j] = all_head_in[0]
            else:
                d_h[j] = all_head_in
    #print(d_h)
    if d_h != {}:
        keys = d_h.keys()
        values = d_h.values()
        repeads = max(len(l) for l in values)
        for _ in range(repeads):
            for key in keys:
                if type(d_h[key]) != float:
                    arr = d_h[key]
                    for val in d_h[key]:
                        if val in single_im_h:
                            arr.remove(val)
                            d_h[key] = arr
                            #print(type(arr[0] != float) and len(arr[0]) == 1)
                            if type(arr[0] != float) and len(arr) == 1:
                                d_h[key] = arr[0]
                                single_im_h[key] = arr[0]
    #print(values)
    #print(d_h[0])
    #print(d_h)
    exit_flag = False
    if d_h == {}:
        return True
    else:
        for key in keys:
            arr = d_h[key]
            if arr == []:
                continue 
            exit_flag = type(arr[0]) == float
        return exit_flag
        
    
def correct_comparison(h, a, b):
    # Подбирает ключевые точки под соответствующие боксы
    N_ob = len(h)
    single_im_h = [0] * N_ob
    single_im_a = [0] * N_ob
    for j in range(N_ob):
        if b[j] == 0:
            break
        else:
            xmin, ymin, xmax, ymax = b[j][0], b[j][1], b[j][2], b[j][3]
            for g in range(N_ob):
                if h[g] != 0:
                    x_h, y_h = h[g][0], h[g][1]
                    if (xmin < x_h < xmax) and (ymin < y_h < ymax):
                        single_im_h[j] = [x_h, y_h]
                        
            for g in range(N_ob):
                if a[g] != 0:
                    x_a, y_a = a[g][0], a[g][1]
                    if (xmin < x_a < xmax) and (ymin < y_a < ymax):
                        single_im_a[j] = [x_a, y_a]
    
    return single_im_h, single_im_a, b
        
        
def conv_x(old):
    # Переводит координату х в новую систему координат
    old_min = new_min = 0
    old_range = 100 - 0  
    new_range = 1920 - 0 # 1920 - 0
    
    converted = (((old - old_min) * new_range) / old_range) + new_min
    return converted

def conv_y(old):
    # Переводит координату у в новую систему координат
    old_min = new_min = 0
    old_range = 100 - 0  
    new_range = 1080 - 0 # 1080 - 0
    
    converted = (((old - old_min) * new_range) / old_range) + new_min
    return converted
        
def join_keypoints(head, abdomen):
    # Соединяет ключевые точки головы и брюшка в одну запись
    single_im_k = []
    for j in range(len(head)):
        if abdomen[j] != 0:
            x_a, y_a = abdomen[j][0], abdomen[j][1]
            x_h, y_h = head[j][0], head[j][1]
            single_im_k.append([x_a, y_a, x_h, y_h])
                
        else:
            single_im_k.append(0)
    return single_im_k
    
 
def create_dataset(root_path, json_path): # Не рабочая функция
    keypoints_path = root_path + '/keypoints'
    bboxes_path = root_path + '/bboxes'
    dir_size = len(glob.glob(root_path + '/images' + '/*'))
    for i in [keypoints_path, bboxes_path]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i)
    
    head_list, abdomen_list, bboxes_list = read_json(json_path)
    head_list, abdomen_list, bboxes_list = correct_comparison(head_list, abdomen_list, bboxes_list)
    keypoints = join_keypoints(head_list, abdomen_list)
    real_im_number = 0
    for i in reversed(range(dir_size)):
        bb_filename = bboxes_path + '/bbox' + str(real_im_number) + '.txt'
        write_txt(bboxes_list[i], bb_filename)
        k_filename = keypoints_path + '/keypoint' + str(real_im_number) + '.txt'
        write_txt(keypoints[i], k_filename)
        real_im_number += 1
        
        
if __name__ == '__main__':
    #root_path - is a forder, where folger with images and a json file with annotation lies. it will create there two folders for bboxes amd keypoints txt files.
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', nargs='?', default='/home/ubuntu/ant_detection/new_train_data', help="Specify directory to create dataset", type=str)
    parser.add_argument('json_path', nargs='?', default='/home/ubuntu/ant_detection/new_train_data/new.json', help="Specify path to json file", type=str)
    args = parser.parse_args()
    ROOT = args.root_path
    JSON = args.json_path
    read_json(ROOT, JSON)
    #create_dataset(ROOT, JSON)
