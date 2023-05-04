import cv2
import numpy as np
import time
import copy
import yaml
import argparse
from collections import OrderedDict
from universal_RCNN_test import read_yaml
#moskovskaya_ed@rrcki.ru

def draw_circle_input(event, x, y, flags, param):
    global img
    global cache
    global real_coords
    global pix_coords
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cache = copy.deepcopy(img)
        pix_coords.append([x, y])
        print('Пиксельные координаты (x:',x,',y:',y,')')
        real_x, real_y = map(float, input("Реальные координаты (x,y): ").split(","))
        real_coords.append([real_x, real_y])
        str1 = '(x:'+ str(x) + ',y:'+ str(y) + ')'
        str2 = '(x:'+ str(real_x) + ',y:'+ str(real_y) + ')'
        cv2.putText(img,str1 , (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness=1)
        cv2.circle(img,(x,y),3,(0, 0, 255),-1)
        cv2.putText(img,str2 , (x, y-20), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 255, 0), thickness=1)
        
    if event == cv2.EVENT_MBUTTONDBLCLK:
        img = copy.deepcopy(cache)
        pix_coords.pop()
        real_coords.pop()
        cv2.imshow('src', img)
        
def draw_circle(event, x, y, flags, param):
    global coords
    global img
    global cache
    global real_coords
    global pix_coords
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cache = copy.deepcopy(img)
        pix_coords.append([x, y])
        print('Пиксельные координаты (x:',x,',y:',y,')')
        str1 = '(x:'+ str(x) + ',y:'+ str(y) + ')'
        cv2.putText(img,str1 , (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness=1)
        cv2.circle(img,(x,y),3,(0, 0, 255),-1)
        real = find_point_of_matrix(coords, [x, y])
        str2 = '(x:'+ str(round(real[0],2)) + ',y:'+ str(round(real[1],2)) + ')'
        cv2.putText(img,str2 , (x, y-25), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 255, 0), thickness=2)
        
    if event == cv2.EVENT_MBUTTONDBLCLK:
        img = copy.deepcopy(cache)
        pix_coords.pop()
        cv2.imshow('src', img)
        
        
def save_coords_to_yaml(yaml_path, p_coords, r_coords, video_path):
    with open(yaml_path, 'w') as f:
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        yaml.dump(OrderedDict({'video_path': video_path, 'pixel_coordinates': p_coords, 'real_coordinates': r_coords}), f)
        
def read_coords_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        datas = list(yaml.safe_load_all(f))
        return datas[0]['pixel_coordinates'], datas[0]['real_coordinates']
    
    
def correlate_points(video_path, num_points, yaml_path):
    cap = cv2.VideoCapture(args.video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(args.video_path)
        cv2.waitKey(1000)
        print("Openning the file...")
    
    flag, frame = cap.read()
    if flag:
        global img  
        global cache  
        img = frame
        global real_coords
        global pix_coords
        real_coords = []
        pix_coords = []
        #img = cv2.imread(path, 1)
        cache = copy.deepcopy(img)
        cv2.namedWindow('src', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('src',draw_circle_input)
        exit_flag = True
        while(len(pix_coords) != args.num_points):
            cv2.imshow('src',img)
            if cv2.waitKey (100) == ord ('q'): # Нажмите q, чтобы выйти
                break
            #cv2.imshow('src',img)
        cv2.destroyAllWindows()
    if len(pix_coords) == args.num_points:
        print("/-----------------------------------------------------------------------/")
        print("INFO: Нажмите Esс чтобы не сохранять координаты и Enter чтобы сохранить")
        print("/-----------------------------------------------------------------------/")
        while(1):
            cv2.namedWindow('src', cv2.WINDOW_NORMAL)
            cv2.imshow('src', img)
            next_step = cv2.waitKey(0)
            if next_step == 27:
                print("_______УДАЛЕНО_______")
                break
            if next_step == 13:
                save_coords_to_yaml(args.yaml_path, pix_coords, real_coords, args.video_path)
                print("_______СОХРАНЕНО_______")
                break
        cv2.destroyAllWindows()
    
    return img
        
def find_point_of_matrix(coords, point):
    src_pts, dst_pts = coords
    #src_pts = np.array([[277, 123], [1250, 66], [1382, 1000],[20, 996]], dtype=np.float32)
    #dst_pts = np.array([[0, 0], [25, 0], [25, 25],[0, 25]], dtype=np.float32)
    matrix, mask = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts))
    matches_mask = mask.ravel().tolist() 
    #h,w,d = img.shape
    pts = np.float32(np.array([[point]])) 
    dst = cv2.perspectiveTransform(pts, matrix)
    return dst[0][0]
    
def find_matrix(coords):
    src_pts, dst_pts = coords
    #src_pts = np.array([[277, 123], [1250, 66], [1382, 1000],[20, 996]], dtype=np.float32)
    #dst_pts = np.array([[0, 0], [25, 0], [25, 25],[0, 25]], dtype=np.float32)
    matrix, mask = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts))
    return matrix

def save_matrix(yaml_path, matrix):
    matrix = matrix.tolist()
    print(matrix)
    with open(yaml_path, 'w') as f:
        yaml.dump({'matrix': matrix}, f)
        
def read_matrix(yaml_path):
    with open(yaml_path) as f:
        datas = list(yaml.safe_load_all(f))
        return datas[0]['matrix']
    
def find_points(im, yaml_path):
    global coords
    coords = read_coords_yaml(yaml_path)
    matrix = find_matrix(coords)
    path = yaml_path[:yaml_path.rfind('/')] + '/matrix.yml'
    save_matrix(path, matrix)
    print("INFO: Матрица сохранена")
    global img  
    global cache  
    global real_coords
    global pix_coords
    img = im
    real_coords = []
    pix_coords = []
    cache = copy.deepcopy(img)
    cv2.namedWindow('src', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('src',draw_circle)
    exit_flag = True
    
    print("INFO: Нажмите q, чтобы выйти")
    while(1):
        cv2.imshow('src',img)
        if cv2.waitKey (100) == ord ('q'):
            break
    cv2.destroyAllWindows()
    
    
def change_detections_to_real(detection_yaml, matrix_yaml):
    #Менятся порядок записи в файл, посмотри, как это влияет
    ANT_DATA = read_yaml(detection_yaml)
    print("Данные считаны")
    matrix = read_matrix(matrix_yaml)
    print("Матрица считана")
    matrix = np.array(matrix, dtype=np.float32)
    for frame in ANT_DATA['frames']:
        orig_bb = np.resize(np.float32(np.array(frame['bboxes'])), (len(frame['bboxes']), 2, 2))
        print(frame['bboxes'])
        orig_kps = np.resize(np.float32(np.array(frame['keypoints'])), (len(frame['bboxes']), 2, 2))
        print(cv2.perspectiveTransform(orig_bb, matrix).shape)
        frame['bboxes'] = np.resize(cv2.perspectiveTransform(orig_bb, matrix), (len(frame['bboxes']), 4)).tolist()
        frame['keypoints'] = cv2.perspectiveTransform(orig_kps, matrix).tolist()
        
    print("Преобразования закончены")
    name = detection_yaml[detection_yaml.rfind('/'):detection_yaml.rfind('.')]
    new_detection_yaml = detection_yaml[:detection_yaml.rfind('/')] + name + '_real_coords' + '.yml'
    with open(new_detection_yaml, 'w') as f:
        #data = yaml.dump({'name': filename}, f)
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        yaml.dump(OrderedDict({'name': ANT_DATA['name'], 'FPS': ANT_DATA['FPS'], 'weight': ANT_DATA['weight'], 'height': ANT_DATA['height']}), f)
        print(type(ANT_DATA['frames']))
        data = OrderedDict({"frames":ANT_DATA['frames']})
        yaml.dump(data, f)
    
if __name__ == '__main__':
    print("INFO: двойной клик левой кнопкой мыши поставит точку, двойной клик колесика отменит последнюю нарисованную точку")
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', nargs='?', default="/home/ubuntu/ant_detection/problems/full_video/18.08.20_Fp2_плос2.mp4", help="path to video for dynamic density analysis", type=str)
    parser.add_argument('--yaml_path', nargs='?', default='/home/ubuntu/ant_detection/problems/full_video/real_coords.yml', help="outputfile path with pixels and real coords", type=str)
    parser.add_argument('--num_points', nargs='?', default=4, help="number of points", type=int)
    args = parser.parse_args()
    
    #img = correlate_points(args.video_path, args.num_points, args.yaml_path)
    #find_points(img, args.yaml_path)
    change_detections_to_real("/home/ubuntu/ant_detection/problems/Track_gap/video0.yml", "/home/ubuntu/ant_detection/problems/Track_gap/matrix.yml")
