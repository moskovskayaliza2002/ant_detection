import cv2
import numpy as np
import time
import copy
import yaml
import argparse
from collections import OrderedDict

#moskovskaya_ed@rrcki.ru

def draw_circle(event, x, y, flags, param):
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
        
        
def save_coords_to_yaml(yaml_path, p_coords, r_coords, video_path):
    with open(yaml_path, 'w') as f:
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        yaml.dump(OrderedDict({'video_path': video_path, 'pixel_coordinates': p_coords, 'real_coordinates': r_coords}), f)
        
        
if __name__ == '__main__':
    print("INFO: двойной клик левой кнопкой мыши поставит точку, двойной клик колесика отменит последнюю нарисованную точку")
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', nargs='?', default="/home/ubuntu/ant_detection/problems/full_video/18.08.20 Fp2 плос2.mp4", help="path to video for dynamic density analysis", type=str)
    parser.add_argument('--yaml_path', nargs='?', default='/home/ubuntu/ant_detection/problems/full_video/real_coords.yml', help="outputfile path with pixels and real coords", type=str)
    parser.add_argument('--num_points', nargs='?', default=2, help="number of points", type=int)
    args = parser.parse_args()
    
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
        cv2.setMouseCallback('src',draw_circle)
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
