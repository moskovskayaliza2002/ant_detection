import cv2
import numpy as np
import time
import copy
import yaml
import argparse
from collections import OrderedDict
   
def draw_circle(event, x, y, flags, param):
    global img
    global cache
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cache = copy.deepcopy(img)
        coords.append([x, y])
        print('(x:',x,',y:',y,')')
        str1 = '(x:'+ str(x) + ',y:'+ str(y) + ')'
        cv2.putText(img,str1 , (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness=1)
        cv2.circle(img,(x,y),3,(0, 0, 255),-1)
        
    if event == cv2.EVENT_MBUTTONDBLCLK:
        img = copy.deepcopy(cache)
        coords.pop()
        cv2.imshow('src', img)
        
def save_coords_to_yaml(yaml_path, coords, video_path):
    with open(yaml_path, 'a') as f:
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        yaml.dump(OrderedDict({'video_path': video_path, 'coordinates': coords}), f)
    
    
def show_polygon(im, coords, yml_path, v_path):
    #image = cv2.imread(path)
    window_name = 'polygon'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Polygon corner points coordinates
    pts = np.array(coords, np.int32)
    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    # Green color in BGR
    color = (0, 255, 0)
    thickness = 3
    image = cv2.polylines(im, [pts], isClosed, color, thickness)
    
    print("Нажмите Esс чтобы не сохранять координаты и Enter чтобы сохранить")
    while(1):
        cv2.imshow('polygon', image)
        next_step = cv2.waitKey(0)
        if next_step == 27:
            break
        if next_step == 13:
            save_coords_to_yaml(yml_path, coords, v_path)
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("INFO: двойной клик левой кнопкой мыши поставит точку, двойной клик колесика отменит последнюю нарисованную точку")
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', nargs='?', default='/home/ubuntu/ant_detection/dynamic_density/cut6s_pred.mp4', help="path to video for dynamic density analysis", type=str)
    parser.add_argument('--yaml_path', nargs='?', default='/home/ubuntu/ant_detection/dynamic_density/coods.yml', help="outputfile path with coordinates of rectangle points", type=str)
    ##path = '/home/ubuntu/ant_detection/polygon_data/Test_data/images/image446.png'
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
        #img = cv2.imread(path, 1)
        cache = copy.deepcopy(img)
        cv2.namedWindow('src', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('src',draw_circle)
        coords = []
        exit_flag = True
        while(len(coords) != 4):
            cv2.imshow('src',img)
            if cv2.waitKey (100) == ord ('q'): # Нажмите q, чтобы выйти
                break
            #cv2.imshow('src',img)
        cv2.destroyAllWindows()
        if len(coords) == 4:
            show_polygon(frame, coords, args.yaml_path, args.video_path)
        print(coords)
    
    
