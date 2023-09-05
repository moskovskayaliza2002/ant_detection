#from dynamic_density import read_tracks_from_txt
from from_pixels_to_real_coords import read_matrix
import argparse
import numpy as np
import cv2
import re
import copy


def read_tracks_from_txt(path):
    tracks = []
    last_no = 0
    with open(path) as f:
        for i in f:
            #a = list(map(float, i[:-2].split(' ')))
            a = i[:-2].split(' ')
            no = int(a[0])
            frame_ind = int(a[1])
            ind = int(a[2])
            color = a[3]
            track = [float(i) for i in a[4:]]
            track = np.array(track).reshape((-1, 5)).tolist()
            tracks.append(track)
    return tracks


def get_line_coords(video_path):
    print("INFO: Поставьте две точки. двойной клик левой кнопкой мыши поставит точку, двойной клик колесика отменит последнюю нарисованную точку")
    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print("Openning the file...")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps
    durationInMinutes = round(durationInSeconds / 60, 2)
    
    flag, frame = cap.read()
    exit_flag = False
    while not exit_flag:
        if flag:
            global img  
            global cache
            global coords
            img = frame
            #img = cv2.imread(path, 1)
            cache = copy.deepcopy(img)
            cv2.namedWindow('src', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('src',draw_circle)
            coords = []
            exit_flag = False
            while(len(coords) != 2):
                cv2.imshow('src',img)
                if cv2.waitKey (100) == ord ('q'): # Нажмите q, чтобы выйти
                    break
                #cv2.imshow('src',img)
            cv2.destroyAllWindows()
            if len(coords) == 2:
                exit_flag = show_line(frame, coords)
        else:
            flag, frame = cap.read()
            
    return coords, durationInMinutes


def side_of_point(x, y, line):
    x1, y1 = line[0][0], line[0][1]
    x2, y2 = line[1][0], line[1][1]
    d = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if d < 0:
        return 'l'
    else:
        return 'r'
    
def replace(string, char_list):
    for char in char_list:
        pattern = char + '{2,}'
        string = re.sub(pattern, char, string)
    return string


def draw_circle(event, x, y, flags, param):
    global img
    global cache
    global coords
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
        
def show_line(im, coords):
    #image = cv2.imread(path)
    window_name = 'line'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Polygon corner points coordinates
    pts = np.array(coords, np.int32)
    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    # Green color in BGR
    color = (0, 255, 0)
    thickness = 3
    image = cv2.polylines(im, [pts], isClosed, color, thickness)
    
    result = False
    print("INFO: Нажмите Enter чтобы продолжить расчет или Ctrl + c чтобы завершить программу")
    while(1):
        cv2.imshow('line', image)
        next_step = cv2.waitKey(0)
        if next_step == 27:
            break
        if next_step == 13:
            result = True
            break
            
    cv2.destroyAllWindows()
    return result
'''
def times_of_crossing_line(way):
    crosses = 0
    last_el = way[0]
    for status in way[1:]:
        if last_el != status:
            crosses += 1
        last_el = status
    return crosses
'''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', nargs='?', default="/home/ubuntu/ant_detection/problems/another_full_video/plos.yml", help="Specify path to video", type=str)
    
    args = parser.parse_args()
    filename = args.video_path[args.video_path.rfind('/')+1:args.video_path.rfind('.')]
    matrix_path = args.video_path[:args.video_path.rfind('/')] + '/' + filename + '_real_coords_matrix.yml'
    tracks_path = args.video_path[:args.video_path.rfind('/')] + '/' + filename + '_tracks.txt'
    
    print('***********************ОПРЕДЕЛЕНИЕ ИНТЕНСИВНОСТИ ДВИЖЕНИЯ*********************')
    matrix = read_matrix(matrix_path)
    matrix = np.array(matrix, dtype=np.float32)
    pix_points, durationInMinutes = get_line_coords(args.video_path)
    pix_points = np.array(pix_points, dtype=np.float32)
    real_points = cv2.perspectiveTransform(pix_points[None, :, :], matrix)
    tracks = read_tracks_from_txt(tracks_path)
    
    all_crosses = 0
    for track in tracks:
        way = [side_of_point(point[0], point[1], real_points[0]) for point in track]
        short_way = replace(''.join(way), ['l', 'r'])
        all_crosses += len(short_way) - 1
    
    print(f'Интенсивность движения на дороге составила {all_crosses} муравьев за {durationInMinutes} минут')
    print('********************КОНЕЦ ОПРЕДЕЛЕНИЯ ИНТЕНСИВНОСТИ ДВИЖЕНИЯ******************')
