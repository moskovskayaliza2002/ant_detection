from read_tracks_from_yml import read_yaml
import cv2
import yaml
from collections import OrderedDict
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
from shapely.geometry import Point, Polygon
import time
from from_pixels_to_real_coords import read_matrix

def read_tracks_from_txt(path):
    tracks = []
    last_no = 0
    with open(path) as f:
        for i in f:
            dic = {}
            #a = list(map(float, i[:-2].split(' ')))
            a = i[:-2].split(' ')
            no = int(a[0])
            frame_ind = int(a[1])
            ind = int(a[2])
            color = a[3]
            track = [float(i) for i in a[4:]]
            track = np.array(track).reshape((-1, 5)).tolist()
            dic[frame_ind] = track
            tracks.append(dic)
    return tracks
    
    
def read_coords_yaml(yaml_path, matrix_path):
    with open(yaml_path) as f:
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        datas = list(yaml.safe_load_all(f))
        print(datas[0]['coordinates'])
        #return datas[0]['coordinates']
    coords = np.array([datas[0]['coordinates']], dtype='float32')
    matrix = read_matrix(matrix_path)
    matrix = np.array(matrix, dtype=np.float32)
    tranf_to_real = cv2.perspectiveTransform(coords.reshape(-1, 1, 2), matrix)
    tranf_to_real = np.array(tranf_to_real.reshape((-1, 4, 2)))
    return tranf_to_real[0].tolist()

'''
coords = [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
point = [x, y, a, v, w]
https://ru.stackoverflow.com/questions/464787/%D0%A2%D0%BE%D1%87%D0%BA%D0%B0-%D0%B2%D0%BD%D1%83%D1%82%D1%80%D0%B8-%D0%BC%D0%BD%D0%BE%D0%B3%D0%BE%D1%83%D0%B3%D0%BE%D0%BB%D1%8C%D0%BD%D0%B8%D0%BA%D0%B0
'''
def is_points_in(coords, point):
    result = False
    x = point[0]
    y = point[1]
    j = 4 - 1
    for i in range(4):
        if ((coords[i][1] < y and coords[j][1] >= y or coords[j][1] < y and coords[i][1] >= y) and
         (coords[i][0] + (y - coords[i][1]) / (coords[j][1] - coords[i][1]) * (coords[j][0] - coords[i][0]) < x)):
            result = not result
        j = i
    return result
    
'''    
def split_1_min(start_frame, data, FPS):
    distance = FPS * 60
    tracks_minuta = []
    counter = 0
    new_track = []
    print("**************************new track************************")
    print(f'shape {distance, 5}')
    for track in data:
        
        if track['frame_idx'] == start_frame:
            new_track = [[-1, -1, -1, -1, -1]] * distance
            if len(track['track']) >= distance:
                new_track = track['track'][:distance]
            else:
                end = np.array([[-1, -1, -1, -1, -1]] * (distance - len(track['track'])))
                new_track = np.row_stack((np.array(track['track']), end)).tolist()
                #new_track[:len(track['track'])] = track['track']
            
            if np.array(new_track).shape != (distance, 5):
                print("Размер изменился в пункте 1")
            
        elif track['frame_idx'] < start_frame:
            new_track = [[-1, -1, -1, -1, -1]] * distance
            diffence = start_frame - track['frame_idx']
            if len(track['track'][diffence:]) != 0:
                #new_track = [[-1, -1, -1, -1, -1]] * distance
                if len(track['track'][diffence:]) >= distance:
                    new_track = track['track'][diffence:diffence + distance]
                else:
                    end = np.array([[-1, -1, -1, -1, -1]] * (distance - len(track['track'][diffence:])))
                    new_track = np.row_stack((np.array(track['track'][diffence:]), end)).tolist()
                    #new_track[:len(track['track'])] = track['track'][diffence:]
                    
                
            if np.array(new_track).shape != (distance, 5):
                print("Размер изменился в пункте 2")
                
        elif track['frame_idx'] > start_frame:
            new_track = [[-1, -1, -1, -1, -1]] * distance
            diffence = track['frame_idx'] - start_frame
            if len(track['track']) >= distance - diffence:
                if distance - diffence > 0:
                    new_track[diffence:] = track['track'][:distance-diffence]
                if np.array(new_track).shape != (distance, 5):
                    print("Размер изменился в пункте 3.1", np.array(new_track).shape)
    
            else:
                begin = np.array([[-1,-1,-1,-1,-1]] * diffence)
                begin_center = np.vstack((begin, np.array(track['track'])))
                ending =  np.array([[-1,-1,-1,-1,-1]] * (distance - len(track['track']) - diffence))
                new_track = np.vstack((begin_center, ending)).tolist()
                if np.array(new_track).shape != (distance, 5):
                    print("Размер изменился в пункте 3.2", np.array(new_track).shape)
            
        if new_track != [[-1, -1, -1, -1, -1]] * distance:
            counter += 1
            tracks_minuta.append([new_track])
    
    print(f"Количество треков {counter}")
    return tracks_minuta

'''
def split_1_min(start_frame, data, FPS):
    distance = FPS * 60
    tracks_minuta = []
    counter = 0
    new_track = []
    print("**************************new track************************")
    print(f'shape {distance, 5}')
    for track in data:
        frame_ind = int(list(track.keys())[0])
        tr = list(track.values())[0]
        if frame_ind == start_frame:
            new_track = [[-1, -1, -1, -1, -1]] * distance
            if len(tr) >= distance:
                new_track = tr[:distance]
            else:
                end = np.array([[-1, -1, -1, -1, -1]] * (distance - len(tr)))
                new_track = np.row_stack((np.array(tr), end)).tolist()
                #new_track[:len(track.value())] = track['track']
            
            if np.array(new_track).shape != (distance, 5):
                print("Размер изменился в пункте 1")
            
        elif frame_ind < start_frame:
            new_track = [[-1, -1, -1, -1, -1]] * distance
            diffence = start_frame - frame_ind
            if len(tr[diffence:]) != 0:
                #new_track = [[-1, -1, -1, -1, -1]] * distance
                if len(tr[diffence:]) >= distance:
                    new_track = tr[diffence:diffence + distance]
                else:
                    end = np.array([[-1, -1, -1, -1, -1]] * (distance - len(tr[diffence:])))
                    new_track = np.row_stack((np.array(tr[diffence:]), end)).tolist()
                    #new_track[:len(tr)] = tr[diffence:]
                    
                
            if np.array(new_track).shape != (distance, 5):
                print("Размер изменился в пункте 2")
                
        elif frame_ind > start_frame:
            new_track = [[-1, -1, -1, -1, -1]] * distance
            diffence = frame_ind - start_frame
            if len(tr) >= distance - diffence:
                if distance - diffence > 0:
                    new_track[diffence:] = tr[:distance-diffence]
                if np.array(new_track).shape != (distance, 5):
                    print("Размер изменился в пункте 3.1", np.array(new_track).shape)
    
            else:
                begin = np.array([[-1,-1,-1,-1,-1]] * diffence)
                begin_center = np.vstack((begin, np.array(tr)))
                ending =  np.array([[-1,-1,-1,-1,-1]] * (distance - len(tr) - diffence))
                new_track = np.vstack((begin_center, ending)).tolist()
                if np.array(new_track).shape != (distance, 5):
                    print("Размер изменился в пункте 3.2", np.array(new_track).shape)
            
        if new_track != [[-1.0, -1.0, -1.0, -1.0, -1.0]] * distance:
            counter += 1
            tracks_minuta.append([new_track])
    
    return tracks_minuta


def check_track(track, coords):
    BOUNDARY_STEPS = 5
    steps_out_of_area = 0
    times_he_walk_in = 0
    any_point_in_area = False
    for point in track:
        if type(point) == float or type(point) == int:
            pass
        else:
            if is_points_in(coords, [point[0], point[1]]) and not any_point_in_area:
                times_he_walk_in += 1
            
            if not is_points_in(coords, [point[0], point[1]]):
                steps_out_of_area += 1
            
            if is_points_in(coords, [point[0], point[1]]):
                if steps_out_of_area < BOUNDARY_STEPS and steps_out_of_area != 0:
                    times_he_walk_in -= 1
                steps_out_of_area = 0
            
            any_point_in_area = is_points_in(coords, [point[0], point[1]])
    return times_he_walk_in

def check_track_distance(track, coords):
    DISTANCE_OUT_BONDARY = 0.01
    last_time_in_area = False
    times_he_walk_in = 0
    points_out_of_boundary = []
    times_he_walk_out = 0
    for point in track:
        if type(point) == int:
            pass
        else:
            if is_points_in(coords, [point[0], point[1]]) and not last_time_in_area:
                times_he_walk_in += 1
            
            if not is_points_in(coords, [point[0], point[1]]):
                times_he_walk_out += 1
            
            if is_points_in(coords, [point[0], point[1]]):
                if not check_dictance(coords, point, DISTANCE_OUT_BONDARY) and times_he_walk_out != 0:
                    times_he_walk_in -= 1
                times_he_walk_out = 0
            
            last_time_in_area = is_points_in(coords, [point[0], point[1]])
                
    return times_he_walk_in
                
    
def check_dictance(coords, point, DISTANCE_OUT_BONDARY):
    flag = False
    poly = Polygon([(coords[0][0], coords[0][1]), (coords[1][0], coords[1][1]), (coords[2][0], coords[2][1]), (coords[3][0], coords[3][1])])
    po = Point(point[0], point[1])
    distance = poly.exterior.distance(po)
    print(f"-----------------------------РАССТОЯНИЕ: {distance}-------------------------------------")
    flag = distance >= DISTANCE_OUT_BONDARY
    return flag
        
'''
tracks - tracks in current minute
area - coordinates of area
'''
def counter_per_min(tracks, area):
    NUM_ANTS = 0
    #print("количество треков в минуту: ", len(tracks))
    for track in tracks:
        #NUM_ANTS += check_track(track, area)
        NUM_ANTS += check_track(track, area)
            
    return NUM_ANTS


def draw_graficks(density, csv_path, name):
    if csv_path == "":
        plt.title("Динамическая плотность") # заголовок
        plt.xlabel("Номер минуты") # ось абсцисс
        plt.ylabel("Количество особей") # ось ординат
        plt.grid() # включение отображение сетки
        plt.plot(range(len(density)), density, linestyle = '--', color='r',linewidth = 3, label='Автоматический подсчет')
        plt.legend(loc = 'best')
        plt.show()
    else:
        #сделай функцию считывания cvc файла
        real_data, real_data_tracks, density_truth = read_cvc(csv_path, name)
        #real_data.append(0)
        #plt.ion()
        #presicion(real_data, density)
        plt.title("Динамическая плотность") # заголовок
        plt.xlabel("Номер минуты") # ось абсцисс
        plt.ylabel("Количество особей") # ось ординат
        plt.grid() # включение отображение сетки
        plt.plot(range(len(density)), density, linestyle = '--', color='r',linewidth = 3, label='Автоматический подсчет')
        plt.plot(range(len(real_data)), real_data, color='b',linestyle = ':', linewidth = 3,label='Ручной подсчет')
        plt.plot(range(len(real_data_tracks)), real_data_tracks, color='g',  linewidth = 3, linestyle = '-.', label='Ручной с треками')
        plt.plot(range(len(density_truth)), density_truth, color='black',  linewidth = 3, linestyle = ':', label='Проверка')
        
        plt.legend(loc = 'best')
        plt.show()
        #plt.savefig(path + '/density.png')
    
def presicion(real_data, pred_data):
    gt = np.array(real_data)
    pr = np.array(pred_data)
    error = gt - pr
    print(gt)
    print(pr)
    print(error)
    # 1 - Среднеквадратическое отклонение ошибки прогнозируемой модели
    pres = (1 - np.mean(np.abs(error) / np.abs(pr))) * 100
    print(f"Точность определения динамической плотности составляет {pres}%")
    
    
    
def read_cvc(path, name):
    # YOU MUST PUT sheet_name=None TO READ ALL CSV FILES IN YOUR XLSM FILE
    #df = pd.read_excel(path, sheet_name='кол-во_мур-в_в_квадрате')
    #density = df['empty_center'].values[:11].tolist()
    #density_tracks = df['empty_center_tracks'].values[:11].tolist()
    #density_truth = df['empty_center_truth'].values[:11].tolist()
    
    df = pd.read_excel(path, sheet_name=name)
    lenth = len(df["time / type"].values) - 2
    density = df["Ручной"].values[:lenth].tolist()
    density_tracks = df['Ручной с треками'].values[:lenth].tolist()
    density_truth = df['Проверка'].values[:lenth].tolist()
    return density, density_tracks, density_truth
    
def count_all_minutas(coord_yaml, matrix_path, tracks_path, video_path):
    cap = cv2.VideoCapture(video_path)
    print("INFO: open video...")
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    #FPS = int(cap.get(cv2.CAP_PROP_FPS))
    FPS = 30
    number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    area = read_coords_yaml(coord_yaml, matrix_path)
    print("INFO: readimg tracks...")
    #all_tracks = read_yaml(tracks_yaml)
    all_tracks = read_tracks_from_txt(tracks_path)
    #print(all_tracks)
    print(f"ВСЕГО ТРЕКОВ В ФАЙЛЕ {len(all_tracks)}")
    print(f"ВСЕГО КАДРОВ: {number_of_frames}")
    print(FPS)
    distance = FPS * 60
    all_density = []
    counter = 0
    print(distance)
    for i in range(1, int(number_of_frames) + 1, distance):
        #print(f'from {i} to {i + distance}')
        tracks_minute = []
        if i != 1 and i + distance < number_of_frames:
            tracks_minute = np.squeeze(split_1_min(i+(1*counter), all_tracks, FPS)).tolist()
            print(f'from {i+(1*counter)} to {i+(1*counter) + distance} frames')
        else:
            tracks_minute = np.squeeze(split_1_min(i, all_tracks, FPS)).tolist()
            print(f'from {i} to {i + distance} frames')
        print(f'треков за {counter} минуту: {len(tracks_minute)}')
        #tracks_minute = split_1_min(i, all_tracks, FPS)
        #print("Итоговый массив", np.array(tracks_minute).shape)
        #print('tracks: ', tracks_minute)
        ANTS = counter_per_min(tracks_minute, area)
        print(f'За {counter} минуту плотность составила {ANTS}')
        counter += 1
        all_density.append(ANTS)
    return all_density
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks_path', nargs='?', default="/home/ubuntu/ant_detection/problems/another_full_video/empty_center_tracks.txt", help="Specify yaml track path", type=str)
    parser.add_argument('--coord_yaml', nargs='?', default="/home/ubuntu/ant_detection/problems/another_full_video/plos.yml", help="Specify yaml coords path", type=str)
    parser.add_argument('--input_video_path', nargs='?', default="/home/ubuntu/ant_detection/problems/another_full_video/empty_center.mp4", help="Specify input video path", type=str)
    #parser.add_argument('csv_path', nargs='?', default="/home/ubuntu/ant_detection/problems/dynamic_density.xlsx", help="Specify path to gt data", type=str)
    parser.add_argument('--csv_path', nargs='?', default="", help="Specify path to gt data", type=str)
    
    sec_start = time.time()
    struct_start = time.localtime(sec_start)
    start_time = time.strftime('%d.%m.%Y %H:%M', struct_start)
    
    args = parser.parse_args()
    name = args.input_video_path[args.input_video_path.rfind('/')+1:args.input_video_path.rfind('.')]
    matrix_path = args.input_video_path[:args.input_video_path.rfind('/')] + '/' + name + '_real_coords_matrix.yml'
    #path = '/home/ubuntu/ant_detection/dynamic_density/'
    draw_graficks(count_all_minutas(args.coord_yaml, matrix_path, args.tracks_path, args.input_video_path), args.csv_path, name)
    
    sec_finish = time.time()
    struct_finish = time.localtime(sec_finish)
    finish_time = time.strftime('%d.%m.%Y %H:%M', struct_finish)
    print(f'Started {start_time} Finished {finish_time}')
    
    #read_cvc("/home/ubuntu/ant_detection/dynamic_density/18.08.20 Fp2' плос2.xlsx")
    
