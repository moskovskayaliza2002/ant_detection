from read_tracks_from_yml import read_yaml
import cv2
import yaml
from collections import OrderedDict
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd

def read_coords_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        datas = list(yaml.safe_load_all(f))
        return datas[0]['coordinates']

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


def check_track(track, coords):
    steps_out_of_area = 0
    times_he_walk_in = 0
    any_point_in_area = False
    for point in track:
        if is_points_in(coords, [point[0], point[1]]) and not any_point_in_area:
            times_he_walk_in += 1
        
        if not is_points_in(coords, [point[0], point[1]]):
            steps_out_of_area += 1
        
        if is_points_in(coords, [point[0], point[1]]):
            if steps_out_of_area < 4:
                times_he_walk_in -= 1
            steps_out_of_area = 0
            
        any_point_in_area = is_points_in(coords, [point[0], point[1]])
        '''
        elif is_points_in(coords, [point[0], point[1]]):
            any_point_in_area = True
        elif not is_points_in(coords, [point[0], point[1]]):
            any_point_in_area = False
        '''
    #if times_he_walk_in != 0:
        #print(f"один муравей побывал тут {times_he_walk_in} раз")
    return times_he_walk_in
        
'''
tracks - tracks in current minute
area - coordinates of area
'''
def counter_per_min(tracks, area):
    NUM_ANTS = 0
    #print("количество треков в минуту: ", len(tracks))
    for track in tracks:
        NUM_ANTS += check_track(track, area)
            
    return NUM_ANTS


def draw_graficks(density, path, csv_path):
    #сделай функцию считывания cvc файла
    real_data = read_cvc(csv_path)
    real_data.append(0)
    #f, ax = plt.subplots(1, 2, figsize=(40, 20))
    #plt.ion()
    plt.title("Динамическая плотность") # заголовок
    plt.xlabel("Минута") # ось абсцисс
    plt.ylabel("Количество особей") # ось ординат
    plt.grid() # включение отображение сетки
    plt.plot(range(len(density)), density, 'r--')
    plt.plot(range(len(real_data)), real_data, 'b--')
    '''
    ax[0].set_title('Автоматический подсчет', fontsize=12)
    ax[0].plot(range(len(density)), density, "r-")  # построение графика
    ax[0].grid(True)
    ax[0].set_xlabel('минута')
    ax[0].set_ylabel('количество особей')
    
    ax[1].set_title('Ручной подсчет', fontsize=12)
    ax[1].plot(range(len(real_data)), real_data, "r-")
    ax[1].grid(True)
    ax[1].set_xlabel('минута')
    ax[1].set_ylabel('количество особей')
    '''
    plt.legend(['Автоматический подсчет', 'Ручной подсчет'], loc = 'best')
    plt.show()
    #plt.savefig(path + '/density.png')
    
   
def read_cvc(path):
    # YOU MUST PUT sheet_name=None TO READ ALL CSV FILES IN YOUR XLSM FILE
    df = pd.read_excel(path, sheet_name='Лист1')
    density = df['кол-во муравьев зашедших в квадрат за минуту'].values.tolist()
    print
    return density
    

    # prints all sheets
    #print(df)
    
def count_all_minutas(coord_yaml, tracks_yaml, video_path):
    cap = cv2.VideoCapture(video_path)
    print("INFO: open video...")
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    area = read_coords_yaml(coord_yaml)
    print("INFO: readimg tracks...")
    all_tracks = read_yaml(tracks_yaml)
    print(f"ВСЕГО ТРЕКОВ В ФАЙЛЕ {len(all_tracks)}")
    distance = FPS * 60
    all_density = []
    counter = 1
    for i in range(0, int(number_of_frames), distance):
        #print(f'from {i} to {i + distance}')
        tracks_minute = np.squeeze(split_1_min(i, all_tracks, FPS)).tolist()
        #tracks_minute = split_1_min(i, all_tracks, FPS)
        #print("Итоговый массив", np.array(tracks_minute).shape)
        #print('tracks: ', tracks_minute)
        ANTS = counter_per_min(tracks_minute, area)
        print(f'За {counter} минуту плотность составила {ANTS}')
        print(f'from {i} to {i + distance}')
        counter += 1
        all_density.append(ANTS)
    return all_density
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracks_yaml', nargs='?', default="/home/ubuntu/ant_detection/dynamic_density/prombem_2minute_tracks.yml", help="Specify yaml track path", type=str)
    parser.add_argument('coord_yaml', nargs='?', default="/home/ubuntu/ant_detection/dynamic_density/coods_21m.yml", help="Specify yaml coords path", type=str)
    parser.add_argument('input_video_path', nargs='?', default="/home/ubuntu/ant_detection/dynamic_density/prombem_2minute.mp4", help="Specify input video path", type=str)
    parser.add_argument('csv_path', nargs='?', default="/home/ubuntu/ant_detection/videos/18.08.20 Fp2' плос2.xlsx", help="Specify path to gt data", type=str)
    #parser.add_argument('out_video_path', nargs='?', default='/home/ubuntu/ant_detection/dynamic_density/cut6s_tracks.mp4', help="Specify output video path", type=str)
    args = parser.parse_args()
    path = '/home/ubuntu/ant_detection/dynamic_density/'
    draw_graficks(count_all_minutas(args.coord_yaml, args.tracks_yaml, args.input_video_path), path, args.csv_path)
    
    #read_cvc("/home/ubuntu/ant_detection/dynamic_density/18.08.20 Fp2' плос2.xlsx")
    
