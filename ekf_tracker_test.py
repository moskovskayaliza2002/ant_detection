import yaml
import argparse
from universal_RCNN_test import read_yaml
import matplotlib.pyplot as plt
import numpy as np
from ekf import multiEKF
import matplotlib.image as mpimg
import cv2
import os
import shutil
import time
import gc
from from_pixels_to_real_coords import read_matrix
import math
from filterpy.common import Q_continuous_white_noise

ARROW_LEN = 50
D_ANT_COLOR = 'w'
D_ANT_SYM = 'o'
ANT_SCORE_MIN = 0.9
MEKF = None
#R_diag = np.array([1.22, 1.75, 0.39])
R_diag = np.array([0.0028, 0.0014, 0.39]) #angle = 0.6257
l = 0.00001


#Модель “кусочного” белого шума
'''
Q = np.array([[0.001, 0, 0, 0.03, 0], 
              [0.001, 0.001, 0, 0.03, 0],
              [0, 0, 0, 0, 0],
              [0.03, 0.03, 0, 1, 0],
              [0, 0, 0, 0, 0]])
'''

#TODO:Модель непрерывного белого шума
'''
Q = np.array([[0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]])
'''

#Подобранная экспериментально 
Q = np.array([[0.001, 0, 0, 0, 0], 
              [0, 0.001, 0, 0, 0],
              [0, 0, 0.08, 0, 0],
              [0, 0, 0, 0.001, 0],
              [0, 0, 0, 0, 0.0001]])

#Q_diag = np.array([l, l, l, l, l])
dt = 0.1
# коэфф для евклидова
#mh = 75
#коэфф для махалонобиса
#mh = 12
#коэфф для временного порога
mh = 2
P_limit = np.inf


#read data from yolo7 detector
def read_txt(path):
    data = []
    name, fps, weight, height = 0, 0, 0, 0
    bb_one_frame, bs_one_frame, kp_one_frame = [], [], []
    last_frame = 1
    num_lines = 0
    with open(path, 'r') as f:
        num_lines = sum(1 for line in f)
    print(num_lines)
    with open(path, 'r') as f:
        for i, s in enumerate(f):
            if i == 0:
                name = s[:-1]
            elif i == 1:
                fps = round(float(s[:-1]))
            elif i == 2:
                weight = int(s[:-1])
            elif i == 3:
                height = int(s[:-1])
            else:
                l = s[:-1].split(' ')
                frame = int(l[0])
                if last_frame != frame:
                    data.append(OrderedDict({'frame': last_frame, 'bboxes': bb_one_frame, 'bboxes_scores': bs_one_frame, 'keypoints': kp_one_frame}))
                    bb_one_frame, bs_one_frame, kp_one_frame = [], [], []
                    last_frame = frame
                num_ants = (len(l) - 1) // 9
                bbox = list(filter(int, l[1:num_ants * 4 + 1]))
                bbox_scores = list(filter(float, l[num_ants * 4 + 1: num_ants * 5 + 1]))
                kps = list(filter(int, l[num_ants * 5 + 1:]))
                bb_one_frame.append(bbox)
                bs_one_frame.append(bbox_scores[0])
                kp_one_frame.append(kps)
                if i == num_lines - 1:
                    data.append(OrderedDict({'frame': frame, 'bboxes': bb_one_frame, 'bboxes_scores': bs_one_frame, 'keypoints': kp_one_frame}))
    d = OrderedDict({'name': name, 'FPS': fps, 'weight': weight, 'height': height, 'frames': data})
    return d


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Функция обработки фрейма, для сохранения
def proceed_frame_cv2(frame, frame_v, W, H, dt, inv_matrix):
    global MEKF
    ants = get_ants(frame, dt)
    image = plot_ants_cv2(frame_v, ants)
    if MEKF is None:
        MEKF = multiEKF(ants, R_diag, Q, dt, mh, P_limit, W, H, int(frame['frame']), inv_matrix)
    else:
        MEKF.proceed(ants, dt, int(frame['frame']))
    #print("Траекторий построилось: ", len(MEKF.EKFS))
    image = MEKF.draw_tracks_cv2(image)
    return image
    

# Функция обработки фрейма, c визуализацией
def proceed_frame(frame, W, H, ax, dt):
    global MEKF
    ants = get_ants(frame, dt) 
    #commit for no vis
    plot_ants(ax, ants, H, dt)
    
    if MEKF is None:
        MEKF = multiEKF(ants, R_diag,  Q, dt, mh, P_limit, W, H, int(frame['frame']))
    else:
        MEKF.proceed(ants, dt, int(frame['frame']))
    #commit for no vis (2)
    MEKF.draw_tracks(H, ax, 'r')
    MEKF.draw_speed(ax)

# Функция обработки фрейма, без визуализации 
def proceed_frame_nv(frame, W, H, dt):
    global MEKF
    ants = get_ants(frame, dt) 
    if MEKF is None:
        MEKF = multiEKF(ants, R_diag,  Q, dt, mh, P_limit, W, H, int(frame['frame']))
    else:
        MEKF.proceed(ants, dt, int(frame['frame']))

def get_ants(frame, dt):
    ants = []
    delta_t = dt
    '''
    for kp, score in zip(frame['keypoints'], frame['bboxes_scores']):
        if score < ANT_SCORE_MIN:
            continue
        cx = (kp[0][0] + kp[1][0])/2 # TODO use box center instead
        cy = (kp[0][1] + kp[1][1])/2            
        a = np.arctan2(kp[1][1]-kp[0][1], kp[1][0]-kp[0][0])
        ant = [score, cx, cy, a, 0, 0]
        ants.append(ant)
    return np.array(ants)
    '''
    for bb, kp, score in zip(frame['bboxes'], frame['keypoints'], frame['bboxes_scores']):
        if score < ANT_SCORE_MIN:
            continue
        cx = (bb[0] + bb[2])/2 
        cy = (bb[1] + bb[3])/2            
        a = np.arctan2(kp[1][1]-kp[0][1], kp[1][0]-kp[0][0])
        ant = [score, cx, cy, a, 0, 0]
        ants.append(ant)
    return np.array(ants)

'''
ants - [[p, x, y, a]]
'''
def plot_ants(ax, ants, H, dt, color = D_ANT_COLOR):            
    ants = get_ants(frame, dt)                                
    for i in range(ants.shape[0]):
        ax.plot(ants[i,1], ants[i,2], color+D_ANT_SYM, alpha = ants[i,0])
        #ax.arrow(ants[i,1], ants[i,2], ARROW_LEN * np.cos(ants[i,3]), ARROW_LEN * np.sin(ants[i,3]), color = color, alpha = ants[i,0])
        
def plot_ants_cv2(frame, ants, color = D_ANT_COLOR):
    image = frame
    for i in range(ants.shape[0]):
        image = cv2.circle(image, (int(ants[i, 1]), int(ants[i, 2])), radius=5, color=(255,255,255), thickness=-1)
    return image
            
        
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def quartile_filter(arr):
    new_mask = np.array([])
    for i, ar in enumerate(arr):
        Q1 = np.percentile(ar, 25)
        Q3 = np.percentile(ar, 75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        a = np.where((ar > upper_limit) | (ar < lower_limit), None, ar)
        new_mask = np.append(new_mask, a)
    new_mask = np.asarray(new_mask).reshape(5, -1)
    withot_outliers = []
    for j in range(new_mask[0].shape[0]):
        if new_mask[0][j] != None and new_mask[1][j] != None and new_mask[2][j] != None and new_mask[3][j] != None and new_mask[4][j] != None:
            withot_outliers.append([new_mask[0][j], new_mask[1][j], new_mask[2][j], new_mask[3][j], new_mask[4][j]])
    
    withot_outliers = np.asarray(withot_outliers, dtype=np.float64)
    return withot_outliers.T
        
    
def plot_gist(d_mean_ants, d_mean_steps, d_ns_mean_ants, d_ns_mean_steps, path):
    lenth = max([len(d_mean_ants), len(d_mean_steps), len(d_ns_mean_ants), len(d_ns_mean_steps)])
    none_array = [None] * lenth
    #дополни массивы none тогда они будут одной длины, или попробуй список массивов
    data = [d_mean_ants, d_mean_steps, d_ns_mean_ants, d_ns_mean_steps]
    fig, ax = plt.subplots()
    ax.set_title('Распределение скоростей')
    parts = ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    medians = np.percentile(data, [50], axis=1)

    ax.scatter([1, 2, 3, 4], medians, marker='o', color='white', s=30, zorder=3)
    
    labels = ['Ост./особи', 'Ост./шаги', 'Б.Ост/особи', 'Б.Ост/шаги']
    
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
    plt.savefig(path)

def lenth_of_traj(traj, dt):
    first_p = traj[0]
    traj = traj[1:]
    all_speed = []
    for point in traj:
        l = math.dist(first_p, point)
        speed = l/dt
        all_speed.append(speed)
        first_p = point
    #print(f"Скорость по особям: {sum(all_speed)/len(all_speed)}")
    return all_speed, sum(all_speed)/len(all_speed)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #file_ = 'cut6s'
    #file_ = 'cut50s'
    #file_ = 'empty_center'
    file_ = "18.08.20_Fp2_плос2"
    #file_ = "video1"
    #file_ = "prombem_2minute"
    #file_ = "video0"
    #file_ = "empty_center"
    #file_ = "FILE0009 1"
    #file_ = "mean_speed"
    '''
    parser.add_argument('--yaml_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/{file_}.yml', help="Full path to yaml-file with ant data", type=str)
    parser.add_argument('--video_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/{file_}.mp4', help="Full path to video file", type=str)
    parser.add_argument('--pic_save_path', nargs='?', default=f'/windows/d/frames_track', help="Full path to directory to save frames", type=str)
    parser.add_argument('--tracks_save_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/{file_}_tracks.yml', help="Full path to directory to save trackes in yaml", type=str)
    '''
    parser.add_argument('--yaml_path', nargs='?', default=f'/home/ubuntu/ant_detection/problems/full_video/{file_}_real_coords.yml', help="Full path to yaml-file with ant data", type=str)
    parser.add_argument('--video_path', nargs='?', default=f'/home/ubuntu/ant_detection/problems/full_video/{file_}.mp4', help="Full path to video file", type=str)
    #parser.add_argument('--tracks_save_path', nargs='?', default=f'/home/ubuntu/ant_detection/problems/another_full_video/{file_}_tracks.txt', help="Full path to directory to save trackes in yaml", type=str)
    parser.add_argument('--visualisation', nargs='?', default=True, help="Make visualization or file with tracks only", type=bool)
    args = parser.parse_args()
    yaml_path = args.yaml_path
    video_path = args.video_path
    
    sec_start = time.time()
    struct_start = time.localtime(sec_start)
    start_time = time.strftime('%d.%m.%Y %H:%M', struct_start)
    
    name = video_path[video_path.rfind('/'):video_path.rfind('.')]
    tracks_save_path = video_path[:video_path.rfind('/')] + name + '_tracks' + '.txt'
    path_to_matrix = video_path[:video_path.rfind('/')] + name + "_real_coords_matrix.yml"
    print(f"INFO: matrix read at {path_to_matrix}")
    print(f"INFO: tracks saving path: {tracks_save_path}")
    matrix = read_matrix(path_to_matrix)
    matrix = np.array(matrix, dtype=np.float32)
    inv_matrix = np.linalg.inv(matrix)
    
    print("----------------Построение траекторий----------------")
    print(f"INFO: Loading data from {yaml_path}...")
    if yaml_path[-3:] == 'txt':
        ANT_DATA = read_txt(yaml_path)
    else:
        ANT_DATA = read_yaml(yaml_path)    
    #print(d.keys() for d in ANT_DATA['frames'])
    print(type(ANT_DATA['FPS']))
    dt = 1/int(float(ANT_DATA['FPS']))

    print(f"INFO: Loading video {args.video_path}...")
    cap = cv2.VideoCapture(args.video_path)   
    while not cap.isOpened():
        cap = cv2.VideoCapture(args.video_path)
        cv2.waitKey(1000)
        print("Openning the file...")
        
    maxim_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 1
    ax = None
    #print(f"всего {len(ANT_DATA['frames'])}")
    name = args.video_path[args.video_path.rfind('/'):args.video_path.rfind('.')]
    new_filename = args.video_path[:args.video_path.rfind('/')] + name + '_real_tracks' + '.mp4'
    #new_filename = args.video_path[:args.video_path.rfind('/')] + name + '_tracks' + '.mp4'
    
    '''
    if args.visualisation:
        fig, ax = plt.subplots()  
        plt.ion()
        plt.show(block=False)
        for frame in ANT_DATA['frames']:    
            ax.clear()
            print('Frame:', count, '/', maxim_frames)
            ret, frame_v = cap.read()
            ax.imshow(frame_v)
            ax.set_title(f"Frame {list(frame.keys())[0]}")
            plt.xlim(0, ANT_DATA['weight'])
            plt.ylim(0, ANT_DATA['height'])
            proceed_frame(frame, ANT_DATA['weight'], ANT_DATA['height'], ax, dt)
            plt.savefig(pic_save_path + '/frame' + str(count) + '.png')
            count += 1
            plt.pause(0.1)
            plt.show()
    '''
    printProgressBar(0, maxim_frames, prefix = 'Progress:', suffix = 'of frames processed', length = 50)
    if args.visualisation:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(w), int(h))
        out = cv2.VideoWriter(new_filename, fourcc, fps, size, True)
        for frame in ANT_DATA['frames']:
            #if count % 1000 == 0:
            #    print('Frame:', count, '/', maxim_frames)
            printProgressBar(count, maxim_frames, prefix = 'Progress:', suffix = 'of frames processed', length = 50)
            ret, frame_v = cap.read()
            pred_im = proceed_frame_cv2(frame, frame_v, w, h, dt, inv_matrix)
            #cv2.imshow('image', frame_v)
            #cv2.waitKey(0)
            out.write(pred_im)
            count += 1
        out.release()
        cap.release()
    else:
        del cap
        print("Начало обработки")
        for frame in ANT_DATA['frames']:
            #ret, frame_v = cap.read()
            proceed_frame_nv(frame, ANT_DATA['weight'], ANT_DATA['height'], dt)
     
    del ANT_DATA
    gc.collect()
    print("Запись треков")
    MEKF.write_tracks(tracks_save_path)
    
    # распределения ошибок
    '''
    all_errors = []
    for ekf in MEKF.EKFS:
        for err in ekf.error:
            all_errors.append(err)
    
    for err in MEKF.deleted_ants_error: 
        all_errors.append(err) 
    all_errors = np.asarray(all_errors, dtype=np.float64)
    
    safe_p = '/home/ubuntu/ant_detection/boxplot/'
    '''
    '''
    plt.clf()
    plt.boxplot(all_errors[:, 0])
    plt.savefig(safe_p + 'x' + '.png')
    plt.clf()
    plt.boxplot(all_errors[:, 1])
    plt.savefig(safe_p + 'y' + '.png')
    plt.clf()
    plt.boxplot(all_errors[:, 2])
    plt.savefig(safe_p + 'a' + '.png')
    plt.clf()
    plt.boxplot(all_errors[:, 3])
    plt.savefig(safe_p + 'v' + '.png')
    plt.clf()
    plt.boxplot(all_errors[:, 4])
    plt.savefig(safe_p + 'w' + '.png')
    '''
    #new_arr = quartile_filter(all_errors.T)
    #cov_matrix = np.matmul(new_arr, new_arr.T)
    #print(np.cov(all_errors))
    
    '''
    C = [[0 for i in range(5)] for j in range(5)]
    for i in range(5):
        for j in range(5):
            if i == j:
                C[i][j] = np.cov(all_errors[:, i], bias=True)[0]
            else:
                C[i][j] = np.cov(all_errors[:, i], all_errors[:, j], ddof=0)[0][1]
    
    for st in C:
        print(st)
    
    cov_matrix = np.matmul(all_errors.T, all_errors)
    for st in cov_matrix:
        print(st)
    '''
    sec_finish = time.time()
    struct_finish = time.localtime(sec_finish)
    finish_time = time.strftime('%d.%m.%Y %H:%M', struct_finish)
    
    print(f'Started {start_time} Finished {finish_time}')
