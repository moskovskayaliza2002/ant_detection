import yaml
import argparse
from RCNN_overlay_test import read_yaml
import matplotlib.pyplot as plt
import numpy as np
from ekf import multiEKF
import matplotlib.image as mpimg
import cv2
import os
import shutil

ARROW_LEN = 50
D_ANT_COLOR = 'w'
D_ANT_SYM = 'o'
ANT_SCORE_MIN = 0.70
MEKF = None
R_diag = np.array([1.22, 1.75, 0.39])
l = 0.00001

Q = np.array([[5, 0, 0, 0, 0], 
              [0, 5, 0, 0, 0],
              [0, 0, 5, 0, 0],
              [0, 0, 0, 10, 0],
              [0, 0, 0, 0, 10]])
'''
Q = np.array([[2.63795021e+00, 4.31565031e-02, 4.34318028e-03, -2.20547258e+00, 2.13165056e-01],
              [4.31565031e-02, 2.17205029e+00, -2.80823458e-03, -8.82016377e+00, -7.06690445e-02],
              [4.34318028e-03, -2.80823458e-03, 2.03181860e-02, 4.16669103e-02, 4.60097662e-01],
              [-2.20547258e+00, -8.82016377e+00, 4.16669103e-02, 2.16764535e+03, -1.92685776e-01],
              [2.13165056e-01, -7.06690445e-02, 4.60097662e-01, -1.92685776e-01, 1.41432123e+01]])
'''
#Q_diag = np.array([l, l, l, l, l])
dt = 0.1
# евклидово
mh = 75
#коэфф для махалонобиса
#mh = 12
P_limit = np.inf

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
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #file_ = 'cut6s'
    #file_ = 'cut50s'
    #file_ = 'empty_center'
    file_ = "18.08.20 Fp2' плос2"
    #file_ = "video4"
    #file_ = "prombem_2minute"
    
    '''
    parser.add_argument('--yaml_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/{file_}.yml', help="Full path to yaml-file with ant data", type=str)
    parser.add_argument('--video_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/{file_}.mp4', help="Full path to video file", type=str)
    parser.add_argument('--pic_save_path', nargs='?', default=f'/windows/d/frames_track', help="Full path to directory to save frames", type=str)
    parser.add_argument('--tracks_save_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/{file_}_tracks.yml', help="Full path to directory to save trackes in yaml", type=str)
    '''
    parser.add_argument('--yaml_path', nargs='?', default=f'/windows/d/ant_detection/dynamic_density/{file_}.yml', help="Full path to yaml-file with ant data", type=str)
    parser.add_argument('--video_path', nargs='?', default=f'/windows/d/ant_detection/dynamic_density/{file_}.mp4', help="Full path to video file", type=str)
    parser.add_argument('--pic_save_path', nargs='?', default=f'/windows/d/frames_track', help="Full path to directory to save frames", type=str)
    parser.add_argument('--tracks_save_path', nargs='?', default=f'//windows/d/ant_detection/dynamic_density/{file_}_tracks.yml', help="Full path to directory to save trackes in yaml", type=str)
    parser.add_argument('--visualisation', nargs='?', default=True, help="Make visualization or file with tracks only", type=bool)
    
    args = parser.parse_args()
    print(f"Loading data from {args.yaml_path}...")
    ANT_DATA = read_yaml(args.yaml_path)    
    #print(d.keys() for d in ANT_DATA['frames'])
    dt = 1/ANT_DATA['FPS']
    pic_save_path = args.pic_save_path
    
    if os.path.exists(pic_save_path):
        shutil.rmtree(pic_save_path)
    os.mkdir(pic_save_path)

    print(f"Loading video {args.video_path}...")
    cap = cv2.VideoCapture(args.video_path)        
    maxim_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 1
    ax = None
    if args.visualisation:
        fig, ax = plt.subplots()  
        plt.ion()
        plt.show(block=False)
        for frame in ANT_DATA['frames']:    
            ax.clear()
            print('Frame:', count, '/', maxim_frames)
            ret, frame_v = cap.read()
            print(frame_v.shape)
            ax.imshow(frame_v)
            ax.set_title(f"Frame {list(frame.keys())[0]}")
            plt.xlim(0, ANT_DATA['weight'])
            plt.ylim(0, ANT_DATA['height'])
            proceed_frame(frame, ANT_DATA['weight'], ANT_DATA['height'], ax, dt)
            plt.savefig(pic_save_path + '/frame' + str(count) + '.png')
            count += 1
            plt.pause(0.1)
            plt.show()
    else:
        for frame in ANT_DATA['frames']:    
            ret, frame_v = cap.read()
            proceed_frame_nv(frame, ANT_DATA['weight'], ANT_DATA['height'], dt)
        
    MEKF.write_tracks(args.tracks_save_path)
        
    all_errors = []
    for ekf in MEKF.EKFS:
        for err in ekf.error:
            all_errors.append(err)
    
    for err in MEKF.deleted_ants_error: 
        all_errors.append(err) 
    all_errors = np.asarray(all_errors, dtype=np.float64)
    
    safe_p = '/home/ubuntu/ant_detection/boxplot/'
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
    new_arr = quartile_filter(all_errors.T)
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
