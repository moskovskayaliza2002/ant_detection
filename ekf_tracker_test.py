import yaml
import argparse
from RCNN_overlay_test import read_yaml
import matplotlib.pyplot as plt
import numpy as np
from ekf import multiEKF
import matplotlib.image as mpimg
import cv2

ARROW_LEN = 50
D_ANT_COLOR = 'w'
D_ANT_SYM = 'o'
ANT_SCORE_MIN = 0.4
MEKF = None
R_diag = np.array([1.69, 3.76, 1.86])
Q_diag = np.array([1, 1, 1, 3, 3])
dt = 0.1
mh = 20
P_limit = np.inf

def proceed_frame(frame, W, H, ax):
    global MEKF
    ants = get_ants(frame)    
    plot_ants(ax, ants, H)
    
    if MEKF is None:
        MEKF = multiEKF(ants, R_diag,  Q_diag, dt, mh, P_limit, W, H)
    else:
        MEKF.proceed(ants)
    MEKF.draw_tracks(H, ax, 'r')

def get_ants(frame):
    ants = []
    
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
def plot_ants(ax, ants, H, color = D_ANT_COLOR):            
    ants = get_ants(frame)                                
    for i in range(ants.shape[0]):
        ax.plot(ants[i,1], ants[i,2], color+D_ANT_SYM, alpha = ants[i,0])
        ax.arrow(ants[i,1], ants[i,2], ARROW_LEN * np.cos(ants[i,3]), ARROW_LEN * np.sin(ants[i,3]), color = color, alpha = ants[i,0])
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #file_ = 'cut6s'
    file_ = 'cut50s'
    
    parser.add_argument('--yaml_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/inputs/{file_}.yml', help="Full path to yaml-file with ant data", type=str)
    parser.add_argument('--video_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/inputs/{file_}.mp4', help="Full path to video file", type=str)
    parser.add_argument('--pic_save_path', nargs='?', default=f'/home/ubuntu/ant_detection/frames_track/', help="Full path to directory to save frames", type=str)
    args = parser.parse_args()
    print(f"Loading data from {args.yaml_path}...")
    ANT_DATA = read_yaml(args.yaml_path)    
    #print(d.keys() for d in ANT_DATA['frames'])
    dt = 1/ANT_DATA['FPS']
    pic_save_path = args.pic_save_path
    
    print(f"Loading video {args.video_path}...")
    cap = cv2.VideoCapture(args.video_path)        
    
    count = 1
    fig, ax = plt.subplots()      
    for frame in ANT_DATA['frames']:                          
        ax.clear()
        print('Frame:', count)
        ret, frame_v = cap.read()
        #frame_v = np.flip(frame_v, (0,2))
        #print(frame_v.shape)
        ax.imshow(frame_v)
        
        ax.set_title(f"Frame {list(frame.keys())[0]}")
        plt.xlim(0, ANT_DATA['weight'])
        plt.ylim(0, ANT_DATA['height'])
        proceed_frame(frame, ANT_DATA['weight'], ANT_DATA['height'], ax)
        
        plt.savefig(pic_save_path + 'frame' + str(count) + '.png')
        count += 1
        plt.pause(0.1)
        #plt.show()
