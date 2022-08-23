import yaml
import argparse
from RCNN_overlay_test import read_yaml
import matplotlib.pyplot as plt
import numpy as np
from ekf import multiEKF
import matplotlib.image as mpimg
import cv2

ARROW_LEN = 50
D_ANT_COLOR = 'k'
D_ANT_SYM = 'o'
ANT_SCORE_MIN = 0.0
MEKF = None
R_diag = np.array([10, 10, 10])
Q_diag = np.array([10, 10, 10, 30, 30])
dt = 0.1
mh = 15
P_limit = 1000

def proceed_frame(frame, H, ax):
    global MEKF
    ants = get_ants(frame)    
    plot_ants(ax, ants, H)
    
    if MEKF is None:
        MEKF = multiEKF(ants, R_diag,  Q_diag, dt, mh, P_limit)
    else:
        MEKF.proceed(ants)
    MEKF.draw_tracks(H, ax, 'r')

def get_ants(frame):
    ants = []
    for k, v in frame.items():
        for kp, score in zip(v['keypoints'], v['bboxes_scores']):
            if score < ANT_SCORE_MIN:
                continue
            cx = (kp[0][0] + kp[1][0])/2
            cy = (kp[0][1] + kp[1][1])/2            
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
        ax.plot(ants[i,1], H-ants[i,2], color+D_ANT_SYM, alpha = ants[i,0])
        ax.arrow(ants[i,1], H-ants[i,2], ARROW_LEN * np.cos(ants[i,3]), ARROW_LEN * np.sin(ants[i,3]), color = color, alpha = ants[i,0])
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', nargs='?', default='//home/anton/Projects/ant_detection/videos/short.yml', help="Full path to yaml-file with ant data", type=str)
    parser.add_argument('--video_path', nargs='?', default='//home/anton/Projects/ant_detection/videos/short.mp4', help="Full path to video file", type=str)
    args = parser.parse_args()
    print(f"Loading data from {args.yaml_path}...")
    ANT_DATA = read_yaml(args.yaml_path)    
    #print(d.keys() for d in ANT_DATA['frames'])
    dt = 1/ANT_DATA['FPS']
    
    print(f"Loading video {args.video_path}...")
    cap = cv2.VideoCapture(args.video_path)        
    
    fig, ax = plt.subplots()      
    for frame in ANT_DATA['frames']:                          
        ax.clear()
        ret, frame_v = cap.read()
        frame_v = np.flip(frame_v, (0,2))
        #print(frame_v.shape)
        ax.imshow(frame_v)
        
        ax.set_title(f"Frame {list(frame.keys())[0]}")
        ax.set_xlim(0, ANT_DATA['weight'])
        ax.set_ylim(0, ANT_DATA['height'])
        proceed_frame(frame, ANT_DATA['height'], ax)
        
        plt.pause(0.1)
        #plt.show()
