import cv2
import numpy as np
import yaml
from collections import OrderedDict
import argparse
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import ffmpeg    
import os

'''
def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
        
    return rotateCode
'''
        
def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 
 
def proceed_frame(frame_num, data):
    ekfs_track = []
    for track in data:
        if frame_num == track['frame_idx']:
            ekfs_track.append([track['track'][0]])
        elif frame_num == track['frame_idx'] + len(track['track']):
            ekfs_track.append(track['track'])
        elif frame_num > track['frame_idx'] and frame_num < (track['frame_idx'] + len(track['track'])):
            ekfs_track.append(track['track'][:frame_num - track['frame_idx']])
        else:
            ekfs_track.append(0)
    return ekfs_track
            
def draw_tracks(ekfs_track, color, ax):
    for i, track in enumerate(ekfs_track):
        if track != 0:
            tr = np.array(track)
            print(tr)
            ax.plot(tr[:,0], tr[:,1], color = color[i])
    
    
def draw_speed(ekfs_track, ax, dt = 0.2, color = 'w', N = 3):
        for track in ekfs_track:
            if track != 0:
                x = [track[-1][0]]
                y = [track[-1][1]]
                a = [track[-1][2]]
                v = track[-1][3]
                w = track[-1][4]
                for i in range(N):
                    new_a = a[-1] + w * dt
                    new_x = x[-1] + v * np.cos(new_a) * dt
                    new_y = y[-1] + v * np.sin(new_a) * dt
                    a.append(new_a)
                    x.append(new_x)
                    y.append(new_y)
                ax.plot(x, y, color = color, linestyle = '--')
            
            
def read_yaml(yml_filename):
    with open(yml_filename) as f:
        #yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        datas = list(yaml.safe_load(f))
        f.close()
    if datas == []:
        return []
    else:
        return datas[0]
    
def visualize_from_yml(yml_path, video_path):#, pred_video_path):
    safe_video_path = '/home/ubuntu/ant_detection/dynamic_density/new_video'
    frames_to_save = 4000
    print("INFO: reading yaml...")
    data = read_yaml(yml_path)
    print("INFO: proceeding...")
    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print("Openning the file...")
    
    #rot_code = check_rotation(video_path)
    maxim_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(w), int(h))
    #out = cv2.VideoWriter(pred_video_path, fourcc, fps, size, True)
    
    color = cm.rainbow(np.linspace(0, 1, len(data)))
    fig, ax = plt.subplots(figsize=(2, 1), frameon=False)
    fig.set_size_inches(12,6)
    ax.set_axis_off()
    fig.add_axes(ax)
    count = 0
    while True:
        flag, frame = cap.read()
        if flag:
            ax.clear()
            #ax.imshow(correct_rotation(frame, cv2.ROTATE_180))
            ax.imshow(frame)
            ax.set_title(f"Frame")
            plt.xlim(0, w)
            plt.ylim(h, 0)
            current_tracks = proceed_frame(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), data)
            draw_tracks(current_tracks, color, ax)
            draw_speed(current_tracks, ax)
            if count < frames_to_save:
                plt.savefig(safe_video_path + "/file%02d.png" % count, dpi=150)
                count += 1
            plt.pause(0.1)
            
        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #Если прочитали все кадры - выходим из цикла
            break
            
    #out.release()
    cap.release()   
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yml_path', nargs='?', default="/home/ubuntu/ant_detection/videos/cut50s_tracks.yml", help="Specify yaml track path", type=str)
    parser.add_argument('input_video_path', nargs='?', default="/home/ubuntu/ant_detection/videos/cut50s.mp4", help="Specify input video path", type=str)
    #parser.add_argument('out_video_path', nargs='?', default='/home/ubuntu/ant_detection/dynamic_density/cut6s_tracks.mp4', help="Specify output video path", type=str)
    args = parser.parse_args()
    
    visualize_from_yml(args.yml_path, args.input_video_path)
