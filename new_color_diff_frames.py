import numpy as np
import cv2
import argparse
import os
import shutil
import pandas as pd

def generate_diff_image(NUM_FRAMES_IN_DATA, video_path, saving_frame_path):
    if not os.path.exists(saving_frame_path):
            os.mkdir(saving_frame_path)
            os.mkdir(saving_frame_path+'/orig')
            os.mkdir(saving_frame_path+'/diff')
    else:
        shutil.rmtree(saving_frame_path)
        os.mkdir(saving_frame_path)
        os.mkdir(saving_frame_path+'/orig')
        os.mkdir(saving_frame_path+'/diff')
            
    cap = cv2.VideoCapture(video_path)
    max_amound_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ind_frames = np.random.randint(low=2000, high=max_amound_of_frames-1, size=NUM_FRAMES_IN_DATA)
    print(f"индексы кадров: {ind_frames}")
    
    for i, ind in enumerate(ind_frames):
        orig_file = saving_frame_path + '/orig/' + str(ind) + '_orig_image.png'
        filename = saving_frame_path + '/diff/' + str(ind) + '_diff_image.png'
        
        #reading 3 following frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind-1)
        flag1, curr_frame = cap.read()
        frame = curr_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind-2)
        flag2, previous_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
        flag3, following_frame = cap.read()
        
        #resize to HD
        curr_frame = cv2.resize(curr_frame, (1920, 1080))
        previous_frame = cv2.resize(previous_frame, (1920, 1080))
        following_frame = cv2.resize(following_frame, (1920, 1080))
        
        #turn current frame to gray scale
        gray_curr_frame = curr_frame[:, :, 0]
        gray_previous_frame = previous_frame[:, :, 0]
        gray_following_frame = following_frame[:, :, 0]
        #gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) #не работает
        
        # finding the difference
        diff_1 = cv2.absdiff(src1=gray_previous_frame, src2=gray_curr_frame)
        diff_2 = cv2.absdiff(src1=gray_curr_frame, src2=gray_following_frame)
        diff_1 = np.where(diff_1 > 10, diff_1, 0)
        diff_2 = np.where(diff_2 > 10, diff_1, 0)
        # Конвертация в 3 канала
        diff_1_c = cv2.cvtColor(src=diff_1, code=cv2.COLOR_GRAY2RGB)
        diff_2_c = cv2.cvtColor(src=diff_2, code=cv2.COLOR_GRAY2RGB)
        # Создание маски 
        lower_white = np.array([0,0,15])
        upper_white = np.array([255,255,255])
        mask = cv2.inRange(diff_1_c, lower_white, upper_white)
        blured_mask = cv2.GaussianBlur(src=mask, ksize=(3,3), sigmaX=0)
        mask_inv = cv2.bitwise_not(blured_mask)
        
        #поиск контуров на маске
        contours, hierarchy = cv2.findContours(blured_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = np.zeros(frame.shape)
        cv2.drawContours(img_contours, contours, -1, (0,255,0), thickness=cv2.FILLED)
        
        #cv2.imshow("old_counters", img_contours)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        areas = []
        ind = []
        contours = np.array(contours, dtype=object)
        for j, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 40:
                ind.append(j)
        new_contours = []
        for j in ind:
            new_contours.append(contours[j])
            
        vis = np.zeros(frame.shape, dtype='uint8')
        cv2.drawContours(vis, new_contours, -1, (255,255,255), thickness=cv2.FILLED)
        vis = vis[:, :, 0]
        
        #cv2.imshow('new_contours', vis)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        # Extract the dimensions of the original image
        '''
        rows, cols, channels = frame.shape
        frame = frame[0:rows, 0:cols]
        red_image = np.zeros(frame.shape, dtype='uint8')
        red_image[:]=(0,0,255)
        '''
 
        # Bitwise-OR mask and original image
        '''
        alpha = 0.5
        overlay1 = frame.copy()
        print(red_image.shape)
        print(vis.shape)
        print(blured_mask.shape)
        ants = cv2.bitwise_or(frame, red_image, mask = vis)
        ants = ants[0:rows, 0:cols]
        background = cv2.addWeighted(ants, alpha, overlay1, 1, 0)
        cv2.imshow("background", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        bb = frame.copy()
        for c in new_contours:
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            cv2.rectangle(bb,(x,y),(x+w,y+h),(255,0,0),2)
          
        '''
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        #save
        cv2.imwrite(orig_file, frame)
        cv2.imwrite(filename, bb)

        print(f"Generated {i+1}/{NUM_FRAMES_IN_DATA} frames")
    
def check_deleted():
    new_dir = '/windows/d/ant_detection/compute_files/Aljona_Pospelova/Прессилябрис интенсивность/dataset1/frames'
    diff_path = '/windows/d/ant_detection/compute_files/Aljona_Pospelova/Прессилябрис интенсивность/dataset1/diff'
    orig_path = '/windows/d/ant_detection/compute_files/Aljona_Pospelova/Прессилябрис интенсивность/dataset1/orig'
    for f in os.scandir(diff_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            filename = f.path[f.path.rfind('/')+1:]
            number = filename[:filename.index('_')]
            image = cv2.imread(orig_path + '/' + number + '_orig_image.png')
            cv2.imwrite(new_dir + '/' + number + '_orig_image.png', image)
    
if __name__ == "__main__":
    NUM_FRAMES_IN_DATA = 5
    video_path = ''
    saving_frame_path = ''
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', nargs='?', default='/windows/d/ant_detection/compute_files/Aljona_Pospelova/Pratensis дороги 1 и 2/Intensivnost_1.mp4', help="Specify the full path to video", type=str)
    parser.add_argument('saving_path', nargs='?', default='/windows/d/ant_detection/compute_files/Aljona_Pospelova/Pratensis дороги 1 и 2/dataset1', help="Specify the full path to saving directory", type=str)
    parser.add_argument('num_images', nargs='?', default=200, help="Specify the number of generated images", type=int)
    args = parser.parse_args()
    
    NUM_FRAMES_IN_DATA = args.num_images
    video_path = args.video_path
    saving_frame_path = args.saving_path
    
    generate_diff_image(NUM_FRAMES_IN_DATA, video_path, saving_frame_path)
    '''
    check_deleted()
