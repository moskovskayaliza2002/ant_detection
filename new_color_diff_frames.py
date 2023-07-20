import numpy as np
import cv2
import argparse
import os
import shutil

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
        blured_mask = cv2.GaussianBlur(src=mask, ksize=(5,5), sigmaX=0)
        mask_inv = cv2.bitwise_not(blured_mask)
        
        # Extract the dimensions of the original image
        rows, cols, channels = frame.shape
        frame = frame[0:rows, 0:cols]
        
        red_image = np.zeros((1080,1920,3), np.uint8)
        red_image[:]=(0,0,255)
 
        # Bitwise-OR mask and original image
        alpha = 0.5
        overlay1 = frame.copy()
        ants = cv2.bitwise_or(frame, red_image, mask = blured_mask)
        ants = ants[0:rows, 0:cols]
        background = cv2.addWeighted(ants, alpha, overlay1, 1, 0)
        
        '''
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow("mask", blured_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow("ants", ants)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

        cv2.imshow("background", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
        #save
        cv2.imwrite(orig_file, frame)
        cv2.imwrite(filename, background)
        
        #alpha = 0.4
        #overlay1 = frame.copy()
        #image_new1 = cv2.addWeighted(diff_1_c, alpha, frame, 1 - alpha, 0)
        #image_new2 = cv2.addWeighted(diff_2_c, alpha, image_new1, 1 - alpha, 0)
        #diff = cv2.bitwise_or(diff_1, diff_2)
        #percentiles1 = np.percentile(diff_1, [0, 25, 75, 100])
        #percentiles2 = np.percentile(diff_2, [0, 25, 75, 100])
        #targets = np.geomspace(10, 255, 4)
        #b = np.interp(diff_1, percentiles1, targets).astype(np.uint8)
        #g = gray_curr_frame #np.zeros_like(diff_1)
        #r = np.interp(diff_2, percentiles2, targets[::-1]).astype(np.uint8)
        
        
        #b = np.maximum(gray_curr_frame, np.interp(diff_1, percentiles1, targets).astype(np.uint8))
        #g = gray_curr_frame
        #r = np.maximum(gray_curr_frame, np.interp(diff_2, percentiles2, targets[::-1]).astype(np.uint8))
        
        #result = cv2.merge([b, g, r])
        #cv2.imshow("diff", result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        # set 3 chanels
        #new_image[:, :, 0] = gray_curr_frame #B
        #new_image[:, :, 1] = np.maximum(gray_curr_frame, diff_1) #G
        #new_image[:, :, 2] = np.maximum(gray_curr_frame, diff_2) #R
        
        #save new_image
        #cv2.imwrite(filename, new_image)
        print(f"Generated {i+1}/{NUM_FRAMES_IN_DATA} frames")
    
    
    
if __name__ == "__main__":
    NUM_FRAMES_IN_DATA = 5
    video_path = ''
    saving_frame_path = ''

    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', nargs='?', default='/windows/d/ant_detection/compute_files/Aljona_Pospelova/Прессилябрис площадки 1-4/Площадка 3  .mp4', help="Specify the full path to video", type=str)
    parser.add_argument('saving_path', nargs='?', default='/windows/d/ant_detection/compute_files/Aljona_Pospelova/Прессилябрис площадки 1-4/dataset3', help="Specify the full path to saving directory", type=str)
    parser.add_argument('num_images', nargs='?', default=30, help="Specify the number of generated images", type=int)
    args = parser.parse_args()
    
    NUM_FRAMES_IN_DATA = args.num_images
    video_path = args.video_path
    saving_frame_path = args.saving_path
    
    generate_diff_image(NUM_FRAMES_IN_DATA, video_path, saving_frame_path)
        
