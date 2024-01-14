import cv2
import os
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt

def create_data(pref, dataset_path, video_path):
    #серые изображения трех последовательных кадров по трем каналам
    img_or_path = dataset_path + "images/"
    img_foll_path = dataset_path + "new_images/"
    if not os.path.exists(img_foll_path):
        os.mkdir(img_foll_path)
    else:
        shutil.rmtree(img_foll_path)
        os.mkdir(img_foll_path)
    cap = cv2.VideoCapture(video_path)
    max_amound_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in os.scandir(img_or_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            number = int(f.path[f.path.rfind("e") + 1 : f.path.rfind(".")])
            #reading 3 following frames
            if number == 1:
                #begin of video, previous_frame = current
                flag1, curr_frame = cap.read()
                previous_frame = curr_frame.copy()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number)
                flag3, following_frame = cap.read()
                
            elif number == max_amound_of_frames:
                #end of video, following_frame = current
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
                flag1, curr_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 2)
                flag2, previous_frame = cap.read()
                following_frame = curr_frame.copy()
            
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
                flag1, curr_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 2)
                flag2, previous_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number)
                flag3, following_frame = cap.read()
            
            #resize to HD
            curr_frame = cv2.resize(curr_frame, (1920, 1080))
            previous_frame = cv2.resize(previous_frame, (1920, 1080))
            following_frame = cv2.resize(following_frame, (1920, 1080))
            
            #turn current frame to gray scale
            gray_curr_frame = curr_frame[:, :, 0]
            gray_previous_frame = previous_frame[:, :, 0]
            gray_following_frame = following_frame[:, :, 0]
            
            #merge into 1 bgr
            bgr = cv2.merge((gray_previous_frame, gray_curr_frame, gray_following_frame))
            
            #save
            cv2.imwrite(img_foll_path + pref + "image" + str(number) + ".png", bgr)
    
    
def create_data_diff(pref, dataset_path, video_path):
    #серые изображения трех последовательных кадров по трем каналам
    img_or_path = dataset_path + "images/"
    img_foll_path = dataset_path + "new_images_diff/"
    if not os.path.exists(img_foll_path):
        os.mkdir(img_foll_path)
    else:
        shutil.rmtree(img_foll_path)
        os.mkdir(img_foll_path)
    cap = cv2.VideoCapture(video_path)
    max_amound_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in os.scandir(img_or_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            number = int(f.path[f.path.rfind("e") + 1 : f.path.rfind(".")])
            #reading 3 following frames
            if number == 1:
                #begin of video, previous_frame = current
                flag1, curr_frame = cap.read()
                previous_frame = curr_frame.copy()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number)
                flag3, following_frame = cap.read()
                
            elif number == max_amound_of_frames:
                #end of video, following_frame = current
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
                flag1, curr_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 2)
                flag2, previous_frame = cap.read()
                following_frame = curr_frame.copy()
            
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
                flag1, curr_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 2)
                flag2, previous_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number)
                flag3, following_frame = cap.read()
            
            #resize to HD
            curr_frame = cv2.resize(curr_frame, (1920, 1080))
            previous_frame = cv2.resize(previous_frame, (1920, 1080))
            following_frame = cv2.resize(following_frame, (1920, 1080))
            
            #turn current frame to gray scale
            gray_curr_frame = curr_frame[:, :, 0]
            gray_previous_frame = previous_frame[:, :, 0]
            gray_following_frame = following_frame[:, :, 0]
            
            #find difference
            diff_1 = cv2.absdiff(src1=gray_previous_frame, src2=gray_curr_frame)
            diff_2 = cv2.absdiff(src1=gray_curr_frame, src2=gray_following_frame)
            
            #merge into 1 bgr
            bgr = cv2.merge((diff_1, gray_curr_frame, diff_2))
            
            #save
            cv2.imwrite(img_foll_path + pref + "image" + str(number) + ".png", bgr)
            
def create_data_areas(pref, dataset_path, video_path):
    #серые изображения трех последовательных кадров по трем каналам
    img_or_path = dataset_path + "images/"
    img_foll_path = dataset_path + "new_images_area/"
    if not os.path.exists(img_foll_path):
        os.mkdir(img_foll_path)
    else:
        shutil.rmtree(img_foll_path)
        os.mkdir(img_foll_path)
    cap = cv2.VideoCapture(video_path)
    max_amound_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in os.scandir(img_or_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            number = int(f.path[f.path.rfind("e") + 1 : f.path.rfind(".")])
            #reading 3 following frames
            if number == 1:
                #begin of video, previous_frame = current
                flag1, curr_frame = cap.read()
                previous_frame = curr_frame.copy()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number)
                flag3, following_frame = cap.read()
                        
            elif number == max_amound_of_frames:
                #end of video, following_frame = current
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
                flag1, curr_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 2)
                flag2, previous_frame = cap.read()
                following_frame = curr_frame.copy()
            
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
                flag1, curr_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number - 2)
                flag2, previous_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, number)
                flag3, following_frame = cap.read()
                    
            #resize to HD
            curr_frame = cv2.resize(curr_frame, (1920, 1080))
            previous_frame = cv2.resize(previous_frame, (1920, 1080))
            following_frame = cv2.resize(following_frame, (1920, 1080))
                    
            #turn current frame to gray scale
            gray_curr_frame = curr_frame[:, :, 0]
            gray_previous_frame = previous_frame[:, :, 0]
            gray_following_frame = following_frame[:, :, 0]
                    
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
            img_contours = np.zeros(curr_frame.shape)
            cv2.drawContours(img_contours, contours, -1, (0,255,0), thickness=cv2.FILLED)
                
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
                        
            vis = np.zeros(curr_frame.shape, dtype='uint8')
            cv2.drawContours(vis, new_contours, -1, (255,255,255), thickness=cv2.FILLED)
            vis = vis[:, :, 0]
                
            # Extract the dimensions of the original image
            rows, cols, channels = curr_frame.shape
            curr_frame = curr_frame[0:rows, 0:cols]
            red_image = np.zeros(curr_frame.shape, dtype='uint8')
            red_image[:]=(0,0,255)
            
            # Bitwise-OR mask and original image
            alpha = 0.3
            overlay1 = curr_frame.copy()
            ants = cv2.bitwise_or(curr_frame, red_image, mask = vis)
            ants = ants[0:rows, 0:cols]
            background = cv2.addWeighted(ants, alpha, overlay1, 1, 0)
            #cv2.imshow("background", background)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            #save
            cv2.imwrite(img_foll_path + pref + "image" + str(number) + ".png", background)
    cap.release()
    cv2.destroyAllWindows()
            
        
def visualize(image, bboxes, keypoints, scores, image_original=None, bboxes_original=None, keypoints_original=None, show_flag = True):
    # Рисует на изображении предсказанные и настоящие боксы и ключевые точки.
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # !!!!! ВОЗМОЖНО НУЖНО УДАЛИТЬ !!!!!
    fontsize = 12
    keypoints_classes_ids2names = {0: 'A', 1: 'H'}
    for idx, bbox in enumerate(bboxes):
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,0,255), 2)
        org = (bbox[0] - 3, bbox[1] - 3)
        image = cv2.putText(image.copy(), str(round(scores[idx], 2)), org , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            overlay = image.copy()
            overlay = cv2.circle(overlay, tuple(kp), 2, (0,0,255), 10)
            # try to make transparent
            alpha = 0.5
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    if image_original is None and keypoints_original is None:
        if show_flag:
            plt.figure(figsize=(40,40))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.show(block=True)
        else:
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                orig_overlay = image_original.copy()
                orig_overlay = cv2.circle(orig_overlay, tuple(kp), 2, (0,255,0), 10)
                alpha = 0.5
                image_original = cv2.addWeighted(orig_overlay, alpha, image_original, 1 - alpha, 0)
                #image_original = cv2.circle(image_original, tuple(kp), 2, (0,255,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                
        f, ax = plt.subplots(1, 2, figsize=(40, 20))
        
        #image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        ax[0].imshow(image_original)
        ax[0].set_title('Original annotations', fontsize=fontsize)
        
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[1].imshow(image)
        ax[1].set_title('Predicted annotations', fontsize=fontsize)
        plt.show(block=True)
        
        
def read_boxes(bbox_path):
    # Считывает боксы
    bboxes_original = []
    with open(bbox_path) as f:
        for i in f:
            x_min, y_min, x_max, y_max = map(int, i[:-1].split(' '))
            bboxes_original.append([x_min, y_min, x_max, y_max])
    return bboxes_original


def read_kps(kp_path):
    kp_original = []
    with open(kp_path) as f:
        for i in f:
            a_x, a_y, h_x, h_y = map(int, i[:-1].split(' '))
            kp_original.append([[a_x, a_y], [h_x, h_y]])
    return kp_original


def check_annot(pref, number, dataset_path):
    bb_path = dataset_path + "bboxes/" + pref + "bbox" + str(number) + ".txt"
    kp_path = dataset_path + "keypoints/" + pref + "keypoint" + str(number) + ".txt"
    im_path = dataset_path + "new_images/image" + str(number) + ".png"
    	
    image = cv2.imread(im_path)
    bb = read_boxes(bb_path)
    kp = read_kps(kp_path)
    scores = [1] * len(bb)
    
    visualize(image, bb, kp, scores, show_flag = True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', nargs='?', default='', help="Specify the full path to video", type=str)
    parser.add_argument('--data_path', nargs='?', default='', help="Specify the full path to data", type=str)
    parser.add_argument('--pref', nargs='?', default='', help="Specify prefix of saving", type=str)
    args = parser.parse_args()
    
    create_data_areas(args.pref, args.data_path, args.video_path)
    #create_data(args.pref, args.data_path, args.video_path)
    #check_annot("1_", 2041, "/windows/d/ant_detection/compute_files/Aljona_Pospelova/doroga_rufy/dataset/")
