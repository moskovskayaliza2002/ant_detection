from random import randint, uniform
import cv2
import os
import glob
import argparse
import shutil
import numpy as np


def gauss_noise(image, gauss_var):
    
    mean = 0
    sigma = gauss_var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    res = image + gauss
    noisy = np.clip(res, 0, 255).astype(np.uint8)
    
    return noisy
    

def generator_images(im_size, min_ants, max_ants, min_body_r, max_body_r, image_path, p, gauss_var_min, gauss_var_max):
    color = (0,0,0)
    number_of_ants = randint(min_ants, max_ants)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, im_size)
    bboxes = []
    keypoints = []
    centeres = []
    for i in range(number_of_ants):
        body_radius = randint(min_body_r, max_body_r)
        head_radius = int(p * body_radius)
        flag_centers = False
        while not flag_centers:
            flag_centers = True
            
            x_center_body = randint(body_radius + 2 * head_radius, im_size[0] - (body_radius + 2 * head_radius))
            y_center_body = randint(body_radius + 2 * head_radius, im_size[0] - (body_radius + 2 * head_radius))
            
            flag = False
            while not flag:
                x_center_head = randint(body_radius + 2 * head_radius, im_size[0] - (body_radius + 2 * head_radius))
                y_center_head = randint(body_radius + 2 * head_radius, im_size[0] - (body_radius + 2 * head_radius))
                
                if (x_center_head - x_center_body)**2 + (y_center_head - y_center_body)**2 == (body_radius + head_radius)**2:
                    flag = True
            
            if i != 0:
                max_cond_r = max_body_r*2 + int(max_body_r*p)
                for j in range(len(centeres)):
                    if (x_center_body - centeres[j][0])**2 + (y_center_body - centeres[j][1])**2 <= (max_cond_r)**2:
                        flag_centers = False
                    if (x_center_head - centeres[j][0])**2 + (y_center_head - centeres[j][1])**2 <= (max_cond_r)**2:
                        flag_centers = False  
            
        centeres.append([(x_center_head + x_center_body)//2, (y_center_body + y_center_head)//2])   
            
        img = cv2.circle(img, (x_center_body, y_center_body), body_radius, color, -1)
        img = cv2.circle(img, (x_center_head, y_center_head), head_radius, color, -1)
            
        x_min = min(x_center_body - body_radius, x_center_head - head_radius)
        y_min = min(y_center_body - body_radius, y_center_head - head_radius)
        x_max = max(x_center_body + body_radius, x_center_head + head_radius)
        y_max = max(y_center_body + body_radius, y_center_head + head_radius)
            
            
        bboxes.append([x_min, y_min, x_max, y_max])
        keypoints.append([x_center_body, y_center_body, x_center_head, y_center_head])
        variance = randint(gauss_var_min, gauss_var_max)
        img = gauss_noise(img, variance)
        
    return keypoints, bboxes, img

def write_txt(list_of_lists, filename):
    str_list = []
    for i in list_of_lists:
        s = ' '.join(map(str, i)) + "\n"
        str_list.append(s)
    with open(filename, 'w') as file:
        file.writelines(str_list)
        file.close()

def create_dataset(amound_of_data, root_path, background_path, im_size, min_ants, max_ants, body_radius, head_radius, p, gauss_var_min, gauss_var_max):
    all_files = []
    if background_path == None:
        background_path = root_path + '/background_im'
    saving_path_tr = root_path + '/train_data'
    saving_path_te = root_path + '/test_data'
    for i in [saving_path_tr, saving_path_te]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i)
        
    image_path_tr = saving_path_tr + '/images'
    keypoints_path_tr = saving_path_tr + '/keypoints'
    bboxes_path_tr = saving_path_tr + '/bboxes'
    
    image_path_te = saving_path_te + '/images'
    keypoints_path_te = saving_path_te + '/keypoints'
    bboxes_path_te = saving_path_te + '/bboxes'
    
    for i in [image_path_tr, keypoints_path_tr, bboxes_path_tr, image_path_te, keypoints_path_te, bboxes_path_te]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i)
    
    for f in os.scandir(background_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            all_files.append(f.path)
    
    dir_size = len(glob.glob(background_path + '/*'))
    for i in range(amound_of_data):
        index = randint(0, dir_size-1)
        image_p = all_files[index]
        k, bb, image = generator_images(im_size, min_ants, max_ants, body_radius, head_radius, image_p, p, gauss_var_min, gauss_var_max)
        im_filename = image_path_tr + '/image' + str(i) + '.png'
        cv2.imwrite(im_filename, image)
        bb_filename = bboxes_path_tr + '/bbox' + str(i) + '.txt'
        write_txt(bb, bb_filename)
        k_filename = keypoints_path_tr + '/keypoint' + str(i) + '.txt'
        write_txt(k, k_filename)
        
    test_amound = int(amound_of_data * 0.2)
    
    for i in range(test_amound):
        index = randint(0, dir_size-1)
        image_p = all_files[index]
        k, bb, image = generator_images(im_size, min_ants, max_ants, body_radius, head_radius, image_p, p, gauss_var_min, gauss_var_max)
        im_filename = image_path_te + '/image' + str(i) + '.png'
        cv2.imwrite(im_filename, image)
        bb_filename = bboxes_path_te + '/bbox' + str(i) + '.txt'
        write_txt(bb, bb_filename)
        k_filename = keypoints_path_te + '/keypoint' + str(i) + '.txt'
        write_txt(k, k_filename)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', nargs='?', default='/home/ubuntu/ant_detection', help="Specify main directory", type=str)
    parser.add_argument('background_path', nargs='?', default=None, help="Specify file with background images", type=str)
    parser.add_argument('amound_of_data', nargs='?', default=100, help="Specify the number of generated images", type=int)
    parser.add_argument('im_size', nargs='?', default=(320,320), help="Specify the size of generated images", type=tuple)
    parser.add_argument('min_ants', nargs='?', default=5, help="Specify the minimum amound of ants per image", type=int)
    parser.add_argument('max_ants', nargs='?', default=11, help="Specify the maximum amound of ants per image", type=int)
    parser.add_argument('min_body_r', nargs='?', default=7, help="Specify the minimum radius of the body", type=int)
    parser.add_argument('max_body_r', nargs='?', default=10, help="Specify the maximum radius of the head", type=int)
    parser.add_argument('p', nargs='?', default=0.6, help="Show the proportion of body radius to head radius. [0,1]", type=float)
    parser.add_argument('gauss_var_min', nargs='?', default=100, help="Left bound of the Gaussian distribution variance", type=int)
    parser.add_argument('gauss_var_max', nargs='?', default=900, help="Right bound of the Gaussian distribution variance", type=int)
    args = parser.parse_args()
    
    root_path = args.root_path
    background_path = args.background_path
    amound_of_data = args.amound_of_data
    im_size = args.im_size
    min_ants = args.min_ants
    max_ants = args.max_ants
    min_body_r = args.min_body_r
    max_body_r = args.max_body_r
    p = args.p
    gauss_var_min = args.gauss_var_min
    gauss_var_max = args.gauss_var_max
    
    create_dataset(amound_of_data, root_path, background_path, im_size, min_ants, max_ants, min_body_r, max_body_r, p, gauss_var_min, gauss_var_max)
    
