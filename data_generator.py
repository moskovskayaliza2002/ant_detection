from random import randint
import cv2
import math
import os
import glob
import argparse

def generator_images(im_size, min_ants, max_ants, body_radius, head_radius, image_path):
    
    color = (0,0,0)
    number_of_ants = randint(min_ants, max_ants)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, im_size)
    bboxes = []
    keypoints = []
    centeres = []
    for i in range(number_of_ants):
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
                for j in range(len(centeres)):
                    if (x_center_body - centeres[j][0])**2 + (y_center_body - centeres[j][1])**2 <= (body_radius+head_radius)**2:
                        flag_centers = False
                    if (x_center_head - centeres[j][0])**2 + (y_center_head - centeres[j][1])**2 <= (body_radius+head_radius)**2:
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
        
    return keypoints, bboxes, img

def write_txt(list_of_lists, filename):
    str_list = []
    for i in list_of_lists:
        s = ' '.join(map(str, i)) + "\n"
        str_list.append(s)
    with open(filename, 'w') as file:
        file.writelines(str_list)
        file.close()

def create_dataset(amound_of_data, root_path, im_size, min_ants, max_ants, body_radius, head_radius):
    all_files = []
    background_path = root_path + '/background_im'
    saving_path = root_path + '/synthetic_data'
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    
    for f in os.scandir(background_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            all_files.append(f.path)
    
    dir_size = len(glob.glob(background_path + '/*'))
    print(dir_size)
    for i in range(amound_of_data):
        index = randint(0, dir_size-1)
        image_p = all_files[index]
        k, bb, image = generator_images(im_size, min_ants, max_ants, body_radius, head_radius, image_p)
        im_filename = saving_path + '/image' + str(i) + '.png'
        cv2.imwrite(im_filename, image)
        bb_filename = saving_path + '/bboxes' + str(i) + '.txt'
        write_txt(bb, bb_filename)
        k_filename = saving_path + '/keypoints' + str(i) + '.txt'
        write_txt(k, k_filename)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', nargs='?', default='/home/ubuntu/ant_detection', help="Specify main directory", type=str)
    parser.add_argument('amound_of_data', nargs='?', default=5, help="Specify the number of generated images", type=int)
    parser.add_argument('im_size', nargs='?', default=(320,320), help="Specify the size of generated images", type=tuple)
    parser.add_argument('min_ants', nargs='?', default=5, help="Specify the minimum amound of ants per image", type=int)
    parser.add_argument('max_ants', nargs='?', default=10, help="Specify the maximum amound of ants per image", type=int)
    parser.add_argument('body_radius', nargs='?', default=10, help="Specify the radius of the body", type=int)
    parser.add_argument('head_radius', nargs='?', default=7, help="Specify the radius of the head", type=int)
    args = parser.parse_args()
    
    root_path = args.root_path
    amound_of_data = args.amound_of_data
    im_size = args.im_size
    min_ants = args.min_ants
    max_ants = args.max_ants
    body_radius = args.body_radius
    head_radius = args.head_radius
    
    create_dataset(amound_of_data, root_path, im_size, min_ants, max_ants, body_radius, head_radius)
    