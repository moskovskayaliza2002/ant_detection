import random
from overlay_intersection import read_boxes, write_bbox, find_bbox, find_kp, resize_bboxes_kps
import os
import cv2
import shutil
import argparse

class Sizes:
    
    Width_orig = 1920
    Height_orig = 1080
    
    Min_new_w = 100
    Min_new_h = 50
    
    Max_new_w = 200 # bc 1920 // 4 = 480
    Max_new_h = 100 # bc 1080 // 4 = 270
    
def crop_image(image):
    
    x_start = random.randint(0, Sizes.Width_orig - Sizes.Max_new_w)
    y_start = random.randint(0, Sizes.Height_orig - Sizes.Max_new_h)
    
    new_w = random.randint(Sizes.Min_new_w, Sizes.Max_new_w)
    new_h = random.randint(Sizes.Min_new_h, Sizes.Max_new_h)
    
    crop = image[y_start : y_start + new_h, x_start : x_start + new_w]
    crop = cv2.resize(crop, (300, 300), interpolation = cv2.INTER_AREA)
    
    return crop, x_start, y_start, x_start + new_w, y_start + new_h
    
def vis_check(im, bboxes, keypoints):
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        im = cv2.rectangle(im.copy(), start_point, end_point, (255,0,0), 1)
    for kps in keypoints:
        h = (kps[0], kps[1])
        a = (kps[2], kps[3])
        overlay = im.copy()
        overlay = cv2.circle(overlay, h, 2, (255,0,0), 10)
        overlay = cv2.circle(overlay, a, 2, (255,0,0), 10)
        # try to make transparent
        alpha = 0.5
        im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
        
    cv2.imshow('im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser.add_argument('new_root_path', nargs='?', default="/home/ubuntu/ant_detection/dataset/augmentation", help="Specify the path to the folder to create new data", type=str)
    parser.add_argument('old_root_path', nargs='?', default="/home/ubuntu/ant_detection/real_im_annot", help="Specify the path to the folder to original data", type=str)
    args = parser.parse_args()
    new_root_path = args.new_root_path
    old_root_path = args.old_root_path
    #new_root_path = '/home/ubuntu/ant_detection/dataset/augmentation'
    #old_root_path = '/home/ubuntu/ant_detection/real_im_annot'
    new_keypoints_path = new_root_path + '/keypoints'
    new_bboxes_path = new_root_path + '/bboxes'
    new_images_path = new_root_path + '/images'
    for i in [new_keypoints_path, new_bboxes_path, new_images_path]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i) 
    
    #c count + 1 начнется нумерация
    counter = 863
    for f in os.scandir(old_root_path + '/images'):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            original_image = cv2.imread(f.path)
            #image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            number = int(f.path[f.path.rfind('e') + 1 : f.path.rfind('.')])
            #print(number)
            original_bboxs = find_bbox(number, old_root_path)
            _, original_keypoints = find_kp(number, old_root_path)
            for i in range(70):
                cr_im, left_x, left_y, right_x, right_y = crop_image(original_image)
                ants_counter, noise_counter, new_bb, new_kp = resize_bboxes_kps(original_bboxs, original_keypoints, left_x, left_y, right_x, right_y)
                if ants_counter != 0 and noise_counter == 0:
                    #resize_im = cv2.resize(cr_im, (300, 300), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(new_images_path + '/image' + str(counter + 1) + '.png', cr_im)
                    write_bbox(new_bb, new_bboxes_path + '/bbox' + str(counter + 1) + '.txt')
                    write_bbox(new_kp, new_keypoints_path + '/keypoint' + str(counter + 1) + '.txt')
                    #vis_check(cr_im, new_bb, new_kp)
                    counter += 1
