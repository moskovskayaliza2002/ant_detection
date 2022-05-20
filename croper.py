import os
import cv2
import argparse
import xml.etree.ElementTree as ET
import torch

def read_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    list_with_single_boxes = []
    for boxes in root.iter('object'):
        
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text) 
        xmin = int(boxes.find("bndbox/xmin").text) 
        ymax = int(boxes.find("bndbox/ymax").text) 
        xmax = int(boxes.find("bndbox/xmax").text)
        
        list_with_single_boxes.append([xmin, ymin, xmax, ymax])
    return list_with_single_boxes

# проверка что ни одна граница обрезанного изображения не попадает в бокс
def verification(boxes, left_x, left_y, right_x, right_y):
    flag = True
    for i in boxes:
    # i[0] = xmin
    # i[1] = ymin
    # i[2] = xmax
    # i[3] = ymax
        if i[0] < left_x < i[2] and i[1] < left_y < i[3]:
            flag = False
            break
        elif i[0] < right_x < i[2] and i[1] < right_y < i[3]:
            flag = False
            break
        elif i[0] < left_x < i[2] and i[1] < right_y < i[3]:
            flag = False
            break
        elif i[0] < right_x < i[2] and i[1] < left_y < i[3]:
            flag = False
            break
        elif i[0] < right_x < i[2] and right_y > i[3]:
            flag = False
            break
        elif i[0] < left_x < i[2] and right_y > i[3]:
            flag = False
            break
        elif left_x < i[0] and i[1] < right_y < i[3]:
            flag = False
            break
        elif right_x > i[2] and i[1] < right_y < i[3]:
            flag = False
            break
        elif i[0] < left_x < i[2] and left_y < i[1]:
            flag = False
            break
        elif i[0] < right_x < i[2] and left_y < i[1]:
            flag = False
            break
        elif right_x > i[2] and i[1] < left_y < i[3]:
            flag = False
            break
        elif left_x < i[0] and i[1] < left_y < i[3]:
            flag = False
            break
    return flag


def resize_bboxes(bboxes, left_x, left_y, right_x, right_y):
    new_list_bboxes = []
    k = right_y - right_x
    for i in bboxes:
        ymin, xmin, ymax, xmax = i[1], i[0], i[3], i[2]
        # проверка на то, что это не бокс за границей обрезанной области
        if not(xmax < left_x or ymax < left_y or ymin > right_y or xmin > right_x):
            new_list_bboxes.append([xmin - left_x, ymin - left_y, xmax - left_x, ymax - left_y])
            
    return new_list_bboxes
        

def write_bbox(bbox, filename):
    str_list = []
    for i in bbox:
        s = ' '.join(map(str, i)) + "\n"
        str_list.append(s)
    with open(filename, 'w') as file:
        file.writelines(str_list)
        file.close()

def crop_data(root_path):
    
    # Prepare folder for crop images
    dir = os.path.join(root_path, 'croped_data')
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    # Prepare crop area    
    width, height = 500, 500
    x, y = 1400, 200
    
    # Read images
    orig_data_path = root_path + 'FILE0001'
    counter = 0
    for f in os.scandir(orig_data_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'jpg':
            original_image = cv2.imread(f.path)
            crop_img = original_image[y:y+height, x:x+width]
            xml_path = f.path[:-3] + 'xml'
            real_boxes = read_xml(xml_path)
            check_bboxes = verification(real_boxes, x, y, x+width, y+height)
            if check_bboxes:
                im_filename = dir + '/croped' + str(counter) + '.jpg'
                cv2.imwrite(im_filename, crop_img)
                #тут еще сохранить новые боксы
                new_bboxes = resize_bboxes(real_boxes, x, y, x+width, y+height)
                txt_filename = dir + '/croped' + str(counter) + '.txt'
                write_bbox(new_bboxes, txt_filename)
                print('сохранено')
                counter += 1
            else:
                print('удалено')
                
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', nargs='?', default='/home/lizamoscow/ant_detection/', help="Specify main directory", type=str)
    args = parser.parse_args()
    root_path = args.root_path
    crop_data(root_path)
