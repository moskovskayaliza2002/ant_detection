import json
import numpy as np

def read_json(path):
    
    head_list = [] # shape [N_im, N_ob, 2]
    abdomen_list = [] # shape [N_im, N_ob, 2]
    bboxes_list = [] # shape [N_im, N_ob, 4]
    
    with open(path) as f:
        data = json.load(f)
    
    #print(len(data[0]['kp-1']))
    #print(data)
    for img in data:
        single_head_list = []
        single_abdomen_list = []
        
        for kps in img['kp-1']:
            label = kps['keypointlabels'][0]
            kp_x = kps['x']
            kp_y = kps['y']
            
            if label == 'Abdomen':
                single_abdomen_list.append([kp_x, kp_y])
            elif label == 'Head':
                single_head_list.append([kp_x, kp_y])
                
        head_list.append(single_head_list)
        abdomen_list.append(single_abdomen_list)
        
        single_bboxes_list = []
        for bboxes in img['label']:
            #print(bboxes)
            xmin = bboxes['x']
            ymin = bboxes['y']
            xmax = bboxes['x'] + bboxes['width']
            ymax = bboxes['y'] + bboxes['height']
            
            if bboxes['rectanglelabels'][0] == 'Ant':
                single_bboxes_list.append([xmin, ymin, xmax, ymax])
            
        bboxes_list.append(single_bboxes_list)
    print(data[2])
    #print(f'bboxes_list {len(bboxes_list[0]), len(bboxes_list[1]), len(bboxes_list[2])}')
    #print(f'\nhead_list {len(head_list[0]), len(head_list[1]), len(head_list[2])}')
    #print(f'\nabdomen_list {len(abdomen_list[0]), len(abdomen_list[1]), len(abdomen_list[2])}')
        
if __name__ == '__main__':
    read_json('/home/ubuntu/Downloads/project-3-at-2022-06-15-13-32-4d657fe5.json')
