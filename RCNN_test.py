import argparse
from torch.utils.data import Dataset, DataLoader
from RCNN_model import ClassDataset, get_model, visualize
import cv2
import torch
import glob
from utils import collate_fn
from torchvision.transforms import functional as F
import numpy as np
import torchvision
from crop_into4 import crop_one_im, read_boxes


def ver(bboxes, kp, l1, l2, r1, r2, crop_w, crop_h):
    flag_l1 = True
    flag_l2 = True
    flag_r1 = True
    flag_r2 = True
    for i in bboxes:
        xmin = i[0]
        ymin = i[1]
        xmax = i[2]
        ymax = i[3]
        if xmin < crop_w < xmax and ymax < crop_h:
            flag_l1 = flag_r1 = False
        if xmin < crop_w < xmax and ymin > crop_h:
            flag_l2 = flag_r2 = False
        if ymin < crop_h < ymax and xmax < crop_w:
            flag_l1 = flag_l2 = False
        if ymin < crop_h < ymax  and xmin > crop_w:
            flag_r1 = flag_r2 = False
        if ymin < crop_h < ymax and xmin < crop_w < xmax:
            flag_l1 = flag_r1 = flag_l2 = flag_r2 = False
    
    if flag_l1 and flag_l2 and flag_r1 and flag_r2 == False:
        return False
    else:
        return True


def get_out_kp_bb(out, left_x, left_y, keypoints, bboxes):
    scores = out[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > nms_threshold)[0].tolist() # Indexes of boxes with scores > 0.7
    post_nms_idxs = torchvision.ops.nms(out[0]['boxes'][high_scores_idxs], out[0]['scores'][high_scores_idxs], iou_threshold).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
    #keypoints = []
    for kps in out[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        int_kps = [list(map(int, kp[:2])) for kp in kps]
        x_a, y_a = int_kps[0][0], int_kps[0][1]
        x_h, y_h = int_kps[1][0], int_kps[1][1]
        keypoints.append([[x_a + left_x, y_a + left_y], [x_h + left_x, y_h + left_y]])

    #bboxes = []
    for bbox in out[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        xmin, ymin, xmax, ymax = map(int, bbox.tolist())
        bboxes.append([xmin + left_x, ymin + left_y, xmax + left_x, ymax + left_y])
    
    return keypoints, bboxes
    

def test_one_from4(im_path, model, flag, nms_threshold, iou_threshold):
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    #input_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_im = F.to_tensor(input_im)
    orig_bb = []
    orig_kp = []
    if flag:
        number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
        im_root = im_path[:im_path.rfind('/')]
        test_root = im_root[:im_root.rfind('/') + 1]
        b_path = test_root + 'bboxes/bbox' + number + '.txt'
        orig_bb = read_boxes(b_path)
        k_path = test_root + 'keypoints/keypoint' + number + '.txt'
        kp = read_boxes(k_path)
        for i in kp:
            orig_kp.append([[i[0], i[1]], [i[2], i[3]]])
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    left_1, left_2, right_1, right_2, c_w, c_h = crop_one_im(img)
    if ver(orig_bb, kp, left_1, left_2, right_1, right_2, c_w, c_h):
        with torch.no_grad():
            model.to(device)
            model.eval()
            out_l1 = model([F.to_tensor(cv2.cvtColor(left_1, cv2.COLOR_BGR2RGB))])
            out_l2 = model([F.to_tensor(cv2.cvtColor(left_2, cv2.COLOR_BGR2RGB))])
            out_r1 = model([F.to_tensor(cv2.cvtColor(right_1, cv2.COLOR_BGR2RGB))])
            out_r2 = model([F.to_tensor(cv2.cvtColor(right_2, cv2.COLOR_BGR2RGB))])
    
        kp_l1, bb_l1 = get_out_kp_bb(out_l1, 0, 0, [], [])
        kp_l2, bb_l2 = get_out_kp_bb(out_l2, 0, c_h, kp_l1, bb_l1)
        kp_r1, bb_r1 = get_out_kp_bb(out_r1, c_w, 0, kp_l2, bb_l2)
        kp_r2, bb_r2 = get_out_kp_bb(out_r2, c_w, c_h, kp_r1, bb_r1)
            
        visualize(img, bb_r2, kp_r2, img, orig_bb, orig_kp)
    else:
        print("Изображение не подходит")
    
    
def test_batch_from4(model, data_loader_test, flag, nms_threshold, iou_threshold, dir_size):
    print(dir_size)
    print("зашел в функцию")
    for i in range(dir_size):
        print("зашел в цикл")
        iterator = iter(data_loader_test)
        images, targets = next(iterator)
        images = list(image.to(device) for image in images)
        
        image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        gt_bboxes = targets[0]["boxes"]
        gt_kps = targets[0]["keypoints"]
        
        left_1, left_2, right_1, right_2, c_w, c_h = crop_one_im(image)
        print("разрезал")
        bb = targets[0]["boxes"].int().tolist()
        print(f'bb_orig: {bb[0]}')
        kp = targets[0]["keypoints"].int().tolist()
        kp_for_ver = []
        orig_kp = []
        for j in kp:
            kp_for_ver.append([j[0][0], j[0][1], j[1][0], j[1][1]])
            orig_kp.append([[j[0][0], j[0][1]], [j[1][0], j[1][1]]])
        print(f'kp_orig: {orig_kp[0]}')
        if ver(bb, kp_for_ver, left_1, left_2, right_1, right_2, c_w, c_h):
            with torch.no_grad():
                model.to(device)
                model.eval()
                out_l1 = model([F.to_tensor(cv2.cvtColor(left_1, cv2.COLOR_BGR2RGB))])
                out_l2 = model([F.to_tensor(cv2.cvtColor(left_2, cv2.COLOR_BGR2RGB))])
                out_r1 = model([F.to_tensor(cv2.cvtColor(right_1, cv2.COLOR_BGR2RGB))])
                out_r2 = model([F.to_tensor(cv2.cvtColor(right_2, cv2.COLOR_BGR2RGB))])
        
            kp_l1, bb_l1 = get_out_kp_bb(out_l1, 0, 0, [], [])
            kp_l2, bb_l2 = get_out_kp_bb(out_l2, 0, c_h, kp_l1, bb_l1)
            kp_r1, bb_r1 = get_out_kp_bb(out_r1, c_w, 0, kp_l2, bb_l2)
            kp_r2, bb_r2 = get_out_kp_bb(out_r2, c_w, c_h, kp_r1, bb_r1)
            
            #keypoints_pred = np.vstack((kp_l1, kp_l2, kp_r1, kp_r2))
            #bboxes_pred =  np.vstack((bb_l1, bb_l2, bb_r1, bb_r2))
            
            visualize(image, bb_r2, kp_r2, image, bb, orig_kp)
            print("зашел в рисовалку")
        else:
            print("Изображение не подходит")

    
def test_batch(model, data_loader_test, flag, nms_threshold, iou_threshold, dir_size):
    for i in range(dir_size):
        iterator = iter(data_loader_test)
        images, targets = next(iterator)
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            model.to(device)
            model.eval()
            output = model(images)
                
        image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            
        if flag:
            gt_bboxes = targets[0]["boxes"]
            gt_kps = targets[0]["keypoints"]
            #gt_kps = torch.squeeze(gt_kps)
            for i in range(gt_bboxes.size(dim=0)):
                x_min, y_min, x_max, y_max = int(gt_bboxes[i][0].item()), int(gt_bboxes[i][1].item()), int(gt_bboxes[i][2].item()), int(gt_bboxes[i][3].item())
                image = cv2.rectangle(image.copy(),(x_min,y_min),(x_max,y_max),(50,205,50),1)
                x_center_body, y_center_body = int(gt_kps[i][0][0].item()), int(gt_kps[i][0][1].item())
                x_center_head, y_center_head = int(gt_kps[i][1][0].item()), int(gt_kps[i][1][1].item())
                image = cv2.circle(image.copy(), (x_center_body, y_center_body), 2, (50,205,50), -1)
                image = cv2.putText(image.copy(),'A', (x_center_body + 2, y_center_body + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,205,50), 1)
                image = cv2.circle(image.copy(), (x_center_head, y_center_head), 2, (50,205,50), -1)
                image = cv2.putText(image.copy(),'H', (x_center_head + 2, y_center_head + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,205,50), 1)
            
        scores = output[0]['scores'].detach().cpu().numpy()

        high_scores_idxs = np.where(scores > nms_threshold)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], iou_threshold).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        bboxes = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))
            
        visualize(image, bboxes, keypoints)
        
        

def test_of_single_image(im_path, model, flag, nms_threshold, iou_threshold):
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    #input_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_im = F.to_tensor(input_im)
    print(input_im.size())
    
    if flag:
        print(im_path.rfind('e'), im_path.rfind('.'))
        number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
        print(im_path[:im_path.rfind('/')])
        im_root = im_path[:im_path.rfind('/')]
        test_root = im_root[:im_root.rfind('/') + 1]
        print(test_root)
        b_path = test_root + 'bboxes/bbox' + number + '.txt'
        k_path = test_root + 'keypoints/keypoint' + number + '.txt'
        
        with open(b_path) as f:
            for i in f:
                x_min, y_min, x_max, y_max = map(int, i[:-1].split(' '))
                img = cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(50,205,50),1)
        
        with open(k_path) as f:
            for i in f:
                x_center_body, y_center_body, x_center_head, y_center_head = map(int, i[:-1].split(' '))
                img = cv2.circle(img, (x_center_body, y_center_body), 2, (50,205,50), -1)
                img = cv2.putText(img,'A', (x_center_body + 2, y_center_body + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,205,50), 1)
                img = cv2.circle(img, (x_center_head, y_center_head), 2, (50,205,50), -1)
                img = cv2.putText(img,'H', (x_center_head + 2, y_center_head + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,205,50), 1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model([input_im])
         
    scores = output[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > nms_threshold)[0].tolist() # Indexes of boxes with scores > 0.7
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], iou_threshold).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
    
    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))
        
    visualize(img, bboxes, keypoints)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/TEST_ACC_DATA/images/0a302e52-image202.png', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/TEST_ACC_DATA', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('model_path', nargs='?', default='/home/ubuntu/ant_detection/rcnn_models/20220627-144837/best_weights.pth', help="Specify weights path", type=str)
    parser.add_argument('draw_targets', nargs='?', default=True, help="True - will draw targets, False - will not", type=bool)
    parser.add_argument('nms_threshold', nargs='?', default=0.1, help="Non maximum supression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.1, help="IOU threshold for boxes", type=float)
    
    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path
    draw_targets = args.draw_targets
    nms_threshold = args.nms_threshold
    iou_threshold = args.iou_threshold
    
    test_model = get_model(2, model_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if test_data_path[-3:] == 'png':
        #test_of_single_image(test_data_path, test_model, draw_targets, nms_threshold, iou_threshold)
        test_one_from4(test_data_path, test_model, draw_targets, nms_threshold, iou_threshold)
        
    else:
        dataset = ClassDataset(test_data_path, transform=None, demo=False)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        dir_size = int(len(glob.glob(test_data_path + '/images' + '/*')))
        print('ya tyt')
        #test_batch(test_model, data_loader, draw_targets, nms_threshold, iou_threshold, dir_size)
        test_batch_from4(test_model, data_loader, draw_targets, nms_threshold, iou_threshold, dir_size)
