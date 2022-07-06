import argparse
from torch.utils.data import Dataset, DataLoader
from RCNN_model import ClassDataset, get_model
import cv2
import torch
import glob
import os
from utils import collate_fn
from torchvision.transforms import functional as F
import numpy as np
import torchvision
from crop_into4 import crop_one_im, read_boxes
import matplotlib.pyplot as plt


def ver(bboxes, kp, l1, l2, r1, r2, crop_w, crop_h): 
    # Проверяет, что ни один бокс не пересекается линией разреза
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


def get_out_kp_bb(out, left_x, left_y, keypoints, bboxes, nms_threshold, iou_threshold):
    # Функция для маштабирования предказанных координат на новый диапазон
    scores = out[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > nms_threshold)[0].tolist() # Indexes of boxes with scores > 0.7
    post_nms_idxs = torchvision.ops.nms(out[0]['boxes'][high_scores_idxs], out[0]['scores'][high_scores_idxs], iou_threshold).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

    for kps in out[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        int_kps = [list(map(int, kp[:2])) for kp in kps]
        x_a, y_a = int_kps[0][0], int_kps[0][1]
        x_h, y_h = int_kps[1][0], int_kps[1][1]
        keypoints.append([[x_a + left_x, y_a + left_y], [x_h + left_x, y_h + left_y]])

    for bbox in out[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        xmin, ymin, xmax, ymax = map(int, bbox.tolist())
        bboxes.append([xmin + left_x, ymin + left_y, xmax + left_x, ymax + left_y])
    
    return keypoints, bboxes
    

def test_one_from4(im_path, model, flag, nms_threshold, iou_threshold):
    # Функция показывающая предсказывания модели на отдельном изображении (Для модели, что делит на 4 исходное изображение)
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    input_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_im = F.to_tensor(input_im)
    orig_bb = []
    orig_kp = []
    # Получаем настоящие координаты для целого изображения
    number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
    im_root = im_path[:im_path.rfind('/')]
    test_root = im_root[:im_root.rfind('/') + 1]
    b_path = test_root + 'bboxes/bbox' + number + '.txt'
    orig_bb = read_boxes(b_path)
    k_path = test_root + 'keypoints/keypoint' + number + '.txt'
    kp = read_boxes(k_path)
    for i in kp:
        orig_kp.append([[i[0], i[1]], [i[2], i[3]]])
    # Разрезаем изображение на 4    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    left_1, left_2, right_1, right_2, c_w, c_h = crop_one_im(img)
    # Проверяем что не пересекаем боксы линиями разреза и получаем предсказания для каждого нового изображения и объединяем их
    if ver(orig_bb, kp, left_1, left_2, right_1, right_2, c_w, c_h):
        with torch.no_grad():
            model.to(device)
            model.eval()
            out_l1 = model([F.to_tensor(cv2.cvtColor(left_1, cv2.COLOR_BGR2RGB))])
            out_l2 = model([F.to_tensor(cv2.cvtColor(left_2, cv2.COLOR_BGR2RGB))])
            out_r1 = model([F.to_tensor(cv2.cvtColor(right_1, cv2.COLOR_BGR2RGB))])
            out_r2 = model([F.to_tensor(cv2.cvtColor(right_2, cv2.COLOR_BGR2RGB))])
    
        kp_l1, bb_l1 = get_out_kp_bb(out_l1, 0, 0, [], [], nms_threshold, iou_threshold)
        kp_l2, bb_l2 = get_out_kp_bb(out_l2, 0, c_h, kp_l1, bb_l1, nms_threshold, iou_threshold)
        kp_r1, bb_r1 = get_out_kp_bb(out_r1, c_w, 0, kp_l2, bb_l2, nms_threshold, iou_threshold)
        kp_r2, bb_r2 = get_out_kp_bb(out_r2, c_w, c_h, kp_r1, bb_r1, nms_threshold, iou_threshold)
         
        # Визуализация и предсказаний, и таргетов 
        if flag:
            visualize(img, bb_r2, kp_r2, img, orig_bb, orig_kp)
        # Визуализация только таргетов
        else:
            visualize(img, bb_r2, kp_r2)
    else:
        print("Изображение не подходит")
    

def find_bbox(num, root_path):
    # Находит файл с боксами, относящийся к изображению и возвращает считанные значения
    bb = []
    bbox_path = root_path + '/bboxes'
    for f in os.scandir(bbox_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'txt':
            if int(f.path[f.path[:-3].rfind('x') + 1 : f.path[:-3].rfind('.')]) == num:
                bb = read_boxes(f.path)
    
    return bb
    
def find_kp(num, root_path):
    # Находит файл с ключевыми точками, относящийся к изображению и возвращает считанные значения
    kp = []
    orig_kp = []
    kp_path = root_path + '/keypoints'
    for f in os.scandir(kp_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'txt':
            if int(f.path[f.path[:-3].rfind('t') + 1 : f.path[:-3].rfind('.')]) == num:
                kp = read_boxes(f.path)
    for i in kp:
        orig_kp.append([[i[0], i[1]], [i[2], i[3]]])
    
    return orig_kp, kp

    
def test_batch_from4(model, root, flag, nms_threshold, iou_threshold):
    # Функция показывающая предсказывания модели на батче изображений (Для модели, что делит на 4 исходное изображение)
    image_data_path = root + '/images'
    
    for f in os.scandir(image_data_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            original_image = cv2.imread(f.path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            number = int(f.path[f.path.rfind('e') + 1 : f.path.rfind('.')])
            print(f"number {number}")
            orig_bbox = find_bbox(number, root)
            orig_kp, kp = find_kp(number, root)
            # Разрезаем изображение на 4    
            left_1, left_2, right_1, right_2, c_w, c_h = crop_one_im(original_image)
            # Проверяем, что не пересекаем боксы линиями разреза и получаем предсказания для каждого нового изображения и объединяем их
            if ver(orig_bbox, kp, left_1, left_2, right_1, right_2, c_w, c_h):
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
                
                # Визуализация и предсказаний, и таргетов
                if flag:
                    visualize(original_image, bb_r2, kp_r2, original_image, orig_bbox, orig_kp)
                # Визуализация только таргетов
                else:
                    visualize(original_image, bb_r2, kp_r2)
            else:
                print(f"Изображение {f.path} не подходит")
 

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, show_flag = True):
    # Рисует на изображении предсказанные и настоящие боксы и ключевые точки.
    fontsize = 12
    keypoints_classes_ids2names = {0: 'A', 1: 'H'}
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (255,0,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 2, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    if image_original is None and keypoints_original is None:
        if show_flag:
            plt.figure(figsize=(40,40))
            plt.imshow(image)
            plt.show(block=True)
        else:
            return image

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 2, (0,255,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                
        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original annotations', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Predicted annotations', fontsize=fontsize)
        plt.show(block=True)

 
def visualizate_predictions(model, data_loader_test, nms_threshold, iou_threshold):
    # Визуализирует предсказания для батча. Для модели, что учится на целиковых изображениях.
    iterator = iter(data_loader_test)
    images, targets = next(iterator)
    images = list(image.to(device) for image in images)

    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(images)
        
    image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
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
    # Тестирование на одной изображении. Для модели, что учится на целиковых изображениях.
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    input_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_im = F.to_tensor(input_im)
    print(input_im.size())
    bb_orig = []
    kp_orig = []
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
                bb_orig.append([x_min, y_min, x_max, y_max])
        
        with open(k_path) as f:
            for i in f:
                x_center_body, y_center_body, x_center_head, y_center_head = map(int, i[:-1].split(' '))
                kp_orig.append([[x_center_body, y_center_body], [x_center_head, y_center_head]])
    
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
    
    if flag:
        visualize(img, bboxes, keypoints, img, bb_orig, kp_orig)
    else:
        visualize(img, bboxes, keypoints)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/TEST_ACC_DATA/images/0a302e52-image202.png', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/TEST_ACC_DATA', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('model_path', nargs='?', default='/home/ubuntu/ant_detection/rcnn_models/20220628-124306/best_weights.pth', help="Specify weights path", type=str)
    parser.add_argument('draw_targets', nargs='?', default=True, help="True - will draw targets, False - will not", type=bool)
    parser.add_argument('nms_threshold', nargs='?', default=0.5, help="Non maximum supression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.2, help="IOU threshold for boxes", type=float)
    
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
        #visualizate_predictions(test_model, data_loader, nms_threshold, iou_threshold)
        test_batch_from4(test_model, test_data_path, draw_targets, nms_threshold, iou_threshold)
