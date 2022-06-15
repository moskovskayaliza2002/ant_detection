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
                image = cv2.putText(image.copy(),'B', (x_center_body + 2, y_center_body + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,205,50), 1)
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
                img = cv2.putText(img,'B', (x_center_body + 2, y_center_body + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,205,50), 1)
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
    #parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/ant_detection/test_data/images/image2.png', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/test_data', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('model_path', nargs='?', default='/home/ubuntu/ant_detection/rcnn_models/20220615-114114/best_weights.pth', help="Specify weights path", type=str)
    parser.add_argument('draw_targets', nargs='?', default=True, help="True - will draw targets, False - will not", type=bool)
    parser.add_argument('nms_threshold', nargs='?', default=0.5, help="Non maximum supression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.3, help="IOU threshold for boxes", type=float)
    
    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path
    draw_targets = args.draw_targets
    nms_threshold = args.nms_threshold
    iou_threshold = args.iou_threshold
    
    test_model = get_model(2, model_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if test_data_path[-3:] == 'png':
        test_of_single_image(test_data_path, test_model, draw_targets, nms_threshold, iou_threshold)
        
    else:
        dataset = ClassDataset(test_data_path, transform=None, demo=False)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        dir_size = int(len(glob.glob(test_data_path + '/*')))
        
        test_batch(test_model, data_loader, draw_targets, nms_threshold, iou_threshold, dir_size)
