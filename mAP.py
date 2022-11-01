import pandas as pd
import numpy as np


def intersection_over_union(boxA, boxB):
    #Считает IoU для двух боксов
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = round(iou, 2)
    return iou  

'''
GT_values - [N, 4]
pred_values - [M, 4] - прямоугольники сортируются в порядке убывания достоверности наличия в них объектов
'''
def IoU_matrix(GT_values, pred_values, iou_tresh):
    correct = []
    all_prec = []
    all_recall = []
    iou_matrix = np.zeros((GT_values.shape[0], pred_values.shape[0]))
    TP = 0 # количество перекрытий больше порога
    FP = 0 # количество перекрытий меньше порога
    FN = 0 # количество необраруженных объектов
    for i in range(GT_values.shape[0]):
        for j in range(pred_values.shape[0]):
            iou_matrix[i][j] = intersection_over_union(GT_values[i], pred_values[j])
            if iou_matrix[i][j] >= iou_tresh:
                correct.append(1)
            else:
                correct.append(0)
        
        precision = np.count_nonzero(correct) / (i + 1)
        recall = np.count_nonzero(correct) / GT_values.shape[0]
        all_prec.append(precision)
        all_recall.append(recall)
        #np.append(table['precision'], precision)
        #np.append(table['recall'], recall)
        
    area = np.trapz(all_prec, all_recall)
    return area
    
def get_real_bboxes(im_path):
    number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
    im_root = im_path[:im_path.rfind('/')]
    test_root = im_root[:im_root.rfind('/') + 1]
    b_path = test_root + 'bboxes/bbox' + number + '.txt'
    orig_bb = read_boxes(b_path)
    return orig_bb
    

def sort_bboxes(scores, bboxes):
    scores_bboxes = np.stack((pred_sc, pred_b), axis=-1)
    scores_bboxes.tolist()
    scores_bboxes.sort(reverse=True)
    new_bboxes = scores_bboxes[:, 1]
    return new_bboxes


if __name__ == '__main__':
    GT_values = np.array([[300, 320, 140, 344], [23, 56, 34, 78]])
    pred_values = np.array([[300, 318, 140, 344], [24, 58, 35, 79]])
    iou_tresh = 0.1
    IoU_matrix(GT_values, pred_values, iou_tresh)
    
    parser = argparse.ArgumentParser()
    file_ = 'cut50s'
    parser.add_argument('--video_path', nargs='?', default='', help="path to folder with images and annot", type=str)
    parser.add_argument('--iuo_tresh', nargs='?', default=0.5, help="treshold for TP and FP", type=float)
    args = parser.parse_args()
    real_bboxes_path = args.annot_path + '/bboxes'
    images_path = args.annot_path + '/images'
    dir_size = len(glob.glob(real_bboxes_path + '/*'))
    iuo_tresh = args.iuo_tresh
    average_precision = 0
    for f in os.scandir(images_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            print(f'Изображение №{counter + 1} из {dir_size}')
            _, pred_b, _, pred_sc = one_image_test(f.path, model, device, False, conf_threshold, nms_threshold, iou_threshold, delta_w, delta_h, False)
            annot_bboxes = get_real_bboxes(f.path)
            
            pred_sort_bboxes = sort_bboxes(pred_sc, pred_b)
            average_precision += IoU_matrix(annot_bboxes, pred_sort_bboxes, iuo_tresh)
            
    mAP = average_precision / dir_size
    print(f"Mean average precision {mAP}")
