import pandas as pd
import numpy as np
import argparse
import glob
import os
from mAP import count_mAP


def scan(path):
    folders = []
    for directory in os.scandir(path):
        if not directory.is_file():
            folders.append(directory.name)
            
    print(folders)
    return folders
        
def evaluate(dir_path, datasets_path, conf_threshold, nms_threshold, iou_threshold, iuo_tresh, overlay_w, overlay_h, splits_vertical, splits_horizontal):
    folders = scan(dir_path) 
    df_model = []
    df_dataset = []
    df_mAP = []
    for mod in folders:
        model_path = dir_path + "/" + mod + '/full_weights.pth'
        for data in datasets_path:
            dataset_path = data + '/images'
            _, _, _, mAP = count_mAP(conf_threshold, nms_threshold, iou_threshold, dataset_path, iuo_tresh, model_path, overlay_w, overlay_h, splits_vertical, splits_horizontal)
            #model_name = mod[mod.rfind('/') + 1:]
            dataset_name = data[data.rfind('/') + 1:]
            
            df_dataset.append(dataset_name)
            df_mAP.append(mAP)
            
            if len(df_mAP) % len(datasets_path) == 1:
                df_model.append(mod)
            #if len(df_model) > 0 and mod == df_model[-1]:
            #    df_model.append("")
            else:
                df_model.append("")
           
    df = pd.DataFrame({'model': df_model,'dataset': df_dataset, 'mAP': df_mAP})
    df.to_csv(dir_path + '/out.csv', index=False)
    print(df)
    return df

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path', nargs='?', default="/home/ubuntu/ant_detection/new_dataset/rcnn_models", help="Specify the path either to the folder with models weights", type=str)
    parser.add_argument('--iuo_tresh', nargs='?', default=0.5, help="treshold for TP and FP", type=float)
    parser.add_argument('conf_threshold', nargs='?', default=0.8, help="Confident threshold for boxes", type=float)
    parser.add_argument('nms_threshold', nargs='?', default=0.3, help="Non maximum suppression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.3, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    parser.add_argument('splits_vertical', nargs='?', default=4, help="Num of pictures in w-axis", type=int)
    parser.add_argument('splits_horizontal', nargs='?', default=3, help="Num of pictures in h-axis", type=int)
    args = parser.parse_args()
    datasets_path = ['/home/ubuntu/ant_detection/dataset/Test_data', '/home/ubuntu/ant_detection/test_on_article_data/indoor', '/home/ubuntu/ant_detection/test_on_article_data/outdoor']
    #datasets_path = ['/home/ubuntu/ant_detection/delete/data1','/home/ubuntu/ant_detection/delete/data2','/home/ubuntu/ant_detection/delete/data3']
    evaluate(args.models_path, datasets_path, args.conf_threshold, args.nms_threshold, args.iou_threshold, args.iuo_tresh, args.overlay_w, args.overlay_h, args.splits_vertical, args.splits_horizontal)
