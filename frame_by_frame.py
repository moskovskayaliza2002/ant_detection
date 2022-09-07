import cv2
import yaml
from RCNN_overlay_test import read_yaml, visualize, one_image_test
import argparse

'''
    Функция visualize:
        Вход: image - cv2 изображение
              bboxes - список боксов [N, 4] где N - количество муравьев на изображении
              keypoints - список ключевых точек [N, 2, 2] где N - количество муравьев на изображении (пример с одним муравьем: [[x_a, y_a], [x_h, y_h]])
              show_flag - Обязательно передавать False. (при False функция вернет изображение, при True просто покажет)

'''

def vis_frame_by_frame(video_path, yaml_path):
    data = read_yaml(yaml_path)
    cap = cv2.VideoCapture(video_path)
    
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print("Openning the file...")
    
    maxim_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cur_pos_frame = None

    print("*******************************FILE INFO*******************************")
    print('Name:', data['name'], '\nFPS:', data['FPS'], '\nWeight:', data['weight'], '\nHeight:', data['height'])
    #print(f'Name: {data['name']}')
    while True:
        cv2.destroyAllWindows()
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            pred_bboxes = data['frames'][int(pos_frame-1)]['bboxes']
            pred_keypoints = data['frames'][int(pos_frame-1)]['keypoints']
            pred_scores = data['frames'][int(pos_frame-1)]['bboxes_scores']
            pred_image = visualize(frame, pred_bboxes, pred_keypoints, pred_scores, show_flag = False)
            win_name = "frame " + str(int(pos_frame)) + " from " + str(int(maxim_frames))
            #cv2.resizeWindow(win_name, 1920, 1080)
            pred_image = cv2.resize(pred_image, (1920, 1000))
            cur_pos_frame = int(pos_frame)
            cv2.imshow(win_name, pred_image)
        
        next_step = cv2.waitKey(0)
        
        if next_step == 83: # ->
            continue 
        
        if next_step == 81: # <-
            if cur_pos_frame - 2 < 0:
                print("нет предыдущих изображений, гружу следующее")
                continue
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos_frame-2)
        
        if next_step == 27: # esc
            break
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #Если прочитали все кадры - выходим из цикла
            break
        
    cap.release()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', nargs='?', default='/home/ubuntu/ant_detection/videos/inputs/cut50s.mp4', help="Specify the full path to video", type=str)
    parser.add_argument('yaml_path', nargs='?', default='/home/ubuntu/ant_detection/videos/inputs/cut50s.yml', help="Specify the full path to video", type=str)
    
    args = parser.parse_args()
    video_path = args.video_path
    yaml_path = args.yaml_path
    
    vis_frame_by_frame(video_path, yaml_path)
