import cv2
import yaml
from RCNN_overlay_test import read_yaml, visualize
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
    frames = [0] * int(maxim_frames)
   
    image_to_show = None
    pos_to_show = None
    pos_frame = 0
    click_back = 0
    click_forward = 0
    while True:
        cv2.destroyAllWindows()
        if pos_to_show is None:
        #Чтобы считать самый первый кадр
            flag, frame = cap.read()
            if flag:
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                pred_bboxes = data['frames'][int(pos_frame)-1][pos_frame]['bboxes']
                pred_keypoints = data['frames'][int(pos_frame)-1][pos_frame]['keypoints']
                pred_scores = data['frames'][int(pos_frame)-1][pos_frame]['bboxes_scores']
                pred_image = visualize(frame, pred_bboxes, pred_keypoints, pred_scores, show_flag = False)
                win_name = "frame " + str(int(pos_frame)) + " from " + str(int(maxim_frames))
                #cv2.resizeWindow(win_name, 1920, 1080)
                pred_image = cv2.resize(pred_image, (1920, 1000))
                frames[int(pos_frame)-1] = pred_image
                cv2.imshow(win_name, pred_image)
        else:
            if type(frames[pos_to_show]) != int:
            # Тут показываются кадры, что уже были показаны до этого
                win_name = "frame " + str(int(pos_to_show + 1)) + " from " + str(int(maxim_frames))
                cv2.imshow(win_name, image_to_show)
            else:
            # Для показа следующего кадра
                flag, frame = cap.read()
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                pred_bboxes = data['frames'][int(pos_frame)-1][pos_frame]['bboxes']
                pred_keypoints = data['frames'][int(pos_frame)-1][pos_frame]['keypoints']
                pred_scores = data['frames'][int(pos_frame)-1][pos_frame]['bboxes_scores']
                pred_image = visualize(frame, pred_bboxes, pred_keypoints, pred_scores, show_flag = False)
                win_name = "frame " + str(int(pos_frame)) + " from " + str(int(maxim_frames))
                pred_image = cv2.resize(pred_image, (1920, 1000))
                click_back = 0
                click_forward = 0
                frames[int(pos_frame)-1] = pred_image
                cv2.imshow(win_name, pred_image)
                
        next_step = cv2.waitKey(0)    
        
        if next_step == 83: # ->
            click_forward += 1 
            
        if next_step == 81: # <-
            click_back += 1
            if int(pos_frame) - click_back + click_forward < 0:
                click_back -= 1
                pos_to_show = 0
                print("нет предыдущих изображений")
                continue
         
        if next_step == 27: # esc
            break
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #Если прочитали все кадры - выходим из цикла
            break
        
        # Обновляется позиция следующего показываемого кадра
        pos_to_show = int(pos_frame) - 1 - click_back + click_forward
        image_to_show = frames[pos_to_show]
    
    cap.release()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', nargs='?', default='/home/ubuntu/ant_detection/videos/inputs/short.mp4', help="Specify the full path to video", type=str)
    parser.add_argument('yaml_path', nargs='?', default='/home/ubuntu/ant_detection/videos/inputs/predicted.yml', help="Specify the full path to video", type=str)
    
    args = parser.parse_args()
    video_path = args.video_path
    yaml_path = args.yaml_path
    
    vis_frame_by_frame(video_path, yaml_path)
