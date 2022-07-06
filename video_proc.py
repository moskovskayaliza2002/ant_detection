import cv2
from RCNN_overlay_test import one_image_test
from RCNN_model import get_model
import argparse
import torch

def read_video(filename):
    cap = cv2.VideoCapture(filename)
    list_of_images = []
    while not cap.isOpened():
        cap = cv2.VideoCapture(filename)
        cv2.waitKey(1000)
        print("Openning the file...")
        
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            #cv2.imshow('video', frame)
            list_of_images.append(frame)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(pos_frame, " frames")
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
    cap.release()
    cv2.destroyAllWindows()
    
    return list_of_images
        

def predict(filename, model_path, draw_targets, nms_threshold, iou_threshold, delta_w, delta_h):
    images = read_video(filename)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("не могу загрузить модель")
    model = get_model(2, model_path)
    print("могу")
    future_video = []
    size = 0
    for image in images:
        print(f'type {type(image)}')
        height, width, layers = image.shape
        size = (width,height)
        print("умираю в предсказании")
        pred_im = one_image_test(image, model, device, False, nms_threshold, iou_threshold, delta_w, delta_h, False)
        print("умираю не в предсказании")
        future_video.append(pred_im)
     
    new_filename = filename[:filename.rfind('/')] + '/predicted.avi'
    out = cv2.VideoWriter(new_filename,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(future_video)):
        out.write(future_video[i])
    out.release()
    
    return new_filename
  
def show_video(name):
    cap = cv2.VideoCapture(name)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
  

def full_video(filename, model_path, nms_threshold, iou_threshold, delta_w, delta_h):
    cap = cv2.VideoCapture(filename)
    list_of_images = []
    model = get_model(2, model_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    new_filename = filename[:filename.rfind('/')] + '/predicted.mp4'
    #out = cv2.VideoWriter(new_filename,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(w), int(h))
    
    out = cv2.VideoWriter(new_filename, fourcc, fps, size, True)
    
    while not cap.isOpened():
        cap = cv2.VideoCapture(filename)
        cv2.waitKey(1000)
        print("Openning the file...")
        
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    maxim_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            #cv2.imshow('video', frame)
            
            pred_im = one_image_test(frame, model, device, False, nms_threshold, iou_threshold, delta_w, delta_h, False)
            list_of_images.append(pred_im)
            out.write(pred_im)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(f'{pos_frame} frame from {maxim_frames}')
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    show_video(new_filename)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', default='/home/ubuntu/ant_detection/videos/inputs/short.mp4', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('model_path', nargs='?', default='/home/ubuntu/ant_detection/rcnn_models/20220628-124306/best_weights.pth', help="Specify weights path", type=str)
    parser.add_argument('draw_targets', nargs='?', default=True, help="True - will draw targets, False - will not", type=bool)
    parser.add_argument('nms_threshold', nargs='?', default=0.3, help="Non maximum supression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.4, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    
    args = parser.parse_args()
    filename = args.filename
    model_path = args.model_path
    draw_targets = args.draw_targets
    nms_threshold = args.nms_threshold
    iou_threshold = args.iou_threshold
    delta_w = args.overlay_w
    delta_h = args.overlay_h
    
    full_video(filename, model_path, nms_threshold, iou_threshold, delta_w, delta_h)
    
    #file_v = predict(filename, model_path, draw_targets, nms_threshold, iou_threshold, delta_w, delta_h)
    #show_video(file_v)
    
    
    
