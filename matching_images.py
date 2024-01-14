import cv2
import argparse
import glob

def find_frame(video_path, images_path, labels_path, new_images_path, new_labels_path):
    cap = cv2.VideoCapture(video_path)
    dir_size = len(glob.glob(dirr_path + '/*')) - 1
    for f in os.scandir(dirr_path):
        number = f.path[f.path.rfind("/") + 1 : f.path.rfind(".")]
        image = cv2.imread(f.path)
        frame = comparation(cap, image)
        shutil.copyfile(f.path, new_images_path + "/" + str(frame) + ".png")
        shutil.copyfile(labels_path + "/" + number + ".txt", new_labels_path + "/" + str(frame) + ".txt")
        
        
def comparation(cap, image):
    max_metric = -1
    frame_num = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # Calculate the histograms, and normalize them
            hist_image = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_frame = cv2.calcHist([frame], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            # Find the metric value
            metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
            if metric_val > max_metric:
                max_metric = metric_val
                frame_num = int(pos_frame)
    return frame_num
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', nargs='?', type=str, default='', help='path to source video')
    parser.add_argument('--data', nargs='?', type=str, default='', help='path to directory with images and labels')
    parser.add_argument('--new_path', nargs='?', type=str, default='', help='path to new directory with data')
    args = parser.parse_args()
    video = args.video
    data = args.data
    new_path = args.new_path
    
    for f in [new_path + '/labels', new_path + '/images', new_path + '/labels/train', new_path + '/labels/val', new_path + '/images/train', new_path + '/images/val']):
        if not os.path.exists(f):
            os.mkdir(f)
        else:
            shutil.rmtree(f)
            os.mkdir(f)
    
    find_frame(video, data + '/images/train', data + '/labels/train', new_path + '/images/train', new_path + '/labels/train')
    find_frame(video, data + '/images/val', data + '/labels/val', new_path + '/images/val', new_path + '/labels/val')
    
