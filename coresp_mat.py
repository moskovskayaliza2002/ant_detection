import numpy as np
import argparse
from RCNN_overlay_test import read_yaml
from ekf_traker_test import get_ants
from ekf import multi_mahalanobis
'''
# Z_prev = [[x, y, a, v, w]]
def get_predict(Z_prev, dt):
    Z_next = np.array([])
    for i, ant in enumerate(Z_prev):
        F = np.array([[1, 0, -np.sin(ant[2]) * ant[3] * dt, np.cos(ant[2]) * dt, 0],
                      [0, 1, np.cos(ant[2]) * ant[3] * dt, np.sin(ant[2]) * dt, 0],
                      [0, 0, 1, 0, dt],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        np.append(Z_next, np.matmul(F, ant))
    return Z_next

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    file_ = 'cut50s'
    parser.add_argument('--yaml_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/inputs/{file_}.yml', help="Full path to yaml-file with ant data", type=str)
    parser.add_argument('--video_path', nargs='?', default=f'/home/ubuntu/ant_detection/videos/inputs/{file_}.mp4', help="Full path to video file", type=str)
    args = parser.parse_args()
    ANT_DATA = read_yaml(args.yaml_path)
    dt = 1/ANT_DATA['FPS']
    error = np.array([])
    #predicted_ants = np.array([])
    prev_pred = None
    for frame in ANT_DATA['frames']:
        real_ants = get_ants(frame, dt)
        predicted_ants = get_predict(real_ants, dt))
        if prev_pred:
            np.append(real_ants - prev_pred)
        prev_pred = predicted_ants
            
    cap = cv2.VideoCapture(args.video_path)
'''
ARROW_LEN = 50
D_ANT_COLOR = 'w'
D_ANT_SYM = 'o'
ANT_SCORE_MIN = 0.8
MEKF = None
R_diag = np.array([1.69, 3.76, 1.86])
l = 0.00001

Q = np.array([[2.63795021e+00, 4.31565031e-02, 4.34318028e-03, -2.20547258e+00, 2.13165056e-01],
              [4.31565031e-02, 2.17205029e+00, -2.80823458e-03, -8.82016377e+00, -7.06690445e-02],
              [4.34318028e-03, -2.80823458e-03, 2.03181860e-02, 4.16669103e-02, 4.60097662e-01],
              [-2.20547258e+00, -8.82016377e+00, 4.16669103e-02, 2.16764535e+03, -1.92685776e-01],
              [2.13165056e-01, -7.06690445e-02, 4.60097662e-01, -1.92685776e-01, 1.41432123e+01]])
#Q_diag = np.array([l, l, l, l, l])
dt = 0.1
mh = 10
P_limit = np.inf

def proceed_frame(frame, W, H, ax, dt):
    global MEKF
    ants = get_ants(frame, dt)    
    
    MEKF = multiEKF(ants, R_diag,  Q, dt, mh, P_limit, W, H)
    for ekf in MEKF.EKFS:
        ekf.predict()
    old_values = self.get_all_ants_data_as_array()
