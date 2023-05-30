import numpy as np
import argparse

def read_tracks_from_txt(path):
    tracks = []
    last_no = 0
    with open(path) as f:
        for i in f:
            dic = {}
            a = list(map(float, i[:-2].split(' ')))
            no = a[0]
            frame_ind = a[1]
            a = np.array(a[2:]).reshape((-1, 5)).tolist()
            dic[frame_ind] = a
            tracks.append(dic)
    return tracks
 
 
def mean_speed_for_ant(traj, dt):
    first_p = traj[0]
    traj = traj[1:]
    all_speed = []
    len_of_traj = 0
    for point in traj:
        l = math.dist(first_p, point)
        len_of_traj += len_of_traj
        speed = l/dt
        all_speed.append(speed)
        first_p = point
    #print(f"Скорость по особям: {sum(all_speed)/len(all_speed)}")
    return all_speed, sum(all_speed)/len(all_speed)


def lenth_of_traj(traj):
    first_p = traj[0]
    traj = traj[1:]
    len_of_traj = 0
    for point in traj:
        l = math.dist(first_p, point)
        len_of_traj += len_of_traj
    return len_of_traj

    
def count_mean_speed(all_tracks, dt = 1/29.9):
    #Средняя скорость треков с остановок
    d_mean_ants = []
    d_mean_steps = []
    #Средняя скорость треков без остановок
    d_ns_mean_ants = []
    d_ns_mean_steps = []
    mean_speed = []
    #Длины каждой траектории
    lenthes = []
    #Время остановок
    time_of_stops = []
    #Время движения
    time_of_movement = []
    v_min = 0.005
    mean_speed_by_dist = []
    for ant in all_tracks:
        frame = list(ant.keys())[0]
        track = list(ant.values())[0]
        curr_pts = []
        mean_ants = []
        speed = 0
        lenth = len(track)
        for num, tr in enumerate(track):
            if tr[3] > v_min:
                speed += tr[3]
                curr_pts.append([tr[0], tr[1]])
            mean_ants.append([tr[0], tr[1]])#для подсчета распределения скоростей с остановками
        ant_speeds, mean_ant_speed = mean_speed_for_ant(mean_ants, dt)
        lenthes.append(lenth_of_traj(mean_ants))
        ns_ant_speeds, ns_mean_ant_speed = mean_speed_for_ant(curr_pts, dt)
            
        d_ns_mean_ants.append(ns_mean_ant_speed)
        d_mean_ants.append(mean_ant_speed)
            
        d_mean_steps += ant_speeds
        d_ns_mean_steps += ns_ant_speeds
        if speed != 0:
            mean_speed.append(speed/lenth)
        
    print(f"Средняя скорость по Калману: {sum(mean_speed)/len(mean_speed)} м/c")
    print(f"Средняя скорость по дистанции: {sum(d_ns_mean_ants)/len(d_ns_mean_ants)} м/c")


if __name__ == '__main__':
    # Нужно реализровать вывод таблицы: для каждого трека: длина пути, время остановок, время пути, средняя скорость без остановок, средняя скорость с остановками.
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks_path', nargs='?', default="/home/ubuntu/ant_detection/problems/another_full_video/empty_center_tracks.txt", help="Specify yaml track path", type=str)
    args = parser.parse_args()
    
    all_tracks = read_tracks_from_txt(args.tracks_path)
    count_mean_speed(all_tracks)
