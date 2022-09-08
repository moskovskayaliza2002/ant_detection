import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.stats import plot_covariance_ellipse
from matplotlib.pyplot import cm

MAX_AM_ANTS = 15
TRESH_STEPS = 25
ARROW_LEN = 50

def HJacobian(x):
    H = [[1, 0, 0 ,0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0]]
    return np.array(H)

def Hx(x):
    # we measure exactly the state values
    return x[:3]

def substract_angles(target, source):
    return np.arctan2(np.sin(target-source), np.cos(target-source))

def residual(a, b):
    y = a - b
    y[2] = substract_angles(a[2], b[2])
    return y

class AntEKF(ExtendedKalmanFilter):
    # dim_x: x, y, a, v, w    
    X_SIZE = 5
    # dim_z: x, y, a
    Z_SIZE = 3
    
    '''
    start x - [x, y, a, v, w]
    p - probabil of detection
    R_diag - [rx, ry, ra]
    Q_diag - [qx, qy, qa, qv, qw]
    dt - 1/fps
    '''
    def __init__(self, start_x, p, R_diag, Q_diag, color, dt):
        super(AntEKF, self).__init__(AntEKF.X_SIZE, AntEKF.Z_SIZE)
                
        self.dt = dt 
        self.x = start_x
        self.P = np.eye(AntEKF.X_SIZE) #* (1 - p) # maybe not good idea
        self.R = np.diag(R_diag)
        self.Q = np.diag(Q_diag)
        self.color = color
        
        self.no_update_steps = 0
        
        self.track = [np.copy(self.x)]
                
        
    def predict(self, u = 0):
        # just write our movement equations
        
        self.x[0] = self.x[0] + self.x[3] * np.cos(self.x[2]) * self.dt
        self.x[1] = self.x[1] + self.x[3] * np.sin(self.x[2]) * self.dt
        self.x[2] = self.x[2] + self.x[4] * self.dt
        self.x[3] = self.x[3] # no change
        self.x[4] = self.x[4] # no change
        
        self.F = np.array([[1, 0, -np.sin(self.x[2]) * self.x[3] * self.dt, np.cos(self.x[2]) * self.dt, 0],
                      [0, 1, np.cos(self.x[2]) * self.x[3] * self.dt, np.sin(self.x[2]) * self.dt, 0],
                      [0, 0, 1, 0, self.dt],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        
        self.P = self.F @ self.P @ self.F.T + self.Q
                        
        self.no_update_steps+=1
        # TODO: add x to history
        self.track.append(np.copy(self.x))
        
    '''
    new_value - [x, y, a]
    '''
    def update2(self, new_value):
        #z = np.array([new_value])
        z = np.array(new_value)        
        self.update(z, HJacobian = HJacobian, Hx = Hx, residual = residual)                        
        self.no_update_steps = 0
        # TODO: rewrite last history element
        self.track[-1] = np.copy(self.x)

        
'''
Calculates pairvise mahalanobis distances between new values x, old values y with correspondence to old values covariations
    x - new values, array of [K, N]
    y - old values, array of [M, N]
    Sm - inverted cov matrixes for y, array of [M, N, N]
returns
    array [K, M]
'''
def multi_mahalanobis(x, y, Sm):
    
    xx = np.tile(x, (y.shape[0],1))
    yy = np.repeat(y, x.shape[0], axis = 0)                        
    SSm = np.repeat(Sm, x.shape[0], axis = 0)        
            
    d = xx - yy        
    de = np.expand_dims(d, 1)
    dee = np.expand_dims(d, 2)                
    
    D = np.matmul(de, SSm)              
    D = np.sqrt( np.matmul(D, dee) )        
    D = D.reshape( (x.shape[0], y.shape[0]), order = 'F' )
    
    return D


class multiEKF(object):
        
    '''
    start_values [[p, x, y, a, v, w]] - initial positions of all ants
    R_diag - [sigma^2 x, sigma^2 y, sigma^2 a] - measurment errors
    Q_diag - [nx, ny, na, nv, nw] - noises for predicion
    dt [seconds] - rate of filter, typically 1/fps
    mahalanobis_thres - mahalanobis disnace at which count ants the same
    P_limit - limitation for covariance, if it is higher - remove that filter
    '''
    def __init__(self, start_values, R_diag, Q_diag, dt, mahalanobis_thres, P_limit, xlim, ylim):
        
        self.mahalanobis_thres = mahalanobis_thres
        
        self.R_diag = R_diag
        self.Q_diag = Q_diag
        #self.color = color
        self.dt = dt
        self.P_limit = P_limit
        self.xlim = xlim
        self.ylim = ylim
        
        self.EKFS = []
        self.color = iter(cm.rainbow(np.linspace(0, 1, MAX_AM_ANTS)))
        for i in range(start_values.shape[0]):
            c = next(self.color)
            ekf = AntEKF(start_values[i][1:], start_values[i][0], self.R_diag, self.Q_diag, c, self.dt)                                    
            self.EKFS.append(ekf)                        

    '''
    new values - [[p, x, y, a, v, w]]
    '''
    def proceed(self, new_values):
        # 1. predict previous
        for ekf in self.EKFS:
            ekf.predict()
        
        # 2. calc correspondence
        old_values = self.get_all_ants_data_as_array()
        
        # with angles
        #inv_covs = np.array([np.linalg.inv(ekf.P[:3,:3]) for ekf in self.EKFS])
        #correspondence_matrix = multi_mahalanobis(new_values[:,1:4], old_values[:,:3], inv_covs)
        
        # without angles
        inv_covs = np.array([np.linalg.inv(ekf.P[:2,:2]) for ekf in self.EKFS])
        correspondence_matrix = multi_mahalanobis(new_values[:,1:3], old_values[:,:2], inv_covs)
        
        # store indexes of all ants, and then delete those which is taken for update, the rest will be new ants
        new_objects = list(range(new_values.shape[0]))
        
        # 3. update where correspondence is
        while True:
            # find minimal value from matrix
            minimal_distance = np.unravel_index(np.argmin(correspondence_matrix, axis=None), correspondence_matrix.shape)
            
            if correspondence_matrix[minimal_distance] > self.mahalanobis_thres:
                break # no more ants satisfy threshold
            
            ekf_ind = minimal_distance[1]
            val_ind = minimal_distance[0]
            
            # update filter
            self.EKFS[ekf_ind].update2(new_values[val_ind, 1:4])
            
            # 'remove' values from matrix
            correspondence_matrix[val_ind,:] = np.inf
            correspondence_matrix[:,ekf_ind] = np.inf
            # and from 'new_objects'
            new_objects.remove(val_ind)
        
        # 4. add new filters for new objects
        for ind in new_objects:            
            #new_x = np.zeros(5)
            #new_x[:3] = new_values[ind][1:]
            ekf = AntEKF(new_values[ind, 1:], new_values[ind][0], self.R_diag, self.Q_diag, next(self.color), self.dt)                                    
            self.EKFS.append(ekf)
                
        # 5. forget bad filters (long no update, huge covs, etc.) 
        
        ## huge cov
        old_colors = []
        if self.P_limit != np.inf:
            filters_to_remove = []
            for i, ekf in enumerate(self.EKFS):
                if np.any(ekf.P[:2,:2] > self.P_limit):
                    filters_to_remove.append(i)
                    
            for index in sorted(filters_to_remove, reverse=True):
                color_to_use_AGAIN = self.EKFS[index].color
                old_colors.append(color_to_use_AGAIN)
                del self.EKFS[index]
                
                
        ## too long not update
        obj_to_remove = []
        for i, ekf in enumerate(self.EKFS):
            #print(f'Муравей {i} цвета {ekf.color} не обновлялся шагов: {ekf.no_update_steps}')
            if ekf.no_update_steps >= TRESH_STEPS:
                obj_to_remove.append(i)
        for index in sorted(obj_to_remove, reverse=False):
            color_to_use_AGAIN = self.EKFS[index].color
            old_colors.append(color_to_use_AGAIN)
            del self.EKFS[index]
        
        for x in self.color:
            old_colors.append(x)
        self.color = iter(old_colors)
        old_colors = []         
    
    
    def get_all_ants_data_as_array(self):
        ants = []
        for ekf in self.EKFS:
            ants.append(ekf.x)
        return np.array(ants)
    
    
    def draw_tracks(self, H, ax, color = None):
        #color = iter(cm.rainbow(np.linspace(0, 1, len(self.EKFS))))
        for ekf in self.EKFS:
            # plot track
            track = np.array(ekf.track)
            c = ekf.color 
            x = ekf.x[0]
            y = ekf.x[1]
            a = ekf.x[2]
            delta_a = ekf.R[2][2]
            
            #angle errors as arrows
            #ax.arrow(x, y, ARROW_LEN * np.cos(a + delta_a), ARROW_LEN * np.sin(a + delta_a), color = c, ls = '--')
            #ax.arrow(x, y, ARROW_LEN * np.cos(a - delta_a), ARROW_LEN * np.sin(a - delta_a), color = c, ls = '--')
            
            ax.plot(track[:,0], track[:,1], color = c)
            # plot ellipse
            plot_covariance_ellipse((ekf.x[0], ekf.x[1]), ekf.P[0:2, 0:2], std=self.mahalanobis_thres, facecolor=c, alpha=0.2, xlim=(0,self.xlim), ylim=(self.ylim,0), ls=None, edgecolor=c)
            #plot triangles
            point_A = (x, y)
            point_B = (x + ARROW_LEN * np.cos(a + delta_a), y + ARROW_LEN * np.sin(a + delta_a))
            point_C = (x + ARROW_LEN * np.cos(a - delta_a), y + ARROW_LEN * np.sin(a - delta_a))
            p = plt.Polygon((point_A, point_B, point_C), fill=True,closed=True, facecolor=c, alpha=0.2, edgecolor=c)
            ax.add_patch(p)
            
    # plot speed
    def draw_speed(self, ax, dt = 0.2, color = 'w', N = 3):
        for ekf in self.EKFS:
            x = [ekf.x[0]]
            y = [ekf.x[1]]
            a = [ekf.x[2]]
            v = ekf.x[3]
            w = ekf.x[4]
            for i in range(N):
                new_a = a[-1] + w * dt
                new_x = x[-1] + v * np.cos(new_a) * dt
                new_y = y[-1] + v * np.sin(new_a) * dt
                a.append(new_a)
                x.append(new_x)
                y.append(new_y)
            ax.plot(x, y, color = color, linestyle = '--')
        

import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
if __name__ == '__main__':
    
    # Multi Mahalanobis Test
    
    x_max = 1500
    y_max = 1000
    
    N_set = 10
    nA = int(np.random.uniform(N_set-5, N_set+5))
    nB = int(np.random.uniform(N_set-5, N_set+5))
    
    set_A_x = np.random.uniform(0, x_max, nA)    
    set_A_y = np.random.uniform(0, y_max, nA)
    
    P = np.array([[100, 30],
                  [30, 300]])
    
    AP = []
    for i in range(nA):
        AP.append(P * np.random.uniform(1,3))
        
    invAP = np.array([np.linalg.inv(P) for P in AP])
    AP = np.array(AP)
        
    
    set_B_x = np.random.uniform(0, x_max, nB)
    set_B_y = np.random.uniform(0, y_max, nB)
    
    setA = np.array([set_A_x, set_A_y]).T
    setB = np.array([set_B_x, set_B_y]).T
    
    plt.cla()
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.plot(setA[:,0], setA[:,1], '.b')
    plt.plot(setB[:,0], setB[:,1], '.r')
    
    for i in range(nA):
        plot_covariance_ellipse((setA[i,0], setA[i,1]), AP[i])
    
    correspondence_matrix = multi_mahalanobis(setB, setA, invAP)
    
    '''
    correspondence_matrix = np.zeros((nB, nA))
    for i in range(nA):
        for j in range(nB):
            correspondence_matrix[j,i] = mahalanobis(setB[j], setA[i], invAP[i])
    '''
    #print(correspondence_matrix - correspondence_matrix_m)
    
    while True:
        
        minimal_distance = np.unravel_index(np.argmin(correspondence_matrix, axis=None), correspondence_matrix.shape)
        
        if correspondence_matrix[minimal_distance] > 10:
            break # no more ants satisfy threshold
    
        
        A_ind = minimal_distance[1]
        B_ind = minimal_distance[0]
        
        plt.plot([setA[A_ind,0],setB[B_ind,0]],[setA[A_ind,1],setB[B_ind,1]],':g')
        
        correspondence_matrix[B_ind,:] = np.inf
        correspondence_matrix[:,A_ind] = np.inf
        
    
    plt.show()
