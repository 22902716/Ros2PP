import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class KalmanFilter(object):
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")
        
        self.n = F.shape[1]
        self.m = H.shape[0]

        self.F = F
        self.H = H
        self.B = np.zeros((self.n, 2)) if B is None else B  # B should match state dimension and control input dimension
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u=np.zeros((2, 1))):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
    dt = 1.0 / 60
    L = 2.0  # Example vehicle length
    mapname = "CornerHallE"

    # State transition matrix F
    F = np.array([
        [1, 0, -dt*np.sin(0), dt*np.cos(0)],  # Partial derivative wrt x
        [0, 1, dt*np.cos(0), dt*np.sin(0)],   # Partial derivative wrt y
        [0, 0, 1, 0],                         # Partial derivative wrt theta
        [0, 0, 0, 1]                          # Partial derivative wrt v
    ])

    # Control input matrix B (assuming control affects x and y)
    B = np.array([
        [dt*np.cos(0), 0],
        [dt*np.sin(0), 0],
        [0, dt/L],
        [0, dt]
    ])

    # Measurement matrix H
    H = np.array([
        [1, 0, 0, 0],  # Measure x
        [0, 1, 0, 0]   # Measure y
    ])

    Q = np.eye(4) * 0.0004  # Process noise covariance
    R = np.eye(2) * 0.5   # Measurement noise covariance

    # Initial state estimate
    x0 = np.array([0, 0, 0, 0]).reshape(4, 1)

    # Create measurements (e.g., from particle filter)
    measurements = np.loadtxt("particle_filter/cornerHall_1.csv",delimiter=',')
    x_measurements_true = measurements[820:,2]
    y_measurements_true = measurements[820:,3]
    x_measurements_pf = measurements[820:,6]
    y_measurements_pf = measurements[820:,7]
    measurements_pf = np.vstack((x_measurements_pf, y_measurements_pf))

    waypoints = np.loadtxt('maps/' + mapname + '_raceline.csv', delimiter=',')
    points = np.vstack((waypoints[:, 1], waypoints[:, 2])).T




    kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, x0=x0)
    predictions_kal = []

    for i in range(len(measurements_pf[0])):
        u = np.array([0.1, 0.01]).reshape(2, 1)  # Control input: steering angle and acceleration
        # print(measurements[:, i])
        z = measurements_pf[:, i].reshape(2, 1)

        if i == 120:
            Q = np.eye(4) * 0.002  # Process noise covariance
            R = np.eye(2) * 0.5   # Measurement noise covariance
            kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, x0=kf.x)
        if i == 210:
            Q = np.eye(4) * 0.0004  # Process noise covariance
            R = np.eye(2) * 0.5   # Measurement noise covariance
            kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, x0=kf.x)


        kf.predict(u)
        kf.update(z)
        kal_coord = kf.x[:2].flatten()
        print(i)

        poses = np.vstack((x_measurements_pf[i], y_measurements_pf[i])).T
        # print(poses)
        min_dist = np.linalg.norm(poses - points,axis = 1)
        ego_index = np.argmin(min_dist)
        # print((kal_coord[0]+points[ego_index][0])/2) 
        # new_coord = np.array([(kal_coord[0]+points[ego_index][0])/2, (kal_coord[1]+points[ego_index][1])/2])
        new_coord = np.array([kal_coord[0], kal_coord[1]])

        predictions_kal.append(kf.x[:2].flatten())
        predictions_txt = np.array(predictions_kal)
        # plt.scatter(x_measurements_pf[i], y_measurements_pf[i], label='Particle Filter Measurements', c = 'r')
        # plt.scatter(new_coord[0], new_coord[1], label='Kalman Filter Prediction',c = 'k')
        # plt.scatter(x_measurements_true[i], y_measurements_true[i], label='True Measurements', c = 'b',)
        # plt.scatter(points[ego_index, 0], points[ego_index, 1], label='Waypoints', c = 'y')
        # plt.pause(0.05)

    # predictions = np.array(predictions_kal)

    plt.plot(-y_measurements_pf, x_measurements_pf, label='Measurements', c = 'r')
    plt.plot(-predictions_txt[:, 1], predictions_txt[:, 0], label='Kalman Filter Prediction',c = 'k')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Kalman Filter Smoothing')
    # print(x_measurements_pf[:,0].shape)

    save_txt = np.vstack((predictions_txt[:,0], predictions_txt[:,1])).T
    save_txt1 = np.vstack(( x_measurements_pf, y_measurements_pf)).T
    final_txt = np.hstack((save_txt, save_txt1))


    np.savetxt('particle_filter/cornerHall_1_KF.csv', final_txt, delimiter=',')
    plt.show()


def plot_ref_true_pf():
    pf_data = np.loadtxt("Benchmark_car/Imgs/3008_MPCC_CornerHallE_sim_Car_2_car.csv",delimiter=',')
    pf_data = pf_data[:,1:3]
    true_data = np.loadtxt("Benchmark_car/Imgs/3008_MPCC_trueOdom.csv",delimiter=',')
    true_data = true_data[:,0:2]
    reference_data = np.loadtxt("maps/CornerHallE_raceline.csv",delimiter=',')
    reference_data = reference_data[:110,1:3]

    plt.figure()

    # for i in range(len(pf_data)):

        # if int(i/4) > 20:
            
        #     plt.plot(reference_data[int(i/4)-20:int(i/4)+20,0], reference_data[int(i/4)-20:int(i/4)+20,1], label = 'Reference Path', c = 'k')
        #     plt.scatter(pf_data[i,0], pf_data[i,1], label = 'Particle Filter Path', c = 'r')
        #     plt.scatter(true_data[i,0], true_data[i,1], label = 'True Path', c = 'b')
        # else:
        # # plt.plot(reference_data[:,0], reference_data[:,1], label = 'Reference Path', c = 'k')
        # plt.scatter(pf_data[i,0], pf_data[i,1], label = 'Particle Filter Path', c = 'r')
        # plt.scatter(true_data[i,0], true_data[i,1], label = 'True Path', c = 'b')
        # plt.legend()
        # plt.xlabel('x (cm)')
        # plt.ylabel('y (cm)')
        # plt.title('Reference Path vs True Path vs Particle Filter Path')
        # plt.pause(0.1)
        # plt.clf()   

    plt.scatter(-pf_data[:,1], pf_data[:,0], label = 'Particle Filter Path', c = 'r', s = 0.9)
    plt.scatter(-true_data[:,1], true_data[:,0], label = 'True Path', c = 'b', s = 0.9)
    plt.legend()
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.title('Reference Path vs True Path vs Particle Filter Path')
    plt.show()

def test():
    import numpy as np
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    # make data
    data = np.loadtxt("particle_filter\cornerHall_1_KF.csv",delimiter=',')
    pf_data = data[:,2:4]
    Kf_data = data[:,0:2]
    extent = (-3, 4, -4, 3)
    ax.plot(-pf_data[:,1], pf_data[:,0], label = 'Particle Filter', c = 'r')
    ax.plot(-Kf_data[:,1], Kf_data[:,0], label = 'Kalman Filter', c = 'b')

    # inset Axes....
    x1, x2, y1, y2 = 4, 9, 5.3, 5.5  # subregion of the original image
    x11,x12,y11,y12= -0.5,0.1,1,3

    axins = ax.inset_axes(
        [0.3, 0.5, 0.6, 0.15],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.scatter(-pf_data[:,1], pf_data[:,0], label = 'Particle Filter', c = 'r', s = 0.9)
    axins.scatter(-Kf_data[:,1], Kf_data[:,0], label = 'Kalman Filter', c = 'b', s = 0.9)

    axins1 = ax.inset_axes(
        [0.1, 0.1, 0.02, 0.8],
        xlim=(x11, x12), ylim=(y11, y12), xticklabels=[], yticklabels=[])
    
    axins1.scatter(-pf_data[:,1], pf_data[:,0], label = 'Particle Filter', c = 'r', s = 0.9)
    axins1.scatter(-Kf_data[:,1], Kf_data[:,0], label = 'Kalman Filter', c = 'b', s = 0.9)

    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.indicate_inset_zoom(axins1, edgecolor="black")

    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Kalman Filter Smoothing')

    plt.show()

if __name__ == '__main__':
    # example()
    plot_ref_true_pf()
    # test()
