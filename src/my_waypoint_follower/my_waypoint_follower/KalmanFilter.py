import numpy as np
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

        dt = 0.01
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

        Q = np.eye(4) * 0.001  # Process noise covariance
        R = np.eye(2) * 0.05   # Measurement noise covariance

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
        
        
    def kalman(self, x0, steering, dt , mapname):


        return new_coord