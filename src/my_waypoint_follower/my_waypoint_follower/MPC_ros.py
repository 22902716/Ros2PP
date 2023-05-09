#! /usr/bin/env python3

#import necessary classes
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from warnings import WarningMessage
import numpy as np
import math, cmath
import casadi as ca
from argparse import Namespace
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped


class MPCControllerNode (Node):
    def __init__(self, dt, N, speedgain=1.0, L = 0.324):
        #initialise subscriber and publisher node
        super().__init__("MPC_ros")
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        #initialise position vectors 
        self.x = 0.0
        self.y = 0.0
        #quaterion vectors given by the 'ego_racecar/odom' topic later converted into euler
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.ow = 0.0


        self.yaw = 0.0
        self.speed = 0.0
        self.speedgain = speedgain    

        #MPC specific parameters
        self.dt = dt
        self.L = L
        self.N = N # number of steps to predict
        self.nx = 4
        self.nu = 2
        self.u_min = [-0.4,-13]
        self.u_max = [0.4, 13]
        self.drawn_waypoints = []

        #trajectory class
        self.wpts = None
        self.ss = None
        self.total_s = None
        self.vs = None
        
        self.max_distance = 0
        self.distance_allowance = 1
        self.constantspeed = 0.5

        self.load_waypoints()

    
    def callback(self, msg: Odometry):
        cmd = AckermannDriveStamped()
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.ox = msg.pose.pose.orientation.x
        self.oy = msg.pose.pose.orientation.y
        self.oz = msg.pose.pose.orientation.z
        self.ow = msg.pose.pose.orientation.w
        self.currentSpeed_x = msg.twist.twist.linear.x

        self.positionMessage()

        self.dt = 0.01*self.currentSpeed_x+0.1          
        #self.dt = 0.01*self.currentSpeed_x+0.1
        x0 = [self.x,self.y,self.yaw,self.currentSpeed_x]
        steering_angle, x_bar,speed, acc = self.plan(x0)
        
        cmd.drive.speed = speed*self.speedgain
        cmd.drive.steering_angle = steering_angle
        self.get_logger().info("current time step = " + str(self.dt))
        # self.get_logger().info("current speed x = " + str(self.currentSpeed_x)
        #                     + "current speed y = " + str(self.currentSpeed_y)
        #                     + "current speed z = " + str(self.currentSpeed_z))
        # self.get_logger().info("current ind = " + str(self.ego_index) 
        #                        + " current yaw = " + str(self.yaw)
        #                        + "cur_x = " + str(self.x)
        #                        + "cur_y = " + str(self.y)
        #                        + " tar_x = " + str(self.points[self.ego_index][0])
        #                        + "tar_y = " + str(self.points[self.ego_index][1]))
        self.drive_pub.publish(cmd)

        # self.get_logger().info("pose_x = " + str(self.x) + " pose_y = " + str(self.y) + " orientation_z = " + str(self.yaw))
                
    def positionMessage(self):
        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(self.ox,self.oy,self.oz,self.ow)
        return self.x, self.y, self.roll, self.pitch, self.yaw
    
    
    def euler_from_quaternion(self,x, y, z, w):  
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
    def load_waypoints(self):
        """
        loads waypoints
        """
        
        self.waypoints = np.loadtxt('f1_aut_wide_raceline.csv', delimiter=',')
        self.wpts = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        seg_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)

        self.trueSpeedProfile = self.waypoints[:, 5]
        self.vs = self.trueSpeedProfile * self.speedgain  #speed profile

        self.total_s = self.ss[-1]
        self.tN = len(self.wpts)

    def get_timed_trajectory_segment(self, position, dt, n_pts=10):
        pose = np.array([position[0], position[1], position[3]])
        trajectory, distances = [pose], [0]
        for i in range(n_pts-1):
            distance = dt * pose[2]
            
            current_distance = self.calculate_progress(pose[0:2])
            next_distance = current_distance + distance
            distances.append(next_distance)
            
            interpolated_x = np.interp(next_distance, self.ss, self.wpts[:, 0])
            interpolated_y = np.interp(next_distance, self.ss, self.wpts[:, 1])

            interpolated_v = np.interp(next_distance, self.ss, self.vs)
            pose = np.array([interpolated_x, interpolated_y, interpolated_v])
            trajectory.append(pose)
        
        interpolated_waypoints = np.array(trajectory)
        return interpolated_waypoints
    
    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        if idx+1 >= len(self.ss)-1:
            idx = 0
        else:
            idx = idx 
               
        d_ss = self.ss[idx+1] - self.ss[idx]

        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:
                # negative due to floating point precision
                # if the point is very close to the trackline, then the trianlge area is increadibly small
                h = 0
                x = d_ss + d1
                # print(f"Area square is negative: {Area_square}")
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5

        return x, h
    
    def get_trackline_segment(self, point):
        """
        Returns the first index representing the line segment that is closest to the point.
        """
        dists = np.linalg.norm(point - self.wpts, axis=1)

        min_dist_segment = np.argmin(dists)

        return min_dist_segment,dists
        
    def calculate_progress(self, point):
        idx, dists = self.get_trackline_segment(point)
        x, h = self.interp_pts(idx, dists)
        s = self.ss[idx] + x
        
        return s
    
    def estimate_u0(self, reference_path, x0):

        reference_theta = np.arctan2(reference_path[1:, 1] - reference_path[:-1, 1], reference_path[1:, 0] - reference_path[:-1, 0])

        th_dot = calculate_angle_diff(reference_theta) 

        th_dot[0] += (reference_theta[0]- x0[2]) 

        
        speeds = reference_path[:, 2]
        steering_angles = (np.arctan(th_dot) * self.L / speeds[:-2]) / self.dt

        speeds[0] += (x0[3] - reference_path[0, 2] )
        accelerations = np.diff(speeds) / self.dt
        
        u0_estimated = np.vstack((steering_angles, accelerations[:-1])).T
        
        return u0_estimated
    
    def plan(self, x0):

        reference_path = self.get_timed_trajectory_segment(x0, self.dt, self.N+2)
        u0_estimated = self.estimate_u0(reference_path, x0)

        u_bar, x_bar = self.generate_optimal_path(x0, reference_path[:-1].T, u0_estimated)

        speed = x0[3] + u_bar[0][1]*self.dt
        return u_bar[0][0],x_bar, speed,u_bar[0][1] # return the first control action
        
    def generate_optimal_path(self, x0_in, x_ref, u_init):
        """generates a set of optimal control inputs (and resulting states) for an initial position, reference trajectory and estimated control

        Args:
            x0_in (ndarray(3)): the initial pose
            x_ref (ndarray(N+1, 2)): the reference trajectory
            u_init (ndarray(N)): the estimated control inputs

        Returns:
            u_bar (ndarray(N)): optimal control plan
        """
        x = ca.SX.sym('x', self.nx, self.N+1)
        u = ca.SX.sym('u', self.nu, self.N)
        
        speeds = x_ref[2]

        # Add a speed objective cost.
        J = ca.sumsqr(x[:2, :] - x_ref[:2, :])  + ca.sumsqr(x[3, :] - speeds[None, :]) *10
        
        g = []
        for k in range(self.N):
            x_next = x[:,k] + self.f(x[:,k], u[:,k])*self.dt
            g.append(x_next - x[:,k+1])

        initial_constraint = x[:,0] - x0_in 
        g.append(initial_constraint)

        x_init = [x0_in]
        for i in range(1, self.N+1):
            x_init.append(x_init[i-1] + self.f(x_init[i-1], u_init[i-1])*self.dt)

        for i in range(len(u_init)):
            x_init.append(u_init[i])


        x_init = ca.vertcat(*x_init)
        
        lbx = [-ca.inf, -ca.inf, -ca.inf, 0] * (self.N+1) + self.u_min * self.N
        ubx = [ca.inf, ca.inf, ca.inf, 8] * (self.N+1) + self.u_max * self.N
        
        x_nlp = ca.vertcat(x.reshape((-1, 1)), u.reshape((-1, 1)))
        g_nlp = ca.vertcat(*g)
        nlp = {'x': x_nlp,
            'f': J,
            'g': g_nlp}
        

        opts = {'ipopt': {'print_level': 2},
                'print_time': False}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        sol = solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=0, ubg=0)

        x_bar = np.array(sol['x'][:self.nx*(self.N+1)].reshape((self.nx, self.N+1)))
        u_bar = sol['x'][self.nx*(self.N+1):]

        u_return = np.array(u_bar)[:, 0]
        u_return = u_return.reshape((self.N, self.nu))
        
        return u_return, x_bar
    
    def f(self, x, u):
        # define the dynamics as a casadi array
        xdot = ca.vertcat(
            ca.cos(x[2])*x[3],
            ca.sin(x[2])*x[3],
            x[3]/self.L * ca.tan(u[0]),
            u[1]
        )
        return xdot

#_________________________________________________________________________________________________________________________________________

def calculate_angle_diff(angle_vec):
    angle_diff = np.zeros(len(angle_vec)-1)
    for i in range(len(angle_vec)-1):
        angle_diff[i] = sub_angles_complex(angle_vec[i], angle_vec[i+1])
    
    return angle_diff

def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase
        



def main(args = None):
    
    #use - ros2 run teleop_twist_keyboard teleop_twist_keyboard 
    #to move the car manually around the map

    rclpy.init(args = args)
    timestep = 0.1     #look forward time step 0.12
    N = 7               #number of predictions 7
    MPC_controller_node = MPCControllerNode(timestep, N)

    rclpy.spin(MPC_controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node


    
