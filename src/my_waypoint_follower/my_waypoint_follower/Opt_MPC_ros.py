#! /usr/bin/env python3
#import necessary classes
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import math, cmath
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
import time
import casadi as ca

speed_profile_gain = 1.0
maxspeed = 2.
# testmode_txt = "sim_Car"
testmode_txt = "real_Car"
date = "3008"

class MPCNode (Node):
    def __init__(self):
        super().__init__("MPC_ros")
        mapname = "CornerHallE"
        self.mapname  = mapname
        max_iter = 1

        self.planner = MPC(mapname)
        self.ds = dataSave(testmode_txt, mapname, max_iter, speed_profile_gain)
        self.kalFilter = KalmanFilter()
        
        self.joy_sub = self.create_subscription(Joy, "/joy", self.callbackJoy, 10)

        #car subscriber
        self.pose_subscriber = self.create_subscription(Odometry, '/pf/pose/odom', self.callback, 10)

        #chris pf
        # self.pose_subscriber = self.create_subscription(Odometry, '/expected_pose', self.callback, 10)

        #rvis subscriber
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.realOdom, 10)
        # self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.callback, 10)


        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.Joy7 = 0

        self.x0 = [0.1] * 4      #x_pos, y_pos, yaw, speed  
        self.x0_prev = [0.0]*4

        self.cmd_start_timer = time.perf_counter()
        self.get_logger().info("initialised")
        self.start_laptime = time.time()
        self.prev_x0 = [0.1,0.1,0.1,0.1]
        self.predict_steering = 0.0
        self.true_x0 = [0.01,0.01,0.01,0.01]
        self.true_x0_txt = np.zeros((2000,4))
        self.trueOdomCounter = 0

    def realOdom(self, msg:Odometry):
        quat_ori = [msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w]
        yaw = self.euler_from_quaternion(quat_ori[0], quat_ori[1], quat_ori[2], quat_ori[3])

        self.true_x0 = [msg.pose.pose.position.x,
                       msg.pose.pose.position.y,
                       yaw,
                       msg.twist.twist.linear.x]
        
    def save_true_odom(self,x0):
        self.true_x0_txt [self.trueOdomCounter,0:4] = x0
        self.trueOdomCounter += 1

    def savefile(self):
        # for i in range(1000):
            # if (self.true_x0_txt[i,3] == 0):
            #     self.true_x0_txt = np.delete(self.true_x0_txt, slice(i,10000),axis=0)
            #     break
        np.savetxt(f"Benchmark_car/Imgs/{date}_MPC_trueOdom.csv", self.true_x0_txt, delimiter = ',',
                    header="laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error, noise, completion, steering, slip", fmt="%-10f")



    def add_distance(self, x, y, yaw, steering, speed):
        oldx,oldy = x,y
        distance  = speed*0.3
        yaw = yaw + speed/0.324 * np.tan(steering)*0.3

        temp_coord = [x + distance * math.cos(yaw), y + distance * math.sin(yaw)]
        new_x = (oldx+temp_coord[0])/2
        new_y = (oldy+temp_coord[1])/2


        return new_x, new_y

    def callback(self, msg: Odometry):

        lapsuccess = 0 if self.planner.completion < 50 else 1
        laptime = time.time() - self.start_laptime
        self.cmd_current_timer = time.perf_counter()
        # self.get_logger().info("in callback")       

        cmd = AckermannDriveStamped()

        quat_ori = [msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w]
        
        yaw = self.euler_from_quaternion(quat_ori[0], quat_ori[1], quat_ori[2], quat_ori[3])

        # self.x0_prev = self.x0

        self.x0 = [msg.pose.pose.position.x,
                   msg.pose.pose.position.y,
                   yaw,
                   msg.twist.twist.linear.x]
        
        self.save_true_odom(self.true_x0)
        
        self.get_logger().info(f"prev_x = {str(self.x0[0])} prev_x = {str(self.x0[1])} + {str(self.x0[2])} and {str(self.x0[3])}")       

        
        # self.x0[0],self.x0[1] = self.kalFilter.kalman(self.x0[:2], self.x0[3], self.predict_steering, self.planner.dt, self.mapname, self.x0[2])


        # self.x0[0], self.x0[1] = self.add_distance(self.x0[0], self.x0[1], self.x0[2], self.predict_steering, self.x0[3]) 

        self.get_logger().info(f"curr_x = {str(self.x0[0])} curr_y = {str(self.x0[1])} + {str(self.x0[2])} and {str(self.x0[3])}")       
        # self.get_logger().info(f"{str(self.planner.angle_from_ref)} ------ {str(self.planner.steering_weight)}-------{self.x0[2]}")

        if self.x0[0] == 0: 
            self.x0[0] = 0.0001
        if self.x0[1] == 0: 
            self.x0[1] = 0.0001
        if self.x0[2] == 0: 
            self.x0[2] = 0.0001
        

        if self.planner.completion >= 50 or (self.planner.completion >= 10 and self.x0_prev[3]-self.x0[3] > 1.5):
            self.get_logger().info("I finished running the lap")
            self.ds.lapInfo(1, lapsuccess, laptime, self.planner.completion, self.planner.dt_gain, self.planner.dt_constant, laptime)
            self.ds.saveLapInfo()
            self.get_logger().info("Lap info csv saved")
            self.ds.savefile()
            self.get_logger().info("States for the lap saved")
            cmd.drive.speed = 0.0
            cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(cmd)
            self.savefile()
            self.get_logger().info("true odom for the lap saved")

            rclpy.shutdown()    
        else:
            if self.cmd_current_timer - self.cmd_start_timer >= 0.0:
                if self.Joy7 == 1:
                	# self.get_logger().info("controller active")
                            
                    indx, trackErr, speed, steering,x_bar, x_ref = self.planner.plan(self.x0)
                    self.predict_steering = steering
                    cmd.drive.speed = speed
                    #cmd.drive.speed = 1.0
                    cmd.drive.steering_angle = steering
                    self.ds.saveOptimisation(self.x0, x_bar, x_ref)
                    slip = self.slipAngleCalc(self.x0)
                    self.ds.saveStates(laptime, self.x0, self.planner.speed_list[indx], trackErr, 0, self.planner.completion, steering, slip)
                    # self.get_logger().info("speed = " + str(speed) + "steering = " + str(steering))

                    self.drive_pub.publish(cmd)
                else:
                    self.get_logger().info("controller inactive")
                    cmd.drive.speed = 0.0
                    cmd.drive.steering_angle = 0.0
                    self.drive_pub.publish(cmd)
                # self.get_logger().info("i published")
                self.cmd_start_timer = self.cmd_current_timer       
        
        
        

        # self.get_logger().info("pose_x = " + str(self.x) 
        #                        + " pose_y = " + str(self.y) 
        #                        + " orientation_z = " + str(self.yaw))

    def callbackJoy(self, msg: Joy):
        self.Joy7 = msg.buttons[7]
        
    def slipAngleCalc(self, x0):
        x = [x0[0] - self.prev_x0[0]]
        y = [x0[1] - self.prev_x0[1]]
        
        velocity_dir = np.arctan2(y,x)
        slip = np.abs(velocity_dir[0] - x0[2]) *360 / (2*np.pi)
        if slip > 180:
            slip = slip-360

        self.prev_x0 = x0

        return slip

    def euler_from_quaternion(self,x, y, z, w):  
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z # in radians
    
class MPC():
    def __init__(self,mapname):
        #MPC specific parameters
        self.dt = 0.2       #this is initialisation default, require tuning to make sure no crash 
        self.L = 0.324
        self.N = 5          # number of steps to predict
        self.nx = 4
        self.nu = 2
        self.u_min = [-0.4,-13]
        self.u_max = [0.4, 13]
        self.load_waypoints(mapname)
        self.completion = 0.0

        #MPC parameters
        self.dt_gain = 0.1         #tune these two parameters before running
        self.dt_constant = 0.15
        self.steering_weight = 0.9
        self.angle_from_ref = 0.0


    def load_waypoints(self,mapname):
        """
        loads waypoints
        """
        
        self.waypoints = np.loadtxt(f'maps/{mapname}_raceline.csv', delimiter=',')
        # self.waypoints = np.loadtxt(f'maps/{mapname}_centerline.csv', delimiter=',')
        self.wpts = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        seg_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)

        self.vs = self.waypoints[:, 5]*speed_profile_gain
        self.speed_list = self.waypoints[:, 5]*speed_profile_gain

        self.total_s = self.ss[-1]


    def euler_from_quaternion(self,x, y, z, w):  

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z # in radians
    

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
        for i in range(len(reference_path)):
            if reference_path[i][2] == 0.0:
                reference_path[i][2] = 0.1
        steering_angles = (np.arctan(th_dot) * self.L / speeds[:-2]) / self.dt

        speeds[0] += (x0[3] - reference_path[0, 2] )
        accelerations = np.diff(speeds) / self.dt
        
        u0_estimated = np.vstack((steering_angles, accelerations[:-1])).T
        
        return u0_estimated
    
    def plan(self, x0):
        # self.dt = self.dt_gain*x0[3]+self.dt_constant   
        self.dt = 0.1
        reference_path = self.get_timed_trajectory_segment(x0, self.dt, self.N+2)
        u0_estimated = self.estimate_u0(reference_path, x0)
        pose = np.array([x0[0], x0[1]])
        ego_index,min_dists = self.get_trackline_segment(pose)
        self.steering_weight_change(ego_index, x0[2])
        u_bar, x_bar = self.generate_optimal_path(x0, reference_path[:-1].T, u0_estimated)

        
        # self.completion = 100 if ego_index/len(self.wpts) == 0 else round(ego_index/len(self.wpts)*100,2)
        self.completion = round(ego_index/len(self.wpts)*100,2)        
        _,trackErr = self.interp_pts(ego_index, min_dists)
        speed = x0[3] + u_bar[1][1]*self.dt

        return ego_index, trackErr, speed, u_bar[0][0], x_bar, reference_path # return the first control action
    
    def steering_weight_change(self, ego_index,yaw):
        future_reference_pose = self.waypoints[ego_index + 20]
        current_reference_pose = self.waypoints[ego_index]
        angle_from_ref = np.arctan2(future_reference_pose[2] - current_reference_pose[2], future_reference_pose[1] - current_reference_pose[1])

        self.angle_from_ref = np.arctan2(future_reference_pose[2] - current_reference_pose[2], future_reference_pose[1] - current_reference_pose[1])
        current_yaw = yaw
        
        if angle_from_ref-current_yaw < 0.1 and angle_from_ref-current_yaw > -0.1:
            self.steering_weight = 3
        else:
            # self.steering_weight = 0.9  #default worked for maxspeed 2
            self.steering_weight = 1.1

                
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
        # J = ca.sumsqr(x[:2, :] - x_ref[:2, :])  + ca.sumsqr(x[3, :] - speeds[None, :]) *10 + ca.sumsqr(u[0, :] * 0.9)

        J = ca.sumsqr(x[:2, :] - x_ref[:2, :])  + ca.sumsqr(x[3, :] - speeds[None, :]) *10 + ca.sumsqr(u[0, :] * self.steering_weight)

        # J = ca.sumsqr(x[:2, :] - x_ref[:2, :])*100  + ca.sumsqr(x[3, :] - speeds[None, :])

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
        ubx = [ca.inf, ca.inf, ca.inf, maxspeed+1.3] * (self.N+1) + self.u_max * self.N
        
        x_nlp = ca.vertcat(x.reshape((-1, 1)), u.reshape((-1, 1)))
        g_nlp = ca.vertcat(*g)
        nlp = {'x': x_nlp,
            'f': J,
            'g': g_nlp}
        

        opts = {'ipopt': {'print_level': 0},
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
class dataSave:
    def __init__(self, TESTMODE, map_name,max_iter, speedgain):
        self.rowSize = 50000
        self.stateCounter = 0
        self.lapInfoCounter = 0
        self.opt_counter = 0
        self.TESTMODE = TESTMODE
        self.map_name = map_name
        self.max_iter = max_iter
        self.txt_x0 = np.zeros((self.rowSize,10))
        self.txt_lapInfo = np.zeros((max_iter,8))
        self.txt_opt = np.zeros((self.rowSize,48))

        self.speedgain = speedgain
        self.speedgain_txt = speedgain

    def saveOptimisation(self, x0, x0_solution, x0_ref):
        #x0 nx, x0_solution nx*(N+1), x0_ref 2*N+1
        self.txt_opt[self.opt_counter, 0:3] = [x0[0], x0[1], x0[3]]
        self.txt_opt[self.opt_counter, 3:27] = x0_solution.reshape((1,24))
        self.txt_opt[self.opt_counter, 27:48] = x0_ref.reshape((1,-1))
        self.opt_counter += 1

    def saveStates(self, time, x0, expected_speed, tracking_error, noise, completion, steering, slip):
        self.txt_x0[self.stateCounter,0] = time
        self.txt_x0[self.stateCounter,1:4] = [x0[0],x0[1],x0[3]]
        self.txt_x0[self.stateCounter,4] = expected_speed
        self.txt_x0[self.stateCounter,5] = tracking_error
        self.txt_x0[self.stateCounter,6] = noise
        self.txt_x0[self.stateCounter,7] = completion
        self.txt_x0[self.stateCounter,8] = steering
        self.txt_x0[self.stateCounter,9] = slip
        self.stateCounter += 1
        #time, x_pos, y_pos, actual_speed, expected_speed, tracking_error, noise

    def savefile(self):
        for i in range(self.rowSize):
            if (self.txt_x0[i,4] == 0):
                self.txt_x0 = np.delete(self.txt_x0, slice(i,self.rowSize),axis=0)
                break
        np.savetxt(f"Benchmark_car/Imgs/{date}_MPC_sim_{self.map_name}_{self.TESTMODE}_{self.speedgain_txt}.csv", self.txt_x0, delimiter = ',',
                    header="laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error, noise, completion, steering, slip", fmt="%-10f")

        self.txt_x0 = np.zeros((self.rowSize,10))
        self.stateCounter = 0
    
    def lapInfo(self,lap_count, lap_success, laptime, completion, var1, var2, Computation_time):
        self.txt_lapInfo[self.lapInfoCounter, 0] = lap_count
        self.txt_lapInfo[self.lapInfoCounter, 1] = lap_success
        self.txt_lapInfo[self.lapInfoCounter, 2] = laptime
        self.txt_lapInfo[self.lapInfoCounter, 3] = completion
        self.txt_lapInfo[self.lapInfoCounter, 4] = var1
        self.txt_lapInfo[self.lapInfoCounter, 5] = var2
        self.txt_lapInfo[self.lapInfoCounter, 6] = np.mean(self.txt_x0[:,5])
        self.txt_lapInfo[self.lapInfoCounter, 7] = Computation_time
        self.lapInfoCounter += 1
        #lap_count, lap_success, laptime, completion, var1, var2, aveTrackErr, Computation_time

    def saveLapInfo(self):
        var1 = "NA"
        var2 = "NA"
        for i in range(self.rowSize):
            if (self.txt_opt[i,35] == 0):
                self.txt_opt = np.delete(self.txt_opt, slice(i,self.rowSize),axis=0)
                break
        np.savetxt(f"Benchmark_car/csv/{date}_MPC_sol_sim_{self.map_name}_car_data_{self.speedgain_txt}.csv", self.txt_opt,delimiter=',',
                   header = f"x0, x_bar, x_ref", fmt="%-10f")
        
        np.savetxt(f"Benchmark_car/csv/{date}_MPC_sim_{self.map_name}_{self.TESTMODE}_{self.speedgain_txt}.csv", self.txt_lapInfo,delimiter=',',
                   header = f"lap_count, lap_success, laptime, completion, {var1}, {var2}, aveTrackErr, Computation_time", fmt="%-10f")



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
    controller_node = MPCNode()
    # publish_node = PurePursuitPlanner(0.3,speedgain,0,0,0)

    rclpy.spin(controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node

class KalmanFilter():
    def __init__(self):
        dt = 0.2
        L = 0.324
        self.F = np.array([
        [1, 0, -dt*np.sin(0), dt*np.cos(0)],  # Partial derivative wrt x
        [0, 1, dt*np.cos(0), dt*np.sin(0)],   # Partial derivative wrt y
        [0, 0, 1, 0],                         # Partial derivative wrt theta
        [0, 0, 0, 1]                          # Partial derivative wrt v
    ])
        self.H = np.array([
        [1, 0, 0, 0],  # Measure x
        [0, 1, 0, 0]   # Measure y
    ])

        self.n = self.F.shape[1]
        self.m = self.H.shape[0]
        self.B = np.array([
        [dt*np.cos(0), 0],
        [dt*np.sin(0), 0],
        [0, dt/L],
        [0, dt]
    ])
        self.Q = np.eye(4) * 0.9  # Process noise covariance
        self.R = np.eye(2) * 0.01   # Measurement noise covariance
        self.P = np.eye(self.n)
        self.x = np.array([0, 0, 0, 0]).reshape(4, 1)
        self.prev_speed = 0.0

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
        
        
    def kalman(self, x_pf,speed, steering, dt , mapname, theta):
        # u = [0.1,0.01]
        L = 0.324
        # theta  = self.x[2,0]
        dt = 0.15

        self.F = np.array([
        [1, 0, -dt*np.sin(theta)*speed, dt*np.cos(theta)*speed],  
        [0, 1, dt*np.cos(theta)*speed, -dt*np.sin(theta)*speed],   
        [0, 0, 1, dt/L*np.tan(steering)],                         
        [0, 0, 0, 1]                          
    ])

        self.B = np.array([
        [0, 0],
        [0, 0],
        [dt*speed/(L*np.cos(steering)**2), 0],
        [0 ,dt]
    ])
        
        u = np.array([steering, speed]).reshape(2,1)
        # u = np.array([steering, (speed-self.prev_speed)/(dt)]).reshape(2,1)
        # self.prev_speed = speed
        self.predict(u)
        x_pf = np.array(x_pf).reshape(2,1)
        self.update(x_pf)

        waypoints = np.loadtxt('maps/' + mapname + '_raceline.csv', delimiter=',')
        points = np.vstack((waypoints[:, 1], waypoints[:, 2])).T
        # new_coord = np.array(self.x[:2].flatten())

        kal_coord = self.x[:2].flatten()
        # poses = np.vstack((x_pf[0], x_pf[1])).T
        # min_dist = np.linalg.norm(poses - points,axis = 1)
        # ego_index = np.argmin(min_dist)
        # new_coord = np.array([(kal_coord[0]+points[ego_index+4][0]+x_pf[0][0])/3, (kal_coord[1]+points[ego_index+4][1]+x_pf[1][0])/3])
        T_dt = 0.16
        new_coord = np.array([(kal_coord[0]+T_dt*np.cos(theta)*speed), (kal_coord[1])+T_dt*np.sin(theta)*speed])


        return new_coord[0], new_coord[1]
        #return the x_pf for a quick check




if __name__ == "__main__":
    main()
    

