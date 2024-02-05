#! /usr/bin/env python3

#import necessary classes
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import math, cmath
import casadi as ca
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import time



class MPCControllerNode (Node):
    def __init__(self):

        mapname_list = ["gbr", "mco", "esp", "CornerHallE", "f1_aut_wide", "levine_blocked"] 
        mapname = mapname_list[3]
        self.max_iter = 50
        self.is_start = None

        #initialise subscriber and publisher node
        super().__init__("MPC_ros")
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.ego_reset_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        #initialise position vectors 
        self.x0 = [0.0] * 4         #x_pos, y_pos, yaw, speed 
        self.speedgain = 1.
        self.x0_prev = [0.0]*4
        self.iter = 0

        self.planner = MPC(mapname)
        self.ds = dataSave("ros_rviz", mapname, self.max_iter)
        self.crash_counter = 0
        self.success_counter = 0
        self.testmode_list = ["constant", "gain"]
        self.testmode = self.testmode_list[1]
        if self.testmode == "constant":
            self.planner.dt_constant = 0.01
        if self.testmode == "gain":
            self.planner.dt_gain = 0.0000000

        self.cmd_start_timer = time.perf_counter()


    
    def callback(self, msg: Odometry):

        if self.is_start == None:
            self.ego_reset_stop()
            self.start_laptime = time.time() #come back to this put this somewhere else
            self.is_start = 1

        cmd = AckermannDriveStamped()
        lapsuccess = 0 if self.planner.completion<99 else 1
        laptime = time.time() - self.start_laptime
        self.cmd_current_timer = time.perf_counter()


        self.ori_x = [msg.pose.pose.orientation.x,
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z,
                      msg.pose.pose.orientation.w]
        self.yaw = self.planner.euler_from_quaternion(self.ori_x[0],
                                              self.ori_x[1],
                                              self.ori_x[2],
                                              self.ori_x[3])
        self.x0_prev = self.x0
        self.x0 = [msg.pose.pose.position.x,
                   msg.pose.pose.position.y,
                   self.yaw,
                   msg.twist.twist.linear.x]

        indx, trackErr,speed,steering = self.planner.plan(self.x0)
        # self.get_logger().info("i planned")
        cmd.drive.speed = speed*self.speedgain
        cmd.drive.steering_angle = steering
        # self.get_logger().info("current time step: dt = " + str(self.planner.dt))
        # self.get_logger().info("current speed x = " + str(self.currentSpeed_x)
        #                     + "current speed y = " + str(self.currentSpeed_y)
        #                     + "current speed z = " + str(self.currentSpeed_z))
        # self.get_logger().info("current ind = " + str(self.ego_index) 
        #                        + " current yaw = " + str(self.yaw)
        #                        + "cur_x = " + str(self.x)
        #                        + "cur_y = " + str(self.y)
        #                        + " tar_x = " + str(self.points[self.ego_index][0])
        #                        + "tar_y = " + str(self.points[self.ego_index][1]))
        # self.get_logger().info(str(self.planner.completion))

        if self.planner.completion >= 99 :
            
            self.ds.lapInfo(self.iter, lapsuccess, laptime, self.planner.completion, 0, 0, laptime)
            self.get_logger().info("Lap info csv saved")
            self.ds.savefile(self.iter)
            self.get_logger().info("States for the lap saved")
            self.ego_reset_stop()
            self.success_counter += 1
            
            #============================================================
            if self.testmode == "constant":
                self.planner.dt_constant += 0.01
            if self.testmode == "gain":
                self.planner.dt_gain += 0.001

        else:
            if self.cmd_current_timer - self.cmd_start_timer >= 0.02:
                self.drive_pub.publish(cmd)
                # self.get_logger().info("i published")
                self.cmd_start_timer = self.cmd_current_timer

            # self.get_logger().info("i published")
            if (self.x0_prev[3]-self.x0[3]) > 0.4:
                self.get_logger().info("i crashed")
                self.crash_counter += 1
                if self.testmode == "constant":
                    self.planner.dt_constant += 0.01
                if self.testmode == "gain":
                    self.planner.dt_gain += 0.001
                self.ego_reset_stop()
                time.sleep(0.5)
                
            # self.get_logger().info("current speed = " + str(self.x0[3]) + "previous speed = " + str(self.x0_prev[3]))
                self.get_logger().info(str(self.planner.dt_constant)
                               +"+"
                               +str(self.planner.dt_gain)
                               +"*"
                               +str(self.x0[3]))
        if self.crash_counter + self.success_counter > self.max_iter:
            self.ds.saveLapInfo()
            rclpy.shutdown()

        self.ds.saveStates(laptime, self.x0, self.planner.speed_list[indx], trackErr, 0, self.planner.completion)


        # self.get_logger().info("pose_x = " + str(self.x) + " pose_y = " + str(self.y) + " orientation_z = " + str(self.yaw))
                
    def ego_reset_stop(self):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        self.x0 = [0.0] * 4      #x_pos, y_pos, yaw, speed  
        self.ego_reset_pub.publish(msg)

        cmd = AckermannDriveStamped()
        cmd.drive.speed = 0.
        cmd.drive.steering_angle = 0.
        self.drive_pub.publish(cmd)
        # self.iteration_no += 1

        self.get_logger().info("Finished Resetting Vehicle")
#_________________________________________________________________________________________________________________________________________

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
        self.dt_gain = 0.02         #tune these two parameters before running
        self.dt_constant = 0.08


    def load_waypoints(self,mapname):
        """
        loads waypoints
        """
        
        self.waypoints = np.loadtxt(f'maps/{mapname}_raceline.csv', delimiter=',')
        self.wpts = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        seg_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)

        self.trueSpeedProfile = self.waypoints[:, 5]
        self.speed_list = self.waypoints[:, 5]

        self.vs = self.trueSpeedProfile  #speed profile

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
        steering_angles = (np.arctan(th_dot) * self.L / speeds[:-2]) / self.dt

        speeds[0] += (x0[3] - reference_path[0, 2] )
        accelerations = np.diff(speeds) / self.dt
        
        u0_estimated = np.vstack((steering_angles, accelerations[:-1])).T
        
        return u0_estimated
    
    def plan(self, x0):
        self.dt = self.dt_gain*x0[3]+self.dt_constant          
        reference_path = self.get_timed_trajectory_segment(x0, self.dt, self.N+2)
        u0_estimated = self.estimate_u0(reference_path, x0)

        u_bar, x_bar = self.generate_optimal_path(x0, reference_path[:-1].T, u0_estimated)

        pose = np.array([x0[0], x0[1]])
        ego_index,min_dists = self.get_trackline_segment(pose)
        # self.completion = 100 if ego_index/len(self.wpts) == 0 else round(ego_index/len(self.wpts)*100,2)
        self.completion = round(ego_index/len(self.wpts)*100,2)        
        _,trackErr = self.interp_pts(ego_index, min_dists)
        speed = x0[3] + u_bar[0][1]*self.dt

        return ego_index, trackErr, speed, u_bar[0][0] # return the first control action
        
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
class dataSave:
    def __init__(self, TESTMODE, map_name,max_iter):
        self.rowSize = 50000
        self.stateCounter = 0
        self.lapInfoCounter = 0
        self.TESTMODE = TESTMODE
        self.map_name = map_name
        self.max_iter = max_iter
        self.txt_x0 = np.zeros((self.rowSize,8))
        self.txt_lapInfo = np.zeros((max_iter,8))

    def saveStates(self, time, x0, expected_speed, tracking_error, noise, completion):
        self.txt_x0[self.stateCounter,0] = time
        self.txt_x0[self.stateCounter,1:4] = [x0[0],x0[1],x0[3]]
        self.txt_x0[self.stateCounter,4] = expected_speed
        self.txt_x0[self.stateCounter,5] = tracking_error
        self.txt_x0[self.stateCounter,6] = noise
        self.txt_x0[self.stateCounter,7] = completion
        self.stateCounter += 1
        #time, x_pos, y_pos, actual_speed, expected_speed, tracking_error, noise

    def savefile(self, iter):
        for i in range(self.rowSize):
            if (self.txt_x0[i,4] == 0):
                self.txt_x0 = np.delete(self.txt_x0, slice(i,self.rowSize),axis=0)
                break
        np.savetxt(f"Imgs/{self.TESTMODE}/{self.map_name}_{str(iter)}.csv", self.txt_x0, delimiter = ',', header="laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error", fmt="%-10f")

        self.txt_x0 = np.zeros((self.rowSize,8))
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
        np.savetxt(f"csv/MPC_{self.map_name}_{self.TESTMODE}.csv", self.txt_lapInfo,delimiter=',',header = f"lap_count, lap_success, laptime, completion, {var1}, {var2}, aveTrackErr, Computation_time", fmt="%-10f")



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
        
#_________________________________________________________________________________________________________________________
def main(args = None):
    
    #use - ros2 run teleop_twist_keyboard teleop_twist_keyboard 
    #to move the car manually around the map

    rclpy.init(args = args)
    MPC_controller_node = MPCControllerNode()

    rclpy.spin(MPC_controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node

#_________________________________________________________________________________________________________________________
if __name__ == "__main__":
    main()
