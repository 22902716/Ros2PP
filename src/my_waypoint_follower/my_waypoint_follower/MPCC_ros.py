import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import math, cmath
import casadi as ca
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from ReferencePath import ReferencePath as rp
import time

class MPCCControllerNode (Node):
    def __init__(self):

        mapname_list = ["gbr", "mco", "esp", "CornerHallE", "f1_aut_wide", "levine_blocked"] 
        mapname = mapname_list[3]
        max_iter = 1
        self.is_start = None

        #initialise subscriber and publisher node
        super().__init__("MPC_ros")
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.ego_reset_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        #initialise position vectors 
        self.x0 = [0.0] * 4         #x_pos, y_pos, yaw, speed 
        self.speedgain = 1.
        self.iter = 0

        self.planner = MPCC(mapname,"Benchmark")
        self.ds = dataSave("ros_rviz", mapname, max_iter)
        self.cmd_start_timer = time.perf_counter()

    def callback(self, msg: Odometry):

        if self.is_start == None:
            # self.ego_reset()
            self.is_start = 1
            self.is_in = 0
            self.start_laptime = time.time() #come back to this put this somewhere else

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
        self.x0 = [msg.pose.pose.position.x,
                   msg.pose.pose.position.y,
                   self.yaw
                   ]
        speed = msg.twist.twist.linear.x

        _, _,speed,steering, slip, controls, x_bar, x_ref = self.planner.plan(self.x0,laptime, speed)
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
        # self.get_logger().info(str(self.planner.X0_slip[3]))
        self.get_logger().info(str(x_bar))

        if self.planner.completion >= 90:
            
            self.ds.lapInfo(self.iter, lapsuccess, laptime, self.planner.completion, 0, 0, laptime)
            self.get_logger().info("Lap info csv saved")
            self.ds.savefile(self.iter)
            self.get_logger().info("States for the lap saved")
            self.ego_reset_stop()
            self.ds.saveLapInfo()
            rclpy.shutdown()     
        else:
            if self.cmd_current_timer - self.cmd_start_timer >= 0.02:
                self.drive_pub.publish(cmd)
                # self.get_logger().info("i published")
                self.cmd_start_timer = self.cmd_current_timer

        self.ds.saveStates(laptime, self.planner.X0_slip, 0.0, 0.0, 0.0, self.planner.completion, steering, slip)

        self.ds.saveOptimisation(self.planner.X0_slip, x_bar, x_ref)
        # self.ds.saveStates(laptime, self.x0, 0, 0, 0, self.planner.completion, steering, slip)


        # self.get_logger().info("pose_x = " + str(self.x) + " pose_y = " + str(self.y) + " orientation_z = " + str(self.yaw))
                
    def ego_reset_stop(self):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        self.ego_index = None
        self.Tindx = None
        self.x0 = [0.0] * 4      #x_pos, y_pos, yaw, speed  
        self.ego_reset_pub.publish(msg)

        cmd = AckermannDriveStamped()
        cmd.drive.speed = 0.
        cmd.drive.steering_angle = 0.
        self.drive_pub.publish(cmd)
        # self.iteration_no += 1

        self.get_logger().info("Finished Resetting Vehicle")

class MPCC:
    def __init__(self, map_name,TESTMODE):
        print("This is Fast MPCC TEST")
        self.nx = 4 #number of input [x, y, psi, s]
        self.nu = 3 #number of output [delta, v, p],steering(change in yaw angle), change in reference path progress and acceleration

        self.map_name = map_name
        self.TESTMODE = TESTMODE
        self.wheelbase = 0.324
        self.load_waypoints()
        self.scale = 0.

        if self.TESTMODE == "Benchmark":
            self.Max_iter= 1           
        

        self.ds = dataSave(TESTMODE, map_name, self.Max_iter)

        #adjustable params
        #----------------------

        self.dt = 0.25
        self.N = 5  #prediction horizon
        self.mass = 3.8
        self.L = 0.324

        self.delta_min = -0.4
        self.delta_max = 0.4
        self.p_init = 7
        self.p_min = 1
        self.p_max = 10
        self.fmax = 9.81*0.8*self.mass

        self.psi_min = -10
        self.psi_max = 10

        self.weight_progress = 10
        self.weight_lag = 1000
        self.weight_contour = 10
        self.weight_steering = 1.5

        self.v_min = 2 
        self.v_max = 8
        #------------------------

        #initial position
        # position = 1000
        # completion = self.track_lu_table[position,1]
        # xt0 = self.track_lu_table[position,2]
        # yt0 = self.track_lu_table[position,3]
        # phit0 = self.track_lu_table[position,4]
        # self.x0 = np.array([xt0,yt0,phit0])

        self.rp = rp(map_name,w=0.55)
        self.u0 = np.zeros((self.N, self.nu))
        self.X0 = np.zeros((self.N + 1, self.nx))
        self.warm_start = True
        self.x_bar = np.zeros((2, 2))
        self.prev_x0 = [0.0,0.0,0.0]

        self.drawn_waypoints = []
        self.completion = 0.0

        self.problem_setup()


    def load_waypoints(self):
        """
        loads waypoints
        """
        # ***if its a new map uncomment this and generate the new trajectory file***
        # self.track_lu_table, smax = Bezier.generatelookuptable(self.map_name) 
        # exit()
        self.centerline = np.loadtxt('./maps/'+ self.map_name +'_centerline.csv', delimiter=",")
        self.wpts = np.vstack((self.centerline[:,0],self.centerline[:,1])).T
        # self.waypoints = np.loadtxt('./maps/'+ self.map_name +'_raceline.csv', delimiter=",")

        #track_lu_table_heading = ['sval', 'tval', 'xtrack', 'ytrack', 'phitrack', 'cos(phi)', 'sin(phi)', 'g_upper', 'g_lower']

    def euler_from_quaternion(self,x, y, z, w):  
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z # in radians

    def get_trackline_segment(self, point):
        """
        Returns the first index representing the line segment that is closest to the point.
        """
        dists = np.linalg.norm(point - self.wpts, axis=1)

        min_dist_segment = np.argmin(dists)

        if min_dist_segment == len(self.wpts)-1:
            min_dist_segment = 0

        return min_dist_segment,dists
        

    def plan(self, x0,laptime, speed):
        # lower case x0: [x,y,psi,states]
        # upper case X0: [x,y,psi,speed]

        x0, self.X0_slip = self.inputStateAdust(x0, speed)
        x0 = self.build_initial_state(x0)
        x0_ref = self.construct_warm_start_soln(x0) 

        p = self.generate_parameters(x0,self.X0_slip[3])
        controls,self.x_bar = self.solve(p)

        action = np.array([controls[0, 0], controls[0,1]])
        speed,steering = action[1],action[0]

        ego_index,_ = self.get_trackline_segment(x0[0:2])
        self.completion = round(ego_index/len(self.wpts)*100,2)
        slip_angle = self.slipAngleCalc(x0)

        return ego_index, 0, speed,steering, slip_angle, controls, self.x_bar, x0_ref
    
    def slipAngleCalc(self, x0):
        x = [self.X0_slip[0] - self.prev_x0[0]]
        y = [self.X0_slip[1] - self.prev_x0[1]]
        
        velocity_dir = np.arctan2(y,x)
        slip = np.abs(velocity_dir[0] - x0[2]) *360 / (2*np.pi)
        if slip > 180:
            slip = slip-360

        self.prev_x0 = self.X0_slip

        return slip

    def problem_setup(self):
        states = ca.MX.sym('states', self.nx) #[x, y, psi, s]
        controls = ca.MX.sym('controls', self.nu) # [delta, v, p]

        #set up dynamic states of the vehichle
        rhs = ca.vertcat(controls[1] * ca.cos(states[2]), controls[1] * ca.sin(states[2]), (controls[1] / self.wheelbase) * ca.tan(controls[0]), controls[2])  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        
        self.U = ca.MX.sym('U', self.nu, self.N)
        self.X = ca.MX.sym('X', self.nx, (self.N + 1))
        self.P = ca.MX.sym('P', self.nx + 2 * self.N + 1) # Parameters: init state and boundaries of the reference path

        '''Initialize upper and lower bounds for state and control variables'''
        self.lbg = np.zeros((self.nx * (self.N + 1) + self.N*2, 1))
        self.ubg = np.zeros((self.nx * (self.N + 1) + self.N*2, 1))
        self.lbx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
        self.ubx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
                
        x_min, y_min = np.min(self.rp.path, axis=0) - 2
        x_max, y_max = np.max(self.rp.path, axis=0) + 2
        s_max = self.rp.s_track[-1] *1.5
        lbx = np.array([[x_min, y_min, self.psi_min, 0]])
        ubx = np.array([[x_max, y_max, self.psi_max, s_max]])
        for k in range(self.N + 1):
            self.lbx[self.nx * k:self.nx * (k + 1), 0] = lbx
            self.ubx[self.nx * k:self.nx * (k + 1), 0] = ubx

        state_count = self.nx * (self.N + 1)
        for k in range(self.N):
            self.lbx[state_count:state_count + self.nu, 0] = np.array([[-self.delta_max, self.v_min, self.p_min]]) 
            self.ubx[state_count:state_count + self.nu, 0] = np.array([[self.delta_max, self.v_max, self.p_max]])  
            state_count += self.nu

        """Initialise the bounds (g) on the dynamics and track boundaries"""
        self.g = self.X[:, 0] - self.P[:self.nx]  # initial condition constraints
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            k1 = self.f(self.X[:, k], self.U[:, k])
            st_next_euler = self.X[:, k] + (self.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # add dynamics constraint

            self.g = ca.vertcat(self.g, self.P[self.nx + 2 * k] * st_next[0] - self.P[self.nx + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  :represents path boundary constraints
        for k in range(self.N):
            force_lateral = self.U[1,k] **2 / self.L * ca.tan(ca.fabs(self.U[0,k])) *  self.mass
            self.g = ca.vertcat(self.g, force_lateral)


        self.J = 0  # Objective function
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            t_angle = self.rp.angle_lut_t(st_next[3])
            ref_x, ref_y = self.rp.center_lut_x(st_next[3]), self.rp.center_lut_y(st_next[3])
            countour_error = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            lag_error = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.J = self.J + countour_error **2 * self.weight_contour  
            self.J = self.J + lag_error **2 * self.weight_lag
            self.J = self.J - self.U[2, k] * self.weight_progress 
            self.J = self.J - self.U[0, k] * self.weight_steering
            
        optimisation_variables = ca.vertcat(ca.reshape(self.X, self.nx * (self.N + 1), 1),
                                ca.reshape(self.U, self.nu * self.N, 1))

        nlp_prob = {'f': self.J,
                     'x': optimisation_variables,
                       'g': self.g,
                         'p': self.P}
        opts = {"ipopt": {"max_iter": 1500, "print_level": 0}, "print_time": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def build_initial_state(self, current_x):
        x0 = current_x
        x0[2] = self.normalise_psi(x0[2]) 
        x0 = np.append(x0, self.rp.calculate_s(x0[0:2]))

        return x0

    def generate_parameters(self, x0_in, x0_speed):
        p = np.zeros(self.nx + 2 * self.N + 1)
        p[:self.nx] = x0_in

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s_progress = self.X0[k, 3]
            
            right_x = self.rp.right_lut_x(s_progress).full()[0, 0]
            right_y = self.rp.right_lut_y(s_progress).full()[0, 0]
            left_x = self.rp.left_lut_x(s_progress).full()[0, 0]
            left_y = self.rp.left_lut_y(s_progress).full()[0, 0]

            delta_x = right_x - left_x
            delta_y = right_y - left_y

            self.lbg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = min(-delta_x * right_x - delta_y * right_y,
                                    -delta_x * left_x - delta_y * left_y)
            self.ubg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = max(-delta_x * right_x - delta_y * right_y,
                                    -delta_x * left_x - delta_y * left_y)
            
            

            p[self.nx + 2 * k:self.nx + 2 * k + 2] = [-delta_x, delta_y]
            p[-1] = max(x0_speed, 1) # prevent constraint violation
            # p[-1] = x0_speed
        self.lbg[self.nx *2, 0] = - ca.inf
        self.ubg[self.nx *2, 0] = ca.inf
        for k in range(self.N):
            self.lbg[self.nx * (self.N + 1) + self.N + k] = -self.fmax
            self.ubg[self.nx * (self.N + 1) + self.N + k] = self.fmax


        return p

    def solve(self, p):

        x_init = ca.vertcat(ca.reshape(self.X0.T, self.nx * (self.N + 1), 1),
                         ca.reshape(self.u0.T, self.nu * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        self.X0 = ca.reshape(sol['x'][0:self.nx * (self.N + 1)], self.nx, self.N + 1).T
        controls = ca.reshape(sol['x'][self.nx * (self.N + 1):], self.nu, self.N).T

        # print(controls[0, 0], controls[0,1])


        # if self.solver.stats()['return_status'] != 'Solve_Succeeded':
        #     print("Solve failed!!!!!")

        return controls.full(), self.X0
        
    def construct_warm_start_soln(self, initial_state):
        if not self.warm_start: return
        # self.warm_start = False

        self.X0 = np.zeros((self.N + 1, self.nx))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.p_init * self.dt

            psi_next = self.rp.angle_lut_t(s_next).full()[0, 0]
            x_next, y_next = self.rp.center_lut_x(s_next), self.rp.center_lut_y(s_next)

            # adjusts the centerline angle to be continuous
            psi_diff = self.X0[k-1, 2] - psi_next
            psi_mul = self.X0[k-1, 2] * psi_next
            if (abs(psi_diff) > np.pi and psi_mul < 0) or abs(psi_diff) > np.pi*1.5:
                if psi_diff > 0:
                    psi_next += np.pi * 2
                else:
                    psi_next -= np.pi * 2
            self.X0[k, :] = np.array([x_next.full()[0, 0], y_next.full()[0, 0], psi_next, s_next])

        # self.realTimePlot(self.x_bar, self.X0)
        return self.X0

    def inputStateAdust(self,x0, speed):

        X0_slip = [x0[0],x0[1],x0[2], speed]

        return x0, X0_slip


    def normalise_psi(self,psi):
        while psi > np.pi:
            psi -= 2*np.pi
        while psi < -np.pi:
            psi += 2*np.pi
        return psi




class dataSave:
    def __init__(self, TESTMODE, map_name,max_iter):
        self.rowSize = 10000
        self.stateCounter = 0
        self.lapInfoCounter = 0
        self.opt_counter = 0
        self.TESTMODE = TESTMODE
        self.map_name = map_name
        self.max_iter = max_iter
        self.txt_x0 = np.zeros((self.rowSize,10))
        self.txt_lapInfo = np.zeros((max_iter,8))
        self.txt_opt = np.zeros((self.rowSize,51))


    def saveStates(self, time, x0, expected_speed, tracking_error, noise, completion, steering, slip_angle):
        self.txt_x0[self.stateCounter,0] = time
        self.txt_x0[self.stateCounter,1:4] = [x0[0],x0[1],x0[3]]
        self.txt_x0[self.stateCounter,4] = expected_speed
        self.txt_x0[self.stateCounter,5] = tracking_error
        self.txt_x0[self.stateCounter,6] = noise
        self.txt_x0[self.stateCounter,7] = completion
        self.txt_x0[self.stateCounter,8] = steering
        self.txt_x0[self.stateCounter,9] = slip_angle
        self.stateCounter += 1
        #time, x_pos, y_pos, actual_speed, expected_speed, tracking_error, noise, completion, steering, slip_angle

    def saveOptimisation(self, x0, x0_solution, x0_ref):
        #x0 nx, x0_solution nx*(N+1), x0_ref 2*N+1
        self.txt_opt[self.opt_counter, 0:3] = [x0[0], x0[1], x0[3]]
        self.txt_opt[self.opt_counter, 3:27] = x0_solution.reshape((1,24))
        self.txt_opt[self.opt_counter, 27:51] = x0_ref.reshape((1,24))
        self.opt_counter += 1

    def savefile(self, iter):
        for i in range(self.rowSize):
            if (self.txt_x0[i+2,0] == 0):
                self.txt_x0 = np.delete(self.txt_x0, slice(i+2,self.rowSize),axis=0)
                break
        np.savetxt(f"Imgs/MPCC_{self.map_name}_{self.TESTMODE}_{str(iter)}_ros.csv", self.txt_x0, delimiter = ',', header="laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error", fmt="%-10f")

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
            if (self.txt_opt[i,4] == 0):
                self.txt_opt = np.delete(self.txt_opt, slice(i,self.rowSize),axis=0)
                break
        np.savetxt(f"csv/MPCC_sol_{self.map_name}_rviz.csv", self.txt_opt,delimiter=',',header = f"x0, x_bar, x_ref", fmt="%-10f")
        
        np.savetxt(f"csv/MPCC_{self.map_name}_{self.TESTMODE}.csv", self.txt_lapInfo,delimiter=',',header = f"lap_count, lap_success, laptime, completion, {var1}, {var2}, aveTrackErr, Computation_time", fmt="%-10f")



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
    MPCC_controller_node = MPCCControllerNode()

    rclpy.spin(MPCC_controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node


if __name__ == "__main__":
    main()

#_________________________________________________________________________________________________________________________
    
