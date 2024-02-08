#! /usr/bin/env python3
#import necessary classes
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import math
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import time

class PoseSubscriberNode(Node):
    def __init__(self):
        mapname_list = ["gbr", "mco", "esp", "CornerHallE", "f1_aut_wide", "levine_blocked"] 
        mapname = mapname_list[3]
        max_iter = 3
        self.speedgain = 1.

        self.planner = PurePursuit(mapname, speedgain=self.speedgain)
        self.ds = dataSave("ros_rviz", mapname, max_iter)

        super().__init__("waypoint_follower")

        #Subscribes to the current position and speed of the car
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.callback, 10)
        #Publishes the drive commands to the actuator
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        #Publishes the position of the car to reset the position of the vehicle
        self.ego_reset_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.is_start = None
        self.x0 = [0.0] * 4      #x_pos, y_pos, yaw, speed  

        self.iter = 0
        self.cmd_start_timer = time.perf_counter()

    def callback(self, msg:Odometry):
        if self.is_start == None:
            # self.ego_reset()
            self.is_start = 1
            self.is_in = 0
            self.start_laptime = time.time() #come back to this put this somewhere else

        lapsuccess = 0 if self.planner.completion<99 else 1
        laptime = time.time() - self.start_laptime
        self.cmd_current_timer = time.perf_counter()

        cmd = AckermannDriveStamped()

        quat_ori = [msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w]
        
        yaw = self.euler_from_quaternion(quat_ori[0], quat_ori[1], quat_ori[2], quat_ori[3])

        self.x0 = [msg.pose.pose.position.x,
                   msg.pose.pose.position.y,
                   yaw,
                   msg.twist.twist.linear.x]
        
        indx, trackErr, speed, steering = self.planner.action(self.x0)
        cmd.drive.speed = speed*self.speedgain
        cmd.drive.steering_angle = steering
        # self.drive_pub.publish(cmd)

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
                self.get_logger().info("i published")
                self.cmd_start_timer = self.cmd_current_timer
        # self.get_logger().info("current ind = " + str(indx) 
        #                        + " current yaw = " + str(self.x0[2])
        #                        + " tar_ind = " + str(self.planner.Tindx)
        #                        + "cur_x = " + str(self.x0[0])
        #                        + "cur_y = " + str(self.x0[1])
        #                        + " tar_x = " + str(self.planner.points[indx][0])
        #                        + "tar_y = " + str(self.planner.points[indx][1]))


        self.ds.saveStates(laptime, self.x0, self.planner.speed_list[indx], trackErr, 0, self.planner.completion)

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

    def euler_from_quaternion(self,x, y, z, w):  
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z # in radians

class PurePursuit():
    def __init__(self, mapname, wb = 0.324, speedgain = 1.):
        self.waypoints = np.loadtxt('maps/' + mapname + '_raceline.csv', delimiter=',')
        self.points = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T
        self.completion = 0.0

        self.wheelbase = wb                 #vehicle wheelbase                           
        self.speedgain = speedgain  
        self.ego_index = None
        self.Tindx = None

        self.v_gain = 0.07                 #change this parameter for different tracks 
        self.lfd = 0.3  

    def distanceCalc(self,x, y, tx, ty):     #tx = target x, ty = target y
        dx = tx - x
        dy = ty - y
        return np.hypot(dx, dy)
        
    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        seg_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)
        # print(len(self.ss))
        if idx+1 >= len(self.ss):
            idxadd1 = 0
        else: 
            idxadd1 = idx +1
        d_ss = self.ss[idxadd1] - self.ss[idx]

        d1, d2 = math.dist(self.points[idx],self.poses),math.dist(self.points[idxadd1],self.poses)

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

    def search_nearest_target(self,x0):

        self.poses = [x0[0],x0[1]]
        self.min_dist = np.linalg.norm(self.poses - self.points,axis = 1)
        self.ego_index = np.argmin(self.min_dist)
        if self.Tindx is None:
            self.Tindx = self.ego_index
        
        self.speed_list = self.waypoints[:, 5]
        self.speed = self.speed_list[self.ego_index]

        self.Lf = self.speed*self.v_gain + self.lfd  # update look ahead distance
        
        # search look ahead target point index
        while self.Lf > self.distanceCalc(x0[0],
                                            x0[1], 
                                            self.points[self.Tindx][0], 
                                            self.points[self.Tindx][1]):

            if self.Tindx + 1 > len(self.points)-1:
                self.Tindx = 0
            else:
                self.Tindx += 1
        
        return self.min_dist, self.ego_index

    def action(self, x0):
        min_dist, indx = self.search_nearest_target(x0)
        _, trackErr = self.interp_pts(indx, min_dist)

        waypoint = np.dot (np.array([np.sin(-x0[2]),np.cos(-x0[2])]),
                           self.points[self.Tindx]-np.array([x0[0], x0[1]]))   
        
        if np.abs(waypoint) < 1e-6:
            return self.speed, 0.
        radius = 1/(2.0*waypoint/self.Lf**2)
        steering_angle = np.arctan(self.wheelbase/radius)
        self.completion = round(self.ego_index/len(self.points)*100,2)

        return indx, trackErr, self.speed, steering_angle

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
        np.savetxt(f"Imgs/{self.map_name}_{self.TESTMODE}_{str(iter)}.csv", self.txt_x0, delimiter = ',', header="laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error", fmt="%-10f")

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
        np.savetxt(f"csv/PP_{self.map_name}_{self.TESTMODE}.csv", self.txt_lapInfo,delimiter=',',header = f"lap_count, lap_success, laptime, completion, {var1}, {var2}, aveTrackErr, Computation_time", fmt="%-10f")

    
def main(args = None):
    
    #use - ros2 run teleop_twist_keyboard teleop_twist_keyboard 
    #to move the car manually around the map
    rclpy.init(args = args)
    controller_node = PoseSubscriberNode()
 
    rclpy.spin(controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node


if __name__ == "__main__":
    main()