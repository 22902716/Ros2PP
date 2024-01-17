#! /usr/bin/env python3
#import necessary classes
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from warnings import WarningMessage
import numpy as np
import yaml
import math
from argparse import Namespace
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import time
import DataSave

class PoseSubscriberNode(Node):
    def __init__(self):
        # mapname = 'CornerHall'
        # mapname = 'f1_aut_wide'
        mapname = 'levine_blocked'
        max_iter = 3

        self.planner = PurePursuit(mapname)
        self.ds = DataSave("ros_rviz", self.mapname, max_iter)

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

    def callback(self, msg:Odometry):
        if self.is_start == None:
            # self.ego_reset()
            self.is_start = 1
            self.is_in = 0
            self.start_laptime = time.time() #come back to this put this somewhere else

        lapsuccess = 0 if self.planner.completion<95 else 1
        laptime = time.time() - self.start_laptime

        if self.planner.completion >= 95:
            
            self.ds.lapInfo(self.iter, lapsuccess, laptime, self.planner.completion, 0, 0, laptime)
            self.ds.savefile(self.iter)
            self.ego_reset_stop()

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
        
        min_dist, indx = self.planner.search_nearest_target(self.x0)
        speed, steering = self.planner.action()
        _, trackErr = self.planner.interp_pts(indx, min_dist)

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
        self.iteration_no += 1

        self.get_logger().info("Finished Resetting Vehicle now at lap " + str(self.iteration_no))







        
    def euler_from_quaternion(self,x, y, z, w):  
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        # t0 = +2.0 * (w * x + y * z)
        # t1 = +1.0 - 2.0 * (x * x + y * y)
        # roll_x = math.atan2(t0, t1)
     
        # t2 = +2.0 * (w * y - z * x)
        # t2 = +1.0 if t2 > +1.0 else t2
        # t2 = -1.0 if t2 < -1.0 else t2
        # pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        # return roll_x, pitch_y, yaw_z # in radians
        return yaw_z # in radians

class PurePursuit():
    def __init__(self, mapname, wb = 0.324, speedgain = 1.):
        self.waypoints = np.loadtxt('maps/' + mapname + '_raceline.csv', delimiter=',')
        self.points = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T

        self.wheelbase = wb                 #vehicle wheelbase                           
        self.speedgain = speedgain  
        self.ego_index = None
        self.Tindx = None

        self.v_gain = 0.12                 #change this parameter for different tracks 
        self.lfd = 0.1  

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

        poses = [x0[0],x0[1]]
        self.min_dist = np.linalg.norm(poses - self.points,axis = 1)
        self.ego_index = np.argmin(self.min_dist)
        if self.Tindx is None:
            self.Tindx = self.ego_index
        
        self.speed_list = self.waypoints[:, 5]
        self.speed = self.speed_list[self.ego_index]

        self.Lf = self.speed*self.v_gain + self.lfd  # update look ahead distance
        
        # search look ahead target point index
        while self.Lf > self.distanceCalc(self.x,
                                            self.y, 
                                            self.points[self.Tindx][0], 
                                            self.points[self.Tindx][1]):

            if self.Tindx + 1 > len(self.points)-1:
                self.Tindx = 0
            else:
                self.Tindx += 1
        
        return self.min_dist, self.ego_index

    def action(self):
        waypoint = np.dot (np.array([np.sin(-self.yaw),np.cos(-self.yaw)]),
                           self.points[self.Tindx]-np.array([self.x, self.y]))   
        
        if np.abs(waypoint) < 1e-6:
            return self.speed, 0.
        radius = 1/(2.0*waypoint/self.Lf**2)
        steering_angle = np.arctan(self.wheelbase/radius)
        self.completion = round(self.ego_index/len(self.points)*100,2)

        return self.speed, steering_angle
    
def main(args = None):
    
    #use - ros2 run teleop_twist_keyboard teleop_twist_keyboard 
    #to move the car manually around the map
    rclpy.init(args = args)
    controller_node = PoseSubscriberNode()
 
    rclpy.spin(controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node


    