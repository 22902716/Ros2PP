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
import matplotlib.pyplot as plt
import time


TESTMODE = "Benchmark"
# TESTMODE = "v_gain"
# TESTMODE = "lfd"
# TESTMODE = "localnoise"
# TESTMODE = "Outputnoise"
# TESTMODE = "control_delay"
# TESTMODE = "perception_delay"
# TESTMODE = " "


class PoseSubscriberNode (Node):
    def __init__(self, wb = 0.324, speedgain = 1.):
        super().__init__("waypoint_follower")
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.ego_reset_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.x = 0.0
        self.y = 0.0
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.ow = 0.0
        self.is_start = None

        self.load_waypoints()

        self.wheelbase = wb                 #vehicle wheelbase                           
        self.speedgain = speedgain          
        self.drawn_waypoints = []
        self.ego_index = None
        self.Tindx = None
        self.yaw = 0.0
        self.speed = 0.0
        self.iteration_no = -1
        self.csv = []

        if TESTMODE == "Benchmark" or TESTMODE == " ":
            self.v_gain = 0.2                 #change this parameter for different tracks 
            self.lfd = 0.2                     #lood forward distance constant
            self.Max_iter = 5
        elif TESTMODE == "localnoise" or TESTMODE == "Outputnoise":
            self.v_gain = 0.2                 #change this parameter for different tracks 
            self.lfd = 0.2                     #lood forward distance constant
            self.Max_iter = 50
        elif TESTMODE == "v_gain":
            self.v_gain = 0.0                 #change this parameter for different tracks 
            self.lfd = 0.2                     #lood forward distance constant
            self.Max_iter = 50
        elif TESTMODE == "lfd":
            self.v_gain = 0.2                 #change this parameter for different tracks 
            self.lfd = 0.0                     #lood forward distance constant
            self.Max_iter = 50
        elif TESTMODE == "control_delay" or TESTMODE == "perception_delay":
            self.v_gain = 0.2                 #change this parameter for different tracks 
            self.lfd = 0.2                     #lood forward distance constant
            self.Max_iter = 10
    
    def callback(self, msg: Odometry):
        if self.is_start == None:
            self.ego_reset()
            self.is_start = 1
            self.is_in = 0
        

        if self.x < -0.01 and self.x > -0.1 and self.y < 1.0 and self.y > -1.0 and self.iteration_no < self.Max_iter and self.is_in == 0:
            self.ego_reset()
            self.is_in = 1
        else:
            self.is_in = 0

        cmd = AckermannDriveStamped()
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.ox = msg.pose.pose.orientation.x
        self.oy = msg.pose.pose.orientation.y
        self.oz = msg.pose.pose.orientation.z
        self.ow = msg.pose.pose.orientation.w
        self.lin_vel= msg.twist.twist.linear.x

        self.search_nearest_target()
        current_pose = self.positionMessage()
        speed,steering_angle= self.action()
        self.csv.append(current_pose)

        cmd.drive.speed = speed*self.speedgain
        cmd.drive.steering_angle = steering_angle
        if self.iteration_no > self.Max_iter:
            self.destroy_node()


        # self.get_logger().info("current ind = " + str(self.ego_index) 
        #                        + " current yaw = " + str(self.yaw)
        #                        + " tar_ind = " + str(self.Tindx)
        #                        + "cur_x = " + str(self.x)
        #                        + "cur_y = " + str(self.y)
        #                        + " tar_x = " + str(self.points[self.ego_index][0])
        #                        + "tar_y = " + str(self.points[self.ego_index][1]))
        self.drive_pub.publish(cmd)



        # self.get_logger().info("pose_x = " + str(self.x) + " pose_y = " + str(self.y) + " orientation_z = " + str(self.yaw))


    def ego_reset(self):
        
        msg = PoseWithCovarianceStamped()
        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        
        self.drawn_waypoints = []
        self.ego_index = None
        self.Tindx = None

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0

        self.ego_reset_pub.publish(msg)
        time.sleep(0.3)
        self.search_nearest_target()


        cmd = AckermannDriveStamped()
        cmd.drive.speed = 0.
        cmd.drive.steering_angle = 0.
        self.drive_pub.publish(cmd)
        self.iteration_no += 1

        if TESTMODE == "localnoise" or TESTMODE == "Outputnoise":
            pass
        elif TESTMODE == "v_gain":
            self.v_gain += 0.01
        elif TESTMODE == "lfd":
            self.lfd += 0.05                     #lood forward distance constant
        elif TESTMODE == "control_delay" or TESTMODE == "perception_delay":
            pass
        np.savetxt("csv/"+self.mapname+'/' +TESTMODE+"/lap" + str(self.iteration_no) +'.csv', self.csv, delimiter=',',header="x,y,yaw,speed profile, actual speed",fmt="%-10f")

        self.csv = []
        self.get_logger().info("Finished Resetting Vehicle now at lap " + str(self.iteration_no))
        self.start_laptime = time.time()



    def positionMessage(self):
        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(self.ox,self.oy,self.oz,self.ow)
        return self.x, self.y, self.yaw, self.speed, self.lin_vel, time.time() - self.start_laptime

    
    
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
        self.mapname = 'f1_aut_wide_raceline'
        self.waypoints = np.loadtxt(self.mapname + '.csv', delimiter=',')
        self.points = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T

    def distanceCalc(self,x, y, tx, ty):     #tx = target x, ty = target y
        dx = tx - x
        dy = ty - y
        return np.hypot(dx, dy)
        

    def search_nearest_target(self):

        self.speed_list = self.waypoints[:, 5]
        poses = [self.x, self.y]
        min_dist = np.linalg.norm(poses - self.points,axis = 1)
        self.ego_index = np.argmin(min_dist)
        if self.Tindx is None:
            self.Tindx = self.ego_index
        

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

    def action(self):
        waypoint = np.dot (np.array([np.sin(-self.yaw),np.cos(-self.yaw)]),
                           self.points[self.Tindx]-np.array([self.x, self.y]))   
        
        if np.abs(waypoint) < 1e-6:
            return self.speed, 0.
        radius = 1/(2.0*waypoint/self.Lf**2)
        steering_angle = np.arctan(self.wheelbase/radius)

        return self.speed, steering_angle
    




def main(args = None):
    
    #use - ros2 run teleop_twist_keyboard teleop_twist_keyboard 
    #to move the car manually around the map
    rclpy.init(args = args)
    controller_node = PoseSubscriberNode()
 
    rclpy.spin(controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node


    
