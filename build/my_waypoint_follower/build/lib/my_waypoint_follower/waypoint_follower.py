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

class PoseSubscriberNode (Node):
    def __init__(self, wb, speedgain):
        super().__init__("waypoint_follower")
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.x = 0.0
        self.y = 0.0
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.ow = 0.0

        self.load_waypoints()

        self.wheelbase = wb                 #vehicle wheelbase                           
        self.max_reacquire = 20.
        self.speedgain = speedgain          
        self.drawn_waypoints = []
        self.ego_index = None
        self.Tindx = None
        self.v_gain = 0.25                  
        self.lfd = 0.3                #lood forward distance
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
    
    def callback(self, msg: Odometry):
        cmd = AckermannDriveStamped()
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.ox = msg.pose.pose.orientation.x
        self.oy = msg.pose.pose.orientation.y
        self.oz = msg.pose.pose.orientation.z
        self.ow = msg.pose.pose.orientation.w

        self.search_nearest_target()
        self.positionMessage()
        speed,steering_angle= self.action()
        
        cmd.drive.speed = speed*self.speedgain
        cmd.drive.steering_angle = steering_angle
        self.get_logger().info("current ind = " + str(self.ego_index) 
                               + " current yaw = " + str(self.yaw)
                               + " tar_ind = " + str(self.Tindx)
                               + "cur_x = " + str(self.x)
                               + "cur_y = " + str(self.y)
                               + " tar_x = " + str(self.points[self.ego_index][0])
                               + "tar_y = " + str(self.points[self.ego_index][1]))
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
    speedgain = 1.

    rclpy.init(args = args)
    controller_node = PoseSubscriberNode(0.324, speedgain)
    # publish_node = PurePursuitPlanner(0.3,speedgain,0,0,0)
    current_x,current_y,current_roll, current_pitch, current_yaw = controller_node.positionMessage()


    rclpy.spin(controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node


    
