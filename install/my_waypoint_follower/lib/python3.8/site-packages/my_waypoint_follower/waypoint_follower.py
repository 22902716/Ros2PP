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
        self.v_gain = 0.12                 #change this parameter for different tracks 
        self.lfd = 0.1                     #lood forward distance constant
        self.yaw = 0.0
        self.speed = 0.0
        self.iteration_no = -1
        self.csv_lap = []
        self.csv_end = []
        self.start_laptime = time.time()
        self.act_x = []
        self.act_y = []
        self.ref_x = [] 
        self.ref_y = []
        self.act_v = []
        self.ref_v = []
        self.trackErr_list = []
        self.lap_time = []



    
    def callback(self, msg: Odometry):
        if self.is_start == None:
            # self.ego_reset()
            self.is_start = 1
            self.is_in = 0
        

        # if self.x < -0.01 and self.x > -0.1 and self.y < 1.0 and self.y > -1.0 and self.iteration_no < self.Max_iter and self.is_in == 0:
        #     self.ego_reset()
        #     self.is_in = 1
        # else:
        #     self.is_in = 0

        cmd = AckermannDriveStamped()
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.ox = msg.pose.pose.orientation.x
        self.oy = msg.pose.pose.orientation.y
        self.oz = msg.pose.pose.orientation.z
        self.ow = msg.pose.pose.orientation.w
        self.lin_vel= msg.twist.twist.linear.x

        self.search_nearest_target()
        self.poses = [self.x,self.y]
        current_pose = self.positionMessage()
        speed,steering_angle= self.action()
        _,trackErr = self.interp_pts(self.ego_index,self.min_dist)

        self.act_x.append(self.x)
        self.act_y.append(self.y)
        self.ref_x.append(self.points[self.ego_index][0])
        self.ref_y.append(self.points[self.ego_index][1])
        self.act_v.append(self.lin_vel)
        self.ref_v.append(self.speed_list[self.ego_index])
        self.trackErr_list.append(trackErr)
        self.lap_time.append(time.time() - self.start_laptime)
         

        self.get_logger().info(str(self.completion))

        if self.completion >= 90:
            self.get_logger().info("I finished running the lap")
            cmd.drive.speed = 0.0
            cmd.drive.steering_angle = steering_angle

            self.act_x = np.array(self.act_x)
            self.act_y = np.array(self.act_y)
            self.ref_x = np.array(self.ref_x)
            self.ref_y = np.array(self.ref_y)
            self.act_v = np.array(self.act_v)
            self.ref_v = np.array(self.ref_v)
            self.trackErr_list = np.array(self.trackErr_list)
            self.lap_time = np.array(self.lap_time)

            save_arr = np.concatenate([self.lap_time[:,None]
                                       ,self.act_x[:,None]
                                       ,self.act_y[:,None]
                                       ,self.ref_x[:,None]
                                       ,self.ref_y[:,None]
                                       ,self.act_v[:,None]
                                       ,self.ref_v[:,None]
                                       ,self.trackErr_list[:,None]
                                       ],axis = 1)
            self.csv_lap = np.array(self.csv_lap)
            np.savetxt("csv/"+self.mapname+'/' +'ROS_1.csv', save_arr, delimiter=',',header="laptime,x,y,x_ref,y_ref,speed,speed_ref,Tracking Error",fmt="%-10f")
            self.drive_pub.publish(cmd)
            self.ego_reset()
            time.sleep(100)
            
        else:
            cmd.drive.speed = speed*self.speedgain
            cmd.drive.steering_angle = steering_angle



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

    
        np.savetxt("csv/"+self.mapname+'/' +'.csv', self.csv_lap, delimiter=',',header="x,y,x_ref,y_ref,speed,speed_ref",fmt="%-10f")

        self.csv = []
        self.pose = []
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
        self.mapname = 'CornerHallE'
        self.waypoints = np.loadtxt(self.mapname + '_raceline.csv', delimiter=',')
        self.points = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T

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

    def search_nearest_target(self):

        self.speed_list = self.waypoints[:, 5]
        poses = [self.x, self.y]
        self.min_dist = np.linalg.norm(poses - self.points,axis = 1)
        self.ego_index = np.argmin(self.min_dist)
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
        self.completion = round(self.ego_index/len(self.points)*100,2)



        return self.speed, steering_angle
    




def main(args = None):
    
    #use - ros2 run teleop_twist_keyboard teleop_twist_keyboard 
    #to move the car manually around the map
    rclpy.init(args = args)
    controller_node = PoseSubscriberNode()
 
    rclpy.spin(controller_node)          #allows the node to always been running 
    rclpy.shutdown()                    #shut dowwn the node


    
