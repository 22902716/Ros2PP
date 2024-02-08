#! /usr/bin/env python3
#import necessary classes
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from warnings import WarningMessage

class PoseSubscriberNode (Node):
	#def __init__(self,x,y,z,ox,oy,oz):
	def __init__(self):
		super().__init__("pose_subscriber")
		# WarningMessage("AM HERE")
		self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.callback, 10)
		# self.x = x
		# self.y = y
		# self.z = z
		# self.ox = ox
		# self.oy = oy
		# self.oz = oz
	
	def callback(self, msg: Odometry):

		# self.x = msg.pose.pose.position.x
		# self.y = msg.pose.pose.position.y
		# self.z = msg.pose.pose.position.z
		# self.ox = msg.pose.pose.orientation.x
		# self.oy = msg.pose.pose.orientation.y
		# self.oz = msg.pose.pose.orientation.z
		# self.get_logger().info("pose_x = " + str(self.x) 
		# 	 + " pose_y = " + str(self.y) 
		# 	 + " orientation_z = " + str(self.oz))
		self.get_logger().info("pose_x = " + str(msg.pose.pose.position.x) 
			 + " pose_y = " + str(msg.pose.pose.position.y) 
			 + " orientation_z = " + str(msg.pose.pose.orientation.z))		




def main(args = None):
	
	#use ros2 run teleop_twist_keyboard teleop_twist_keyboard 
	#to move the car manually around the map

	rclpy.init(args = args)
	# node = PoseSubscriberNode(0,0,0,0,0,0)
	node = PoseSubscriberNode()
	# position = [node.x, node.y, node.z]
	# orientation = [node.ox, node.oy, node.oz]
	rclpy.spin(node)                    #allows the node to always been running 
	rclpy.shutdown()                    #shut dowwn the node


	
