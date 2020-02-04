#!/usr/bin/python3
#ROS Libraries
import rospy
import roslib
#ROS Messages
from edo_core_msgs.msg import MovementCommand
from edo_core_msgs.msg import CartesianPose
from edo_core_msgs.msg import JointStateArray
from edo_core_msgs.msg import MovementFeedback
#Python Libraries
import sys, time, queue
#numpy
import numpy as np
#Manipulation Class
class Manipulation(object):
    #init
    def __init__(self):
        #Parameters
        self.x = 0
        self.y = 0
        self.z = 0
        self.a = 0
        self.e = 0
        self.r = 0
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 45.0, 80, 0.0, 0.0, 0.0]
        self.pickHeight = 100.0
        self.grabHeight = 34.0	
        self.pickWidth = 52.0
        self.grabWidth = 18.0
        self.moveQueue = queue.Queue()
        self.mf = MovementFeedback()
        #Node Cycle rate
        self.loop_rate = rospy.Rate(100)
        #Publishers
        self.pub = rospy.Publisher('/bridge_move', MovementCommand, queue_size = 10) 
        #Subscribers
        self.subCart = rospy.Subscriber('/cartesian_pose', CartesianPose, self.cartCallback)
        self.subJoint = rospy.Subscriber('/usb_jnt_state', JointStateArray, self.jntCallback)   # subscribes to new joint angles
        self.subAck = rospy.Subscriber('/machine_movement_ack', MovementFeedback, self.ackCallback)

        rospy.sleep(0.5)
#functions

    #Creates a Cartesian Move
    def createMove(self, type):
        msg = MovementCommand()
        msg.move_command = 77
        msg.move_type = 74
        msg.ovr = 100
        msg.delay = 0
        if type == "joint":
            msg.target.data_type = 74
        elif type == "cart":
            msg.target.data_type = 88
        else: 
            msg.target.data_type = 74
        msg.target.joints_mask = 127
        return msg

    #Callback function for /cartesian_pose subscriber 
    def cartCallback(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z
        self.a = msg.a
        self.e = msg.e
        self.r = msg.r

    #Prints Cartesian Position (debug)
    def printCartPos(self):
        print("x = ", self.x)
        print("y = ", self.y)
        print("z = ", self.z)
        print("a = ", self.a)
        print("e = ", self.e)
        print("r = ", self.r)

    #Callback function for receiving new joint angles
    def jntCallback(self, msg):
        for x in range(7):
           self.joints[x] = msg.joints[x].position

    #Prints Joint Position (debug)
    def printJntPos(self):
        print(self.joints)

    #Callback function for /machine_movement_ack subscriber
    def ackCallback(self, msg):
        self.mf = msg

    #Prints Latest MovementFeedback Message recieved
    def printMF(self, msg):
        rospy.loginfo(msg)

    #Processes moves in the MoveQueue
    def processQueue(self):
        while not(self.moveQueue.empty()):
            self.mf = MovementFeedback()
            self.pub.publish(self.moveQueue.get_nowait())
            timeout = time.time() + 30 #30 Second timeout for awaiting a MovementAck
            while self.mf.type != 2:
                time.sleep(0.01)    
                if time.time() > timeout: #If it reaches 30 seconds, it moves on to the next message
                    break

    #This function moves the robot to a joint destination (Its default is "home")
    def jointMove(self, jointData = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], debug = False ):
        msg = self.createMove("joint")
        msg.target.joints_data = jointData
        self.moveQueue.put_nowait(msg)
        self.processQueue()
        if debug:
            rospy.loginfo(msg)

    #This function moves the robot to a cartesian destination (Its default is "pick")
    def cartMove(self, x = 400.5, y = 0.05, z = 120, a = 0, e = 180, r = -1.37, gripper = 18, debug = False):
        msg = self.createMove("cart")
        msg.target.joints_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper, 0.0, 0.0, 0.0]
        msg.target.cartesian_data.x = x
        msg.target.cartesian_data.y = y
        msg.target.cartesian_data.z = z
        msg.target.cartesian_data.a = a
        msg.target.cartesian_data.e = e
        msg.target.cartesian_data.r = r
        self.moveQueue.put_nowait(msg)
        self.processQueue()
        if debug:
            rospy.loginfo(msg)

    #Changes gripper width (defaults to "closed")
    def setGripper(self, width = 0):
        rospy.sleep(0.2)
        self.joints[6] = width
        self.jointMove(self.joints, False)
