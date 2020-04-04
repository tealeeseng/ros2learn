import gym
gym.logger.set_level(40) # hide warnings
import time
import numpy as np
import copy
import math
import os
import psutil
import signal
import sys
from scipy.stats import skew
from gym import utils, spaces
from gym_gazebo2.utils import ut_generic, ut_launch, ut_mara, ut_math, ut_gazebo, tree_urdf, general_utils
from gym.utils import seeding
from gazebo_msgs.srv import SpawnEntity
import subprocess
import argparse
import transforms3d as tf3d

# ROS 2
import rclpy
import copy

from rclpy.node import Node

from rclpy.qos import QoSProfile, qos_profile_sensor_data
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing mara joint angles.
from control_msgs.msg import JointTrajectoryControllerState
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.msg import ContactState, ModelState
from std_msgs.msg import String
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from ros2pkg.api import get_prefix_path
from builtin_interfaces.msg import Duration

JOINT_SUBSCRIBER = '/mara_controller/state'
JOINT_PUBLISHER = '/mara_controller/command'

# joint names:
MOTOR1_JOINT = 'motor1'
MOTOR2_JOINT = 'motor2'
MOTOR3_JOINT = 'motor3'
MOTOR4_JOINT = 'motor4'
MOTOR5_JOINT = 'motor5'
MOTOR6_JOINT = 'motor6'
EE_LINK = 'ee_link'

JOINT_ORDER = [MOTOR1_JOINT,MOTOR2_JOINT, MOTOR3_JOINT,
                MOTOR4_JOINT, MOTOR5_JOINT, MOTOR6_JOINT]



class JointsSubscriber(Node):
    def __init__(self):
        super().__init__('JointsSubscriber')
        self.subcription = self.create_subscription(JointTrajectoryControllerState, JOINT_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self.pub = self.create_publisher(JointTrajectory, JOINT_PUBLISHER, qos_profile=qos_profile_sensor_data)



    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        observation_msg = message
        self.get_logger().info(str(observation_msg))

    def stretch(self, joints):
        m_jointOrder = copy.deepcopy(JOINT_ORDER)

        self.pub.publish(ut_mara.getTrajectoryMessage(
            joints,
            m_jointOrder,
            0.3))


def main(args=None):
    rclpy.init(args=args)
    joint_sub = JointsSubscriber()
    rclpy.spin_once(joint_sub)
    joint_sub.stretch(np.array([-1.5,0.75,-.5,0,-1.5,0]))
    rclpy.spin_once(joint_sub)
    

    joint_sub.destroy_node()
    rclpy.shutdown()

    print('END recycler_package.')


def main_():

    print('test')

    urdf = "reinforcement_learning/mara_robot_gripper_140_camera_train.urdf"
    urdfPath = get_prefix_path("mara_description") + "/share/mara_description/urdf/" + urdf

    # Set constants for links
    WORLD = 'world'
    BASE = 'base_robot'
    MARA_MOTOR1_LINK = 'motor1_link'
    MARA_MOTOR2_LINK = 'motor2_link'
    MARA_MOTOR3_LINK = 'motor3_link'
    MARA_MOTOR4_LINK = 'motor4_link'
    MARA_MOTOR5_LINK = 'motor5_link'
    MARA_MOTOR6_LINK = 'motor6_link'
    EE_LINK = 'ee_link'

    LINK_NAMES = [ WORLD, BASE,
                MARA_MOTOR1_LINK, MARA_MOTOR2_LINK,
                MARA_MOTOR3_LINK, MARA_MOTOR4_LINK,
                MARA_MOTOR5_LINK, MARA_MOTOR6_LINK, EE_LINK]
                
    m_linkNames = copy.deepcopy(LINK_NAMES)
 
    _, ur_tree = tree_urdf.treeFromFile(urdfPath)
        # Retrieve a chain structure between the base and the start of the end effector.
    mara_chain = ur_tree.getChain(m_linkNames[0], m_linkNames[-1])
    pos = [ -0.5 , 0.2 , 0.1 ]
    rot = np.array([[0,0,0],[0,0,0],[0,0,0]])

    #     pos,  [[-1.05369001e-03  1.92728881e-05  1.11664069e+00]] rot: [[ 1.00000000e+00 -4.47419942e-06 -2.23629828e-11]
    #  [ 4.47419942e-06  1.00000000e+00  2.87694329e-05]
    #  [-1.06357197e-10 -2.87694329e-05  1.00000000e+00]]
    print(rot[0,1])
    print(general_utils.inverseKinematics(mara_chain, pos, rot))

    # def inverseKinematics(robotChain, pos, rot, qGuess=None, minJoints=None, maxJoints=None):



if __name__ == '__main__':
    main()
