from builtin_interfaces.msg import Duration
from ros2pkg.api import get_prefix_path
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from std_msgs.msg import String
from gazebo_msgs.msg import ContactState, ModelState
from gazebo_msgs.srv import DeleteEntity
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.node import Node
import rclpy
import transforms3d as tf3d
import argparse
import subprocess
from gazebo_msgs.srv import SpawnEntity
from gym.utils import seeding
from gym_gazebo2.utils import ut_generic, ut_launch, ut_mara, ut_math, ut_gazebo, tree_urdf, general_utils
from gym import utils, spaces
from scipy.stats import skew
import sys
import signal
import psutil
import os
import math
import copy
import numpy as np
import time
import gym
from PyKDL import ChainJntToJacSolver  # For KDL Jacobians
import pandas as pd

gym.logger.set_level(40)  # hide warnings


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

JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT,
               MOTOR4_JOINT, MOTOR5_JOINT, MOTOR6_JOINT]


class Robot(Node):
    def __init__(self):
        super().__init__('Robot')
        self.subcription = self.create_subscription(
            JointTrajectoryControllerState, JOINT_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self.pub = self.create_publisher(
            JointTrajectory, JOINT_PUBLISHER, qos_profile=qos_profile_sensor_data)
        
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')

        # delete entity
        self.delete_entity_cli = self.create_client(DeleteEntity, '/delete_entity')
        self.initArm()

    def initArm(self):
        print('test')

        urdf = "reinforcement_learning/mara_robot_gripper_140_camera_train.urdf"
        urdfPath = get_prefix_path("mara_description") + \
            "/share/mara_description/urdf/" + urdf

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

        LINK_NAMES = [WORLD, BASE,
                      MARA_MOTOR1_LINK, MARA_MOTOR2_LINK,
                      MARA_MOTOR3_LINK, MARA_MOTOR4_LINK,
                      MARA_MOTOR5_LINK, MARA_MOTOR6_LINK, EE_LINK]

        EE_POINTS = np.asmatrix([[0, 0, 0]])

        self.m_linkNames = copy.deepcopy(LINK_NAMES)
        self.ee_points = copy.deepcopy(EE_POINTS)
        self.m_jointOrder = copy.deepcopy(JOINT_ORDER)
        self.target_orientation = np.asarray([0., 0.7071068, 0.7071068, 0.]) # arrow looking down [w, x, y, z]


        _, self.ur_tree = tree_urdf.treeFromFile(urdfPath)
        # Retrieve a chain structure between the base and the start of the end effector.
        self.mara_chain = self.ur_tree.getChain(
            self.m_linkNames[0], self.m_linkNames[-1])
        self.numJoints = self.mara_chain.getNrOfJoints()
        # Initialize a KDL Jacobian solver from the chain.
        self.jacSolver = ChainJntToJacSolver(self.mara_chain)

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg = message
        # self.get_logger().info(str(observation_msg))

    def stretch(self, joints):
        # m_jointOrder = copy.deepcopy(JOINT_ORDER)

        self.pub.publish(ut_mara.getTrajectoryMessage(
            joints,
            self.m_jointOrder,
            0.3))

    def take_observation(self, targetPosition):
        """
        Take observation from the environment and return it.
        :return: state.
        """
        # # Take an observation
        rclpy.spin_once(self)
        self.ros_clock = rclpy.clock.Clock().now().nanoseconds

        obs_message = self._observation_msg

        # Check that the observation is not prior to the action
        while obs_message is None or int(str(obs_message.header.stamp.sec)+(str(obs_message.header.stamp.nanosec))) < self.ros_clock:
            rclpy.spin_once(self)
            obs_message = self._observation_msg

        # Collect the end effector points and velocities in cartesian coordinates for the processObservations state.
        # Collect the present joint angles and velocities from ROS for the state.
        agent = {'jointOrder': self.m_jointOrder}
        lastObservations = ut_mara.processObservations(obs_message, agent)
        # Set observation to None after it has been read.
        self._observation_msg = None

        # Get Jacobians from present joint angles and KDL trees
        # The Jacobians consist of a 6x6 matrix getting its from from
        # (joint angles) x (len[x, y, z] + len[roll, pitch, yaw])
        ee_link_jacobians = ut_mara.getJacobians(
            lastObservations, self.numJoints, self.jacSolver)
        if self.m_linkNames[-1] is None:
            print("End link is empty!!")
            return None
        else:
            translation, rot = general_utils.forwardKinematics(self.mara_chain,
                                                               self.m_linkNames,
                                                               lastObservations[:self.numJoints],
                                                               # use the base_robot coordinate system
                                                               baseLink=self.m_linkNames[0],
                                                               endLink=self.m_linkNames[-1])

            current_eePos_tgt = np.ndarray.flatten(
                general_utils.getEePoints(self.ee_points, translation, rot).T)
            eePos_points = current_eePos_tgt - targetPosition

            eeVelocities = ut_mara.getEePointsVelocities(
                ee_link_jacobians, self.ee_points, rot, lastObservations)

            # Concatenate the information that defines the robot state
            # vector, typically denoted asrobot_id 'x'.

            # state = np.r_[np.reshape(lastObservations, -1),
            #               np.reshape(eePos_points, -1),
            #               np.reshape(eeVelocities, -1),
            #               np.reshape(current_eePos_tgt,-1),
            #               np.reshape(rot.reshape(1, 9),-1)]

            #               #np.reshape(self.targetPosition,-1)]

            # return state

            # sample data.
            # translation, [[-0.67344571  0.00105318  0.3273965 ]]  rot, [[ 1.28011341e-06 -5.88501156e-01 -8.08496376e-01]
            #  [-1.00000000e+00 -3.34320541e-07 -1.33997557e-06]
            #  [ 5.18280224e-07  8.08496376e-01 -5.88501156e-01]]
            # [-0.67344571  0.00105318  0.3273965 ]

            # print('translation,',translation,' rot,',rot)

            return current_eePos_tgt

    def spawn_target(self, urdf_obj):
        self.targetPosition = self.sample_position()

        while not self.spawn_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/spawn_entity service not available, waiting again...')

        # modelXml = self.getTargetSdf()
        # modelXml = self.load_random_urdf(urdf_obj)
        modelXml = self.load_coke_can()
        pose = Pose()
        pose.position.x = self.targetPosition[0]
        pose.position.y = self.targetPosition[1]
        pose.position.z = self.targetPosition[2]
        pose.orientation.x = self.target_orientation[1]
        pose.orientation.y= self.target_orientation[2]
        pose.orientation.z = self.target_orientation[3]
        pose.orientation.w = self.target_orientation[0]


        #override previous spawn_request element.
        self.spawn_request = SpawnEntity.Request()
        self.spawn_request.name = "target_"+urdf_obj
        self.spawn_request.xml = modelXml
        self.spawn_request.robot_namespace = ""
        self.spawn_request.initial_pose = pose
        self.spawn_request.reference_frame = "world"

        #ROS2 Spawn Entity
        target_future = self.spawn_cli.call_async(self.spawn_request)
        rclpy.spin_until_future_complete(self, target_future)
        if target_future.result() is not None:
            print('response: %r' % target_future.result())

        return pose

    def load_coke_can(self):
        modelXml = """<?xml version='1.0'?>
                        <sdf version='1.6'>
                            <model name="can1">
                                <include>
                                <static>true</static>
                                <uri>model://coke_can</uri>
                                </include>
                                <gravity>1</gravity>
                            </model>
                            </sdf>"""
        return modelXml

    def getTargetSdf(self):
        modelXml = """<?xml version='1.0'?>
                        <sdf version='1.6'>
                        <model name='target'>
                            <link name='cylinder0'>
                            <pose frame=''>0 0 0 0 0 0</pose>
                            <inertial>
                                <pose frame=''>0 0 0 0 0 0</pose>
                                <mass>5</mass>
                                <inertia>
                                <ixx>1</ixx>
                                <ixy>0</ixy>
                                <ixz>0</ixz>
                                <iyy>1</iyy>
                                <iyz>0</iyz>
                                <izz>1</izz>
                                </inertia>
                            </inertial>
                            <gravity>1</gravity>
                            <velocity_decay/>
                            <self_collide>0</self_collide>
                            <enable_wind>0</enable_wind>
                            <kinematic>0</kinematic>
                            <visual name='cylinder0_visual'>
                                <pose frame=''>0 0 0 0 0 0</pose>
                                <geometry>
                                <sphere>
                                    <radius>0.01</radius>
                                </sphere>
                                </geometry>
                                <material>
                                <script>
                                    <name>Gazebo/Green</name>
                                    <uri>file://media/materials/scripts/gazebo.material</uri>
                                </script>
                                <shader type='pixel'/>
                                </material>
                                <transparency>0.1</transparency>
                                <cast_shadows>1</cast_shadows>
                            </visual>
                            </link>
                            <static>1</static>
                            <allow_auto_disable>1</allow_auto_disable>
                        </model>
                        </sdf>"""
        return modelXml

    def sample_position(self):
            # [ -0.5 , 0.2 , 0.1 ], [ -0.5 , -0.2 , 0.1 ] #sample data. initial 2 points in original setup.
        pos = [-1 * np.random.uniform(0,0.8), np.random.uniform(0,0.8), np.random.uniform(0.2,0.4)]
        print('object pos, ', pos)
        return pos
            # sample_x = np.random.uniform(0,1)

            # if sample > 0.5:
            #     return [ -0.8 , 0.0 , 0.1 ]
            # else:
            #     return [ -0.5 , 0.0 , 0.1 ]
    def load_random_urdf(self, obj):
        urdfPath = get_prefix_path("mara_description") + "/share/mara_description/random_urdfs/" + obj
        urdf_file = open(urdfPath,"r")
        urdf_string = urdf_file.read()
        print("urdf_string:", urdf_string)
        return urdf_string


def generate_joints_for_line(args=None):
    rclpy.init(args=args)
    robot = Robot()
    rclpy.spin_once(robot)

    STEP = 0.1
    data_frame = pd.DataFrame(columns=['m2', 'm3', 'm5', 'x', 'y', 'z'])

    # can we fix motor 1 4,6, to train free moving arm on a line?

    for m2 in np.arange(0, 0.2, STEP):
        for m3 in np.arange(0, -np.pi/2-0.1, -1*STEP):
            for m5 in np.arange(0, -np.pi/2-0.5, -1*STEP):

                # sample data, [-np.pi/2, 0.5, -.5, 0, -1.2, 0]
                robot.stretch(np.array([-np.pi/2, m2, m3, 0, m5, 0]))
                rclpy.spin_once(robot)
                current_eePos_tgt = robot.take_observation([0, 0, 0])
                rclpy.spin_once(robot)

                if 0 <= current_eePos_tgt[2] < 0.2:
                    data = [m2, m3, m5]
                    data.extend(current_eePos_tgt)
                    # print('data,', data)
                    
                    df = pd.Series(data, index=data_frame.columns)
                    # print('df,', df)
                    data_frame = data_frame.append(df, ignore_index=True)
                    robot.get_logger().info(str(data))

    joints_df = data_frame.read_csv(get_prefix_path("mara_description") + "/share/recycler_package/joints_xyz.csv")
    joints_df


    robot.stretch(np.array([-np.pi/2, 0, -np.pi/2-0.1, 0, -np.pi/2-0.5, 0]))
    rclpy.spin_once(robot)
    current_eePos_tgt = robot.take_observation([0, 0, 0])
    rclpy.spin_once(robot)
    print('current_eePos_tgt, ', current_eePos_tgt)


    robot.destroy_node()
    rclpy.shutdown()

    print('END generate_joints_for_line().')



def drop_coke_can(args=None):
    rclpy.init(args=args)
    robot = Robot()
    rclpy.spin_once(robot)

    obj = "coke0"
    pose = robot.spawn_target(obj)
    rclpy.spin_once(robot)

    return pose

    #     pos,  [[-1.05369001e-03  1.92728881e-05  1.11664069e+00]] rot: [[ 1.00000000e+00 -4.47419942e-06 -2.23629828e-11]
    #  [ 4.47419942e-06  1.00000000e+00  2.87694329e-05]
    #  [-1.06357197e-10 -2.87694329e-05  1.00000000e+00]]
    # print(rot[0, 1])
    # print(general_utils.inverseKinematics(mara_chain, pos, rot))  # can't understand rot and generate data for that.

    # def inverseKinematics(robotChain, pos, rot, qGuess=None, minJoints=None, maxJoints=None):

    # TODO: have to trial run first. Pseudo code alike. 
def grab_can_and_drop_delete_entity(args=None):
    rclpy.init(args=args)
    robot = Robot()
    rclpy.spin_once(robot)

    joints = load_joints()


    pose = drop_coke_can()
    # joints_df = pd.read_csv('joints_xyz.csv')
    
    x, y, z = pose.position.x, pose.position.y, pose.position.z

    distance = np.sqrt(x^2+y^2)
    # TODO: how to calculate m1 rotation?
    
    m1 = 0.0


    joints = search_joints(joints, distance)

    if joints is not None:
        m2 = joints['m2']
        m3 = joints['m3']
        m5 = joints['m5']
        robot.stretch([m1, m2, m3, 0.0, m5, 0.0])

    pass


def search_joints(joints, x_distance):
    data =None
    if sum(joints['x']>x_distance)>0:
        data = joints[joints['x']>x_distance][['m2','m3','m5']].iloc[0]
    return data

def load_joints():
    joints_df = pd.read_csv('joints_xyz.csv')
    joints_df['x']=joints_df['x']*-1
    joints_df = joints_df.sort_values(by='x')
    joints = joints_df.drop('index', axis=1)
    return joints

def main(args=None):
    # generate_joints_for_line(args)
    drop_coke_can()
    # grab_can_and_drop_delete_entity()


if __name__ == '__main__':
    main()
