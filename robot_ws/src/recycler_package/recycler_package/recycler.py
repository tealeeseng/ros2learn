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
from gazebo_msgs.msg import ContactsState
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
from ament_index_python.packages import get_package_share_directory
from hrim_actuator_gripper_srvs.srv import ControlFinger
from std_msgs.msg import String, Int32MultiArray
from sensor_msgs.msg import Image
import cv2

gym.logger.set_level(40)  # hide warnings

# FLAG_DEBUG_CAMERA = True
FLAG_DEBUG_CAMERA = False


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
        self.delete_entity_cli = self.create_client(
            DeleteEntity, '/delete_entity')

        # Create a gripper client for service "/hrim_actuation_gripper_000000000004/goal"
        self.gripper = self.create_client(
            ControlFinger, "/hrim_actuator_gripper_000000000004/fingercontrol")

        # Wait for service to be avaiable before calling it
        while not self.gripper.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.initArm()
        self.subscription_img = self.create_subscription(
            Image,
            '/rs_camera/rs_d435/image_raw',
            self.get_img,
            10)
        self.subscription_img  # prevent unused variable warning
        self.subscription_contact = self.create_subscription(ContactState,'/gazebo_contacts',
            self.get_contact,
            qos_profile=qos_profile_sensor_data) # QoS profile for reading (joint) sensors

    def initArm(self):
        print('start initArm()')

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
        self.target_orientation = np.asarray(
            [0., 0.7071068, 0.7071068, 0.])  # arrow looking down [w, x, y, z]
        self.current_joints = np.array([0, 0, 0, 0, 0, 0])

        _, self.ur_tree = tree_urdf.treeFromFile(urdfPath)
        # Retrieve a chain structure between the base and the start of the end effector.
        self.mara_chain = self.ur_tree.getChain(
            self.m_linkNames[0], self.m_linkNames[-1])
        self.numJoints = self.mara_chain.getNrOfJoints()
        # Initialize a KDL Jacobian solver from the chain.
        self.jacSolver = ChainJntToJacSolver(self.mara_chain)

    def gripper_angle(self, angle=1.57):
        req = ControlFinger.Request()
        req.goal_angularposition = angle
        # self.gripper.call(req)
        # rclpy.spin_once(self)

        future = self.gripper.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        # Analyze the result
        if future.result() is not None:
            self.get_logger().info('Goal accepted: %d: ' % future.result().goal_accepted)
        else:
            self.get_logger().error('Exception while calling service: %r' % future.exception())


    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg = message
        # self.get_logger().info(str(observation_msg))

    def moving(self, joints):
        if type(joints) is list:
            joints = np.array(joints)

        # lift arm
        step1 = copy.deepcopy(self.current_joints)
        step1[1] = 0
        self.moving_like_robot(step1)

        # rotate
        step2 = step1
        step2[0] = joints[0]
        self.moving_like_robot(step2)

        # stetch arm
        self.moving_like_robot(joints)

    def moving_like_robot(self, joints):

        STEPS = 10
        source = self.current_joints
        diff = joints - source
        # print('diff, ',diff)
        step_size = diff / STEPS

        for i in range(1, 11):
            self.stretch(source + i * step_size)
            time.sleep(0.1)

        self.current_joints = copy.deepcopy(joints)

    def stretch(self, joints):
        # m_jointOrder = copy.deepcopy(JOINT_ORDER)

        self.pub.publish(ut_mara.getTrajectoryMessage(
            joints,
            self.m_jointOrder,
            0.1))

        rclpy.spin_once(self)

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

    def delete_can(self, target_name):
        # delete entity
        while not self.delete_entity_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_simulation service not available, waiting again...')

        req = DeleteEntity.Request()
        req.name = target_name
        delete_future = self.delete_entity_cli.call_async(req)
        rclpy.spin_until_future_complete(self, delete_future)
        if delete_future.result() is not None:
            self.get_logger().info('delete_future response: %r' % delete_future.result())

    def spawn_target(self, urdf_obj, position):
        # self.targetPosition = self.sample_position()
        self.targetPosition = position

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
        pose.orientation.y = self.target_orientation[2]
        pose.orientation.z = self.target_orientation[3]
        pose.orientation.w = self.target_orientation[0]
        # override previous spawn_request element.
        self.spawn_request = SpawnEntity.Request()
        self.spawn_request.name = urdf_obj
        self.spawn_request.xml = modelXml
        self.spawn_request.robot_namespace = ""
        self.spawn_request.initial_pose = pose
        self.spawn_request.reference_frame = "world"

        # ROS2 Spawn Entity
        target_future = self.spawn_cli.call_async(self.spawn_request)
        rclpy.spin_until_future_complete(self, target_future)
        if target_future.result() is not None:
            print('spawn_request response: %r' % target_future.result())

        return pose

    def load_coke_can(self):
        modelXml = """<?xml version='1.0'?>
                        <sdf version='1.5'>
                            <model name="can1">
                                <include>
                                <static>false</static>
                                <uri>model://coke_can</uri>
                                </include>
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
                            <self_collide>1</self_collide>
                            <enable_wind>0</enable_wind>
                            <kinematic>1</kinematic>
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
                            <static>0</static>
                            <allow_auto_disable>1</allow_auto_disable>
                        </model>
                        </sdf>"""
        return modelXml

    def sample_position(self):
        # [ -0.5 , 0.2 , 0.1 ], [ -0.5 , -0.2 , 0.1 ] #sample data. initial 2 points in original setup.
        pos = [-1 * np.random.uniform(0.1, 0.6),
               np.random.uniform(0.1, 0.6), 0.15]
        print('object pos, ', pos)
        return pos
        # sample_x = np.random.uniform(0,1)

        # if sample > 0.5:
        #     return [ -0.8 , 0.0 , 0.1 ]
        # else:
        #     return [ -0.5 , 0.0 , 0.1 ]
    def load_random_urdf(self, obj):
        urdfPath = get_prefix_path(
            "mara_description") + "/share/mara_description/random_urdfs/" + obj
        urdf_file = open(urdfPath, "r")
        urdf_string = urdf_file.read()
        print("urdf_string:", urdf_string)
        return urdf_string

    def get_img(self, msg):
        # print(type(msg.data))
        # print(len(msg.data))

        self.raw_img = msg.data

        # if self.capturing:            
        #     img = np.array(msg.data).reshape((480, 640, 3))
        #     self.image = cv2.rotate(img, cv2.ROTATE_180)

        
        # #image still upside down. Need to rotate 180 degree?
        # if FLAG_DEBUG_CAMERA:
        #     image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        #     cv2.imshow('RS D435 Camera Image', image)
        #     key = cv2.waitKey(1)
        #     # Press esc or 'q' to close the image window
        #     if key & 0xFF == ord('q') or key == 27:
        #         cv2.destroyAllWindows()

    def captureImage(self):

        img = np.array(self.raw_img).reshape((480, 640, 3))
        self.image = cv2.rotate(img, cv2.ROTATE_180)

    def get_contact(self, msg):
        '''
        Retrieve contact points for gripper fingers
        '''
        if msg.collision1_name in ['mara::left_inner_finger::left_inner_finger_collision',
                                   'mara::right_inner_finger::right_inner_finger_collision']:
            collision1_side = msg.collision1_name
            contact1_positions = len(msg.contact_positions)
            # print(msg.collision1_name)
            # print(len(msg.contact_positions))

def generate_joints_for_length(args=None):
    rclpy.init(args=args)
    robot = Robot()
    rclpy.spin_once(robot)

    STEP = 0.01
    data_frame = pd.DataFrame(columns=['m2', 'm3', 'm5', 'x', 'y', 'z'])

    # can we fix motor 1 4,6, to train free moving arm on a line?

    for change in np.arange(-0.3, 1, STEP):
        #             # sample data, [-np.pi/2, 0.5, -.5, 0, -1.2, 0]
        m2 = 0+change
        m3 = -np.pi/2+change
        m5 = -np.pi/2+change
        robot.stretch(np.array([-np.pi/2, m2, m3, 0, m5, 0]))
        rclpy.spin_once(robot)
        rclpy.spin_once(robot)
        current_eePos_tgt = robot.take_observation([0, 0, 0])
        rclpy.spin_once(robot)

        if 0 <= current_eePos_tgt[2] < 1:
            data = [m2, m3, m5]
            data.extend(current_eePos_tgt)
            # print('data,', data)

            df = pd.Series(data, index=data_frame.columns)
            # print('df,', df)
            data_frame = data_frame.append(df, ignore_index=True)
            robot.get_logger().info(str(data))

    data_frame.to_csv('../resource/joints_xyz.csv', index=False)

    change = 1
    robot.stretch(
        np.array([-np.pi/2, 0+change, -np.pi/2+change, 0, -np.pi/2+change, 0]))
    rclpy.spin_once(robot)
    current_eePos_tgt = robot.take_observation([0, 0, 0])
    rclpy.spin_once(robot)
    # print('current_eePos_tgt, ', current_eePos_tgt)

    robot.destroy_node()
    rclpy.shutdown()

    print('END generate_joints_for_length().')


def generate_joints_for_length_core(robot):
    STEP = 0.01
    for change in np.arange(-0.3, 1, STEP):
        m2 = 0+change
        m3 = -np.pi/2+change
        m5 = -np.pi/2+change
        robot.stretch(np.array([-np.pi/2, m2, m3, 0, m5, 0]))
        rclpy.spin_once(robot)
        end_effector_pose = robot.take_observation([0, 0, 0])


def generate_joints_for_line_outdated(args=None):
    rclpy.init(args=args)
    robot = Robot()
    rclpy.spin_once(robot)

    STEP = 0.05
    data_frame = pd.DataFrame(columns=['m2', 'm3', 'm5', 'x', 'y', 'z'])

    # can we fix motor 1 4,6, to train free moving arm on a line?

    for m2 in np.arange(0, 0.2, STEP):
        for m3 in np.arange(-np.pi/2+0.5, -np.pi/2-0.1, -1*STEP):
            for m5 in np.arange(0, -np.pi/2-0.5, -1*STEP):

                # sample data, [-np.pi/2, 0.5, -.5, 0, -1.2, 0]
                robot.stretch(np.array([-np.pi/2, m2, m3, 0, m5, 0]))
                rclpy.spin_once(robot)
                rclpy.spin_once(robot)
                current_eePos_tgt = robot.take_observation([0, 0, 0])
                rclpy.spin_once(robot)

                if 0.08 <= current_eePos_tgt[2] < 0.12:
                    data = [m2, m3, m5]
                    data.extend(current_eePos_tgt)
                    # print('data,', data)

                    df = pd.Series(data, index=data_frame.columns)
                    # print('df,', df)
                    data_frame = data_frame.append(df, ignore_index=True)
                    robot.get_logger().info(str(data))
    data_frame.to_csv('../resource/joints_xyz.csv', index=False)

    robot.stretch(np.array([-np.pi/2, 0, -np.pi/2-0.1, 0, -np.pi/2-0.5, 0]))
    rclpy.spin_once(robot)
    current_eePos_tgt = robot.take_observation([0, 0, 0])
    rclpy.spin_once(robot)
    print('current_eePos_tgt, ', current_eePos_tgt)

    robot.destroy_node()
    rclpy.shutdown()

    print('END generate_joints_for_line().')


def drop_coke_can(robot=None):
    if robot is None:
        rclpy.init()
        robot = Robot()
        rclpy.spin_once(robot)

    obj = "coke0"
    robot.delete_can(obj)
    pose = robot.spawn_target(obj, robot.sample_position())
    rclpy.spin_once(robot)

    return pose

def drop_coke_can_on(robot, position):

    obj = "coke0"
    robot.delete_can(obj)
    pose = robot.spawn_target(obj, position)
    rclpy.spin_once(robot)

    return pose

    #     pos,  [[-1.05369001e-03  1.92728881e-05  1.11664069e+00]] rot: [[ 1.00000000e+00 -4.47419942e-06 -2.23629828e-11]
    #  [ 4.47419942e-06  1.00000000e+00  2.87694329e-05]
    #  [-1.06357197e-10 -2.87694329e-05  1.00000000e+00]]
    # print(rot[0, 1])
    # print(general_utils.inverseKinematics(mara_chain, pos, rot))  # can't understand rot and generate data for that.

    # def inverseKinematics(robotChain, pos, rot, qGuess=None, minJoints=None, maxJoints=None):


def grab_can_and_drop_delete_entity(robot, pose):
    if robot is None:
        rclpy.init()
        robot = Robot()
        rclpy.spin_once(robot)

    joints = load_joints()

    x, y, z = pose.position.x, pose.position.y, pose.position.z

    distance = np.sqrt(x*x+y*y)
    rotation = np.arctan2(-x, y)
    print("x, y, rotation :", x, y, rotation)

    # reverse sign of x to let it handle things appear on left hand side. +y move along green axis.
    m1 = -np.pi + rotation

    joints = search_joints(joints, distance, 0.2)
    # joints = calculate_joints()

    if joints is not None:
        m2 = joints['m2']
        m3 = joints['m3']
        m5 = joints['m5']
        print('distance m1 m2 m3 m5:', distance, m1, m2, m3, m5)
        robot.moving(np.array([m1, m2, m3, 0.0, m5, 0.0]))
        # rclpy.spin_once(robot)
        # # robot.stretch(np.array([m1, m2, m3, 0.0, m5, 0.0]))
        # rclpy.spin_once(robot)
        # rclpy.spin_once(robot)

    else:
        print('No Joints found.')

    robot.gripper_angle(0.3)
    time.sleep(3)
    robot.moving(np.array([m1+np.pi, m2, m3, 0.0, m5, 0.0]))
    robot.gripper_angle(1.57)


def look_for_can(robot):
    robot.moving([-np.pi*3/4, 0, 0, 0, -np.pi, 0])
    # time.sleep(2)
    print('look_for_can, moved')

    # robot.image
    # robot.img


def search_joints(joints, x_distance, z):
    data = None
    idx = np.where((x_distance < joints['x']) & (
        joints['x'] < x_distance+0.1) & (joints['z'] < z))
    if np.sum(idx) > 0:
        # print('data count:', np.sum(idx))
        data = joints.loc[idx][['m2', 'm3', 'm5']].iloc[0]
        # data = joints.loc[idx][['m2','m3','m5']].median()
    return data


def load_joints():
    joints_df = pd.read_csv(get_package_share_directory(
        'recycler_package')+'/resource/joints_xyz.csv')
    joints_df['x'] = joints_df['x']*-1
    joints = joints_df.sort_values(by='x')
    # joints = joints_df.drop('index', axis=1)
    return joints


def main(args=None):
    rclpy.init(args=args)
    robot = Robot()
    rclpy.spin_once(robot)

    for i in range(5, 12):
        ## for images collection

        if FLAG_DEBUG_CAMERA:
            pose = drop_coke_can_on(robot, [-0.05*i, 0.05*i, 0.15])
        else:
            pose = drop_coke_can(robot)
        
        if FLAG_DEBUG_CAMERA:
            look_for_can(robot)
            time.sleep(1)
            robot.captureImage()
            image = robot.image
            cv2.imwrite(str(i)+'.png', image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('RS D435 Camera Image', image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()

        grab_can_and_drop_delete_entity(robot, pose)


def main_(args=None):
    generate_joints_for_length(args)


if __name__ == '__main__':
    main()
