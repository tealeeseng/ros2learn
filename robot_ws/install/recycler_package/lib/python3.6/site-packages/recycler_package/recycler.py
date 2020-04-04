# ROS 2
import rclpy
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

class JointsSubscriber(Node):
    def __init__(self):
        super().__init__('JointsSubscriber')
        self.subcription = self.create_subscription(JointTrajectoryControllerState, JOINT_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        observation_msg = message
        self.get_logger().info(observation_msg)


def main(args=None):
    rclpy.init(args=args)
    joint_sub = JointsSubscriber()
    rclpy.spin(joint_sub)

    joint_sub.destroy_node()
    rclpy.shutdown()

    print('END recycler_package.')


if __name__ == '__main__':
    main()
