import rclpy
from rclpy.node import Node

from std_msgs.msg import String,Int32MultiArray
from sensor_msgs.msg import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rs_camera/rs_d435/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # print(type(msg.data))
        # print(len(msg.data))
        img = np.array(msg.data).reshape((720,1280))
        cv2.imshow('RS D435 Camera Image', img)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()