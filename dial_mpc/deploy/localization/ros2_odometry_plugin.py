import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from dial_mpc.deploy.localization.base_plugin import BaseLocalizationPlugin

class ROS2OdometryPlugin(BaseLocalizationPlugin, Node):
    def __init__(self, config):
        super().__init__(config)
        Node.__init__(self, 'ros2_odom_plugin')
        self.subscription = self.create_subscription(
            Odometry,
            'odometry',
            self.odom_callback,
            1
        )

        self.qpos = None
        self.qvel = None

    def odom_callback(self, msg):
        self.qpos = msg.pose.pose.position
        self.qvel = msg.twist.twist.linear

    def get_state(self):
        return self.qpos, self.qvel