import time
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from mocap4r2_msgs.msg import RigidBodies, RigidBody

from dial_mpc.deploy.localization.base_plugin import BaseLocalizationPlugin

class Mocap4ROS2InterfaceNode(Node):
    def __init__(self, config):
        Node.__init__(self, "mocap_listener")

        if "debug_mocap" in config and config["debug_mocap"]:
            self.debug = True
            self.odometry_pub = self.create_publisher(Odometry, "odometry", 1)
        else:
            self.debug = False

        self.subscription = self.create_subscription(
            RigidBodies, config["rigid_bodies_topic"], self.rigid_bodies_callback, 10
        )

        self.cutoff_freq = 5.0  # Cut-off frequency of the filter (Hz)
        self.filter_order = 2
        self.fs = 120.0  # Sampling frequency (Hz)
        self.normalized_freq = self.cutoff_freq / (0.5 * self.fs)
        self.buffer_size = int(self.fs / self.cutoff_freq) * 3
        self.data_buffer = []
        self.b, self.a = butter(self.filter_order, self.normalized_freq, btype="low")

        self.up_axis = config["up_axis"]
        assert self.up_axis in ["y", "z"], f"Invalid up_axis: {self.up_axis}, must be y or z"

        self.last_time = None
        self.last_pose = None
        self.last_vel = None
        self.lock = threading.Lock()

    def rigid_bodies_callback(self, msg: RigidBodies):
        body = msg.rigidbodies[0]
        q = [body.pose.orientation.x, body.pose.orientation.y, body.pose.orientation.z, body.pose.orientation.w]
        r1 = R.from_quat(q)

        if self.up_axis == "y":
            r2 = R.from_euler("xyz", [-90.0, 0.0, 180.0], degrees=True)
            rotation = r2 * r1
        elif self.up_axis == "z":
            rotation = r1
        q_final = rotation.as_quat()
        rpy = euler_from_quaternion(q_final, 'rxyz')
        # rpy = rotation.as_euler("xyz")
        
        translation = np.array([body.pose.position.x, body.pose.position.y, body.pose.position.z])
        translation = r2.apply(translation)

        if len(self.data_buffer) == self.buffer_size:
            self.data_buffer.pop(0)
        self.data_buffer.append(np.concatenate([translation, rpy]))

        if len(self.data_buffer) >= 2:
            for i in range (3):
                self.data_buffer[-1][3 + i] = self.align_angle(self.data_buffer[-1][3 + i], self.data_buffer[-2][3 + i])
        if len(self.data_buffer) >= self.filter_order:
            filtered_data = lfilter(self.b, self.a, self.data_buffer, axis=0)
            with self.lock:
                this_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.last_pose = np.concatenate([filtered_data[-1][:3], R.from_euler("xyz", filtered_data[-1][3:]).as_quat(scalar_first=True)])
                self.last_vel = (filtered_data[-1] - filtered_data[-2]) * self.fs
                self.last_time = this_time

            if self.debug:
                odom_msg = Odometry()
                odom_msg.header.stamp = msg.header.stamp
                odom_msg.pose.pose.position.x = self.last_pose[0]
                odom_msg.pose.pose.position.y = self.last_pose[1]
                odom_msg.pose.pose.position.z = self.last_pose[2]
                odom_msg.pose.pose.orientation.x = self.last_pose[4]
                odom_msg.pose.pose.orientation.y = self.last_pose[5]
                odom_msg.pose.pose.orientation.z = self.last_pose[6]
                odom_msg.pose.pose.orientation.w = self.last_pose[3]
                odom_msg.twist.twist.linear.x = self.last_vel[0]
                odom_msg.twist.twist.linear.y = self.last_vel[1]
                odom_msg.twist.twist.linear.z = self.last_vel[2]
                odom_msg.twist.twist.angular.x = self.last_vel[3]
                odom_msg.twist.twist.angular.y = self.last_vel[4]
                odom_msg.twist.twist.angular.z = self.last_vel[5]
                self.odometry_pub.publish(odom_msg)

    def align_angle(self, angle_1, angle_2):
        diff = angle_1 - angle_2
        diff_aligned = np.arctan2(np.sin(diff), np.cos(diff))
        return diff_aligned + angle_2


class Mocap4ROS2Plugin(BaseLocalizationPlugin):
    def __init__(self, config):
        BaseLocalizationPlugin.__init__(self, config)
        rclpy.init()
        self.node = Mocap4ROS2InterfaceNode(config)
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,))
        self.thread.start()

        self.qpos = None
        self.qvel = None
        self.last_time = None

    def __del__(self):
        rclpy.shutdown()
        self.thread.join()

    def get_state(self):
        with self.node.lock:
            if self.node.last_time is not None:
                return np.concatenate([self.node.last_pose, self.node.last_vel])
            else:
                return None

    def get_last_update_time(self):
        return time.time()
