import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
from multiprocessing import shared_memory
from pyvicon_datastream import tools
import struct


class ViconDemo:
    def __init__(self):
        # Vicon DataStream IP and object name
        self.vicon_tracker_ip = "128.2.184.3"  # Replace with your Vicon DataStream IP
        self.vicon_object_name = "lecar_go2"  # Replace with your object name

        # Connect to Vicon DataStream
        self.tracker = tools.ObjectTracker(self.vicon_tracker_ip)
        if self.tracker.is_connected:
            print(f"Connected to Vicon DataStream at {self.vicon_tracker_ip}")
        else:
            print(f"Failed to connect to Vicon DataStream at {self.vicon_tracker_ip}")
            raise Exception(f"Connection to {self.vicon_tracker_ip} failed")

        # Initialize previous values for velocity computation
        self.prev_time = None
        self.prev_position = None
        self.prev_quaternion = None

        # Low-pass filter parameters
        self.cutoff_freq = 5.0  # Cut-off frequency of the filter (Hz)
        self.filter_order = 2
        self.fs = 100.0  # Sampling frequency (Hz)
        self.b, self.a = butter(
            self.filter_order, self.cutoff_freq / (0.5 * self.fs), btype="low"
        )

        # Initialize data buffers for filtering
        self.vel_buffer = []
        self.omega_buffer = []

        
        # Initialize shared memory
        self.shared_mem_name = "mocap_state_shm"
        self.shared_mem_size = 8 + 13 * 8  # 8 bytes for utime (int64), 13 float64s (13*8 bytes)
        try:
            self.state_shm = shared_memory.SharedMemory(name=self.shared_mem_name, create=True, size=self.shared_mem_size)
            print(f"Attach to shared memory '{self.shared_mem_name}' of size {self.shared_mem_size} bytes.")
        except FileExistsError:
            print(f"shared memory does not exist")
        self.state_buffer = self.state_shm.buf

    def get_vicon_data(self):
        position = self.tracker.get_position(self.vicon_object_name)
        if not position:
            print(f"Cannot get the pose of `{self.vicon_object_name}`.")
            return None, None, None

        try:
            obj = position[2][0]
            _, _, x, y, z, roll_ext, pitch_ext, yaw_ext = obj
            current_time = time.time()
            # q = tf_transformations.quaternion_from_euler(roll, pitch, yaw, "rxyz")
            # roll, pitch, yaw = tf_transformations.euler_from_quaternion(q, "sxyz")

            # Position and orientation
            position = np.array([x, y, z]) / 1000.0
            rotation = R.from_euler("XYZ", [roll_ext, pitch_ext, yaw_ext], degrees=False)
            quaternion = rotation.as_quat()  # [x, y, z, w]

            return current_time, position, quaternion
        except Exception as e:
            print(f"Error retrieving Vicon data: {e}")
            return None, None, None

    def compute_velocities(self, current_time, position, quaternion):
        # Initialize velocities
        linear_velocity = np.zeros(3)
        angular_velocity = np.zeros(3)

        if (
            self.prev_time is not None
            and self.prev_position is not None
            and self.prev_quaternion is not None
        ):
            dt = current_time - self.prev_time
            if dt > 0:
                # Linear velocity
                dp = position - self.prev_position
                linear_velocity = dp / dt

                # Angular velocity
                prev_rot = R.from_quat(self.prev_quaternion)
                curr_rot = R.from_quat(quaternion)
                delta_rot = curr_rot * prev_rot.inv()
                delta_angle = delta_rot.as_rotvec()
                angular_velocity = delta_angle / dt
        else:
            # First data point; velocities remain zero
            pass

        # Update previous values
        self.prev_time = current_time
        self.prev_position = position
        self.prev_quaternion = quaternion

        return linear_velocity, angular_velocity

    def low_pass_filter(self, data_buffer, new_data):
        # Append new data to the buffer
        data_buffer.append(new_data)
        # Keep only the last N samples (buffer size)
        buffer_size = int(self.fs / self.cutoff_freq) * 3
        if len(data_buffer) > buffer_size:
            data_buffer.pop(0)
        # Apply low-pass filter if enough data points are available
        if len(data_buffer) >= self.filter_order + 1:
            data_array = np.array(data_buffer)
            filtered_data = lfilter(self.b, self.a, data_array, axis=0)[-1]
            return filtered_data
        else:
            return new_data  # Not enough data to filter; return the new data as is

    def main_loop(self):
        print("Starting Vicon data acquisition...")
        try:
            while True:
                # Get Vicon data
                current_time, position, quaternion = self.get_vicon_data()
                if position is None:
                    time.sleep(0.01)         
                    continue

                # Compute velocities
                linear_velocity, angular_velocity = self.compute_velocities(
                    current_time, position, quaternion
                )

                # Apply low-pass filter to velocities
                filtered_linear_velocity = self.low_pass_filter(
                    self.vel_buffer, linear_velocity
                )
                filtered_angular_velocity = self.low_pass_filter(
                    self.omega_buffer, angular_velocity
                )

                # Prepare data to pack
                utime = int(current_time * 1e6)  # int64
                data_to_pack = [utime]
                data_to_pack.extend(position.tolist())
                data_to_pack.extend(quaternion.tolist())
                data_to_pack.extend(filtered_linear_velocity.tolist())
                data_to_pack.extend(filtered_angular_velocity.tolist())

                # Pack data into shared memory buffer
                struct_format = "q13d"
                struct.pack_into(struct_format, self.state_buffer, 0, *data_to_pack)

                # Optionally, print or process the filtered data
                # print(f"Position: {position}")
                # print(f"Filtered Linear Velocity: {filtered_linear_velocity}")
                # print(f"Filtered Angular Velocity: {filtered_angular_velocity}")
                print(f"Quat:", quaternion)
                print("-" * 50)

                print(f"State:", position)
                print("-" * 50)

                # Sleep to mimic sampling rate
                time.sleep(1.0 / self.fs)

        except KeyboardInterrupt:
            print("Exiting Vicon data acquisition.")
        finally:
            # Close and unlink shared memory
            try:
                self.state_shm.close()
                print(f"Shared memory '{self.shared_mem_name}' closed")
            except:
                pass


if __name__ == "__main__":
    vicon_demo = ViconDemo()
    vicon_demo.main_loop()
