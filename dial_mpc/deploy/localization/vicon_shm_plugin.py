import numpy as np
import time
from dial_mpc.deploy.localization import register_plugin
from dial_mpc.deploy.localization.base_plugin import BaseLocalizationPlugin
from multiprocessing import shared_memory
import struct

class MyPlugin(BaseLocalizationPlugin):
    def __init__(self, config):
        self.time = time.time()
        # Initialize shared memory
        self.shared_mem_name = "mocap_state_shm"
        self.shared_mem_size = 8 + 13 * 8  # 8 bytes for utime (int64), 13 float64s (13*8 bytes)
        self.mocap_shm = shared_memory.SharedMemory(name=self.shared_mem_name, create=False, size=self.shared_mem_size)
        self.state_buffer = self.mocap_shm.buf

    def get_state(self):
        # Unpack data from shared memory
        struct_format = "q13d"
        data = struct.unpack_from(struct_format, self.state_buffer, 0)

        # Extract position, quaternion, linear velocity, and angular velocity
        utime = data[0]
        position = np.array(data[1:4])
        quaternion = np.array(data[4:8])
        quaternion = np.roll(quaternion, 1)  # change quaternion from xyzw to wxyz
        linear_velocity = np.array(data[8:11])
        angular_velocity = np.array(data[11:14])

        # Combine position and quaternion into qpos
        qpos = np.concatenate([position, quaternion])
        # Combine linear and angular velocities into qvel
        qvel = np.concatenate([linear_velocity, angular_velocity])
        self.time = utime
        return np.concatenate([qpos, qvel])

    def get_last_update_time(self):
        return self.time
