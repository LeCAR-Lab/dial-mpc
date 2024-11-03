import time

import numpy as np

from dial_mpc.deploy.localization.base_plugin import BaseLocalizationPlugin


class ROS2OdometryPlugin(BaseLocalizationPlugin):
    def __init__(self, config):
        BaseLocalizationPlugin.__init__(self, config)

    def get_state(self):
        qpos = np.zeros(7)
        qpos[2] = 1.0
        qpos[3] = 1.0
        qvel = np.zeros(6)
        return np.concatenate([qpos, qvel])

    def get_last_update_time(self):
        return time.time()
