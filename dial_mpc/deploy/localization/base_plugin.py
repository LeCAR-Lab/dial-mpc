from typing import Dict, Any


class BaseLocalizationPlugin:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_state(self):
        """
        Returns base qpos (3+4) and qvel (3+3) in a 1D array of size 13.
        Returns None if no update has been received.
        ALL VELOCITIES MUST BE RETURNED IN WORLD FRAME.
        """
        raise NotImplementedError

    def get_last_update_time(self):
        """
        Returns the timestamp (float) of the last update.
        Returns None if no update has been received.
        Used to check if the plugin is still alive.
        """
        raise NotImplementedError
