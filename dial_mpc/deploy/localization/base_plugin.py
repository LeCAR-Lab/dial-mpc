from typing import Dict, Any

class BaseLocalizationPlugin:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_state(self):
        """
        Returns base qpos (3+4) and qvel (3+3) in a 1D array of size 13
        ALL VELOCITIES MUST BE RETURNED IN WORLD FRAME
        """
        raise NotImplementedError