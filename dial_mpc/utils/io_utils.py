import importlib.resources
import os


def get_model_path(robot_name, model_name):
    with importlib.resources.path(f"dial_mpc.models.{robot_name}", model_name) as path:
        return path


def get_example_path(example_name):
    with importlib.resources.path(f"dial_mpc.examples", example_name) as path:
        return path
