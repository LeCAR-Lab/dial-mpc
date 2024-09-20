import subprocess
import sys
import time


def main():
    args = sys.argv[1:]
    arg_list = args if args is not None else []
    subprocess.run(["dial-mpc-sim"] + arg_list)
    delay = 2.0
    time.sleep(delay)
    subprocess.run(["dial-mpc-plan"] + arg_list)
