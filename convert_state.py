import numpy as np
import mujoco
from matplotlib import pyplot as plt

def convert_state():
    # load the state from the file
    data = np.load("data/real_data_jump_good_20240914-213605.npy")
    timestep = data[:, 0] - data[0, 0]
    qpos = data[:, 1:1+19]
    xpos = np.zeros((len(timestep), 13, 3))
    xquat = np.zeros((len(timestep), 13, 4))
    # convert qpos to xpos
    model = mujoco.MjModel.from_xml_path("model/go2.xml")
    data = mujoco.MjData(model)
    # run kinematics to get xpos
    for i in range(len(timestep)):
        data.qpos[:] = qpos[i]
        mujoco.mj_step(model, data)
        xpos[i] = data.xpos[1:]
        xquat[i] = data.xquat[1:]
    # save the xpos and xquat to a file
    np.save("data/go2_xpos.npy", xpos)
    np.save("data/go2_xquat.npy", xquat)


if __name__ == "__main__":
    convert_state()