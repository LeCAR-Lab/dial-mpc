import numpy as np
import mujoco
from matplotlib import pyplot as plt
from copy import deepcopy

def convert_state():
    # env configs
    Nsample = 1
    Hsample = 5

    # load the state from the file
    data = np.load("data/real_data_jump_good_20240914-213605.npy")
    timestep = data[:, 0] - data[0, 0]
    qpos = data[:, 1:1+19]
    qvel = data[:, 1:19:1+19+18]
    us = data[:, 1+19+18:1+19+18+12]
    xpos = np.zeros((len(timestep), 13, 3))
    xquat = np.zeros((len(timestep), 13, 4))
    # convert qpos to xpos
    model = mujoco.MjModel.from_xml_path("model/go2_mjx_force.xml")
    data = mujoco.MjData(model)
    for i in range(len(timestep)):
        # run kinematics to get xpos
        data.qpos[:] = qpos[i]
        data.qvel
        mujoco.mj_step(model, data)
        xpos[i] = data.xpos[1:]
        xquat[i] = data.xquat[1:]
        # generate sampled trajectories
        for j in range(Nsample):
            # create initial state
            state_sample = deepcopy(data)
            for h in range(Hsample):
                # get the action
                pass
    # save the xpos and xquat to a file
    np.save("data/go2_xpos.npy", xpos)
    np.save("data/go2_xquat.npy", xquat)

if __name__ == "__main__":
    convert_state()