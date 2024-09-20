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
    xsite_feet = np.zeros((len(timestep), 4, 3))
    # convert qpos to xpos
    model = mujoco.MjModel.from_xml_path("model/go2_mjx_force.xml")
    data = mujoco.MjData(model)
    FR_foot_id = model.site("FR_foot").id
    FL_foot_id = model.site("FL_foot").id
    RR_foot_id = model.site("RR_foot").id
    RL_foot_id = model.site("RL_foot").id
    for i in range(len(timestep)):
        # run kinematics to get xpos
        data.qpos[:] = qpos[i]
        data.qvel
        mujoco.mj_step(model, data)
        xpos[i] = data.xpos[1:]
        xquat[i] = data.xquat[1:]
        # get feet position
        xsite_feet[i, 0] = data.site_xpos[FR_foot_id]
        xsite_feet[i, 1] = data.site_xpos[FL_foot_id]
        xsite_feet[i, 2] = data.site_xpos[RR_foot_id]
        xsite_feet[i, 3] = data.site_xpos[RL_foot_id]
    # save the xpos and xquat to a file
    np.save("data/go2_xpos.npy", xpos)
    np.save("data/go2_xquat.npy", xquat)
    np.save("data/go2_xsite_feet.npy", xsite_feet)
if __name__ == "__main__":
    convert_state()