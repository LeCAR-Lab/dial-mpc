import numpy as np
import mujoco
from matplotlib import pyplot as plt
from copy import deepcopy


def convert_state():
    # env configs
    Nsample = 1
    Hsample = 5
    exp_idx = 4
    exp_name = [
        "dial_real_jump",
        "dial_real_run",
        "dial_sim_gallop",
        "dial_sim_trot",
        "dial_sim_climb",
    ][exp_idx]
    file_name = [
        "data/real_data_jump_good_20240914-213605.npy",
        "data/real_data_20240914-231915.npy",
        "data/force_flat_gallop_2_2.npy",
        "data/force_flat_trot_-1_-1.npy",
        "data/sim_data_climb.npy",
    ][exp_idx]

    # load the state from the file
    data = np.load(file_name)
    print(data.shape)
    timestep = data[:, 0] - data[0, 0]
    qpos = data[:, 1 : 1 + 19]
    qvel = data[:, 1 : 19 : 1 + 19 + 18]
    us = data[:, 1 + 19 + 18 : 1 + 19 + 18 + 12]
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
    np.save(f"data/{exp_name}_xpos.npy", xpos)
    np.save(f"data/{exp_name}_xquat.npy", xquat)
    np.save(f"data/{exp_name}_xsite_feet.npy", xsite_feet)
    # plot xpos[:, 0, [0,2]], xsite_feet[:, 0, [0,2]], xsite_feet[:, 1, [0,2]], xsite_feet[:, 2, [0,2]], xsite_feet[:, 3, [0,2]]
    plt.plot(xpos[:, 0, 0], xpos[:, 0, 2], label="base")
    plt.plot(xsite_feet[:, 0, 0], xsite_feet[:, 0, 2], label="FR_foot")
    plt.plot(xsite_feet[:, 1, 0], xsite_feet[:, 1, 2], label="FL_foot")
    plt.plot(xsite_feet[:, 2, 0], xsite_feet[:, 2, 2], label="RR_foot")
    plt.plot(xsite_feet[:, 3, 0], xsite_feet[:, 3, 2], label="RL_foot")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    convert_state()
