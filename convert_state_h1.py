import numpy as np
import mujoco
from matplotlib import pyplot as plt
from copy import deepcopy

def convert_state():
    exp_idx = 1
    task = [
        "jogging",
        "push_light", 
    ][exp_idx]
    data = np.load(f"data/h1_data_{task}.npy")
    print(f"data shape: {data.shape}")
    timestep = data[:, 0] - data[0, 0]
    qpos = data[:, 1:1+26]
    print(f"frame number: {len(timestep)}")
    xpos = np.zeros((len(timestep), 20, 3))
    xquat = np.zeros((len(timestep), 20, 4))
    xsite_feet = np.zeros((len(timestep), 2, 3))
    model = mujoco.MjModel.from_xml_path("model/unitree_h1/h1.xml")
    data = mujoco.MjData(model)
    left_foot_id = model.site("left_foot_site").id
    right_foot_id = model.site("right_foot_site").id
    for i in range(len(timestep)):
        data.qpos[:] = qpos[i]
        mujoco.mj_step(model, data)
        xpos[i] = data.xpos[1:]
        xquat[i] = data.xquat[1:]
        xsite_feet[i, 0] = data.site_xpos[left_foot_id]
        xsite_feet[i, 1] = data.site_xpos[right_foot_id]
    np.save(f"data/h1_data_{task}_xpos.npy", xpos)
    np.save(f"data/h1_data_{task}_xquat.npy", xquat)
    np.save(f"data/h1_data_{task}_xsite_feet.npy", xsite_feet)

if __name__ == "__main__":
    convert_state()