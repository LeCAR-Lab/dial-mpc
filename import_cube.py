import os
import sys
# check if the system is windows, if so, add the path of blender
if os.name == "nt":
    packages_path = r"C:\Users\JC-Ba\AppData\Roaming\Python\Python311\Scripts" + r"\\..\\site-packages"
    sys.path.insert(0, packages_path )
import bpy
import pickle
import numpy as np
import matplotlib as mpl
from mathutils import Quaternion, Matrix, Vector, Euler
import time

def import_cube():
    print("import_cube")

    file_prefix = r"C:\Users\JC-Ba\Downloads\code\dial-mpc\data"
    data_original = np.load(f"{file_prefix}\h1_data_push_light.npy")
    xpos_x = data_original[:, 1+26]
    Hrollout = xpos_x.shape[0]
    xpos = np.zeros((Hrollout, 3))
    xpos[:, 0] = xpos_x
    xpos[:, 2] = 0.6
    xquat_wxyz = np.zeros((Hrollout, 4))
    xquat_wxyz[:, 0] = 1.0

    Hrender_start, Hrender_end = [0, 300-40]
    Hrender = Hrender_end - Hrender_start

    link_name = "Cube"
    blender_obj = bpy.data.objects[link_name]
    for frame_idx, (pos, quat) in enumerate(zip(xpos[Hrender_start:Hrender_end], xquat_wxyz[Hrender_start:Hrender_end])):
        # insert position keyframe
        blender_obj.location = pos
        blender_obj.keyframe_insert("location", frame=frame_idx)
        # change rotation mode to quaternion
        blender_obj.rotation_mode = "QUATERNION"
        # blender's quaternion constructor is in wxyz format
        # insert quaternion keyframe, applying transform
        blender_obj.rotation_quaternion = (
            Quaternion((quat[3], quat[0], quat[1], quat[2])) #@ yup_to_zup
        )
        blender_obj.keyframe_insert("rotation_quaternion", frame=frame_idx)

        if frame_idx > Hrender:
            break

if __name__ == "__main__":
    import_cube()
