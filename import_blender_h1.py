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

link_mapper = [
    "pelvis",
    "left_hip_yaw_link", 
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_yaw_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
]


def import_animation():
    dt = 0.02
    Hsample = 40
    Nsample = 8
    Ndiffuse = 4
    Ndownsample = 10 # trajectory downsample rate
    Hdownsample = 3 # frame downsample rate

    exp_idx = 1
    exp_name = [
        "jogging",
        "push_light",
    ][exp_idx]  
    file_name = f"h1_data_{exp_name}"
    Hrender_start, Hrender_end = [
        [0, 310],
        [0, 300-40],
    ][exp_idx]
    Hrender = Hrender_end - Hrender_start

    file_prefix = r"C:\Users\JC-Ba\Downloads\code\dial-mpc\data"

    link_pos_original = np.load(f"{file_prefix}\{file_name}_xpos.npy")
    link_quat_wxyz_original = np.load(f"{file_prefix}\{file_name}_xquat.npy")
    xsite_feet_original = np.load(
        f"{file_prefix}\{file_name}_xsite_feet.npy"
    )
    link_quat_xyzw_original = link_quat_wxyz_original[:, :, [1, 2, 3, 0]]
    # Nlink = link_pos.shape[1]
    # for i in range(Nlink):
    #     link_pos[:, i, 0] = np.sin(np.arange(Hrollout) * dt + i / Nlink * np.pi / 2)
    #     link_quat_xyzw[:, i, 0] = np.sin(
    #         np.arange(Hrollout) * dt + i / Nlink * np.pi / 2
    #     )
    #     link_quat_xyzw[:, i, 1] = np.cos(
    #         np.arange(Hrollout) * dt + i / Nlink * np.pi / 2
    #     )
    link_pos = np.transpose(link_pos_original, (1, 0, 2))
    link_quat_xyzw = np.transpose(link_quat_xyzw_original, (1, 0, 2))
    xsite_feet = np.transpose(xsite_feet_original, (1, 0, 2))
    # isaac gym is Y-up Right-handed coordinate system
    # blender is  Z-up left-handed coordinate system
    # so we need to convert the quaternion from isaac gym to blender
    yup_to_zup = Euler((np.pi / 2, 0, 0), "XYZ").to_quaternion()
    for link_idx, (link_pos_traj, link_quat_traj) in enumerate(
        zip(link_pos, link_quat_xyzw)
    ):
        print(f"Progress: {link_idx}/{len(link_mapper)}")
        print(f"Importing {link_mapper[link_idx]}...")
        link_name = link_mapper[link_idx]
        if link_name not in bpy.data.objects:
            continue
        blender_obj = bpy.data.objects[link_name]
        for frame_idx, (pos, quat) in enumerate(zip(link_pos_traj[Hrender_start:Hrender_end], link_quat_traj[Hrender_start:Hrender_end])):
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

            mesh = bpy.data.meshes.new(f"trace_{frame_idx}")

            if frame_idx > Hrender:
                break

    # generate trajectory to visualize
    # TODO: replace it with the actual trajectory
    xssss_torso = np.zeros((Hrender, Ndiffuse, Nsample, Hsample, 3)) 
    xssss_feet = np.zeros((Hrender, Ndiffuse, Nsample, Hsample, 2, 3))
    for i in range(Hrender):
        xs_torso_ref = link_pos_original[i + Hrender_start : i + Hrender_start + Hsample, 0]+ np.array([0.1, 0.0, 0.0])
        xs_feet_ref = xsite_feet_original[i + Hrender_start : i + Hrender_start + Hsample]
        for j in range(Ndiffuse):
            sigma = 0.1 * (0.5**j)
            xssss_torso[i, j] = (
                xs_torso_ref + np.random.randn(Nsample, Hsample, 3) * sigma
            )
            xssss_feet[i, j] = (
                xs_feet_ref + np.random.randn(Nsample, Hsample, 2, 3) * sigma
            )
            # go through low-pass filter
            alpha = 0.2
            for k in range(1, Hsample):
                xssss_torso[i, j, :, k] = alpha * xssss_torso[i, j, :, k] + (1 - alpha) * xssss_torso[i, j, :, k-1]
                xssss_feet[i, j, :, k] = alpha * xssss_feet[i, j, :, k] + (1 - alpha) * xssss_feet[i, j, :, k-1]

    # visualize the trajectory
    t0 = time.time()
    for i in range(Ndiffuse):

        # create material according to Ndiffuse (i=0, it is light red, i=Ndiffuse-1, it is dark red)
        # Create a new material and assign a color
        material = bpy.data.materials.new(name=f"diffuse_material_{i}")
        k = i / (Ndiffuse - 1)
        # color from light red to dark red
        red = np.array([1, 0, 0, 1.0])
        white = np.array([1, 1, 1, 0.1])
        color = (1-k) * white + k * red
        material.diffuse_color = color  # RGBA values



        # Enable 'Use Nodes' to access material nodes
        material.use_nodes = True
        nodes = material.node_tree.nodes
        principled_bsdf = nodes.get("Principled BSDF")
        if principled_bsdf:
            principled_bsdf.inputs["Base Color"].default_value = color
            # principled_bsdf.inputs['Emission'].default_value = color  # Emission color (optional)
        for j in range(Nsample):
            Ntraj = 3
            traj_names = ["torso", "left_foot_site", "right_foot_site"]
            for k, traj_name in enumerate(traj_names):
                # Create a new curve data-block
                curve_data = bpy.data.curves.new(
                    name=f"{traj_name}_diffuse{i}_sample{j}", type="CURVE"
                )
                delta_t = time.time() - t0
                total_exp = Ndiffuse*Nsample*Ntraj
                current_exp = i*Nsample*Ntraj + j*Ntraj + k + 1
                elapsed_time = delta_t
                left_time = (total_exp - current_exp) * elapsed_time / current_exp
                print(f"total progress: {i*Nsample*Ntraj + j*Ntraj + k}/{Ndiffuse*Nsample*Ntraj}, left time: {left_time:.2f}s, elapsed time: {elapsed_time:.2f}s")
                print(f"Creating {traj_name}_diffuse{i}_sample{j}...")
                curve_data.dimensions = "3D"

                # Adjust the bevel depth to make the curve thicker
                curve_data.bevel_depth = (
                    0.002  # Adjust this value for desired thickness
                )
                curve_data.fill_mode = "FULL"  # Ensures the curve is fully filled

                # Create a new basal spline in that curve
                spline = curve_data.splines.new(type="NURBS")
                if Hsample % Ndownsample == 0:
                    spline.points.add(Hsample//Ndownsample)  # Add points (minus the default point)
                else:
                    spline.points.add(Hsample//Ndownsample-1)  # Add points (minus the default point)
                spline.use_cyclic_u = False  # Ensure the spline is not cyclic
                spline.order_u = 4  # Order can be between 2 and 6
                spline.resolution_u = 12  # Increase for smoother curves

                # Initialize the spline points with the first frame data
                if traj_name == "torso":
                    traj = xssss_torso[:, i, j]
                elif traj_name == "left_foot_site":
                    traj = xssss_feet[:, i, j, :, 0]
                elif traj_name == "right_foot_site":
                    traj = xssss_feet[:, i, j, :, 1]
                for p in range(Hsample):
                    if p % Ndownsample != 0:
                        continue
                    x, y, z = traj[0, p]
                    spline.points[p//Ndownsample].co = (x, y, z, 1)  





                # Set the order of the NURBS spline (degree + 1)
                # spline.order_u = min(4, Hsample)  # Order cannot exceed number of points
                # spline.use_endpoint_u = True


                # Create a new object with the curve data
                curve_object = bpy.data.objects.new(
                    f"{traj_name}_diffuse{i}_sample{j}", curve_data
                )


                # Link the object to the current collection
                bpy.context.collection.objects.link(curve_object)

                # Assign the material to the curve object
                if curve_object.data.materials:
                    # Assign to first material slot
                    curve_object.data.materials[0] = material
                else:
                    # Create a new material slot and assign
                    curve_object.data.materials.append(material)

                # Animate the curve by modifying control points over time
                for frame in range(Hrender):
                    if frame % Hdownsample != 0:
                        continue
                    bpy.context.scene.frame_set(frame)
                    for k in range(Hsample):
                        if k % Ndownsample != 0:
                            continue
                        x, y, z = traj[frame, k]
                        spline.points[k//Ndownsample].co = (x, y, z, 1)


                        # Insert keyframe for the point
                        spline.points[k//Ndownsample].keyframe_insert(data_path="co", frame=frame)




if __name__ == "__main__":
    import_animation()
