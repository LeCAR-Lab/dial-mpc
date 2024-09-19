import sys
import os
# for windows
if os.name == "nt":
    packages_path = r"C:\Users\JC-Ba\AppData\Roaming\Python\Python311\Scripts" + r"\\..\\site-packages"
    sys.path.insert(0, packages_path)
import bpy
import pickle
import numpy as np
import matplotlib as mpl
from mathutils import Quaternion, Matrix, Vector, Euler


link_mapper = [
    "base",
    # "FL_foot_target",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    # "FL_foot",
    # "FR_foot_target",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    # "FR_foot",
    # "RL_foot_target",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    # "RL_foot",
    # "RR_foot_target",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    # "RR_foot",
]


def import_animation():
    dt = 0.02
    Hrender = 50
    link_pos = np.load(r"C:\Users\JC-Ba\Downloads\code\dial-mpc\data\go2_xpos.npy")[:Hrender]
    link_quat_wxyz = np.load(r"C:\Users\JC-Ba\Downloads\code\dial-mpc\data\go2_xquat.npy")[:Hrender]
    link_quat_xyzw = link_quat_wxyz[:, :, [1, 2, 3, 0]]
    link_pos = np.transpose(link_pos, (1, 0, 2))
    link_quat_xyzw = np.transpose(link_quat_xyzw, (1, 0, 2))

    # Isaac Gym is Y-up Right-handed coordinate system
    # Blender is Z-up left-handed coordinate system
    # So we need to convert the quaternion from Isaac Gym to Blender
    yup_to_zup = Euler((np.pi / 2, 0, 0), "XYZ").to_quaternion()
    for link_idx, (link_pos_traj, link_quat_traj) in enumerate(
        zip(link_pos, link_quat_xyzw)
    ):
        link_name = link_mapper[link_idx]
        if link_name not in bpy.data.objects:
            continue
        blender_obj = bpy.data.objects[link_name]
        for frame_idx, (pos, quat) in enumerate(zip(link_pos_traj, link_quat_traj)):
            # Insert position keyframe
            blender_obj.location = pos
            blender_obj.keyframe_insert("location", frame=frame_idx)
            # Change rotation mode to quaternion
            blender_obj.rotation_mode = "QUATERNION"
            # Blender's quaternion constructor is in wxyz format
            # Insert quaternion keyframe, applying transform
            blender_obj.rotation_quaternion = (
                Quaternion((quat[3], quat[0], quat[1], quat[2])) @ yup_to_zup
            )
            blender_obj.keyframe_insert("rotation_quaternion", frame=frame_idx)

    # Generate the actsss array
    Hsample = 20
    Nsample = 10
    Hrollout = link_pos.shape[1]
    actsss = np.zeros([Hrollout, Nsample, Hsample, 3])
    for i in range(Hrollout):
        for j in range(Nsample):
            actsss[i, j, :] = link_pos[0, i]
            actsss[i, j, :, 0] += (
                np.sin(i / Hrollout * np.pi + (i + j) / Hrollout * np.pi / 2) * 0.3
                + np.random.randn(Hsample) * 0.1
                + np.cos(i / Hrollout * np.pi * 2.0 + (i + j) / Hrollout * np.pi / 2) * 0.3
            )
            actsss[i, j, :, 2] += (
                np.cos(i / Hrollout * np.pi + (i + j) / Hrollout * np.pi / 2) * 0.3
                + np.random.randn(Hsample) * 0.1
                + np.cos(i / Hrollout * np.pi * 2.0 + (i + j) / Hrollout * np.pi / 2) * 0.3
            )

    # Create and animate curves based on actsss data
    for j in range(Nsample):
        # Create a new curve data-block
        curve_data = bpy.data.curves.new(name=f'AnimatedCurve_{j}', type='CURVE')
        curve_data.dimensions = '3D'

        # Adjust the bevel depth to make the curve thicker
        curve_data.bevel_depth = 0.005  # Adjust this value for desired thickness
        curve_data.fill_mode = 'FULL'  # Ensures the curve is fully filled

        # Create a new NURBS spline in that curve
        spline = curve_data.splines.new(type='NURBS')
        spline.points.add(Hsample - 1)  # Add points (minus the default point)

        # Initialize the spline points with the first frame data
        for i in range(Hsample):
            x, y, z = actsss[0, j, i]
            spline.points[i].co = (x, y, z, 1)  # The fourth value is the weight (set to 1)

        # Set the order of the NURBS spline (degree + 1)
        spline.order_u = min(4, Hsample)  # Order cannot exceed number of points
        spline.use_endpoint_u = True

        # Create a new object with the curve data
        curve_object = bpy.data.objects.new(f'AnimatedCurveObject_{j}', curve_data)

        # Link the object to the current collection
        bpy.context.collection.objects.link(curve_object)

        # Create a new material and assign a color
        material = bpy.data.materials.new(name=f'CurveMaterial_{j}')
        # Generate a random color based on the index j
        color = (np.random.rand(), np.random.rand(), np.random.rand(), 1.0)
        material.diffuse_color = color  # RGBA values

        # Enable 'Use Nodes' to access material nodes
        material.use_nodes = True
        nodes = material.node_tree.nodes
        principled_bsdf = nodes.get('Principled BSDF')
        if principled_bsdf:
            principled_bsdf.inputs['Base Color'].default_value = color

        # Assign the material to the curve object
        if curve_object.data.materials:
            # Assign to first material slot
            curve_object.data.materials[0] = material
        else:
            # Create a new material slot and assign
            curve_object.data.materials.append(material)

        # Animate the curve by modifying control points over time
        for frame in range(Hrollout):
            bpy.context.scene.frame_set(frame)
            for i in range(Hsample):
                x, y, z = actsss[frame, j, i]
                spline.points[i].co = (x, y, z, 1)
                # Insert keyframe for the control point
                spline.points[i].keyframe_insert(data_path='co', frame=frame)

    # Set the scene frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = Hrollout - 1


if __name__ == "__main__":
    import_animation()
