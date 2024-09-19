import os
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
    Hrender = 10
    link_pos = np.load("/Users/pcy/Downloads/dial-mpc/data/go2_xpos.npy")[:Hrender]
    link_quat_wxyz = np.load("/Users/pcy/Downloads/dial-mpc/data/go2_xquat.npy")[:Hrender]
    link_quat_xyzw = link_quat_wxyz[:, :, [1, 2, 3, 0]]
    # Nlink = link_pos.shape[1]
    # task = "jogging"
    # for i in range(Nlink):
    #     link_pos[:, i, 0] = np.sin(np.arange(Hrollout) * dt + i / Nlink * np.pi / 2)
    #     link_quat_xyzw[:, i, 0] = np.sin(
    #         np.arange(Hrollout) * dt + i / Nlink * np.pi / 2
    #     )
    #     link_quat_xyzw[:, i, 1] = np.cos(
    #         np.arange(Hrollout) * dt + i / Nlink * np.pi / 2
    #     )
    link_pos = np.transpose(link_pos, (1, 0, 2))
    link_quat_xyzw = np.transpose(link_quat_xyzw, (1, 0, 2))

    # isaac gym is Y-up Right-handed coordinate system
    # blender is  Z-up left-handed coordinate system
    # so we need to convert the quaternion from isaac gym to blender
    yup_to_zup = Euler((np.pi / 2, 0, 0), "XYZ").to_quaternion()
    for link_idx, (link_pos_traj, link_quat_traj) in enumerate(
        zip(link_pos, link_quat_xyzw)
    ):
        link_name = link_mapper[link_idx]
        if link_name not in bpy.data.objects:
            continue
        blender_obj = bpy.data.objects[link_name]
        for frame_idx, (pos, quat) in enumerate(zip(link_pos_traj, link_quat_traj)):
            # insert position keyframe
            blender_obj.location = pos
            blender_obj.keyframe_insert("location", frame=frame_idx)
            # change rotation mode to quaternion
            blender_obj.rotation_mode = "QUATERNION"
            # blender's quaternion constructor is in wxyz format
            # insert quaternion keyframe, applying transform
            blender_obj.rotation_quaternion = (
                Quaternion((quat[3], quat[0], quat[1], quat[2])) @ yup_to_zup
            )
            blender_obj.keyframe_insert("rotation_quaternion", frame=frame_idx)

            mesh = bpy.data.meshes.new(f"trace_{frame_idx}")

    Hsample = 20
    Nsample = 10
    Hrollout = link_pos.shape[1]
    actsss = np.zeros([Hrollout, Nsample, Hsample, 3])
    for i in range(Hrollout):
        for j in range(Nsample):
            actsss[i, j, :] = link_pos[0, i]
            actsss[i, j, :, 0] += np.sin(i / Hrollout * np.pi + (i+j) / Hrollout * np.pi / 2)
            actsss[i, j, :, 2] += np.cos(i / Hrollout * np.pi + (i+j) / Hrollout * np.pi / 2)
    num_extra_pts = 10
    new_collection = bpy.data.collections.new("traces")
    bpy.context.scene.collection.children.link(new_collection)
    for frame_idx, action_seqs in enumerate(actsss):
        vertices = []
        vertex_colors = []
        for action_seq in action_seqs:
            prev_pos = action_seq[0][:3]
            # densely interpolate

            total_pts = len(action_seq) * num_extra_pts
            action_colors = mpl.colormaps["jet"](np.linspace(0, 1, total_pts)).tolist()
            p_idx = 0

            for action in action_seq[1:]:
                pos = action[:3]
                for i in range(num_extra_pts):
                    alpha = i / num_extra_pts
                    vertices.append((alpha * pos + (1 - alpha) * prev_pos).tolist())
                    vertex_colors.append(action_colors[p_idx])
                    p_idx += 1
                prev_pos = action[:3]
        mesh = bpy.data.meshes.new(f"trace_{frame_idx}")
        mesh.from_pydata(vertices, [], [])
        mesh.update()
        new_object = bpy.data.objects.new(f"trace_{frame_idx}", mesh)

        colattr = new_object.data.color_attributes.new(
            name="FloatColAttr",
            type="FLOAT_COLOR",
            domain="POINT",
        )
        for v_idx in range(len(vertices)):
            colattr.data[v_idx].color = vertex_colors[v_idx]
        new_collection.objects.link(new_object)
        # add geometry node modifier
        modifier = new_object.modifiers.new(name="GeometryNodes", type="NODES")
        # create a new node group
        node_group = bpy.data.node_groups.new(name="Particle Viewer", type="GeometryNodeTree")
        modifier.node_group = node_group
        # set active object
        bpy.context.view_layer.objects.active = new_object
        bpy.context.active_object.hide_render = True
        bpy.context.active_object.keyframe_insert("hide_render", frame=frame_idx - 1)
        bpy.context.active_object.hide_render = False
        bpy.context.active_object.keyframe_insert("hide_render", frame=frame_idx)
        bpy.context.active_object.hide_render = True
        bpy.context.active_object.keyframe_insert("hide_render", frame=frame_idx + 1)

if __name__ == "__main__":
    import_animation()
