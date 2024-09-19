import os
import bpy
import pickle
import numpy as np
import matplotlib as mpl
from mathutils import Quaternion, Matrix, Vector

object_mapper = {
    "ur5e/base": "base",
    "ur5e/shoulder_link": "shoulder_link",
    "ur5e/upper_arm_link": "upper_arm_link",
    "ur5e/forearm_link": "forearm_link",
    "ur5e/wrist_1_link": "wrist_1_link",
    "ur5e/wrist_2_link": "wrist_2_link",
    "ur5e/wrist_3_link": "wrist_3_link",
    "ur5e/wsg50/base": "WSG50_110",
    "ur5e/wsg50/right_finger": "right_finger",
    "ur5e/wsg50/left_finger": "left_finger",
    "drawer/bottom_drawer": "bottom_drawer",
    "drawer/middle_drawer": "middle_drawer",
    "drawer/top_drawer": "top_drawer",
    "vitamin bottle/": "vitamin",
    "purple pencilcase/": "pencilcase",
    "crayon box/": "crayon",
    "horse toy/": "horse",
    "left_bin/left_bin": "left_bin",
    "right_bin/right_bin": "right_bin",
    "mailbox/mailbox_flag": "mailbox_flag",
    "mailbox/mailbox": "mailbox_body",
    "mailbox/mailbox_lid": "mailbox_lid",
    "amazon package/amazon package": "amazon_package",
    "yellow_block/yellow_block": "yellow_block",
    "closest_box/closest_box": "closest_box",
    "middle_box/middle_box": "middle_box",
    "furthest_box/furthest_box": "furthest_box",
    "catapult/catapult": "catapult frame",
    "catapult/button": "catapult button",
    "catapult/catapult_arm": "catapult arm",
    "school bus toy/school bus toy": "school_bus",
    "red_block/red_block": "red_block",
}


def import_animation(root_path: str):
    anim_data = pickle.load(open(root_path + "anim_data.pkl", "rb"))
    for obj_key, timestamped_obj_poses in anim_data.items():
        if obj_key not in object_mapper:
            continue
        blender_obj_name = object_mapper[obj_key]
        blender_obj = bpy.data.objects[blender_obj_name]
        for frame_idx, (timestamp, pos, euler) in enumerate(timestamped_obj_poses):
            # insert position keyframe
            blender_obj.location = pos
            blender_obj.keyframe_insert("location", frame=frame_idx)
            # change rotation mode to euler
            blender_obj.rotation_mode = "XYZ"
            # insert rotation keyframe
            blender_obj.rotation_euler = euler
            blender_obj.keyframe_insert("rotation_euler", frame=frame_idx)


def import_traces(root_path: str, num_extra_pts: int = 20):
    action_path = os.path.join(root_path, "timestamped_actions.pkl")
    timestamped_actions = pickle.load(open(action_path, "rb"))
    new_collection = bpy.data.collections.new("traces")
    bpy.context.scene.collection.children.link(new_collection)

    for frame_idx, (timestamp, action_seqs) in enumerate(timestamped_actions.items()):
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
        modifier.node_group = bpy.data.node_groups["Particle Viewer.001"]
        # set active object
        bpy.context.view_layer.objects.active = new_object
        bpy.context.active_object.hide_render = True
        bpy.context.active_object.keyframe_insert("hide_render", frame=frame_idx - 1)
        bpy.context.active_object.hide_render = False
        bpy.context.active_object.keyframe_insert("hide_render", frame=frame_idx)
        bpy.context.active_object.hide_render = True
        bpy.context.active_object.keyframe_insert("hide_render", frame=frame_idx + 1)


if __name__ == "__main__":
    # from scripts/export_blender_visualization.py
    root_path = "/path/to/output/dir"
    import_traces(root_path)