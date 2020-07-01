import bpy
import mathutils
import numpy as np
import bmesh
from itertools import permutations
import math
import os
import json

################################
#### Grid Search Parameters ####
################################

bounds_min = [-9.1, -3.5, 0]
bounds_max = [5.0,  4.4, 3]

grid_steps = [1, 5, 2]


####################
#### Scene Data ####
####################

armature = bpy.data.objects["Armature"]
pose = bpy.data.objects["Armature"].pose

fingertip_targets = bpy.data.objects["Fingertips"].children
fingertip_bones = {
    "thumb": pose.bones["finger5joint3"],
    "index": pose.bones["finger4joint3"],
    "middle": pose.bones["finger3joint3"],
    "ring": pose.bones["finger2joint3"],
    "pinky": pose.bones["finger1joint3"],
}

###########################
#### Utility Functions ####
###########################

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR.endswith(".blend"):
    SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)


def count_permutations(n, r):
    result = 1
    for x in range(n - r + 1, n + 1):
        result *= x
    return result


def compute_relative_rotation(bone):
    mode = bone.rotation_mode
    if mode == 'QUATERNION':
        return bone.rotation_quaternion.copy()
    elif mode == 'AXIS_ANGLE':
        return mathutils.Quaternion(bone.rotation_axis_angle)
    else:
        return bone.rotation_euler.to_quaternion()


def check_pose():
    # Ensure that all relvant IK constraints are satisfied
    for target in fingertip_targets:
        name = target.name.lower()
        if name in fingertip_bones:
            target_pos = target.matrix_world.to_translation()
            fingertip_pos = armature.matrix_world @ fingertip_bones[name].tail

            if (target_pos - fingertip_pos).magnitude > 0.001:
                return False
    return True


def render(filename):
    print("\n")
    bpy.context.scene.render.filepath = filename
    bpy.ops.render.render(write_still=True)


def export_targets(filename):
    result = {}
    for target in fingertip_targets:
        target_name = str(target.name)
        result[target_name] = {
            "location": [
                 target.location.x, 
                 target.location.y, 
                 target.location.z
            ] 
        }
    
    with open(filename, "w") as f:
        f.write(json.dumps(result))


def matrix_to_list(matrix):
    return [
        list(matrix[0]),
        list(matrix[1]),
        list(matrix[2]),
        list(matrix[3])
    ]

def export_armature_pose(filename):
    result = {}
    for bone in pose.bones:
        bone_name = str(bone.name)
        result[bone_name] = {"matrix": matrix_to_list(bone.matrix)}

    with open(filename, "w") as f:
        f.write(json.dumps(result))

#####################
#### Grid Search ####
#####################


# Generate grid points
X = np.linspace(bounds_min[0], bounds_max[0], grid_steps[0])
Y = np.linspace(bounds_min[1], bounds_max[1], grid_steps[1])
Z = np.linspace(bounds_min[2], bounds_max[2], grid_steps[2])

X_ = np.broadcast_to(X[:, None, None], grid_steps)
Y_ = np.broadcast_to(Y[None, :, None], grid_steps)
Z_ = np.broadcast_to(Z[None, None, :], grid_steps)

grid_points = np.stack([X_, Y_, Z_], axis=-1).reshape(-1, 3)

# Loop through all combs
targets = [target for target in fingertip_targets if target.name not in ["Palm"]]

num_to_try = count_permutations(grid_points.shape[0], len(targets))
total_tried = 0
pose_id = 0
for conf in permutations(range(grid_points.shape[0]), len(targets)):
    for i in range(len(targets)):
        targets[i].location.x = grid_points[conf[i]][0]
        targets[i].location.y = grid_points[conf[i]][1]
        targets[i].location.z = grid_points[conf[i]][2]

    bpy.context.view_layer.update()

    total_tried += 1
    # if total_tried % 10 == 0:
    print(f"Tried: {total_tried}/{num_to_try}, Exported: {pose_id}", end="\r")
    if not check_pose():
        continue

    render(os.path.join(SCRIPT_DIR, f"output/{pose_id}.png"))
    export_targets(os.path.join(SCRIPT_DIR, f"output/{pose_id}-targets.json"))
    export_armature_pose(os.path.join(SCRIPT_DIR, f"output/{pose_id}-pose.json"))
    pose_id += 1

print()
