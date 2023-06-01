import os
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Euler
import sys
import shutil
from pathlib import Path
import interpolation
import math
from uv_mapping import UV_Map

class Camera():
    def __init__(self):
        self.cam = self.render_settings()
        

    def render_settings(self):       
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.samples = 4

        # camera settings
        cam = bpy.data.scenes["Scene"].objects['Camera']

        return cam
def select_obj(obj, deselect=True, active=True):
    if deselect:
        bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    if active:
        bpy.context.view_layer.objects.active = obj

def get_path(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

def render_animations(save=False):
    if save:
        bpy.ops.wm.save_as_mainfile(filepath=get_path("output\\Anim_Test.blend"))
    folder_path = get_path("Output\\animation_renders")
    shutil.rmtree(folder_path)
    output_path = folder_path + "\\Snowboard"
    bpy.data.scenes['Scene'].render.filepath = output_path
    bpy.ops.render.render(write_still=False, animation=True)

def pretty_setup():
    bpy.ops.wm.open_mainfile(filepath=get_path("Final Scenes and Objects/Snowy Mountain Scene.blend"))
    
    scene = bpy.data.scenes["Scene"]
    mountain = scene.objects["Snowy Mountain"]
    girl = scene.objects["Girl"]

    return (scene, mountain, girl)

def add_snowboard(girl):
    bottom_half_path = get_path("Final Scenes and Objects\\Snowboard_Bottom_Half")
    top_path = get_path("Final Scenes and Objects\\Snowboard_Top")
    texture_path = get_path("Final Scenes and Objects\\Snowboard Texture.png")

    # The following lines use LSCM to map a texture onto 2 halves of a snowboard mesh!
    # Once these lines are run, the mapped object will be permanently created in the
    # directory, so you can comment out these lines after running this function once.
    # ---------------------------------------------------------------------------------
    # UV_Map.map(bottom_half_path, uv_scale=0.5, uv_offset=(0.5, 0.25))
    # UV_Map.map(top_path, uv_scale=0.5, uv_offset=(0, 0.25))
    # ---------------------------------------------------------------------------------

    bpy.ops.import_scene.obj(filepath=bottom_half_path + "_mapped.obj")
    bottom_half = bpy.context.selected_objects[0]

    bpy.ops.import_scene.obj(filepath=top_path + "_mapped.obj")
    top = bpy.context.selected_objects[0]

    select_obj(bottom_half)
    select_obj(top, False)
    bpy.ops.object.join() # top is now the full snowboard
    
    top.location = Vector((0, 0.24, 0.12))
    top.rotation_mode = 'QUATERNION'
    top.rotation_quaternion = Vector((0.517, 0.482, 0.482, 0.517))

    mat = bpy.data.materials.new(name="New_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(texture_path)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

    top.data.materials[0] = mat
    top.data.materials[1] = mat

    select_obj(top)
    select_obj(girl, False)
    bpy.ops.object.join() # Snowboard now merged into the girl
    
def generate_keyframes(scene, girl, cam):
    seconds = 10
    fps = 30
    frames = seconds * fps

    scene.frame_end = frames

    girl_positions = [(84.782, 233.65, 68.729), (83.963, 227.99, 68.552), (81.649, 221.38, 63.816), (84.944, 217.92, 57.625), (89.513, 213.14, 49.042), (95.178, 217.46, 47.805), (97.084, 212.27, 36.642), (99.793, 204.9, 20.779), (99.115, 199.37, 10.004), (101.3, 196.23, -0.50026), (103.93, 192.44, -13.156), (111.07, 184.92, -26.67), (115.38, 177.23, -34.437), (119.19, 169.98, -41.229), (121.66, 163.8, -48.245), (127.19, 149.91, -55.904), (133.42, 139.26, -61.774), (128.01, 128, -53.106), (136.28, 120.76, -49.545), (143.92, 114.07, -46.252), (143.09, 114.15, -46.809), (152.51, 108.56, -44.864), (159.34, 104.5, -43.454), (168.04, 100.32, -46.111), (172.6, 95.966, -44.039), (176.64, 92.105, -42.201), (180.09, 90.106, -38.906), (184.07, 87.832, -34.989), (186.62, 86.377, -32.485), (188.05, 85.562, -30.946), (189.46, 84.741, -29.335)]
    girl_rotations = [(-0.688, 0.018, -0.039, -0.724), (-0.173, -0.345, 0.054, -0.921), (-0.565, -0.289, 0.440, -0.633), (-0.524, -0.300, 0.513, -0.610), (-0.529, -0.139, 0.403, -0.734), (-0.610, 0.182, 0.416, -0.649), (-0.216, 0.302, 0.432, -0.822), (-0.229, 0.183, 0.323, -0.900), (-0.417, 0.308, 0.202, -0.831), (-0.136, 0.418, 0.141, -0.887), (-0.129, 0.474, 0.142, -0.859)]
    cam_positions = [(66.266, 251.12, 82.459), (70.158, 221.58, 81.583), (70.017, 195.61, 40.602), (92.336, 175.01, -14.946), (124.01, 164.66, -40.136), (99.36, 153.4, -60.011), (100.71, 112.11, -62.747), (123.4, 105.66, -46.725), (142.71, 86.477, -41.448), (168.69, 84.472, -40.102), (168.75, 80.671, -36.399)]
    cam_rotations = [(0.334, 0.213, -0.493, -0.775), (0.510, 0.196, -0.234, -0.804), (0.605, 0.573, -0.331, -0.444), (0.470, 0.842, -0.234, -0.124), (0.524, 0.802, 0.244, 0.154), (0.362, 0.460, -0.634, -0.506), (0.430, 0.576, -0.555, -0.419), (0.464, 0.481, -0.532, -0.520), (0.565, 0.537, -0.429, -0.457), (0.452, 0.625, -0.514, -0.376), (0.500, 0.633, -0.462, -0.369)]


    # Add extra elements to cam values because both types of splines don't mesh well 
    # with early and late values
    # girl_positions = [girl_positions[0]] + girl_positions + [girl_positions[-1]]
    girl_rotations = [girl_rotations[0]] + girl_rotations + [girl_rotations[-1]]
    cam_positions = [cam_positions[0]] + cam_positions + [cam_positions[-1]]
    cam_rotations = [cam_rotations[0]] + cam_rotations + [cam_rotations[-1]]

    # Convert to np arrays so one spline can be used for all 3 values
    # thanks to numpy's entrywise operations
    girl_positions = list(map(lambda x : np.array(x), girl_positions))
    girl_rotations = list(map(lambda x : np.array(x), girl_rotations))
    cam_positions = list(map(lambda x : np.array(x), cam_positions))
    cam_rotations = list(map(lambda x : np.array(x), cam_rotations))

    # girl_pos_spline = interpolation.CatmullRomSpline(girl_positions)
    girl_pos_spline = interpolation.BezierSpline(list(np.linspace(0, seconds, len(girl_positions))), girl_positions, 3)
    girl_rot_spline = interpolation.CatmullRomSpline(girl_rotations)
    cam_pos_spline = interpolation.CatmullRomSpline(cam_positions)
    cam_rot_spline = interpolation.CatmullRomSpline(cam_rotations)

    cam.rotation_mode = 'QUATERNION'
    girl.rotation_mode = 'QUATERNION'

    for frame in range(frames):
        t = frame / fps

        girl.location = Vector(list(girl_pos_spline.interp(t)))
        girl.keyframe_insert(data_path='location', frame=frame)


        girl.rotation_quaternion = Vector(list(girl_rot_spline.interp(t)))
        girl.keyframe_insert(data_path='rotation_quaternion', frame=frame)

        cam.location = Vector(list(cam_pos_spline.interp(t)))
        cam.rotation_quaternion = Vector(list(cam_rot_spline.interp(t)))

        cam.keyframe_insert(data_path='location', frame=frame)
        cam.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        
if __name__ == '__main__':
    scene, mountain, girl = pretty_setup()

    cam = Camera().cam

    add_snowboard(girl)
    
    generate_keyframes(scene, girl, cam)

    render_animations()