"""Render V2 hand + cube with collision boxes visible (wireframe) overlaid on visual meshes."""
import mujoco
import numpy as np
from pathlib import Path

# Load model
xml_path = Path(__file__).parent / "mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Set to keyframe 0 (hand + cube at initial pose)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Create renderer (within offscreen buffer limits)
renderer = mujoco.Renderer(model, height=480, width=640)

# Camera setup - use side_high camera
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
# Find camera id
for i in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    print(f"Camera {i}: {name}")
cam.fixedcamid = 0  # first camera

# Render 1: Visual only (group 2 + cube group 1)
scene_option = mujoco.MjvOption()
# Enable groups: 1 (cube), 2 (visual mesh)
scene_option.geomgroup[0] = 0  # group 0 off
scene_option.geomgroup[1] = 1  # group 1 on (cube)
scene_option.geomgroup[2] = 1  # group 2 on (visual meshes)
scene_option.geomgroup[3] = 0  # group 3 off (collision)
scene_option.geomgroup[4] = 0  # group 4 off
scene_option.geomgroup[5] = 0  # group 5 off

renderer.update_scene(data, cam, scene_option)
img_visual = renderer.render()

# Render 2: Collision only (group 3 + cube group 1)
scene_option2 = mujoco.MjvOption()
scene_option2.geomgroup[0] = 0
scene_option2.geomgroup[1] = 1  # cube
scene_option2.geomgroup[2] = 0  # visual off
scene_option2.geomgroup[3] = 1  # collision on
scene_option2.geomgroup[4] = 0
scene_option2.geomgroup[5] = 0
scene_option2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1  # transparent

renderer.update_scene(data, cam, scene_option2)
img_collision = renderer.render()

# Render 3: Both overlaid (visual + collision with transparency)
scene_option3 = mujoco.MjvOption()
scene_option3.geomgroup[0] = 0
scene_option3.geomgroup[1] = 1  # cube
scene_option3.geomgroup[2] = 1  # visual on
scene_option3.geomgroup[3] = 1  # collision on
scene_option3.geomgroup[4] = 0
scene_option3.geomgroup[5] = 0

renderer.update_scene(data, cam, scene_option3)
img_both = renderer.render()

# Save images
from PIL import Image
for name, img in [("visual", img_visual), ("collision", img_collision), ("both", img_both)]:
    Image.fromarray(img).save(f"/home/ll/SRTP/Aero-Hand/collision_viz_{name}.png")
    print(f"Saved collision_viz_{name}.png")

# Also render from palm camera
for i in range(model.ncam):
    cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    if "palm" in cname.lower():
        cam.fixedcamid = i
        break

renderer.update_scene(data, cam, scene_option3)
img_palm = renderer.render()
Image.fromarray(img_palm).save("/home/ll/SRTP/Aero-Hand/collision_viz_palm.png")
print("Saved collision_viz_palm.png")

print("\nDone! Check collision_viz_*.png files")
