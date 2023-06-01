# Snowboarder Animation
#### Jacob Serfaty's *Intro to Computer Graphics* final project

# What did I do for this project?

For this project, I implemented two separate modules.

First, I added Bezier, Hermite, Cardinal, and Catmull-Rom splines (they are in `interpolation.py`). 
I then used these splines in my animation to make the snowboarder move.

Second, I added Least Square Conformal Mapping (they are in `uv_mapping.py`).
I was originally going to use this to map all of my textures, but the requirement
that the meshes had to have a disc topology made this way harder than I expected.
I implemented another module to help with this (Seams, also in `uv_mapping.py`)
that would allow you to easily edit meshes to give them disc topology without changing 
the shape of the mesh. However, though this worked for smaller meshes, I had
trouble using it with larger meshes since I didn't know which edges to choose to make
seams. I eventually settled on using this only for a small part of the animation:
the texture of the snowboard.

My actual animation is just a snowboarder moving along splines and should be self explanitory.

The 3D models I used were all free and all of the materials that are applied to them came 
withe models. Note that the model for the girl who snowboards and the model for the 
mountains are both included in a Blender scene instead of in an obj file because
for some reason, exporting them from blender ruins their materials.

# How to run this project (assuming you have all of the libraries I used):
1. Clone this repo
2. Unzip `Final Scenes and Objects/Snowy Mountain Scene.zip` into `Final Scenes and Objects`
2. Run `main.py`
3. Run `save_video.py`

## 3D Model Sources:
Mountains: https://www.turbosquid.com/3d-models/free-snowy-terrain-landscapes-3d-model/584949

Snowboard: https://www.turbosquid.com/3d-models/snow-board-obj-free/871115

Girl: https://www.turbosquid.com/3d-models/girl-model-1637866

Snow material for the mountains: https://www.blenderkit.com/asset-gallery-detail/f758fcd2-7161-442b-896b-dc987bdef98b/