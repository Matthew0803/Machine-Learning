import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import torch

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3.5, -1.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = None,
    ),
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########################## build ##########################

# create 20 parallel environments
B = 8
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# control all the robots

franka.control_dofs_position(
    position = torch.zeros(3, 9, device=gs.device),
    envs_idx = torch.tensor([1, 5, 7], device=gs.device),
)

for i in range(1000):
    scene.step()