import genesis as gs
gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True,renderer = gs.renderers.Rasterizer())
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build()

for i in range(1000):
    scene.step()