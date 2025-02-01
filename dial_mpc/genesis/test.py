import genesis as gs
gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=False,
    rigid_options = gs.options.RigidOptions(
        dt = 0.02,
        iterations = 1,
        ls_iterations = 1,
    ),
)
go2 = scene.add_entity(gs.morphs.MJCF(
    file = "/home/haoru/research/dial/dial-mpc/dial_mpc/models/unitree_go2/mjx_go2_force.xml"
    # file = "/home/haoru/research/dial/dial-mpc/dial_mpc/models/unitree_go2/mjx_go2_position.xml"
    # file = "/home/haoru/research/dial/dial-mpc/dial_mpc/models/unitree_go2/mjx_scene_force_crate.xml"
    # file = "/home/haoru/research/dial/dial-mpc/dial_mpc/models/unitree_go2/mjx_scene_force.xml"
    # file = "/home/haoru/research/dial/dial-mpc/dial_mpc/models/unitree_h1/mjx_h1_loco.xml"
    # file = "/home/haoru/research/dial/dial-mpc/dial_mpc/models/unitree_h1/h1_real_feet.xml"
))
plane = scene.add_entity(gs.morphs.Plane())

scene.build(n_envs=2048)
for i in range(1000):
    scene.step()