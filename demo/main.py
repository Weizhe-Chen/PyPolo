import math
import numpy as np
import pypolo
import pyvista as pv
from tqdm import tqdm

array = np.load("./data/N17E073.npy").astype(np.float64)
array /= 100
array = array[:, :, np.newaxis]
env = pypolo.utils.TensorMap(array, origin=np.zeros(3), resolution=0.1)

plotter = pv.Plotter(off_screen=True)
terrain = env.grid.warp_by_scalar()
robot_mesh = pv.read("./data/robot_mesh.stl")
robot_mesh = robot_mesh.translate([0, 0, array.max()]).scale(2)

state = np.array([5.0, 5.0, 0, 10.0, 0])
robot = pypolo.robots.DifferentialDriveRobot(hertz=60.0, state=state)

plotter.open_gif("./animation.gif")
num_steps = 100
for i in tqdm(range(num_steps)):
    action = np.array([0.0, np.random.uniform(-0.5, 0.5)])
    robot.take(action)
    updated_robot_mesh = robot_mesh.rotate_z(
        robot.state[2] * 180 / math.pi).translate(
            (robot.state[0], robot.state[1], 0))

    plotter.add_axes()
    plotter.add_mesh(terrain, cmap="terrain", lighting=False)
    plotter.add_mesh(updated_robot_mesh, color="orange", lighting=False)
    plotter.add_bounding_box(line_width=1, color='grey')
    plotter.add_text(f"Timestep: {i:03d}", font_size=18)

    plotter.write_frame()
    plotter.clear()

plotter.close()
