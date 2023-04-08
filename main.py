import os
import numpy as np
import pypolo
import pyvista as pv
from pyvista import examples

array = np.load("./N17E073.npy").astype(np.float64)
array /= 100
array = array[:, :, np.newaxis]
env = pypolo.utils.TensorMap(array, origin=np.zeros(3), resolution=0.1)

plotter = pv.Plotter()
terrain = env.grid.warp_by_scalar()
plotter.add_mesh(terrain, cmap="terrain")
plotter.add_axes(interactive=True)

body_length = 0.5
body_width = 0.3
body_height = 0.2
box = pv.Box(bounds=[
    -body_length / 2,
    body_length / 2,
    -body_width / 2,
    body_width / 2,
    -body_height / 2,
    body_height / 2,
])

plotter.add_mesh(box, color="red")
plotter.show()
