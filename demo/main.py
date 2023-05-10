import numpy as np
import pypolo
import pyvista as pv

array = np.load("./data/N17E073.npy").astype(np.float64)
array /= 100
array = array[:, :, np.newaxis]
env = pypolo.utils.TensorMap(array, origin=np.zeros(3), resolution=0.1)

plotter = pv.Plotter()
terrain = env.grid.warp_by_scalar()
plotter.add_mesh(terrain, cmap="terrain")
plotter.add_axes(interactive=True)
auv = pv.read("./data/auv.stl")
auv = auv.translate([0, 0, array.max()]).scale(2)
plotter.add_mesh(auv, color="orange")
plotter.add_bounding_box(line_width=1, color='grey')
plotter.show()
