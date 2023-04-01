import numpy as np
import pyvista as pv

values = np.linspace(0, 10, 48).reshape((6, 8, 1))
grid = pv.UniformGrid()
grid.dimensions = [s + 1 for s in values.shape]
grid.origin = (0.0, 0.0, 0.0)
grid.spacing = (1.5, 1, 1)
grid.cell_data["values"] = values.flatten()

p = pv.Plotter()
x_axis = pv.Arrow(scale=20, direction=(1, 0, 0))
p.add_mesh(x_axis, color="red", show_edges=True)
y_axis = pv.Arrow(scale=20, direction=(0, 1, 0))
p.add_mesh(y_axis, color="green", show_edges=True)
z_axis = pv.Arrow(scale=20, direction=(0, 0, 1))
p.add_mesh(z_axis, color="blue", show_edges=True)

probe_point = np.random.uniform(low=[0.0, 0.0, 0.0], high=[6.0, 8.0, 0.0])
p.add_mesh(pv.Arrow(start=probe_point, direction=(0, 0, 1), scale=5),
           color="orange",
           show_edges=True)
p.add_mesh(grid, scalars="values", show_edges=True)
print(grid.probe(probe_point).point_data["values"].item())  # pyright: ignore
p.show(cpos="xy")
