import numpy as np
import pyvista

# Make the xyz points
theta = np.linspace(-10 * np.pi, 10 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
points = np.column_stack((x, y, z))

spline = pyvista.Spline(points, 500).tube(radius=0.1)
spline.plot(scalars='arc_length', show_scalar_bar=False)
