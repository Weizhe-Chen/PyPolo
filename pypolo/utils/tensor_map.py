import numpy as np

import pyvista as pv


class TensorMap:

    def __init__(
        self,
        tensor: np.ndarray,
        origin: np.ndarray,
        resolution: float,
    ) -> None:
        self.origin = origin
        self.resolution = resolution
        self.set_tensor(tensor)

    def set_tensor(self, tensor: np.ndarray) -> None:
        assert tensor.ndim == 3, "The tensor must be 3D."
        self.tensor = tensor
        self.num_rows, self.num_cols, self.num_layers = tensor.shape
        self.update_geometry()
        self.grid.cell_data["values"] = self.tensor.flatten()

    def update_geometry(self) -> None:
        self.len_x = self.num_cols * self.resolution
        self.len_y = self.num_rows * self.resolution
        self.min_x = self.origin[0]
        self.max_x = self.origin[0] + self.len_x
        self.min_y = self.origin[1]
        self.max_y = self.origin[1] + self.len_y
        self.grid = pv.UniformGrid(
            dimensions=[
                self.num_rows + 1,
                self.num_cols + 1,
                self.num_layers + 1,
            ],
            origin=self.origin,
            spacing=[self.resolution] * 3,
        )

    def get_values(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        self._check_positions(xs, ys)
        cols = np.floor((xs - self.min_x) / self.resolution).astype(int)
        rows = np.floor((ys - self.min_y) / self.resolution).astype(int)
        return self.tensor[rows, cols]

    def _check_positions(self, xs: np.ndarray, ys: np.ndarray):
        if (self.min_x > xs).any() or (xs >= self.max_x).any():
            raise ValueError("The x values are out of the map.")
        if (self.min_y > ys).any() or (ys >= self.max_y).any():
            raise ValueError("The y values are out of the map.")

    def plot(self, plotter: pv.Plotter = None) -> pv.Plotter:
        if plotter is None:
            plotter = pv.Plotter()
        plotter.add_mesh(
            self.grid,
            scalars="values",
            show_edges=True,
        )
        return plotter
