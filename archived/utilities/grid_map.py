from typing import List

import numpy as np


class GridMap:
    """GridMap is a matrix with Cartesian coordinates."""
    def __init__(self, matrix: np.ndarray, extent: List[float]) -> None:
        """

        Parameters
        ----------
        matrix: np.ndarray, shape=(num_rows, num_cols)
            A 2-D array representing the ground-truth values.
        extent: List[float], [xmin, xmax, ymin, ymax]
            The bounding box of the environment.

        """
        self.matrix = matrix
        self.extent = extent
        self.num_rows, self.num_cols = matrix.shape
        # The matrix is slightly larger than the Cartesian coordinate
        # to prevent out-of-range acces.
        self.eps = 1e-4
        self.x_cell_size = (extent[1] - extent[0]) / self.num_cols + self.eps
        self.y_cell_size = (extent[3] - extent[2]) / self.num_rows + self.eps

    def xs_to_cols(self, xs: np.ndarray) -> np.ndarray:
        """Transform x values to column indices.

        Parameters
        ----------
        xs: np.ndarray, shape=(num_samples,)
            x elements of the sampling locations.

        Returns
        -------
        cols: np.ndarray, shape=(num_samples,)
            Column indices corresponds to `xs`.

        """
        cols = ((xs - self.extent[0]) / self.x_cell_size).astype(int)
        return cols

    def ys_to_rows(self, ys: np.ndarray) -> np.ndarray:
        """Transform y values to row indices.

        Parameters
        ----------
        ys: np.ndarray, shape=(num_samples,)
            y elements of the sampling locations.

        Returns
        -------
        rows: np.ndarray, shape=(num_samples,)
            Row indices corresponds to `ys`.

        """
        rows = ((ys - self.extent[2]) / self.y_cell_size).astype(int)
        return rows

    def get(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Get matrix values at some locations in the Cartesian coordinate.

        Parameters
        ----------
        xs: np.ndarray, shape=(num_samples,)
            x elements of the sampling locations.
        ys: np.ndarray, shape=(num_samples,)
            y elements of the sampling locations.

        Returns
        -------
        values: np.ndarray, shape=(num_samples,)
            Matrix values at the given locations.

        """
        cols = self.xs_to_cols(xs)
        rows = self.ys_to_rows(ys)
        values = self.matrix[rows, cols]
        return values

    def set(self, xs: np.ndarray, ys: np.ndarray, values: np.ndarray) -> None:
        """Set matrix values at some locations in the Cartesian coordinate.

        Parameters
        ----------
        xs: np.ndarray, shape=(num_samples,)
            x elements of the sampling locations.
        ys: np.ndarray, shape=(num_samples,)
            y elements of the sampling locations.
        values: np.ndarray, shape=(num_samples,)
            Matrix values at the given locations.

        Attributes
        ----------
        matrix: np.ndarray, shape=(num_rows, num_cols)
            The elements in the matrix corresponding to the given locations
            will be changed to the given values.

        """
        cols = self.xs_to_cols(xs)
        rows = self.ys_to_rows(ys)
        self.matrix[rows, cols] = values
