import numpy as np


def test_init(grid_map):
    num_cells_x = int(
        (grid_map.extent[1] - grid_map.extent[0]) / grid_map.x_cell_size)
    assert num_cells_x == grid_map.num_cols - 1

    num_cells_y = int(
        (grid_map.extent[3] - grid_map.extent[2]) / grid_map.y_cell_size)
    assert num_cells_y == grid_map.num_rows - 1


def test_xs_to_cols(grid_map):
    xs = np.linspace(
        grid_map.extent[0],
        grid_map.extent[1],
        grid_map.matrix.shape[1],
    )
    actual_cols = grid_map.xs_to_cols(xs)
    expected_cols = np.arange(grid_map.num_cols)
    np.testing.assert_allclose(actual_cols, expected_cols)


def test_ys_to_cols(grid_map):
    ys = np.linspace(
        grid_map.extent[2],
        grid_map.extent[3],
        grid_map.matrix.shape[0],
    )
    actual_rows = grid_map.ys_to_rows(ys)
    expected_rows = np.arange(grid_map.num_rows)
    np.testing.assert_allclose(actual_rows, expected_rows)


def test_get(grid_map):
    x, y = np.ogrid[0:6, 0:7]
    expected = x + y
    xs = np.linspace(
        grid_map.extent[0],
        grid_map.extent[1],
        grid_map.matrix.shape[1],
    )
    ys = np.linspace(
        grid_map.extent[2],
        grid_map.extent[3],
        grid_map.matrix.shape[0],
    )
    xx, yy = np.meshgrid(xs, ys)
    actual = grid_map.get(
        xx.ravel(),
        yy.ravel(),
    ).reshape(*grid_map.matrix.shape)
    np.testing.assert_allclose(actual, expected)


def test_set(grid_map):
    x, y = np.ogrid[1:7, 1:8]
    expected = x + y
    xs = np.linspace(
        grid_map.extent[0],
        grid_map.extent[1],
        grid_map.matrix.shape[1],
    )
    ys = np.linspace(
        grid_map.extent[2],
        grid_map.extent[3],
        grid_map.matrix.shape[0],
    )
    xx, yy = np.meshgrid(xs, ys)
    grid_map.set(xx.ravel(), yy.ravel(), expected.ravel())
    np.testing.assert_allclose(grid_map.matrix, expected)
