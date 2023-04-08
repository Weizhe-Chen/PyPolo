import os
import numpy as np
import pytest
import pyvista as pv
from pypolo.utils import TensorMap


@pytest.fixture
def sample_tensor():
    return np.arange(1200).reshape(40, 30, 1)


@pytest.fixture
def sample_origin():
    return np.array([0, 0, 0])


@pytest.fixture
def sample_resolution():
    return 1.0


@pytest.fixture
def tensor_map(sample_tensor, sample_origin, sample_resolution):
    return TensorMap(sample_tensor, sample_origin, sample_resolution)


def test_set_tensor(tensor_map, sample_tensor):
    tensor_map.set_tensor(sample_tensor)
    assert np.array_equal(tensor_map.tensor, sample_tensor)


def test_set_tensor_raises_error(tensor_map):
    with pytest.raises(AssertionError):
        tensor_map.set_tensor(np.random.rand(40, 30))


def test_update_geometry(tensor_map, sample_tensor):
    tensor_map.set_tensor(sample_tensor)
    assert tensor_map.len_y == pytest.approx(40.0)
    assert tensor_map.len_x == pytest.approx(30.0)
    assert tensor_map.min_y == pytest.approx(0.0)
    assert tensor_map.max_y == pytest.approx(40.0)
    assert tensor_map.min_x == pytest.approx(0.0)
    assert tensor_map.max_x == pytest.approx(30.0)


def test_get_values(tensor_map):
    xs = np.array([0.0, 0.0, 29.9, 29.9])
    ys = np.array([0.0, 39.9, 0.0, 39.9])
    expected = np.array([0, 1170, 29, 1199]).reshape(-1, tensor_map.num_layers)
    actual = tensor_map.get_values(xs, ys)
    assert np.array_equal(actual, expected)
    assert actual.shape == (len(xs), tensor_map.num_layers)


def test_get_values_raises_error(tensor_map):
    xs = np.array([30])
    ys = np.array([40])
    with pytest.raises(ValueError):
        tensor_map.get_values(xs, ys)


def test_tensor_map_visualization(tensor_map, render):
    if not render:
        return
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        tensor_map.grid,
        scalars="values",
        lighting=False,
        show_edges=True,
    )
    plotter.add_axes(interactive=True)
    plotter.add_text("Tensor Map", font_size=24)
    plotter.camera.zoom(1.5)
    viewup = [0.5, 0.5, 1]
    path = plotter.generate_orbital_path(viewup=viewup, shift=50)
    save_path = os.path.join(os.path.dirname(__file__), "tensor_map.gif")
    plotter.open_gif(save_path)
    plotter.orbit_on_path(
        path,
        write_frames=True,
        viewup=[0, 0, 1],
        step=0.05,
    )
    plotter.close()
