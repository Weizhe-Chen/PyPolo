import pytest
import numpy as np


def test_init(dubins_car):
    assert np.isclose(0.1, dubins_car.dt)


def test_check_shape_no_exception(dubins_car):
    try:
        valid_state = np.array([0, 0, 0], dtype=np.float64)
        valid_action = np.array([0.1, 0.1], dtype=np.float64)
        dubins_car.check_shape(valid_state, valid_action)
    except ValueError as e:
        pytest.fail(f"check_shape raised an unexpected exception {e}.")


def test_check_shape_with_exception_on_state(dubins_car):
    with pytest.raises(ValueError):
        invalid_state = np.array([0, 0, 0, 0], dtype=np.float64)
        valid_action = np.array([0.1, 0.1], dtype=np.float64)
        dubins_car.check_shape(invalid_state, valid_action)


def test_check_shape_with_exception_on_action(dubins_car):
    with pytest.raises(ValueError):
        valid_state = np.array([0, 0, 0], dtype=np.float64)
        invalid_action = np.array([0.1], dtype=np.float64)
        dubins_car.check_shape(valid_state, invalid_action)


def test_check_dtype_no_exception(dubins_car):
    try:
        valid_state = np.array([0, 0, 0], dtype=np.float64)
        valid_action = np.array([0.1, 0.1], dtype=np.float64)
        dubins_car.check_dtype(valid_state, valid_action)
    except TypeError as e:
        pytest.fail(f"check_dtype raised an unexpected exception {e}.")


def test_check_dtype_with_exception_on_state(dubins_car):
    with pytest.raises(TypeError):
        invalid_state = np.array([0, 0, 0], dtype=np.int64)
        valid_action = np.array([0.1, 0.1], dtype=np.float64)
        dubins_car.check_dtype(invalid_state, valid_action)


def test_check_dtype_with_exception_on_action(dubins_car):
    with pytest.raises(TypeError):
        valid_state = np.array([0, 0, 0], dtype=np.float64)
        invalid_action = np.array([0.1, 0.1], dtype=np.float32)
        dubins_car.check_dtype(valid_state, invalid_action)


def test_normalize_angle(dubins_car):
    angle = -np.pi - np.pi / 2
    actual_normalized_angle = dubins_car.normalize_angle(angle)
    expected_normalized_angle = np.pi / 2
    assert np.isclose(actual_normalized_angle, expected_normalized_angle)
    angle = np.pi + np.pi / 2
    actual_normalized_angle = dubins_car.normalize_angle(angle)
    expected_normalized_angle = -np.pi / 2
    assert np.isclose(actual_normalized_angle, expected_normalized_angle)


def test_step_linear_velocity(dubins_car):
    # Positive linear velocity.
    state = np.array([0, 0, 0], dtype=np.float64)
    action = np.array([1.0, 0.0], dtype=np.float64)
    actual_next_state = dubins_car.step(state, action)
    expected_next_state = np.array([0.1, 0, 0], dtype=np.float64)
    assert np.allclose(actual_next_state, expected_next_state)
    # Zero linear velocity
    state = np.array([0, 0, 0], dtype=np.float64)
    action = np.array([0.0, 0.0], dtype=np.float64)
    actual_next_state = dubins_car.step(state, action)
    expected_next_state = np.array([0.0, 0, 0], dtype=np.float64)
    assert np.allclose(actual_next_state, expected_next_state)
    # Negative linear velocity
    state = np.array([0, 0, 0], dtype=np.float64)
    action = np.array([-1.0, 0.0], dtype=np.float64)
    actual_next_state = dubins_car.step(state, action)
    expected_next_state = np.array([-0.1, 0, 0], dtype=np.float64)
    assert np.allclose(actual_next_state, expected_next_state)


def test_step_angular_velocity(dubins_car):
    # Turn left.
    state = np.array([0, 0, 0], dtype=np.float64)
    action = np.array([0.0, 1.0], dtype=np.float64)
    actual_next_state = dubins_car.step(state, action)
    expected_next_state = np.array([0, 0, 0.1], dtype=np.float64)
    assert np.allclose(actual_next_state, expected_next_state)
    # Turn right.
    state = np.array([0, 0, 0], dtype=np.float64)
    action = np.array([0.0, -1.0], dtype=np.float64)
    actual_next_state = dubins_car.step(state, action)
    expected_next_state = np.array([0, 0, -0.1], dtype=np.float64)
    assert np.allclose(actual_next_state, expected_next_state)
    # Turn left too much
    state = np.array([0, 0, 0], dtype=np.float64)
    action = np.array([0.0, 3.0 * np.pi / 2.0 * 10], dtype=np.float64)
    actual_next_state = dubins_car.step(state, action)
    expected_next_state = np.array([0, 0, -np.pi / 2.0], dtype=np.float64)
    assert np.allclose(actual_next_state, expected_next_state)
    assert -np.pi < actual_next_state[2] < np.pi
    # Turn right too much
    state = np.array([0, 0, 0], dtype=np.float64)
    action = np.array([0.0, -3.0 * np.pi / 2.0 * 10], dtype=np.float64)
    actual_next_state = dubins_car.step(state, action)
    expected_next_state = np.array([0, 0, np.pi / 2.0], dtype=np.float64)
    assert np.allclose(actual_next_state, expected_next_state)
    assert -np.pi < actual_next_state[2] < np.pi


def test_step_linear_angular_velocity(dubins_car):
    state = np.array([0, 0, 0], dtype=np.float64)
    action = np.array([1.0, 1.0], dtype=np.float64)
    actual_next_state = dubins_car.step(state, action)
    expected_next_state = np.array([0.1, 0, 0.1], dtype=np.float64)
    assert np.allclose(actual_next_state, expected_next_state)
