import parent_import
import numpy as np
import pytest
from MLibrary.utils import euclidean_distance, sigmoid, accuracy, mse


@pytest.mark.parametrize(
    'params, expected', [
        ([1, 3], 2),
        ([3, 1], 2),
        ([-3, -2], 1),
        ([-2, -3], 1),
        ([0, 0], 0)
    ]
)
def test_euclidean_distance(params, expected):
    assert euclidean_distance(params[0], params[1]) == expected


@pytest.mark.parametrize(
    'param, expected', [
        (5, 0.9933071490757153),
        (-5, 0.0066928509242848554),
        (0, 0.5)
    ]
)
def test_sigmoid(param, expected):
    assert sigmoid(param) == expected


@pytest.mark.parametrize(
    'params, expected', [
        ([[1, 1], [1, 1]], 1.0),
        ([[1, 0], [1, 1]], 0.5),
        ([[0, 0], [1, 1]], 0.0)
    ]
)
def test_accuracy(params, expected):
    assert accuracy(np.array(params[0]), np.array(params[1])) == expected


def test_mse_error():
    assert mse(np.array([0.5, 0.6, 0.7, 0.8]), np.array([0.5, 0.7, 0.6, 1.0])) == 0.014999999999999993


def test_mse_no_error():
    assert mse(np.array([1, 1, 1, 1, ]), np.array([1, 1, 1, 1])) == 0.0
