import parent_import
import numpy as np
from MLibrary.utils import euclidean_distance, sigmoid, accuracy, mse


def test_euclidean_distance_pos_a_b():
    assert euclidean_distance(1, 3) == 2


def test_euclidean_distance_pos_b_a():
    assert euclidean_distance(3, 1) == 2


def test_euclidean_distance_neg_a_b():
    assert euclidean_distance(-3, -2) == 1


def test_euclidean_distance_neg_b_a():
    assert euclidean_distance(-2, -3) == 1


def test_euclidean_distance_neg_a_pos_b():
    assert euclidean_distance(-3, 1) == 4


def test_euclidean_distance_pos_a_neg_b():
    assert euclidean_distance(3, -2) == 5


def test_euclidean_distance_null_a_neg_b():
    assert euclidean_distance(0, -3) == 3


def test_euclidean_distance_null_a_pos_b():
    assert euclidean_distance(0, 5) == 5


def test_euclidean_distance_neg_a_null_b():
    assert euclidean_distance(-2, 0) == 2


def test_euclidean_distance_pos_a_null_b():
    assert euclidean_distance(6, 0) == 6


def test_euclidean_distance_null_a_b():
    assert euclidean_distance(0, 0) == 0


def test_sigmoid_pos():
    assert sigmoid(5) == 0.9933071490757153


def test_sigmoid_neg():
    assert sigmoid(-5) == 0.0066928509242848554


def test_sigmoid_nul():
    assert sigmoid(0) == 0.5


def test_accuracy_full():
    assert accuracy(np.array([1, 1]), np.array([1, 1])) == 1.0


def test_accuracy_half():
    assert accuracy(np.array([1, 0]), np.array([1, 1])) == 0.5


def test_accuracy_nul():
    assert accuracy(np.array([0, 0]), np.array([1, 1])) == 0.0


def test_mse_error():
    assert mse(np.array([0.5, 0.6, 0.7, 0.8]), np.array([0.5, 0.7, 0.6, 1.0])) == 0.014999999999999993


def test_mse_no_error():
    assert mse(np.array([1, 1, 1, 1, ]), np.array([1, 1, 1, 1 ])) == 0.0
