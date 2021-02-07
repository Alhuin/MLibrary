from MLibrary.utils import euclidean_distance


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
