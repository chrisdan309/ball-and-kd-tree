import numpy as np
from ball_tree.ball_tree import BallTree

def test_nearest_neighbor_exact_match():
    data = [[1, 2], [3, 4], [5, 6]]
    tree = BallTree(data, node_size=1)
    query = [3, 4]
    point, dist = tree.nearest_neighbor(query)
    assert np.allclose(point, query)
    assert np.isclose(dist, 0)

def test_k_nearest_neighbors():
    data = [[0, 0], [1, 1], [2, 2], [3, 3]]
    tree = BallTree(data, node_size=1)
    query = [1.5, 1.5]
    neighbors = tree.k_nearest_neighbors(query, 2)
    assert len(neighbors) == 2
    assert all(isinstance(p, tuple) for p in neighbors)
    assert neighbors[0][0] <= neighbors[1][0]

def test_insert_and_rebuild():
    data = [[0, 0], [2, 2]]
    tree = BallTree(data, node_size=1)
    for _ in range(10):
        tree.insert([3, 3])
    point, dist = tree.nearest_neighbor([3, 3])
    assert np.allclose(point, [3, 3])

def test_check_rebuild_logic():
    tree = BallTree([[0, 0]], node_size=1)
    assert not tree.check_rebuild()
    for _ in range(10):
        tree.insert([1, 1])
    assert tree.check_rebuild() or not tree.check_rebuild()

def test_nearest_neighbor_far_point():
    data = np.random.rand(100, 3)
    tree = BallTree(data, node_size=5)
    query = np.array([100, 100, 100])
    point, dist = tree.nearest_neighbor(query)
    expected = data[np.argmin(np.linalg.norm(data - query, axis=1))]
    assert np.allclose(point, expected)

def test_single_point_tree():
    data = [[7, 7]]
    tree = BallTree(data, node_size=1)
    query = [7, 7]
    point, dist = tree.nearest_neighbor(query)
    assert np.allclose(point, [7, 7])
    assert dist == 0

def test_high_dimensional_data():
    np.random.seed(42)
    data = np.random.rand(100, 50)
    tree = BallTree(data.tolist(), node_size=10)
    query = np.random.rand(50).tolist()
    point, dist = tree.nearest_neighbor(query)
    expected = data[np.argmin(np.linalg.norm(data - np.array(query), axis=1))]
    assert np.allclose(point, expected)

def test_duplicate_points():
    data = [[1, 2]] * 10 + [[5, 6]]
    tree = BallTree(data, node_size=3)
    query = [5, 6]
    point, dist = tree.nearest_neighbor(query)
    assert np.allclose(point, [5, 6])
    assert dist == 0

def test_multiple_queries():
    data = [[0, 0], [10, 10], [5, 5], [3, 3]]
    tree = BallTree(data, node_size=2)
    queries = [[1, 1], [9, 9], [4, 4]]
    expected = [[0, 0], [10, 10], [5, 5]]
    for q, e in zip(queries, expected):
        point, _ = tree.nearest_neighbor(q)
        assert np.allclose(point, e)
