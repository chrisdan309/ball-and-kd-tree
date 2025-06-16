from kd_tree.kd_tree import KDTree

def test_tree_construction_2d():
    points = [(2.0, 3.0), (5.0, 4.0), (9.0, 6.0)]
    tree = KDTree(points)
    assert tree.k == 2
    assert tree.root is not None
    assert tree.root.point in points

def test_tree_construction_empty():
    tree = KDTree([])
    assert tree.k == 0
    assert tree.root is None

def test_insert_into_empty_tree():
    tree = KDTree([])
    tree.insert((1.0, 2.0))
    assert tree.root is not None
    assert tree.root.point == (1.0, 2.0)

def test_insert_preserves_structure():
    tree = KDTree([(3.0, 3.0)])
    tree.insert((2.0, 4.0))
    tree.insert((4.0, 2.0))
    assert tree.root.left.point == (2.0, 4.0)
    assert tree.root.right.point == (4.0, 2.0)

def test_insert_invalid_dimensions_too_few():
    tree = KDTree([(1.0, 2.0)])
    try:
        tree.insert((1.0,))
        assert False
    except ValueError:
        assert True

def test_insert_invalid_dimensions_too_many():
    tree = KDTree([(1.0, 2.0)])
    try:
        tree.insert((1.0, 2.0, 3.0))
        assert False
    except ValueError:
        assert True

def test_nearest_neighbor_basic():
    points = [(2.0, 3.0), (5.0, 4.0), (9.0, 6.0)]
    tree = KDTree(points)
    dist, pt = tree.nearest_neighbor((9.1, 6.1))
    assert pt == (9.0, 6.0)

def test_nearest_neighbor_after_insert():
    tree = KDTree([(2.0, 3.0)])
    tree.insert((4.0, 3.0))
    dist, pt = tree.nearest_neighbor((4.1, 3.1))
    assert pt == (4.0, 3.0)

def test_k_nearest_neighbors():
    points = [(1, 1), (2, 2), (3, 3), (4, 4)]
    tree = KDTree(points)
    neighbors = tree.k_nearest_neighbors((2.5, 2.5), k=2)
    neighbor_points = [pt for _, pt in neighbors]
    assert (2, 2) in neighbor_points
    assert (3, 3) in neighbor_points
