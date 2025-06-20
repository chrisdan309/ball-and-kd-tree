from kd_tree.kd_tree import KDTree
from ball_tree.ball_tree import BallTree

points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
tree = KDTree(points)

target = (9, 2)
print("1-NN:", tree.nearest_neighbor(target))

k = 3
print(f"{k}-NN:", tree.k_nearest_neighbors(target, k))

ball_tree = BallTree(points)
print("Ball Tree 1-NN:", ball_tree.nearest_neighbor(target))

