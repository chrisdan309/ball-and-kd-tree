import numpy as np
from ball_tree.ball_node import BallNode

class BallTree:
    def __init__(self, data, node_size=10):
        self.root = BallNode(np.array(data), node_size)

    def _nn_search(self, node, target, best_dist, best_point):
        if node is None:
            return best_dist, best_point

        dist_to_center = np.linalg.norm(target - node.center)

        if dist_to_center - node.radius > best_dist:
            return best_dist, best_point

        if node.is_leaf():
            for point in node.points:
                dist = np.linalg.norm(target - point)
                if dist < best_dist:
                    best_dist = dist
                    best_point = point
            return best_dist, best_point

        if np.linalg.norm(target - node.left.center) < np.linalg.norm(target - node.right.center):
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        best_dist, best_point = self._nn_search(first, target, best_dist, best_point)
        best_dist, best_point = self._nn_search(second, target, best_dist, best_point)

        return best_dist, best_point

    def nearest_neighbor(self, target):
        dist, point = self._nn_search(self.root, np.array(target), float('inf'), None)
        return point, dist