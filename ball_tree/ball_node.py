import numpy as np

class BallNode:
    def __init__(self, points, node_size):
        self.points = points
        self.node_size = node_size
        self.left = None
        self.right = None
        self.center = np.mean(points, axis=0)
        self.radius = np.max(np.linalg.norm(points - self.center, axis=1))        
        if len(points) > node_size:
            self._split_points(node_size)

    def _split_points(self, node_size):
        i = 0
        j = np.argmax(np.linalg.norm(self.points - self.points[i], axis=1))
        k = np.argmax(np.linalg.norm(self.points - self.points[j], axis=1))

        p1 = self.points[j]
        p2 = self.points[k]

        dists_to_p1 = np.linalg.norm(self.points - p1, axis=1)
        dists_to_p2 = np.linalg.norm(self.points - p2, axis=1)

        left_points = self.points[dists_to_p1 < dists_to_p2]
        right_points = self.points[dists_to_p1 >= dists_to_p2]

        if len(left_points) == 0 or len(right_points) == 0:
            return

        self.left = BallNode(left_points, node_size)
        self.right = BallNode(right_points, node_size)

    def is_leaf(self):
        return self.left is None and self.right is None