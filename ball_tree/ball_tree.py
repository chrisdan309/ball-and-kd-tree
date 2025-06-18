import numpy as np
import heapq
import math
from ball_tree.ball_node import BallNode
from typing import List, Tuple

class BallTree:
    def __init__(self, data: List[List[float]], node_size: int = 10):
        self.data = [np.array(d) for d in data]
        self.node_size = node_size
        self.last_insertions = 0
        self.rebuild()

    def _distance2(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return np.sum((p1 - p2) ** 2)

    def _nn_search(self, node: BallNode, target: np.ndarray, best_dist: float, best_point: np.ndarray) -> Tuple[float, np.ndarray]:
        if node is None:
            return best_dist, best_point

        dist_to_center = np.linalg.norm(target - node.center)
        if dist_to_center - node.radius > best_dist:
            return best_dist, best_point

        if node.is_leaf():
            for point in node.points:
                dist = self._distance2(target, point)
                if dist < best_dist:
                    best_dist = dist
                    best_point = point
            return best_dist, best_point

        left_dist = self._distance2(target, node.left.center)
        right_dist = self._distance2(target, node.right.center)
        first, second = (node.left, node.right) if left_dist < right_dist else (node.right, node.left)

        best_dist, best_point = self._nn_search(first, target, best_dist, best_point)
        best_dist, best_point = self._nn_search(second, target, best_dist, best_point)

        return best_dist, best_point

    def nearest_neighbor(self, target: List[float]) -> Tuple[List[float], float]:
        target = np.array(target)
        dist, point = self._nn_search(self.root, target, float('inf'), None)
        return point.tolist(), math.sqrt(dist)

    def _knn_search(self, node: BallNode, target: np.ndarray, k: int, heap: List[Tuple[float, np.ndarray]]):
        if node is None:
            return

        if node.is_leaf():
            for point in node.points:
                dist = self._distance2(target, point)
                heapq.heappush(heap, (-dist, tuple(point)))
                if len(heap) > k:
                    heapq.heappop(heap)
            return

        dist_to_center = np.linalg.norm(target - node.center)
        max_dist = float('inf')
        if heap:
            max_dist = math.sqrt(-heap[0][0])
        else:
            max_dist = float('inf')

        if dist_to_center - node.radius > max_dist:
            return

        left_dist = self._distance2(target, node.left.center)
        right_dist = self._distance2(target, node.right.center)
        first, second = (node.left, node.right) if left_dist < right_dist else (node.right, node.left)

        self._knn_search(first, target, k, heap)
        self._knn_search(second, target, k, heap)

    def k_nearest_neighbors(self, target: List[float], k: int) -> List[Tuple[float, List[float]]]:
        target = np.array(target)
        heap = []
        self._knn_search(self.root, target, k, heap)
        return sorted([(math.sqrt(-d), list(p)) for d, p in heap])

    def insert(self, point: List[float]):
        np_point = np.array(point)
        if len(self.data) > 0 and len(np_point) != len(self.data[0]):
            raise ValueError("Invalid point dimensions.")
        self.data.append(np_point)
        self.last_insertions += 1
        if self.check_rebuild():
            self.rebuild()

    def check_rebuild(self) -> bool:
        n = len(self.data)
        if n == 0:
            return False
        h = math.floor(math.log2(n))
        threshold = 2 ** h
        return self.last_insertions >= threshold

    def rebuild(self):
        self.root = BallNode(np.array(self.data), self.node_size)
        self.last_insertions = 0
