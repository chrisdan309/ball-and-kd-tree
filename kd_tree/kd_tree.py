import heapq
from typing import List, Tuple, Optional
from kd_tree.kd_node import KDNode

class KDTree:
    def __init__(self, points: List[Tuple[float]]):
        self.k = len(points[0]) if points else 0
        self.root = self.build(points, depth=0)

    def build(self, points: List[Tuple[float]], depth: int) -> Optional[KDNode]:
        if not points:
            return None
        axis = depth % self.k
        points.sort(key=lambda x: x[axis])
        median_idx = len(points) // 2
        return KDNode(
            point=points[median_idx],
            axis=axis,
            left=self.build(points[:median_idx], depth + 1),
            right=self.build(points[median_idx + 1:], depth + 1)
        )

    def _distance2(self, p1: Tuple[float], p2: Tuple[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(p1, p2))

    def _nn(self, node: KDNode, target: Tuple[float], depth: int, best: Tuple[float, Tuple[float]]) -> Tuple[float, Tuple[float]]:
        if node is None:
            return best
        axis = node.axis
        dist = self._distance2(target, node.point)
        if dist < best[0]:
            best = (dist, node.point)

        diff = target[axis] - node.point[axis]
        close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
        best = self._nn(close, target, depth + 1, best)

        if diff ** 2 < best[0]:
            best = self._nn(away, target, depth + 1, best)

        return best

    def nearest_neighbor(self, target: Tuple[float]) -> Tuple[float, Tuple[float]]:
        best = (float("inf"), None)
        return self._nn(self.root, target, 0, best)

    def _knn(self, node: KDNode, target: Tuple[float], k: int, heap: List[Tuple[float, Tuple[float]]], depth: int):
        if node is None:
            return
        dist = self._distance2(target, node.point)
        heapq.heappush(heap, (-dist, node.point))
        if len(heap) > k:
            heapq.heappop(heap)

        axis = node.axis
        diff = target[axis] - node.point[axis]
        close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._knn(close, target, k, heap, depth + 1)
        if diff ** 2 < -heap[0][0] or len(heap) < k:
            self._knn(away, target, k, heap, depth + 1)

    def k_nearest_neighbors(self, target: Tuple[float], k: int) -> List[Tuple[float, Tuple[float]]]:
        heap = []
        self._knn(self.root, target, k, heap, 0)
        return sorted([(-d, pt) for d, pt in heap])

    def insert(self, point: Tuple[float]):
        if self.k == 0:
            self.k = len(point)

        if len(point) != self.k:
            raise ValueError("Invalid point dimensions.")

        self.root = self._insert(self.root, point, depth=0)

    def _insert(self, node: Optional[KDNode], point: Tuple[float], depth: int) -> KDNode:
        if node is None:
            return KDNode(point=point, axis=depth % self.k)
            
        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, point, depth + 1)
        else:
            node.right = self._insert(node.right, point, depth + 1)
        return node
