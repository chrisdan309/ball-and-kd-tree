from typing import Tuple, Optional

class KDNode:
    def __init__(self, point: Tuple[float], axis: int, left: Optional['KDNode'] = None, right: Optional['KDNode'] = None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right
