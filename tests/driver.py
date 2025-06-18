import numpy as np
import time
import pandas as pd

from kd_tree.kd_tree import KDTree
from ball_tree.ball_tree import BallTree

from sklearn.neighbors import KDTree as SklearnKDTree
from sklearn.neighbors import BallTree as SklearnBallTree

def benchmark(dims, num_points=1000, num_queries=100):
    results = []

    for dim in dims:
        print(f"\nDimensión: {dim}D")

        data = np.random.rand(num_points, dim)
        queries = np.random.rand(num_queries, dim)

        print(f"Mi kd_tree con {dim} dimensiones")
        start = time.perf_counter()
        kd_custom = KDTree([tuple(p) for p in data])
        build_kd_custom = time.perf_counter() - start

        start = time.perf_counter()
        for q in queries:
            kd_custom.nearest_neighbor(tuple(q))
        query_kd_custom = time.perf_counter() - start

        print(f"Mi ball_tree con {dim} dimensiones")
        start = time.perf_counter()
        ball_custom = BallTree(data)
        build_ball_custom = time.perf_counter() - start

        start = time.perf_counter()
        for q in queries:
            ball_custom.nearest_neighbor(q)
        query_ball_custom = time.perf_counter() - start

        print(f"Sklearn kd_tree con {dim} dimensiones")
        start = time.perf_counter()
        kd_sklearn = SklearnKDTree(data)
        build_kd_sklearn = time.perf_counter() - start

        start = time.perf_counter()
        kd_sklearn.query(queries, k=1)
        query_kd_sklearn = time.perf_counter() - start

        print(f"Sklearn ball_tree con {dim} dimensiones")
        start = time.perf_counter()
        ball_sklearn = SklearnBallTree(data)
        build_ball_sklearn = time.perf_counter() - start

        start = time.perf_counter()
        ball_sklearn.query(queries, k=1)
        query_ball_sklearn = time.perf_counter() - start

        results.append({
            "dimension": dim,

            "build_my_kdtree": round(build_kd_custom, 6),
            "query_my_kdtree": round(query_kd_custom, 6),

            "build_my_balltree": round(build_ball_custom, 6),
            "query_my_balltree": round(query_ball_custom, 6),

            "build_sklearn_kdtree": round(build_kd_sklearn, 6),
            "query_sklearn_kdtree": round(query_kd_sklearn, 6),

            "build_sklearn_balltree": round(build_ball_sklearn, 6),
            "query_sklearn_balltree": round(query_ball_sklearn, 6),
        })

    df = pd.DataFrame(results)
    print("\nResultados:")
    print(df.to_string(index=False))

    df.to_csv("tests/results/benchmark_kdtree_balltree_comparison.csv", index=False)
    print("\nResultados guardados tests/results/en benchmark_kdtree_balltree_comparison.csv")

if __name__ == "__main__":
    benchmark([2, 10, 50])
