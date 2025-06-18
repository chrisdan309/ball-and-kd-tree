import numpy as np
import time
import matplotlib.pyplot as plt
from kd_tree.kd_tree import KDTree as CustomKDTree
from ball_tree.ball_tree import BallTree as CustomBallTree

def run_experiment(dimensions, n_points=1000, n_queries=200):
    times = {
        "dim": [],
        "my_kdtree": [],
        "my_balltree": []
    }

    for dim in dimensions:
        print(f"Dimension: {dim}")
        data = np.random.rand(n_points, dim)
        queries = np.random.rand(n_queries, dim)

        # My KDTree
        custom_kd = CustomKDTree([tuple(p) for p in data])
        start = time.perf_counter()
        for q in queries:
            custom_kd.nearest_neighbor(tuple(q))
        times["my_kdtree"].append(time.perf_counter() - start)

        # My BallTree
        custom_ball = CustomBallTree(data)
        start = time.perf_counter()
        for q in queries:
            custom_ball.nearest_neighbor(q)
        times["my_balltree"].append(time.perf_counter() - start)

        times["dim"].append(dim)

    return times

def plot_results(times):
    plt.figure(figsize=(10, 6))
    plt.plot(times["dim"], times["my_kdtree"], label="My KDTree", marker='o')
    plt.plot(times["dim"], times["my_balltree"], label="My BallTree", marker='o')

    plt.title("Maldición de la Dimensionalidad (Tiempo de Consulta 1-NN)")
    plt.xlabel("Dimensión")
    plt.ylabel("Tiempo total (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tests/results/dimensionality_analysis.png")
    plt.show()

if __name__ == "__main__":
    dims = [2, 5, 10, 20, 30, 40, 50, 70, 100]
    resultados = run_experiment(dims)
    plot_results(resultados)
